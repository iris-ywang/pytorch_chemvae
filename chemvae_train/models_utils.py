import logging

import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

from chemvae_train.load_params import ChemVAETrainingParams


def add_activation(activation_param):
    # add activation function
    if activation_param == 'tanh':
        return nn.Tanh()
    elif activation_param == 'relu':
        return nn.ReLU()
    elif activation_param == 'softmax':
        return nn.Softmax(dim=-1)
    else:
        raise ValueError('Invalid activation function specified in the params file.')


class CustomGRUWithSoftmax(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=True):
        super(CustomGRUWithSoftmax, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=batch_first)

    def forward(self, x):
        # Apply GRU normally
        x, _ = self.gru(x)  # x shape: [batch_size, seq_length, hidden_size]

        # Apply softmax element-wise along the last dimension (features)
        x = torch.softmax(x, dim=-1)

        return x


def kl_loss(z_mean_log_var):
    """
    Compute the KL divergence loss for a Variational Autoencoder.

    Args:
        z_mean_log_var (Tensor): Tensor of shape (batch_size, 2 * latent_dim),
                                 where the first half corresponds to z_mean
                                 and the second half to z_log_var.

    Returns:
        Tensor: The KL divergence loss for the batch.
    """
    z_mean, z_log_var = torch.chunk(z_mean_log_var, 2, dim=1)
    kl_divergence = -0.5 * torch.mean(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
    return kl_divergence


def categorical_accuracy(y_pred, y_true):
    """
    Computes categorical accuracy.
    :param y_pred: Predicted probabilities or logits (batch_size, num_classes)
    :param y_true: True labels (batch_size)
    :return: Accuracy value
    """
    y_pred_classes = torch.argmax(y_pred, dim=-2)
    y_true_classes = torch.argmax(y_true, dim=-2)
    # Get class predictions
    correct = (y_pred_classes == y_true_classes).float()  # Compare with true labels
    return correct.mean().item()  # Average accuracy


class WeightAnnealer:
    def __init__(self, schedule, weight_var, weight_orig, weight_name="vae"):
        """
        Dynamically adjust the KL loss weight during training.

        Args:
            schedule (callable): A function to compute the weight at a given epoch.
            weight_var (float): The variable holding the current weight.
            weight_orig (float): The original (maximum) weight.
            weight_name (str): Name of the weight (for logging).
        """
        self.schedule = schedule
        self.weight_var = weight_var
        self.weight_orig = weight_orig
        self.weight_name = weight_name

    def on_epoch_begin(self, epoch):
        new_weight = self.schedule(epoch)
        self.weight_var = new_weight * self.weight_orig
        print(f"Current {self.weight_name} annealer weight: {self.weight_var}")


# Example sigmoid schedule
def sigmoid_schedule(epoch, slope=1.0, start=10):
    return 1.0 / (1.0 + np.exp(slope * (start - epoch)))


class GPUUsageLogger:
    def __init__(self, print_every=100):
        """
        Logs GPU usage after every `print_every` batches.

        Args:
            print_every (int): Frequency of GPU usage logging.
        """
        self.print_every = print_every

    def on_batch_end(self, batch):
        if batch % self.print_every == 0:
            usage = torch.cuda.memory_reserved(0) / 1e9  # Convert to GB
            allocated = torch.cuda.memory_allocated(0) / 1e9  # Convert to GB
            print(f"Batch {batch}: GPU reserved: {usage:.2f} GB, allocated: {allocated:.2f} GB")


def categorical_crossentropy_tf(pred, target):
    # Normalize predictions along the class axis
    pred = pred / pred.sum(dim=-1, keepdim=True)
    # Clip predictions to avoid log(0)
    pred = torch.clamp(pred, min=1e-7, max=1.0 - 1e-7)
    # Compute log probabilities
    log_probs = torch.log(pred)
    # Compute element-wise loss
    elementwise_loss = -target * log_probs
    # Sum over class dimension and take mean over batch and sequence
    loss = elementwise_loss.sum(dim=-1).mean()
    return loss
#
#
# class ManualGRUWithSoftmax(nn.Module):
#     def __init__(self, input_size, hidden_size):
#         super(ManualGRUWithSoftmax, self).__init__()
#         self.gru_cell = nn.GRUCell(input_size=input_size, hidden_size=hidden_size)
#         self.hidden_size = hidden_size
#
#     def forward(self, x):
#         batch_size, seq_length, _ = x.size()
#         h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)  # Initial hidden state
#
#         outputs = []
#         for t in range(seq_length):
#             h_t = self.gru_cell(x[:, t, :], h_t)  # Process one time step
#             outputs.append(h_t.unsqueeze(1))
#
#         return torch.cat(outputs, dim=1)



class ManualGRUCellWithSoftmax(nn.Module):
    def __init__(self, input_size, hidden_size, activation='tanh', recurrent_activation='sigmoid',
                 use_bias=True, dropout=0.0, recurrent_dropout=0.0, reset_after=True):
        super(ManualGRUCellWithSoftmax, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.use_bias = use_bias
        self.reset_after = reset_after

        # Weight matrices
        self.kernel = nn.Parameter(torch.Tensor(input_size, 3 * hidden_size))
        self.recurrent_kernel = nn.Parameter(torch.Tensor(hidden_size, 3 * hidden_size))
        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(3 * hidden_size))
        else:
            self.register_parameter('bias', None)

        # Dropout masks
        self.dropout = nn.Dropout(dropout)
        self.recurrent_dropout = nn.Dropout(recurrent_dropout)

        # Initialization
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.kernel)
        nn.init.orthogonal_(self.recurrent_kernel)
        if self.use_bias:
            nn.init.zeros_(self.bias)

    def forward(self, x, h_tm1):
        # Apply dropout to input
        x = self.dropout(x)

        # Compute x * kernel (input projection)
        matrix_x = torch.matmul(x, self.kernel)
        if self.use_bias:
            matrix_x += self.bias
        x_z, x_r, x_h = torch.split(matrix_x, self.hidden_size, dim=-1)

        # Compute h_tm1 * recurrent_kernel (hidden state projection)
        matrix_inner = torch.matmul(h_tm1, self.recurrent_kernel)
        recurrent_z, recurrent_r, recurrent_h = torch.split(matrix_inner, self.hidden_size, dim=-1)

        # Compute gates
        z = F.sigmoid(x_z + recurrent_z)  # Update gate
        r = F.sigmoid(x_r + recurrent_r)  # Reset gate

        # Compute candidate hidden state
        if self.reset_after:
            recurrent_h = r * recurrent_h
        hh = self.apply_activation(x_h + recurrent_h)

        # Final hidden state
        h = z * h_tm1 + (1 - z) * hh
        return h

    def apply_activation(self, x):
        if self.activation == 'tanh':
            return torch.tanh(x)
        elif self.activation == 'relu':
            return F.relu(x)
        elif self.activation == 'softmax':
            return F.softmax(x, dim=-1)
        elif self.activation is None:
            return x
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")


class ManualGRUWithSoftmax(nn.Module):
    def __init__(self, input_size, hidden_size, activation='tanh'):
        super(ManualGRUWithSoftmax, self).__init__()
        self.hidden_size = hidden_size
        self.gru_cell = ManualGRUCellWithSoftmax(input_size, hidden_size, activation=activation)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)  # Initial hidden state

        outputs = []
        for t in range(seq_length):
            h_t = self.gru_cell(x[:, t, :], h_t)  # Process one time step
            outputs.append(h_t.unsqueeze(1))

        return torch.cat(outputs, dim=1)

