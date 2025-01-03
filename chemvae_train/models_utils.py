import logging

import torch.nn as nn
import torch
import numpy as np

from chemvae_train.load_params import ChemVAETrainingParams


def add_activation(activation_param):
    # add activation function
    if activation_param == 'tanh':
        return nn.Tanh()
    elif activation_param == 'relu':
        return nn.ReLU()
    elif activation_param == 'softmax':
        return nn.Softmax(dim=2)
    else:
        raise ValueError('Invalid activation function specified in the params file.')


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


def categorical_crossentropy_onehot(pred, target):
    """
    Compute categorical cross-entropy loss for one-hot encoded targets.
    Replicate CategoricalCross-Entropy loss by Keras.

    Args:
        pred: Predicted probabilities, shape (batch_size, num_classes, seq_length).
        target: One-hot encoded true targets, shape (batch_size, num_classes, seq_length).

    Returns:
        Tensor: Mean categorical cross-entropy loss.
    """

    pred = torch.nn.functional.softmax(pred, dim=1)  # Apply softmax along the class dimension
    log_probs = torch.log(pred + 1e-8)  # Avoid log(0)
    # Multiply log_probs with one-hot targets and sum over the class axis
    loss = -torch.sum(target * log_probs, dim=1)  # Sum over the class dimension

    # Mean over batch and sequence
    return loss.mean()