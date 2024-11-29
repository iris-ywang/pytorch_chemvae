import torch
from torch import nn

from chemvae_train.models import VAEAutoEncoder
from chemvae_train.load_params import ChemVAETrainingParams, load_params
from chemvae_train.models_utils import kl_loss, WeightAnnealer, sigmoid_schedule, GPUUsageLogger

import numpy as np


if __name__ == '__main__':
    random_seeds = np.random.seed(10)
    X_np = np.random.randint(2, size=(200, 35, 120)).astype("float32")

    train_loader = torch.utils.data.DataLoader(
        dataset=X_np,
        batch_size=100,
        shuffle=False,
    )

    # create an instance of ChemVAETrainingParams with default values
    training_params = load_params()

    # create an instance of Autoencoder with the training parameters
    # autoencoder_model = VAEAutoEncoder(training_params).cuda(gpu)
    autoencoder_model = VAEAutoEncoder(training_params)

    # compile the autoencoder model
    ## loss function of the autoencoder model is set to 'categorical_crossentropy'. Use the Pytorch loss for categorical_crossentropy:
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        autoencoder_model.parameters(), lr=training_params.lr
    )

    # set up callbacks
    # Initialize the annealer
    kl_weight = 1.0  # Initial weight for KL loss
    weight_annealer = WeightAnnealer(
        schedule=lambda epoch: sigmoid_schedule(epoch, slope=1.0, start=10),
        weight_var=kl_weight,
        weight_orig=1.0
    )
    gpu_logger = GPUUsageLogger(print_every=50)

    outputs = []
    losses = []
    for epoch in range(training_params.epochs):
        # for loop over train_loader with both ith batch_idx and ith X data
        for batch_idx, X in enumerate(train_loader):
            weight_annealer.on_epoch_begin(epoch)

            optimizer.zero_grad()
            x_pred, z_mean_log_var = autoencoder_model(torch.tensor(X))
            recon_loss = loss_function(x_pred, torch.tensor(X))
            kl_div = kl_loss(z_mean_log_var)

            total_loss = recon_loss + kl_div
            total_loss.backward()
            optimizer.step()

            gpu_logger.on_batch_end(batch_idx)

        losses.append(total_loss)
        outputs.append((epoch, X, x_pred,))





