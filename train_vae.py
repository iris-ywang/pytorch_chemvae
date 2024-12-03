import os
from datetime import datetime
from functools import partial

import numpy as np
import torch
import logging
from torch import nn

from chemvae_train.models import VAEAutoEncoder
from chemvae_train.load_params import ChemVAETrainingParams, load_params
from chemvae_train.models_utils import kl_loss, WeightAnnealer, sigmoid_schedule, GPUUsageLogger
from chemvae_train.data_utils import DataPreprocessor


def load_data(model_fit_batch_size: int, X_train: np.array):
    """Load the data for the model fit training process."""

    # Swap 2nd and 3rd axis to match the input shape of the model
    X_train = np.swapaxes(X_train, 1, 2)

    train_loader = torch.utils.data.DataLoader(
        X_train,
        batch_size=model_fit_batch_size,
        shuffle=False,
        num_workers=2,
    )
    return train_loader


def load_optimiser(params: ChemVAETrainingParams):
    """Load the optimizer for the training process. Returns a partial function with the learning rate set."""
    if params.optim == "adam":
        optimizer = partial(torch.optim.Adam, lr=params.lr)
    elif params.optim == "rmsprop":
        optimizer = partial(torch.optim.RMSprop, lr=params.lr, rho=params.momentum)
    elif params.optim == "sgd":
        optimizer = partial(torch.optim.SGD, lr=params.lr, momentum=params.momentum)
    else:
        raise NotImplemented("Please define valid optimizer")
    return optimizer


def load_model(params: ChemVAETrainingParams):
    """Load the model for the training process."""
    autoencoder_model = VAEAutoEncoder(params)

    if params.reload_model:
        autoencoder_model.load_state_dict(torch.load(params.vae_weights_file))
        
    return autoencoder_model


def save_model(params, vae_model, batch_id, batch_size_per_loop):
    if params.vae_weights_file:
        filename = params.vae_weights_file
        chunk_batch_filename = params.vae_weights_file[:-3] + f"_{(batch_id + 1) * batch_size_per_loop}.h5"
        torch.save(vae_model.state_dict(), filename)
        torch.save(vae_model.state_dict(), chunk_batch_filename)
        logging.info(f"Model weights saved to {filename} and {chunk_batch_filename}.")
    else:
        today_date = datetime.today().strftime('%Y-%m-%d')
        filename = f"model_weights_{today_date}.pth"
        torch.save(vae_model.state_dict(), filename)
        logging.info(f"Model weights saved to {filename}.")


def train(params: ChemVAETrainingParams):
    # Load data
    data_preprocessor = DataPreprocessor()
    X_train_all, X_test_all = data_preprocessor.vectorize_data(params)

    chunk_size_per_loop, n_chunks, chunk_start_id = \
        data_preprocessor.get_model_fit_chunk_size_and_starting_chunk_id(params)

    for batch_id in range(chunk_start_id, n_chunks):
        logging.info(f"Training batch id over model fit func: {batch_id} out of {n_chunks}")

        # load chunk size data
        X_train_chunk, X_test_chunk = data_preprocessor.generate_loop_chunk_data_for_model_fit(
            if_paired=params.paired_output,
            current_chunk_id=batch_id,
            n_chunks=n_chunks,
            chunk_size=chunk_size_per_loop,
        )
        train_loader = load_data(model_fit_batch_size=params.loop_over_fit_batch_size, X_train=X_train_chunk)

        # set up training model
        autoencoder_model = load_model(params)

        # compile the autoencoder model
        ## loss function of the autoencoder model is set to 'categorical_crossentropy'.
        # Use the Pytorch loss for categorical_crossentropy:
        loss_function = nn.CrossEntropyLoss()
        optimizer = load_optimiser(params)(autoencoder_model.parameters())

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
        for epoch in range(params.epochs):
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

        logging.info(f"Training batch id {batch_id} completed. Saving model weights.")
        save_model(params, autoencoder_model, batch_id, chunk_size_per_loop)

    return losses, outputs


if __name__ == '__main__':
    # create an instance of ChemVAETrainingParams with default values
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-e", "--exp_file",
    #                     help="experiment file", default="exp.json")
    # parser.add_argument("-d", "--directory",
    #                     help="exp directory", default=None)
    # args = vars(parser.parse_args())

    # config logging to be compatible with the pytorch
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.info("Logging started")

    current_dir = os.getcwd()
    args = {"exp_file": "./trained_models/zinc/exp.json", "directory": current_dir}

    if args["directory"] is not None:
        args['exp_file'] = os.path.join(args["directory"], args['exp_file'])

    training_params = load_params(args['exp_file'])

    # train the model
    losses, outputs = train(training_params)
    print(losses, outputs)