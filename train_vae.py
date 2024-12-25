import csv
import os
from datetime import datetime
from functools import partial

import numpy as np
import torch
import logging
from torch import nn

from chemvae_train.load_params import ChemVAETrainingParams, load_params
from chemvae_train.models import VAEAutoEncoder
from chemvae_train.models_utils import (
    kl_loss,
    WeightAnnealer,
    sigmoid_schedule,
    GPUUsageLogger,
    categorical_accuracy,
)
from chemvae_train.data_utils import DataPreprocessor


def load_data(model_fit_batch_size: int, X_train: np.array, X_test: np.array):
    """Load the data for the model fit training process."""

    # Swap 2nd and 3rd axis to match the input shape of the model
    X_train = np.swapaxes(X_train, 1, 2)
    X_test = np.swapaxes(X_test, 1, 2)

    train_loader = torch.utils.data.DataLoader(
        X_train,
        batch_size=model_fit_batch_size,
        shuffle=False,
        num_workers=2,
    )

    test_loader = torch.utils.data.DataLoader(
        X_test,
        batch_size=model_fit_batch_size,
        shuffle=False,
        num_workers=2,
    )
    return train_loader, test_loader


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


def load_model(params: ChemVAETrainingParams, evaluating=False):
    """Load the model for the training process."""
    autoencoder_model = VAEAutoEncoder(params)

    if params.reload_model or evaluating:
        logging.info(f"Loading data from {params.vae_weights_file}")
        autoencoder_model.load_state_dict(torch.load(params.vae_weights_file))
    else:
        logging.info("Initializing a new set of model weights...")
    return autoencoder_model


def save_model(params, vae_model, batch_id, batch_size_per_loop):
    if params.vae_weights_file:
        filename = params.vae_weights_file
        chunk_batch_filename = params.vae_weights_file[:-3] + f"_{(batch_id + 1) * batch_size_per_loop}.pth"
        torch.save(vae_model.state_dict(), filename)
        torch.save(vae_model.state_dict(), chunk_batch_filename)
        logging.info(f"Model weights saved to {filename} and {chunk_batch_filename}. \n")
    else:
        today_date = datetime.today().strftime('%Y-%m-%d')
        filename = f"model_weights_{today_date}.pth"
        torch.save(vae_model.state_dict(), filename)
        logging.info(f"Model weights saved to {filename}. \n")


def train(params: ChemVAETrainingParams):
    """Train the ChemVAE model, the full workflow."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load data
    data_preprocessor = DataPreprocessor()
    data_preprocessor.vectorize_data(params)

    chunk_size_per_loop, n_chunks, chunk_start_id = \
        data_preprocessor.get_model_fit_chunk_size_and_starting_chunk_id(params)

    # set up training model
    autoencoder_model = load_model(params).to(device)

    # compile the autoencoder model
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

    # ##
    # Training loop - chunk by chunk
    for chunk_id in range(chunk_start_id, n_chunks):
        logging.info(f"Training batch id over model fit func: {chunk_id} out of {n_chunks}")

        # load chunk size data
        X_train_chunk, X_test_chunk = data_preprocessor.generate_loop_chunk_data_for_model_fit(
            if_paired=params.paired_output,
            current_chunk_id=chunk_id,
            n_chunks=n_chunks,
            chunk_size=chunk_size_per_loop,
        )
        train_loader, test_loader = load_data(
            model_fit_batch_size=params.model_fit_batch_size,
            X_train=X_train_chunk,
            X_test=X_test_chunk
        )

        train_results = {"loss": [], "x_pred_loss": [], "kl_loss": [], "categorical_accuracy": []}
        num_train_samples = len(train_loader.dataset)

        for epoch in range(params.epochs):
            weight_annealer.on_epoch_begin(epoch)

            # for loop over train_loader with both ith batch_idx and ith X data
            for batch_idx, X in enumerate(train_loader):
                autoencoder_model.train()
                optimizer.zero_grad()
                x_true = X.to(device)
                x_pred, z_mean_log_var = autoencoder_model(x_true)
                recon_loss = loss_function(x_pred, x_true)
                kl_div = kl_loss(z_mean_log_var)

                total_loss = recon_loss + kl_div
                total_loss.backward()
                optimizer.step()

                gpu_logger.on_batch_end(batch_idx)

                # Accumulate losses
                train_results["loss"].append(total_loss.item() * len(X))  # Scaled by batch size
                train_results["x_pred_loss"].append(recon_loss.item() * len(X))
                train_results["kl_loss"].append(kl_div.item() * len(X))
                train_results["categorical_accuracy"].append(categorical_accuracy(x_pred, x_true))

            # Compute epoch-level losses (mean per sample)
            train_loss = sum(train_results["loss"]) / num_train_samples
            train_x_pred_loss = sum(train_results["x_pred_loss"]) / num_train_samples
            train_kl_loss = sum(train_results["kl_loss"]) / num_train_samples
            train_accuracy = sum(train_results["categorical_accuracy"]) / len(train_results["categorical_accuracy"])
            logging.info(f"Current chunk: {chunk_id}, epoch: {epoch}, loss: {train_loss}.")

            if params.history_file is not None:

                # Validation step
                autoencoder_model.eval()  # Set model to evaluation mode
                val_results = {
                    "val_loss": [], "val_x_pred_loss": [], "val_kl_loss": [], "val_categorical_accuracy": []
                }
                num_val_samples = len(test_loader.dataset)

                with torch.no_grad():  # Disable gradient computation for validation
                    for x_batch in test_loader:
                        x_batch = x_batch.to(device)
                        x_pred_val, z_mean_log_var_val = autoencoder_model(x_batch)
                        recon_loss_val = loss_function(x_pred_val, torch.tensor(x_batch))
                        kl_div_val = kl_loss(z_mean_log_var_val)
                        total_loss_val = recon_loss_val + kl_div_val

                        # Accumulate losses
                        val_results["val_loss"].append(total_loss_val.item() * len(x_batch))
                        val_results["val_x_pred_loss"].append(recon_loss_val.item() * len(x_batch))
                        val_results["val_kl_loss"].append(kl_div_val.item() * len(x_batch))
                        val_results["val_categorical_accuracy"].append(categorical_accuracy(x_pred_val, x_batch))

                # Compute epoch-level validation losses
                val_loss = sum(val_results["val_loss"]) / num_val_samples
                val_x_pred_loss = sum(val_results["val_x_pred_loss"]) / num_val_samples
                val_kl_loss = sum(val_results["val_kl_loss"]) / num_val_samples
                val_accuracy = sum(val_results["val_categorical_accuracy"]) / len(val_results["val_categorical_accuracy"])

                # Prepare data to be logged in history csv file
                epoch_results = {
                    "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "epoch": epoch,
                    "loss": train_loss,
                    "val_loss": val_loss,
                    "val_x_pred_loss": val_x_pred_loss,
                    "val_kl_loss": val_kl_loss,
                    "val_categorical_accuracy": val_accuracy,
                    "annuealer_weight": weight_annealer.weight_var,
                    "x_pred_loss": train_x_pred_loss,
                    "kl_loss": train_kl_loss,
                    "categorical_accuracy": train_accuracy,
                }

                with open(params.history_file, "a") as f:
                    writer = csv.DictWriter(f, fieldnames=epoch_results.keys())
                    if epoch == 0:  # Write header only for the first epoch
                        writer.writeheader()
                    writer.writerow(epoch_results)

        logging.info(f"Training batch id {chunk_id} completed. Saving model weights.")
        save_model(params, autoencoder_model, chunk_id, chunk_size_per_loop)

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
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
        filename=f"loggings_on_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt",
    )
    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    logger = logging.getLogger(__name__)


    logging.info("Logging started.")

    current_dir = os.getcwd()
    args = {"exp_file": "./trained_models/zinc/exp.json", "directory": current_dir}

    if args["directory"] is not None:
        os.chdir(args["directory"])  # change to the directory where the experiment file is located

    training_params = load_params(args['exp_file'])

    # train the model
    losses, outputs = train(training_params)
    print(losses, outputs)