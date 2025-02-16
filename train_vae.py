import csv
import os
from datetime import datetime
from functools import partial

import numpy as np
import torch
import logging

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from chemvae_train.load_params import ChemVAETrainingParams, load_params
from chemvae_train.models import VAEAutoEncoder
from chemvae_train.models_utils import (
    kl_loss,
    WeightAnnealer,
    sigmoid_schedule,
    GPUUsageLogger,
    categorical_accuracy,
    categorical_crossentropy_tf,
)
from chemvae_train.data_utils import DataPreprocessor
from utils.utils import logging_set_up


def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    # Explicitly setting seed to make sure that models created in two processes start from same random weights
    # torch.manual_seed(0)


def load_data(model_fit_batch_size: int, X: np.array):
    """Load the data for the model fit training process."""
    if torch.cuda.is_available():
        sampler_train = DistributedSampler(X, shuffle=True)
    else:
        sampler_train = None

    # set dtype to float32
    X = torch.tensor(X, dtype=torch.float32)

    data_loader = torch.utils.data.DataLoader(
        X,
        batch_size=model_fit_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        sampler=sampler_train,
    )
    return data_loader


def load_multiple_test_loader(model_fit_batch_size: int, data: dict):
    test_loaders = {}
    for key, array in data.items():
        test_loaders[key] = load_data(model_fit_batch_size, array)
    return test_loaders


def load_optimiser(params: ChemVAETrainingParams):
    """Load the optimizer for the training process. Returns a partial function with the learning rate set."""
    if params.optim == "adam":
        optimizer = partial(torch.optim.Adam, lr=params.lr, betas=(params.momentum, 0.999))
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


def save_model(params, vae_model, batch_id, batch_size_per_loop, gpu_id=None):
    if torch.cuda.is_available():
        vae_model = vae_model.module
        if gpu_id != 0:
            return
    if params.vae_weights_file:
        filename = params.vae_weights_file
        chunk_batch_filename = params.vae_weights_file[:-4] + f"_{(batch_id + 1) * batch_size_per_loop}.pth"
        torch.save(vae_model.state_dict(), filename)
        torch.save(vae_model.state_dict(), chunk_batch_filename)
        logging.info(f"Model weights saved to {filename} and {chunk_batch_filename}. \n")
    else:
        today_date = datetime.today().strftime('%Y-%m-%d')
        filename = f"model_weights_{today_date}.pth"
        torch.save(vae_model.state_dict(), filename)
        logging.info(f"Model weights saved to {filename}. \n")


def train(params: ChemVAETrainingParams, gpu_id=0, n_gpus=None):
    """Train the ChemVAE model, the full workflow."""
    # set device to cuda of id = gpu_id if available, else to cpu
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")
    # Load data
    data_preprocessor = DataPreprocessor()
    data_preprocessor.vectorize_data(params)

    chunk_size_per_loop, n_chunks, chunk_start_id = \
        data_preprocessor.get_model_fit_chunk_size_and_starting_chunk_id(params)

    data_preprocessor.generate_training_chunks(params, n_chunks)

    if params.paired_output:
        data_preprocessor.generate_fixed_test_pairs(chunk_size_per_loop, random_state=params.RAND_SEED)
        test_data_dict = data_preprocessor.Xp_test_all
    else:
        test_data_dict = {"Unpaired": data_preprocessor.X_test_all}

    test_loaders_dict = load_multiple_test_loader(
        model_fit_batch_size=params.model_fit_batch_size, data=test_data_dict
    )
    # set up training model
    autoencoder_model = load_model(params).to(device)

    # compile the autoencoder model
    loss_function = categorical_crossentropy_tf
    optimizer = load_optimiser(params)(autoencoder_model.parameters())

    # set up callbacks
    # Initialize the annealer
    kl_weight = params.kl_loss_weight  # Initial weight for KL loss
    weight_annealer = WeightAnnealer(
        schedule=lambda epoch: sigmoid_schedule(
            epoch, slope=params.anneal_sigmod_slope, start=params.vae_annealer_start
        ),
        weight_var=kl_weight,
        weight_orig=kl_weight
    )
    gpu_logger = GPUUsageLogger(print_every=100)

    if torch.cuda.is_available():
        autoencoder_model = DDP(autoencoder_model, device_ids=[gpu_id])

    # ##
    # Training loop - chunk by chunk
    for chunk_id in range(chunk_start_id, n_chunks):
        print(f"Training batch id over model fit func: {chunk_id} out of {n_chunks}")

        # load chunk size data
        X_train_chunk = data_preprocessor.generate_loop_chunk_data_for_model_fit(
            if_paired=params.paired_output,
            current_chunk_id=chunk_id,
        )
        batch_size = params.model_fit_batch_size
        train_loader = load_data(
            model_fit_batch_size=batch_size,
            X=X_train_chunk,
        )

        num_train_samples = len(train_loader.dataset)

        for epoch in range(params.epochs):
            train_results = {"loss": [], "x_pred_loss": [], "kl_loss": [], "categorical_accuracy": []}
            weight_annealer.on_epoch_begin(epoch)

            # for loop over train_loader with both ith batch_idx and ith X data
            for batch_idx, X in enumerate(train_loader):
                autoencoder_model.train()
                optimizer.zero_grad()
                x_true = X.to(device)
                x_pred, z_mean_log_var = autoencoder_model(x_true)
                recon_loss = loss_function(x_pred, x_true)
                kl_div = kl_loss(z_mean_log_var)

                kl_weight = weight_annealer.weight_var  # Dynamically adjust weight
                total_loss = recon_loss + kl_weight * kl_div
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
            print(
                f"Current chunk: {chunk_id}, epoch: {epoch}, \n"
                f"total loss: {total_loss}, reconstruction loss: {recon_loss}, "
                f"kl loss: {kl_div}, kl weight: {kl_weight},"
            )
            logging.info(
                f"Average Train loss: {train_loss}, x_pred_loss: {train_x_pred_loss}, kl_loss: {train_kl_loss}."
                f"accuracy: {train_accuracy}.")

            if params.history_file is not None:
                print("Evaluation start time: ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                # Validation step
                autoencoder_model.eval()  # Set model to evaluation mode

                for key, test_loader in test_loaders_dict.items():
                    val_results = {
                        "val_loss": [], "val_x_pred_loss": [], "val_kl_loss": [], "val_categorical_accuracy": []
                    }
                    num_val_samples = len(test_loader.dataset)

                    with torch.no_grad():  # Disable gradient computation for validation
                        for x_batch in test_loader:
                            x_batch = x_batch.to(device)
                            x_pred_val, z_mean_log_var_val = autoencoder_model(x_batch)
                            recon_loss_val = loss_function(x_pred_val, x_batch)
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
                        "Test set type": key,
                    }

                    with open(params.history_file, "a") as f:
                        writer = csv.DictWriter(f, fieldnames=epoch_results.keys())
                        if epoch == 0:  # Write header only for the first epoch
                            writer.writeheader()
                        writer.writerow(epoch_results)
                    print("Epoch evaluation results: ", epoch_results)
                print("Evaluation end time: ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        logging.info(f"Training batch id {chunk_id} completed. Saving model weights.")
        save_model(params, autoencoder_model, chunk_id, chunk_size_per_loop, gpu_id)
    return


def main(rank: int, world_size: int, training_params: ChemVAETrainingParams):
    ddp_setup(rank=rank, world_size=world_size)
    train(training_params, gpu_id=rank, n_gpus=world_size)
    destroy_process_group()
    return


if __name__ == '__main__':
    # create an instance of ChemVAETrainingParams with default values
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-e", "--exp_file",
    #                     help="experiment file", default="exp.json")
    # parser.add_argument("-d", "--directory",
    #                     help="exp directory", default=None)
    # args = vars(parser.parse_args())

    logger = logging_set_up("zinc_training")
    logging.info("Logging started.")

    current_dir = os.getcwd()
    args = {"exp_file": "./trained_models/chembl204/exp.json", "directory": current_dir}

    if args["directory"] is not None:
        os.chdir(args["directory"])  # change to the directory where the experiment file is located

    training_params = load_params(args['exp_file'])

    # train the model
    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
        logging.info(f"World size: {world_size}")
        mp.spawn(main, args=(world_size, training_params), nprocs=world_size, join=True)
    else:
        train(training_params)

    logging.info("Training completed.")
