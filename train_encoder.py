import csv
import os
from datetime import datetime
from functools import partial

import numpy as np
import torch
import logging

from torch import nn, Tensor
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import TensorDataset

from chemvae_train.fp_models import FPEncoderToDeltaY
from chemvae_train.load_params import ChemVAETrainingParams, load_params

from chemvae_train.data_utils import DataPreprocessor
from utils.utils import (
    logging_set_up,
)

from train_vae import (
    save_model,
    load_optimiser,
    load_multiple_test_loader,
    ddp_setup,
    save_optimiser,
)


def load_data_from_tensor(model_fit_batch_size: int, X: TensorDataset):
    """Load the data for the model fit training process."""
    if torch.cuda.is_available():
        sampler_train = DistributedSampler(X, shuffle=True)
    else:
        sampler_train = None

    data_loader = torch.utils.data.DataLoader(
        X,
        batch_size=model_fit_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        sampler=sampler_train,
    )
    return data_loader


def load_model(params: ChemVAETrainingParams, evaluating=False):
    """Load the model for the training process."""
    model = FPEncoderToDeltaY(params)

    if params.reload_model or evaluating:

        if params.pre_trained_weights_file is not None and params.loop_over_fit_batch_id == 0 and not evaluating:
            weights_path = params.pre_trained_weights_file
        else:
            weights_path = params.vae_weights_file

        logging.info(f"Loading data from {weights_path}")
        # autoencoder_model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        model.load_state_dict(torch.load(weights_path))
    else:
        print("Initializing a new set of model weights...")
    return model


def get_tensors_for_cross_entropy_loss(y_pred: Tensor, y_true: Tensor):
    # Map torch.sign outputs to valid class indices
    y_pred_sign = torch.sign(y_pred)  # Values: -1, 0, 1
    y_true_sign = torch.sign(y_true)  # Values: -1, 0, 1

    # Map {-1, 0, 1} to {0, 1, 2}
    y_pred_mapped = (y_pred_sign + 1).long()
    y_true_mapped = (y_true_sign + 1).long()

    # Ensure y_pred is reshaped correctly for Cross-Entropy Loss
    # y_pred should have shape [batch_size, num_classes]
    y_pred_logits = torch.stack([
        (y_pred_sign == -1).float(),
        (y_pred_sign == 0).float(),
        (y_pred_sign == 1).float()
    ], dim=1)
    return y_pred_logits, y_true_mapped


def weighted_total_loss(mse_loss, sign_loss, mse_weight=0.3, sign_loss_scalar=100):
    return mse_weight * mse_loss + (1 - mse_weight) * (sign_loss * sign_loss_scalar)


def train(params: ChemVAETrainingParams):
    """Train the ChemVAE model, the full workflow."""
    # set device to cuda of id = gpu_id if available, else to cpu
    if torch.cuda.is_available():
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
    else:
        local_rank = None
        global_rank = "None"
    device = torch.device(f"cuda:{local_rank} out of {global_rank}" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    # Load data
    data_preprocessor = DataPreprocessor()
    data_preprocessor.vectorize_data(params)

    chunk_size_per_loop, n_chunks, chunk_start_id = \
        data_preprocessor.get_model_fit_chunk_size_and_starting_chunk_id(params)

    data_preprocessor.generate_training_chunks(params, n_chunks)

    total_global_epochs = params.epochs
    epoch_start_id = params.epochs_start_idx
    logging.info(f"Total number of epochs: {total_global_epochs}. Starting from epoch: {epoch_start_id}.")

    if params.paired_output:
        data_preprocessor.generate_fixed_test_pairs(chunk_size_per_loop, random_state=params.RAND_SEED)
        test_data_dict = data_preprocessor.Xp_test_all
    else:
        test_data_dict = {"Unpaired": data_preprocessor.X_test_all}

    test_loaders_dict = load_multiple_test_loader(
        model_fit_batch_size=params.model_fit_batch_size, data=test_data_dict
    )  # TODO
    # set up training model
    oneway_model = load_model(params).to(device)

    # compile the single way model which compresses the pair of FP first and then forward to the y values.
    # using MSE loss for the y values
    loss_function_mse = nn.MSELoss(reduction="sum")
    loss_function_crossentropy = nn.CrossEntropyLoss()

    optimizer = load_optimiser(params)(oneway_model.parameters())
    if params.optimiser_file and os.path.exists(params.optimiser_file):
        optimizer.load_state_dict(torch.load(params.optimiser_file))
        logging.info(f"Optimizer state loaded from {params.optimiser_file}.")

    torch.nn.utils.clip_grad_norm_(oneway_model.parameters(), max_norm=1.0)

    # set up callbacks
    # Initialize the annealer
    # kl_weight = params.kl_loss_weight  # Initial weight for KL loss
    # weight_annealer = WeightAnnealer(
    #     schedule=lambda epoch: sigmoid_schedule(
    #         epoch,
    #         slope=30 / total_global_epochs,
    #         start=total_global_epochs / 2.5,
    #     ),
    #     weight_var=kl_weight,
    #     weight_orig=kl_weight
    # )

    if torch.cuda.is_available():
        oneway_model = DDP(oneway_model, device_ids=[local_rank])

    # ##
    # Training loop - chunk by chunk
    for epoch in range(epoch_start_id, total_global_epochs):
        print(f"Training epoch {epoch} out of {total_global_epochs}.")
        # weight_annealer.on_epoch_begin(epoch)

        for chunk_id in range(chunk_start_id, n_chunks):
            print(f"Training batch id over model fit func: {chunk_id} out of {n_chunks}")

            # load chunk size data
            data_preprocessor.X_all = None  # clear memory
            X_train_chunk = data_preprocessor.generate_loop_chunk_data_for_model_fit(
                if_paired=params.paired_output,
                current_chunk_id=chunk_id,
                if_required_y=params.if_fp_one_way,
            )

            if params.if_fp_one_way:
                # Unpack the tuple
                X, y = X_train_chunk
                X_train_chunk = TensorDataset(Tensor(X), Tensor(y))

            batch_size = params.model_fit_batch_size
            train_loader = load_data_from_tensor(
                model_fit_batch_size=batch_size,
                X=X_train_chunk,
            )

            num_train_samples = len(train_loader.dataset)

            train_results = {"loss": [], "y_pred_mse": [], "y_pred_sign": []}

            # for loop over train_loader with both ith batch_idx and ith X data
            for batch_idx, X in enumerate(train_loader):
                oneway_model.train()
                optimizer.zero_grad()
                y_true = X[1].to(device)
                x_true = X[0].to(device)
                y_pred = oneway_model(x_true)


                mse_loss = loss_function_mse(y_pred, y_true)
                y_pred_sign_prob, y_true_sign = get_tensors_for_cross_entropy_loss(y_pred, y_true)
                sign_loss = loss_function_crossentropy(y_pred_sign_prob, y_true_sign)

                total_loss = weighted_total_loss(mse_loss, sign_loss)

                total_loss.backward()
                optimizer.step()

                # Accumulate losses
                train_results["loss"].append(total_loss.item() * len(X))  # Scaled by batch size
                train_results["y_pred_mse"].append(mse_loss.item() * len(X))
                train_results["y_pred_sign"].append(sign_loss.item() * len(X))

            train_loss = sum(train_results["loss"]) / num_train_samples
            train_y_pred_mse = sum(train_results["y_pred_mse"]) / num_train_samples
            train_y_pred_sign = sum(train_results["y_pred_sign"]) / num_train_samples
            print(
                f"Current chunk: {chunk_id}, epoch: {epoch}, gpu: {global_rank}: \n "
                f"Average Train loss: {train_loss}, train_y_pred_mse: {train_y_pred_mse}, "
                f"train_y_pred_sign: {train_y_pred_sign}. ")

            if params.history_file is not None:
                print("Evaluation start time: ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                # Validation step
                oneway_model.eval()  # Set model to evaluation mode

                for key, test_loader in test_loaders_dict.items():
                    val_results = {
                        "val_loss": [], "val_y_pred_sign": [], "val_y_pred_sign": [],
                    }
                    num_val_samples = len(test_loader.dataset)

                    with torch.no_grad():  # Disable gradient computation for validation

                        for x_batch in test_loader:
                            y_true_val = x_batch[1].to(device)
                            x_true_val = x_batch[0].to(device)
                            y_pred_val = oneway_model(x_true_val)

                            mse_loss_val = loss_function_mse(y_pred_val, y_true_val)
                            y_pred_sign_prob_val, y_true_sign_val = get_tensors_for_cross_entropy_loss(y_pred_val, y_true_val)
                            sign_loss_val = loss_function_crossentropy(y_pred_sign_prob_val, y_true_sign_val)
                            total_loss_val = weighted_total_loss(mse_loss_val, sign_loss_val)

                            # Accumulate losses
                            val_results["val_loss"].append(total_loss_val.item() * len(x_batch))
                            val_results["y_pred_mse"].append(mse_loss_val.item() * len(x_batch))
                            val_results["y_pred_sign"].append(sign_loss_val.item() * len(x_batch))

                    # Compute epoch-level validation losses
                    val_loss = sum(val_results["val_loss"]) / num_val_samples
                    val_y_mse_loss = sum(val_results["y_pred_mse"]) / num_val_samples
                    val_y_sign_loss = sum(val_results["y_pred_sign"]) / num_val_samples

                    # Prepare data to be logged in history csv file
                    epoch_results = {
                        "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "epoch": epoch,
                        "chunk": chunk_id,
                        "loss": train_loss,
                        "val_loss": val_loss,
                        "val_y_pred_mse": val_y_mse_loss,
                        "val_y_pred_sign": val_y_sign_loss,
                        "y_pred_mse": train_y_pred_mse,
                        "y_pred_sign": train_y_pred_sign,
                        "Test set type": key,
                    }
                with open(params.history_file, "a") as f:
                    writer = csv.DictWriter(f, fieldnames=epoch_results.keys())
                    if epoch == 0:  # Write header only for the first epoch
                        writer.writeheader()
                    writer.writerow(epoch_results)
                print(f"Epoch-level evaluation results on test data type {key}: ", epoch_results)
            print("Evaluation end time: ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        logging.info(f"Training batch id {chunk_id} completed. Saving model weights.")
        save_model(params, oneway_model, chunk_id, chunk_size_per_loop, global_rank)
        save_optimiser(params, optimizer, global_rank)

        # clear memory
        del train_loader
        del X_train_chunk

    # delete memory intensive variable
    del data_preprocessor
    return


def main(training_params: ChemVAETrainingParams, logging_filename_prefix=None):
    logger = logging_set_up(logging_filename_prefix)
    logging.info("Logging started.")

    ddp_setup()
    train(training_params)
    destroy_process_group()
    return


def run_train_vae(exp_file_path):
    # create an instance of ChemVAETrainingParams with default values
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-e", "--exp_file",
    #                     help="experiment file", default="exp.json")
    # parser.add_argument("-d", "--directory",
    #                     help="exp directory", default=None)
    # args = vars(parser.parse_args())

    current_dir = os.getcwd()
    args = {"exp_file": exp_file_path, "directory": current_dir}  # check

    if args["directory"] is not None:
        os.chdir(args["directory"])  # change to the directory where the experiment file is located

    training_params = load_params(args['exp_file'])
    logging_prefix_filename = training_params.name

    # train the model
    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
        logging.info(f"World size: {world_size}. Training with Torchrun.")
        main(training_params, logging_prefix_filename)
        # torchrun --standalone --nproc_per_node=3 train_vae.py
    else:
        logger = logging_set_up(logging_prefix_filename)  # check
        logging.info("Logging started.")

        train(training_params)

    logging.info("Training completed.")

if __name__ == '__main__':
    run_train_vae(exp_file_path="./trained_models/chembl4016/exp_oneway.json")
