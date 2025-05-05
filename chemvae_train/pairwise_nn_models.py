import os
import logging

import numpy as np
import torch.multiprocessing as mp

from torch.distributed import destroy_process_group
from torch.utils.data import TensorDataset

from chemvae_train.data_utils import DataPreprocessor, get_x_and_y_from_paired_data_in_vstacked_shape
from model_evaluations.vae_utils import get_torch_of_eval_data
from train_encoder import load_model, load_data_from_tensor, get_tensors_for_cross_entropy_loss, weighted_total_loss
from train_vae import load_optimiser, ddp_setup
from chemvae_train.load_params import ChemVAETrainingParams

import torch
import torch.nn as nn

from utils.utils import logging_set_up


class ChEMBLToDeltaYNN:

    def __init__(self, params: ChemVAETrainingParams):
        self.params = params
        self.optimizer = None
        self.model = None
        logger = logging_set_up()  # check

    @staticmethod
    def make_pairs_from_all_data_and_pair_ids(data: np.array, pair_ids: list):
        """pair_ids is a list of tuples, each tuple contains two indices of the data array.
        Create two sub-arrays, one with the first tuple item from all tuples, and the
        other with the second tuple item from all tuples.
        Then, use these two sub-arrays to create pairs of data using data_processor.make_combination_pairs.
        """
        X = DataPreprocessor.make_pair_array_from_pair_ids(data, pair_ids)
        X, Y = get_x_and_y_from_paired_data_in_vstacked_shape(X)
        return X, Y

    def fit(self, X, y):
        if torch.cuda.is_available():
            world_size = torch.cuda.device_count()
            logging.info(f"World size: {world_size}. Training with Torchrun.")
            mp.spawn(
                self._mp_train_wrapper, args=(world_size, X, y),
                nprocs=world_size, join=True
            )
        else:
            logging.info("No GPU available. Using CPU.")
            self.train(X, y)
        return self

    def _mp_train_wrapper(self, rank: int, world_size: int, X, y):
        ddp_setup(rank=rank, world_size=world_size)
        self.train(X, y, gpu_id=rank)
        destroy_process_group()

    def train(self, X, y, gpu_id=0):
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        logging.info(f"Device: {device}")

        params = self.params

        if self.model is None:
            oneway_model = load_model(params).to(device)
        else:
            oneway_model = self.model.to(device)

        if self.optimizer is None:
            optimizer = load_optimiser(params)(oneway_model.parameters())
            if params.optimiser_file and os.path.exists(params.optimiser_file):
                optimizer.load_state_dict(torch.load(params.optimiser_file))
                logging.info(f"Optimizer state loaded from {params.optimiser_file}.")
        else:
            optimizer = self.optimizer

        # compile the single way model which compresses the pair of FP first and then forward to the y values.
        # using MSE loss for the y values
        self.loss_function_mse = nn.MSELoss(reduction="sum")
        self.loss_function_crossentropy = nn.CrossEntropyLoss()
        torch.nn.utils.clip_grad_norm_(oneway_model.parameters(), max_norm=1.0)

        total_global_epochs = self.params.epochs
        epoch_start_id = self.params.epochs_start_idx
        batch_size = self.params.model_fit_batch_size

        Xy_train = TensorDataset(torch.Tensor(X), torch.Tensor(y))
        train_loader = load_data_from_tensor(
            model_fit_batch_size=batch_size,
            X=Xy_train,
        )

        train_results = {"loss": [], "y_pred_mse": [], "y_pred_sign": []}
        print(f"(MSE weight, Sign loss scalar) = {weighted_total_loss.__defaults__}")
        for epoch in range(epoch_start_id, total_global_epochs):
            print(f"Training epoch {epoch} out of {total_global_epochs}.")
            # for loop over train_loader with both ith batch_idx and ith X data
            num_train_samples = len(train_loader.dataset)
            for batch_idx, X in enumerate(train_loader):
                oneway_model.train()
                optimizer.zero_grad()
                y_true = X[1].to(device)
                x_true = X[0].to(device)
                y_pred = oneway_model(x_true)

                mse_loss = self.loss_function_mse(y_pred, y_true)
                y_pred_sign_prob, y_true_sign = get_tensors_for_cross_entropy_loss(y_pred, y_true)
                sign_loss = self.loss_function_crossentropy(y_pred_sign_prob, y_true_sign)

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
                f"Current epoch: {epoch}, gpu: {gpu_id}: \n "
                f"Average Train loss: {train_loss}, train_y_pred_mse: {train_y_pred_mse}, "
                f"train_y_pred_sign: {train_y_pred_sign}. ")

        self.model = oneway_model # TODO: can be replaced
        self.optimizer = optimizer  # TODO: can be replaced
        return self

    def predict(self, X):
        oneway_model = self.model
        X_test_torch = get_torch_of_eval_data(X)

        oneway_model.eval()
        Y_pred = oneway_model(X_test_torch)
        return Y_pred[:, 0].tolist()
