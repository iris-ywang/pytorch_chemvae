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
        self.Xy_train = None
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

            # Reloading temporarily save model from mp training
            # Load the model trained by rank 0
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model = load_model(self.params).to(device)
            self.model.load_state_dict(torch.load("_trained_model_temp_file.pt", map_location=device))

        else:
            logging.info("No GPU available. Using CPU.")
            self.train(X, y)
        return self

    def _mp_train_wrapper(self, rank: int, world_size: int, X, y):
        ddp_setup(rank=rank, world_size=world_size)
        self.train(X, y, gpu_id=rank)
        if rank == 0:
            torch.save(self.model.state_dict(), "_trained_model_temp_file.pt")
        destroy_process_group()
        return self

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

        print(f"(MSE weight, Sign loss scalar) = {weighted_total_loss.__defaults__}")
        for epoch in range(epoch_start_id, total_global_epochs):
            print(f"Training epoch {epoch} out of {total_global_epochs}.")
            # for loop over train_loader with both ith batch_idx and ith X data
            num_train_samples = len(train_loader.dataset)
            train_results = {"loss": [], "y_pred_mse": [], "y_pred_sign": []}
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
        self.Xy_train = Xy_train
        return self

    def predict(self, X):
        device = next(self.model.parameters()).device  # Automatically get model's device
        oneway_model = self.model.to(device)
        X_test_torch = get_torch_of_eval_data(X).to(device)
        # self.check_if_test_tensor_contains_training_tensor(self.Xy_train.tensors[0], X)
        oneway_model.eval()
        with torch.no_grad():
            Y_pred = oneway_model(X_test_torch)
        return Y_pred[:, 0].tolist()

    @staticmethod
    def check_if_test_tensor_contains_training_tensor(train_tensor: TensorDataset, test_array: np.array):
        """train_tensor and test_tensor will both be in shape of (n_samples, 1024, 2).
        For each item in test_tensor, check if it is in train_tensor. if so, print
        the index of the item in test_tensor. """

        train_array = train_tensor.numpy()
        doggy_list = []
        for j in range(len(test_array)):
            test_j = test_array[j]
            for i in range(len(train_array)):
                if (test_j == train_array[i]).all():
                    # print(f"Item {j} in test tensor is found in training tensor.")
                    # print the index of the item in train_tensor
                    index = np.where(train_array == test_j)[0][0]
                    print(f"Item {j} in test tensor is found in training tensor at index {index, i}.")
                    doggy_list.append((j, i))
        if doggy_list:
            input(f"Found {len(doggy_list)} matching items between train and test tensors: {doggy_list}.")

            # train_idx = find_matching_index(train_array, test_j)
            # if train_idx is not None:
            #     doggy_list.append(train_idx)
            #     input(train_idx)

def find_matching_index(a1, a2):
    """ Check if the second array exists in the first array and return the index.
:param a1: numpy array of shape (x, 1024, 2)
:param a2: numpy array of shape (1024, 2)
:return: Index i (0 <= i < x) if a2 exists in a1, otherwise None
"""
    for i in range(a1.shape[0]):
        if np.array_equal(a1[i], a2):
            return i
    return None