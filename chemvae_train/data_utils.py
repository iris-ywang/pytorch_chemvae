"""Module for supporting utility functions for the train_vae module."""


import pandas as pd
import numpy as np
import yaml

import logging
from dataclasses import dataclass
from chemvae_train.load_params import ChemVAETrainingParams
from mol_utils import mol_utils as mu


@dataclass
class DataPreprocessor:
    X_train_all: np.array = None
    X_test_all: np.array = None

    def vectorize_data_chembl(self, params: ChemVAETrainingParams):
        # For Morgan FP, MAX_LEN = 1024.
        MAX_LEN = params.data_width
        assert MAX_LEN == 1024

        chembl_data = pd.read_csv(params.data_file)
        logging.info(f'Training set size is {len(chembl_data)}')
        if params["paired_output"]:
            params.data_height = 2
        else:
            params.data_height = 1

        X = chembl_data.iloc[:, 2:].to_numpy(dtype=np.float32)

        # Shuffle the data
        np.random.seed(params.RAND_SEED)
        rand_idx = np.arange(np.shape(X)[0])
        np.random.shuffle(rand_idx)

        # Set aside the validation set
        TRAIN_FRAC = 1 - params.val_split
        num_train = int(X.shape[0] * TRAIN_FRAC)

        if num_train % params.model_fit_batch_size != 0:
            num_train = num_train // params.model_fit_batch_size * \
                params.model_fit_batch_size

        train_idx, test_idx = rand_idx[: int(num_train)], rand_idx[int(num_train):]

        if params.test_idx_file is not None:
            np.save(params.test_idx_file, test_idx)

        X_train, X_test = X[train_idx], X[test_idx]
        logging.info(f'shape of input vector : {np.shape(X_train)}')
        logging.info('Training set size is {}, after filtering to max length of {}'.format(
            np.shape(X_train), MAX_LEN))

        self.X_train_all = X_train
        self.X_test_all = X_test

        return X_train, X_test

    def vectorize_data(self, params: ChemVAETrainingParams):
        # @out : Y_train /Y_test : each is list of datasets.
        #        i.e. if reg_tasks only : Y_train_reg = Y_train[0]
        #             if logit_tasks only : Y_train_logit = Y_train[0]
        #             if both reg and logit_tasks : Y_train_reg = Y_train[0], Y_train_reg = 1
        #             if no prop tasks : Y_train = []

        MAX_LEN = params.data_width
        if params.paired_output:
            MAX_LEN = int(MAX_LEN / 2)

        CHARS = yaml.safe_load(open(params.char_file))
        params.data_height = len(CHARS)
        NCHARS = len(CHARS)
        CHAR_INDICES = dict((c, i) for i, c in enumerate(CHARS))

        ## Load data if no properties

        smiles = mu.load_smiles_and_data_df(params.data_file, MAX_LEN)

        logging.info(f'Training set size is {len(smiles)}')
        logging.info(f'first smiles: {smiles[0]}')
        logging.info(f'total chars: {NCHARS}')

        logging.info('Vectorization...')
        X = mu.smiles_to_hot(smiles, MAX_LEN, params.PADDING, CHAR_INDICES, NCHARS)

        logging.info(f'Total Data size {X.shape[0]}')
        if np.shape(X)[0] % params.model_fit_batch_size != 0:
            X = X[:np.shape(X)[0] // params.model_fit_batch_size
                  * params.model_fit_batch_size]

        np.random.seed(params.RAND_SEED)
        rand_idx = np.arange(np.shape(X)[0])
        np.random.shuffle(rand_idx)

        TRAIN_FRAC = 1 - params.val_split
        num_train = int(X.shape[0] * TRAIN_FRAC)

        if num_train % params.model_fit_batch_size != 0:
            num_train = num_train // params.model_fit_batch_size * \
                params.model_fit_batch_size

        train_idx, test_idx = rand_idx[: int(num_train)], rand_idx[int(num_train):]

        if params.test_idx_file is not None:
            np.save(params.test_idx_file, test_idx)

        X_train, X_test = X[train_idx], X[test_idx]
        logging.info(f'shape of input vector : {np.shape(X_train)}')
        logging.info('Training set size is {}, after filtering to max length of {}'.format(
            np.shape(X_train), MAX_LEN))

        self.X_train_all = X_train
        self.X_test_all = X_test

        return X_train, X_test


    # def interleave_halves(x: tf.Tensor):
    #     """Function to take in a KerasTensor of shape (dim_a, n) and
    #     return a tensor of that double the length of shape (2*dim_a, n),
    #     but with their elements intersected.
    #
    #     For example, if the input keras tensor has its first half being
    #     [[1,2,3,4]] and the second half being [[10, 20, 30, 40]], the
    #     output will be [[1, 10, 2, 20, 3, 30, 4, 40]]."""
    #     dim_a = x.shape[1]
    #     half_dim_a = int(dim_a // 2)
    #     x_a = x[:, :half_dim_a]
    #     x_b = x[:, half_dim_a:]
    #     stacked = tf.stack([x_a, x_b], axis=2)
    #
    #     interleaved_pair = tf.reshape(stacked, [tf.shape(x_a)[0], -1])
    #     return interleaved_pair


    # def swap_halves(x: tf.Tensor):
    #     """
    #     Function to take in a 3D (?) KerasTensor of shape (dim_a, dim_b) and
    #     return a tensor of the same shape, but the first half in dim_a
    #     swaps position with the second half by the 2nd (?) axis.
    #     """
    #     dim_a = x.shape[1]
    #     half_dim_a = dim_a // 2
    #     return tf.concat([x[half_dim_a:], x[:half_dim_a]], axis=0)


    @staticmethod
    def make_pairs(train_set: np.array, test_set: np.array):
        if len(train_set.shape) == 2:
            train_set = np.expand_dims(train_set, axis=1)
            test_set = np.expand_dims(test_set, axis=1)
            extra_step = True

        train_set_size, train_set_length, train_set_dim= train_set.shape
        test_set_size, test_set_length, test_set_dim = test_set.shape

        train_set_pairs = np.zeros((
            train_set_size * train_set_size, 2*train_set_length, train_set_dim
        ))
        test_set_pairs = np.zeros((
            train_set_size * test_set_size * 2, 2*test_set_length, test_set_dim
        ))

        for i in range(train_set_size):
            for j in range(train_set_size):
                train_set_pairs[i * train_set_size + j] = np.concatenate((train_set[i], train_set[j]), axis=0)
        for i in range(test_set_size):
            for j in range(train_set_size):
                test_set_pairs[i * train_set_size * 2 + j] = np.concatenate((train_set[j], test_set[i]), axis=0)
                test_set_pairs[i * train_set_size * 2 + train_set_size + j] = np.concatenate((test_set[i], train_set[j]), axis=0)

        if extra_step == 1.0:
            train_set_pairs = train_set_pairs.swapaxes(1, 2)
            test_set_pairs = test_set_pairs.swapaxes(1, 2)

        return train_set_pairs, test_set_pairs

    def generate_loop_chunk_data_for_model_fit(
            self, if_paired,
            current_chunk_id, n_chunks, chunk_size,
    ):
        """

        Args:
            if_paired: if paired_output is True, make pairs of the input data
            X_train_all: from loading all data * 0.9
            X_test_all:  from loading all data * 0.1
            current_chunk_id: the nth loop / looped batch to go through
            n_chunks: total number of loops required
            chunk_size: the batch size per loop over model.fit()
            callbacks:

        Returns:
            X_train: training data for the loop
            X_test: test data for the loop
        """
        X_train_all = np.array(self.X_train_all)
        X_test_all = np.array(self.X_test_all)

        # Batch data per loop
        if current_chunk_id == n_chunks - 1:
            X_train = X_train_all[current_chunk_id * chunk_size:]
            X_test = X_test_all[current_chunk_id * int(chunk_size / 10):]
            logging.info(f"Training loop chunk index is from {current_chunk_id * chunk_size} to {len(X_train_all)} \n"
                         f"Test loop chunk index is from {current_chunk_id * int(chunk_size / 10)} to {len(X_test_all)}")
        else:
            X_train = X_train_all[current_chunk_id * chunk_size:(current_chunk_id + 1) * chunk_size]
            X_test = X_test_all[current_chunk_id * int(chunk_size / 10):(current_chunk_id + 1) * int(chunk_size / 10)]
            logging.info(f"Training loop chunk index is from {current_chunk_id * chunk_size} to {(current_chunk_id + 1) * chunk_size} \n"
                         f"Test loop chunk index is from {current_chunk_id * int(chunk_size / 10)} to {(current_chunk_id + 1) * int(chunk_size / 10)}")

        # if paired_output is True, make pairs of the input data
        if if_paired:
            X_train, X_test = self.make_pairs(X_train, X_test)
        logging.info(f"Size of training and test size: {X_train.shape}, {X_test.shape}")

        # checking for NaN values in the data
        if np.isnan(X_train).any():
            logging.warning("\n!!!!!!! \n NaN values in training data \n !!!!!!!\n")
        else:
            logging.info("No NaN values in training data")
        if np.isnan(X_test).any():
            logging.warning("\n!!!!!!!\n NaN values in test data\n !!!!!!!\n")
        else:
            logging.info("No NaN values in test data")

        return X_train, X_test

    def get_model_fit_chunk_size_and_starting_chunk_id(self, params: ChemVAETrainingParams):

        # if total_data_size is 10000, but data_size_for_all_loops is 6000,
        # that means we will only train on 6000 samples in this run.py process.
        # If batch_size_per_loop = loop_over_fit_batch_size is 2000, then
        # n_batch_per_run is 3, then we will train on 3 batches/chunks of 2000 samples.
        # if the training is paused after 2 chunks are finished, and if n_batch_per_run=2,
        # then the next run will start from the 3rd chunk.

        if params.data_size_for_all_loops:
            batch_size_per_loop = params.loop_over_fit_batch_size
            logging.info(f"Total available training data size is: {len(self.X_train_all)}, but "
                         f"only {params.data_size_for_all_loops} samples will be used for training.")
        else:
            logging.info(f"Total available training data size is: {len(self.X_train_all)}for training.")
            batch_size_per_loop = len(self.X_train_all)

        if params.data_size_for_all_loops:
            n_batch_per_run = int(params.data_size_for_all_loops // batch_size_per_loop)
            logging.info(f"The training data in this run will be split into {n_batch_per_run} "
                         f"chunks of size {batch_size_per_loop}. Model.fit() will be called "
                         f"{n_batch_per_run} times on {n_batch_per_run} chunks separately. "
                         f"i.e. the model will be trained repeatedly but with limited view on "
                         f"the whole training data. \n")
        else:
            n_batch_per_run = 1
            logging.info(f"Number of training chunks: {n_batch_per_run}")

        # Skip previously completed batches until the specified loop_over_fit_batch_id
        batch_start_id = 0
        if params.loop_over_fit_batch_id is not None:
            batch_start_id = int(params.loop_over_fit_batch_id)
            logging.info(f"\n Skipping Batch {list(range(batch_start_id))} as the start "
                             f"batch_id is specified at {batch_start_id} \n ")
        else:
            logging.info(f"\n Starting from the first batch {batch_start_id} \n ")

        return batch_size_per_loop, n_batch_per_run, batch_start_id
