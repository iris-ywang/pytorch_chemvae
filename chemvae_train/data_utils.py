"""Module for supporting utility functions for the train_vae module."""
from random import random

import pandas as pd
import numpy as np
import yaml

import logging
from dataclasses import dataclass

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer

from chemvae_train.load_params import ChemVAETrainingParams
from mol_utils import mol_utils as mu


@dataclass
class DataPreprocessor:
    X_train_all: np.array = None
    X_test_all: np.array = None
    Xy_train_all: np.array = None
    training_chunk_indice: list = None
    Xp_test_all: dict = None

    def vectorize_data(self, params: ChemVAETrainingParams):
        if params.if_smiles:
            return self.vectorize_smiles_data(params)
        else:
            return self.vectorize_data_chembl(params)

    def vectorize_data_chembl(self, params: ChemVAETrainingParams):
        # For Morgan FP, MAX_LEN = 1024.
        MAX_LEN = params.data_width
        assert MAX_LEN == 1024

        chembl_data = pd.read_csv(params.data_file)
        logging.info(f'Training set size is {len(chembl_data)}')
        if params.paired_output:
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
        self.Xy_train_all = chembl_data.iloc[train_idx, 1:].to_numpy(dtype=np.float32)

        return X_train, X_test

    def vectorize_smiles_data(self, params: ChemVAETrainingParams):
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
    def make_permutation_pairs(train_set):
        if len(train_set.shape) == 2:
            train_set = np.expand_dims(train_set, axis=2)
        train_set_size, train_set_length, train_set_dim = train_set.shape
        train_set_pairs = np.zeros((
            train_set_size * train_set_size, train_set_length, 2 * train_set_dim
        ))
        for i in range(train_set_size):
            for j in range(train_set_size):
                train_set_pairs[i * train_set_size + j] = np.concatenate((train_set[i], train_set[j]), axis=1)
        return train_set_pairs

    @staticmethod
    def make_combination_pairs(train_set: np.array, test_set: np.array):
        if len(train_set.shape) == 2:
            train_set = np.expand_dims(train_set, axis=2)
            test_set = np.expand_dims(test_set, axis=2)

        train_set_size, train_set_length, train_set_dim = train_set.shape
        test_set_size, test_set_length, test_set_dim = test_set.shape

        test_set_pairs = np.zeros((
            train_set_size * test_set_size * 2, test_set_length, 2 * test_set_dim
        ))

        for i in range(test_set_size):
            for j in range(train_set_size):
                test_set_pairs[i * train_set_size * 2 + j] = np.concatenate((train_set[j], test_set[i]), axis=1)
                test_set_pairs[i * train_set_size * 2 + train_set_size + j] = np.concatenate((test_set[i], train_set[j]), axis=1)

        return test_set_pairs

    def generate_loop_chunk_data_for_model_fit(self, if_paired, current_chunk_id):
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

        # Batch data per loop
        chunk_indices = self.training_chunk_indice[current_chunk_id]
        X_train = X_train_all[chunk_indices]
        logging.info(
            f"Training loop chunk index is the {current_chunk_id}th starting from index {chunk_indices[0]} \n"
        )

        # if paired_output is True, make pairs of the input data
        if if_paired:
            X_train = self.make_permutation_pairs(X_train)
        logging.info(f"Size of training size: {X_train.shape}")

        # checking for NaN values in the data
        self._check_nan_value_in_array(X_train)

        return X_train

    @staticmethod
    def _check_nan_value_in_array(X: np.array):
        if np.isnan(X).any():
            logging.warning("\n!!!!!!! \n NaN values in training data \n !!!!!!!\n")
        else:
            logging.info("No NaN values in training data")

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

    def generate_training_chunks(self, params: ChemVAETrainingParams, n_chunks):
        random_state = params.RAND_SEED
        if params.if_stratify:
            logging.info(f"Stratified KFold with {n_chunks} chunks with random state {random_state}")
            # Stratified KFold
            training_subset_size = params.data_size_for_all_loops
            chunk_indice = self.stratify_regression_data(
                self.Xy_train_all[:training_subset_size],
                k=n_chunks,
                n_bins=n_chunks * 2,
                strategy='quantile',
                random_state=random_state
            )
        else:
            logging.info(
                f"No stratification will be applied. {n_chunks} chunks will be split by "
                f"the order of the shuffled dataset."
            )
            chunk_indice = np.array_split(np.arange(params.data_size_for_all_loops), n_chunks)
            chunk_indice = [fold.tolist() for fold in chunk_indice]

        self.training_chunk_indice = chunk_indice
        return chunk_indice

    def generate_fixed_test_pairs(self, chunk_size: int, random_state=20):
        if self.training_chunk_indice is None:
            raise ValueError("Training chunk indices are not generated yet. "
                             "Please run generate_training_chunks() first")
        X_test_all = self.X_test_all

        Xp_permutate = self.make_permutation_pairs(X_test_all)
        self._check_nan_value_in_array(Xp_permutate)
        logging.info(f"Size of permutation pairs: {Xp_permutate.shape}")

        X_train_first_chunk = self.X_train_all[self.training_chunk_indice[0]]
        Xp_train_test_first_chunk = self.make_combination_pairs(X_train_first_chunk, X_test_all)
        self._check_nan_value_in_array(Xp_train_test_first_chunk)
        logging.info(f"Size of combination pairs: {Xp_train_test_first_chunk.shape}")

        X_train_chunk_sampling_indices = self.sample_from_chunks(
            self.training_chunk_indice, chunk_size, random_state=random_state
        )
        X_train_chunk_sampling = self.X_train_all[X_train_chunk_sampling_indices]
        Xp_train_chunk_sampling = self.make_combination_pairs(X_train_chunk_sampling, X_test_all)
        self._check_nan_value_in_array(Xp_train_chunk_sampling)
        logging.info(f"Size of combination pairs: {Xp_train_chunk_sampling.shape}")

        self.Xp_test_all = {
            "Permutate": Xp_permutate,
            "Combination_first_chunk": Xp_train_test_first_chunk,
            "Combination_chunk_sampling": Xp_train_chunk_sampling
        }
        return self.Xp_test_all

    @staticmethod
    def stratify_regression_data(data, k=5, n_bins=10, strategy='quantile', random_state=None):
        """
        Splits a regression dataset into k stratified chunks based on the distribution of the target variable.

        Parameters:
        - data (numpy.ndarray): A 2D NumPy array where the first column is `y` (target) and the rest are features.
        - k (int): Number of chunks.
        - n_bins (int): Number of bins to stratify `y`. More bins = better stratification but risk of sparsity.
        - strategy (str): Binning strategy ('quantile', 'uniform', 'kmeans'). Default is 'quantile'.
        - random_state (int, optional): Random seed for reproducibility.

        Returns:
        - List of lists, where each sublist contains indices for one chunk.
        """
        y = data[:, 0]  # Target values

        # Convert continuous target into discrete bins
        discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
        y_binned = discretizer.fit_transform(y.reshape(-1, 1)).astype(int).flatten()

        # Apply StratifiedKFold to create k chunks
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
        chunks = [[] for _ in range(k)]

        for fold_idx, (_, test_idx) in enumerate(skf.split(data, y_binned)):
            chunks[fold_idx] = test_idx.tolist()

        return chunks

    @staticmethod
    def sample_from_chunks(chunks, N, random_state=None):
        """
        Randomly samples x = N/k indices from each chunk.
        Returns:
        - List of sampled indices.
        """
        k = len(chunks)  # Number of chunks
        x = N // k  # Number of samples per chunk

        # Random seed will be set by params.RAND_SEED
        # if random_state is not None:
        #     random.seed(random_state)

        sampled_indices = []
        for chunk in chunks:
            sampled_indices.extend(
                np.random.choice(chunk, size=min(x, len(chunk)), replace=False).tolist()
            )

        return sampled_indices
