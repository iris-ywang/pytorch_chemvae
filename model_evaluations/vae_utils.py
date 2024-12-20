import logging
from typing import Optional

import mol_utils.mol_utils as mu
import random
import yaml

from chemvae_train.models import VAEAutoEncoder
from chemvae_train.load_params import load_params, ChemVAETrainingParams
from train_vae import load_model

import numpy as np
import pandas as pd
import os
from mol_utils.mol_utils import fast_verify


class VAEUtils(object):
    params: ChemVAETrainingParams
    autoencoder: VAEAutoEncoder

    chars: str
    char_indices: dict
    indices_char: dict
    max_length: int
    data_height: int
    smile: list
    smiles_for_encoding: list
    smiles_one_hot_for_encoding: np.array
    test_idxs: Optional[np.array]

    Z: np.array
    Z_mu: np.array
    Z_std: np.array
    Z_unstandardised: np.array
    x_pred: np.array

    def __init__(
            self,
            exp_file,
            working_directory=None,
            test_idx_file=None,
            autoencoder_file=None,
            chembl=True,
    ):
        # files
        if working_directory is not None:
            logging.info("Setting working directory to: " + working_directory)
            os.chdir(working_directory)
        else:
            # set default working directory as the project folder path
            os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            logging.info("Current default working directory: ", os.getcwd())

        # load parameters
        self.params = load_params(exp_file, False)
        if autoencoder_file is not None:
            print(f"Changing encoder file to: {autoencoder_file} \n")
            self.params.vae_weights_file = autoencoder_file

        # char stuff
        chars = yaml.safe_load(open(self.params.char_file))
        self.chars = chars
        self.params.data_height = len(chars)
        if self.params.paired_output:
            self.max_length = int(self.params.data_width / 2)
        else:
            self.max_length = self.params.data_width
        self.char_indices = dict((c, i) for i, c in enumerate(chars))
        self.indices_char = dict((i, c) for i, c in enumerate(chars))

        # autoencoder
        self.autoencoder = load_model(self.params, evaluating=True)

        # self.encode, self.decode = self.enc_dec_functions()
        self.data = None

        # Load data without normalization as dataframe
        df = pd.read_csv(self.params.data_file)
        if chembl:
            df.iloc[:, 0] = df.iloc[:, 0].str.strip()
            df = df[df.iloc[:, 0].str.len() <= self.max_length]
            self.smiles = df.iloc[:, 0].tolist()

            # self.reg_tasks = df.iloc[:, 1:]
            # print("Available regression tasks: ", self.reg_tasks.columns)

            if df.shape[1] > 1:
                self.data = df.iloc[:, 1:]

        if test_idx_file is not None:
            self.test_idxs = np.load(test_idx_file).astype(int)
            logging.info("Loaded text index file from given file name.")
        else:
            try:
                self.test_idxs = np.load(self.params.test_idx_file).astype(int)
                logging.info("Loaded text index file from params file.")
            except FileNotFoundError:
                self.test_idxs = None

        self.estimate_estandarization(self.test_idxs)

        return

    def estimate_estandarization(self, test_idxs=None):

        # Load test set for encoding
        if test_idxs is None:
            logging.info("Encoding latent rep for random molecules from all data...")
            smiles = self.random_molecules(size=50000)
        else:
            print("Encoding latent rep for test set...")
            smiles = [self.smiles[i] for i in test_idxs]

        batch = 2500
        Z = np.zeros((len(smiles), self.params.hidden_dim))
        x_pred = np.zeros((len(smiles), self.max_length, self.params.data_height))

        self.smiles_for_encoding = smiles
        self.smiles_one_hot_for_encoding = self.smiles_to_hot(smiles)

        if self.params.paired_output:
            Z = np.zeros((len(smiles), self.max_length * self.params.data_height))  # self.params.data_height = 1

        self.autoencoder.eval()
        for chunk in self.chunks(list(range(len(smiles))), batch):
            # smiles_tr = list(set(self.smiles) - set(smiles))
            # pair_1 = np.concatenate((one_hot[0], one_hot[1]), axis=0)
            # pair_1 = np.reshape(pair_1, (1, pair_1.shape[0], pair_1.shape[1]))
            # output = self.encode(pair_1, False)
            sub_smiles = [smiles[i] for i in chunk]
            sub_one_hot = [self.smiles_one_hot_for_encoding[i] for i in chunk]

            if self.params.paired_output:
                # TODO: Check flatten and reshape error
                Z[chunk, :] = sub_one_hot.reshape((len(sub_smiles), self.max_length * self.params.data_height))
                continue
            x_pred_chunk, z_mean_log_var_chunk = self.autoencoder(sub_one_hot)
            Z[chunk, :] = z_mean_log_var_chunk
            x_pred[chunk, :] = x_pred_chunk

        if self.params.paired_output:
            self.Z = Z.astype(sub_one_hot.dtype)
            print('Paired output is True. No encoding performed. '
                  'VAEUtils.Z will be one-hot of smiles.')
            return

        self.Z_mu = np.mean(Z, axis=0)
        self.Z_std = np.std(Z, axis=0)
        self.Z_unstandardised = Z
        self.x_pred = x_pred

        logging.info("Standardizing latent rep...")
        self.Z = self.standardize_z(Z)

        logging.info('Finished encoding!')
        return

    def standardize_z(self, z):
        return (z - self.Z_mu) / self.Z_std

    def unstandardize_z(self, z):
        return (z * self.Z_std) + self.Z_mu

    def perturb_z(self, z, noise_norm, constant_norm=False):
        if noise_norm > 0.0:
            noise_vec = np.random.normal(0, 1, size=z.shape)
            noise_vec = noise_vec / np.linalg.norm(noise_vec)
            if constant_norm:
                return z + (noise_norm * noise_vec)
            else:
                noise_amp = np.random.uniform(
                    0, noise_norm, size=(z.shape[0], 1))
                return z + (noise_amp * noise_vec)
        else:
            return z

    def smiles_distance_z(self, smiles, z0):
        x = self.smiles_to_hot(smiles)
        z_rep = self.encode(x)
        return np.linalg.norm(z0 - z_rep, axis=1)

    def prep_mol_df(self, smiles, z):
        df = pd.DataFrame({'smiles': smiles})
        sort_df = pd.DataFrame(df[['smiles']].groupby(
            by='smiles').size().rename('count').reset_index())
        df = df.merge(sort_df, on='smiles')
        df.drop_duplicates(subset='smiles', inplace=True)
        df = df[df['smiles'].apply(fast_verify)]
        if len(df) > 0:
            df['mol'] = df['smiles'].apply(mu.smiles_to_mol)
        if len(df) > 0:
            df = df[pd.notnull(df['mol'])]
        if len(df) > 0:
            df['distance'] = self.smiles_distance_z(df['smiles'], z)
            df['frequency'] = df['count'] / float(sum(df['count']))
            df = df[['smiles', 'distance', 'count', 'frequency', 'mol']]
            df.sort_values(by='distance', inplace=True)
            df.reset_index(drop=True, inplace=True)
        return df

    def z_to_smiles(
            self,
            z,
            decode_attempts=250,
            noise_norm=0.0,
            constant_norm=False,
            early_stop=None
    ):
        if not (early_stop is None):
            Z = np.tile(z, (25, 1))
            Z = self.perturb_z(Z, noise_norm, constant_norm)
            X = self.decode(Z)
            smiles = self.hot_to_smiles(X, strip=True)
            df = self.prep_mol_df(smiles, z)
            if len(df) > 0:
                low_dist = df.iloc[0]['distance']
                if low_dist < early_stop:
                    return df

        Z = np.tile(z, (decode_attempts, 1))
        Z = self.perturb_z(Z, noise_norm)
        X = self.decode(Z)
        smiles = self.hot_to_smiles(X, strip=True)
        df = self.prep_mol_df(smiles, z)
        return df


    def smiles_to_hot(self, smiles, canonize_smiles=True, check_smiles=False):
        if isinstance(smiles, str):
            smiles = [smiles]

        if canonize_smiles:
            smiles = [mu.canon_smiles(s) for s in smiles]

        if check_smiles:
            smiles = mu.smiles_to_hot_filter(smiles, self.char_indices)

        p = self.params
        z = mu.smiles_to_hot(
            smiles,
            self.max_length,
            p.PADDING,
            self.char_indices,
            p.data_height,
        )
        return z

    def hot_to_smiles(self, hot_x, strip=False):
        smiles = mu.hot_to_smiles(hot_x, self.indices_char)
        if strip:
            smiles = [s.strip() for s in smiles]
        return smiles

    def random_idxs(self, size=None):
        if size is None:
            return [i for i in range(len(self.smiles))]
        else:
            return random.sample([i for i in range(len(self.smiles))], size)

    def random_molecules(self, size=None):
        if size is None:
            return self.smiles
        else:
            return random.sample(self.smiles, size)

    @staticmethod
    def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]
