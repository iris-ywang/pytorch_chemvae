import logging
import json
from collections import OrderedDict

from model_evaluations.vae_utils import VAEUtils
import numpy as np

from sklearn.metrics import jaccard_score


def one_hot_via_max_prob(arr: np.array):
    """Takes a 2D numpy array. Return a numpy array of the same shape, but each row (axis=1) has
    its maximum value set to 1, and all other values set to 0"""
    return np.where(arr == np.max(arr, axis=1)[:, None], 1, 0)


def vae_sa_similarity(
        demo_size=5,
        eval_size=500,
        autoencoder_file=None,
        test_idx_file='../models/zinc/test_idx.npy',
        exp_file='../models/zinc/exp.json',
        working_directory=None,
    ):
    vae_sa = VAEUtils(
        exp_file=exp_file,
        working_directory=working_directory,
        test_idx_file=autoencoder_file,
        autoencoder_file=test_idx_file,
        chembl=True,
    )

    metrics = []
    X = vae_sa.smiles_for_encoding[-eval_size:]
    Xoh = vae_sa.smiles_one_hot_for_encoding[-eval_size:]
    Z_standardised = vae_sa.Z[-eval_size:]
    Z_unstandardised = vae_sa.Z_unstandardised[-eval_size:]
    Xoh_pred = vae_sa.x_pred[-eval_size:]
    X_r = vae_sa.hot_to_smiles(Xoh_pred)

    metrics_oh_all_eval = jaccard_score(Xoh, one_hot_via_max_prob(Xoh_pred), average='micro')
    print(f"Similarity (one-hot) for all evaluation set: {metrics_oh_all_eval}")

    for i in range(demo_size):

        print(f"\n Calculating similarity for molecule {i}...")
        print("Original smiles     :", X[i])
        print("Reconstructed smiles:", X_r[i])

        metrics_oh = jaccard_score(Xoh[i], one_hot_via_max_prob(Xoh_pred[i]), average='micro')

        print(f"Similarity (one-hot): {metrics_oh}")
        print(f"Standardised Z: {Z_standardised[i]}")
        print(f"Unstandardised Z: {Z_unstandardised[i]}")

    for i in range(eval_size):
        metrics_oh = jaccard_score(Xoh[i], one_hot_via_max_prob(Xoh_pred[i]), average='micro')
        metrics.append(metrics_oh)

    return metrics


def main_sa(model_folder_path: str, metrics_filename: str):

    # model_train_size = 12600
    # base_path = '../models/zinc/'
    # test_idx_file_path = '../models/zinc/test_idx.npy'
    # encoder_file = base_path + f'zinc_encoder_iris2_{model_train_size}.h5'
    # decoder_file = base_path + f'zinc_decoder_iris2_{model_train_size}.h5'
    # metrics_filename = f"sa_model_iris2_{model_train_size}_similarity_scores_testsize_{size}.npy"

    exp_file_path = model_folder_path + "exp.json"
    exp_file_dict = json.loads(open(exp_file_path).read(),
                         object_pairs_hook=OrderedDict)
    test_idx_file_path = exp_file_dict['test_idx_file']
    autoencoder_file = exp_file_dict['vae_weights_file']

    metrics_jaccard = vae_sa_similarity(
        demo_size=5,
        eval_size=500,
        exp_file=exp_file_path,

        autoencoder_file=None,
        test_idx_file=None,
        working_directory=None,
    )
    np.save(model_folder_path + f"{metrics_filename}", metrics_jaccard)
    print("Finished!")

    return metrics_jaccard


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.info("Logging started.")

    main_sa(
        model_folder_path="trained_models/zinc/",
        metrics_filename="zinc_similarity_testsize_500.npy"
    )
    # main_pa()