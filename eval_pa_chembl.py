import logging
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from chemvae_train.data_utils import DataPreprocessor
from chemvae_train.load_params import ChemVAETrainingParams, load_params
from model_evaluations.pa_eval_utils import LatentRepViaFPVAE, run
from submodules.pairwise_formulation.pa_basics.import_data import kfold_splits
from train_vae import load_model


def main(params: ChemVAETrainingParams):
    # Load the data
    data_preprocessor = DataPreprocessor()
    data_preprocessor.vectorize_data(params)
    train_test = data_preprocessor.X_all[:params.data_size_for_all_loops]

    # Load the FP VAE model
    fp_autoencoder = load_model(params, evaluating=True)

    # Prepare pairs
    train_test_splits_dict = kfold_splits(train_test=train_test, fold=10)
    pairing_method = LatentRepViaFPVAE(fp_autoencoder, params.hidden_dim).get_latent_rep

    metrics_per_dataset = run(
        train_test_splits_dict=train_test_splits_dict,
        ML_reg=RandomForestRegressor(random_state=1, n_jobs=-1),
        # ML_reg=LinearRegression(),  # for debugging purpose only
        pairing_method=pairing_method,
        percentage_of_top_samples=0.1,  # top-performing as in top 10%
    )


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.info("Logging started.")

    current_dir = os.getcwd()
    args = {"exp_file": "./trained_models/chembl204/exp.json", "directory": current_dir}

    if args["directory"] is not None:
        os.chdir(args["directory"])  # change to the directory where the experiment file is located

    training_params = load_params(args['exp_file'])
    main(training_params)