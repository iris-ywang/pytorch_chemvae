import logging
import os

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression

from chemvae_train.data_utils import DataPreprocessor
from chemvae_train.load_params import ChemVAETrainingParams, load_params
from model_evaluations.pa_eval_utils import LatentRepViaFPVAE, run
from submodules.pairwise_formulation.pa_basics.import_data import kfold_splits
from train_vae import load_model


def main(params: ChemVAETrainingParams, n_qsar_test_size=None):
    # Load the data
    data_preprocessor = DataPreprocessor()
    data_preprocessor.vectorize_data(params)
    if n_qsar_test_size is None:
        n_qsar_test_size = params.data_size_for_all_loops
    train_test = data_preprocessor.X_all[:int(n_qsar_test_size)]

    # Load the FP VAE model
    fp_autoencoder = load_model(params, evaluating=True)

    # Prepare pairs
    train_test_splits_dict = kfold_splits(train_test=train_test, fold=10)
    pairing_method = LatentRepViaFPVAE(fp_autoencoder, params.hidden_dim).get_latent_rep

    metrics_per_dataset = run(
        train_test_splits_dict=train_test_splits_dict,
        ML_reg=RandomForestRegressor(random_state=1, n_jobs=-1),
        # ML_reg=LinearRegression(),  # for debugging purpose only
        ML_cls=RandomForestClassifier(random_state=1, n_jobs=-1),
        pairing_method=pairing_method,
        percentage_of_top_samples=0.1,  # top-performing as in top 10%
    )


def run_eval(
        exp_file_path,
        n_test_size=200,
        specific_dataset_path=None,
        specific_model_path=None,
):

    current_dir = os.getcwd()
    args = {"exp_file": exp_file_path, "directory": current_dir}

    if args["directory"] is not None:
        os.chdir(args["directory"])  # change to the directory where the experiment file is located

    training_params = load_params(args['exp_file'])

    #### if changing params:

    # Avoid mismatch between data_file path and training params exp.json file,
    # because the shuffle state and val_split will be different.
    if specific_dataset_path is not None:
        training_params.data_file = specific_dataset_path
        logging.warning(f"Using dataset: {specific_dataset_path}")

    if specific_model_path is not None:
        training_params.vae_weights_file = specific_model_path
        logging.warning(f"Using model weights: {specific_model_path}")

    main(training_params, n_test_size)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.info("Logging started.")
    run_eval("./trained_models/chembl4016/exp.json", 40)