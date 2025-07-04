import logging
import os
import warnings

import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression

from chemvae_train.data_utils import DataPreprocessor
from chemvae_train.load_params import ChemVAETrainingParams, load_params
from chemvae_train.pairwise_nn_models import ChEMBLToDeltaYNN
from model_evaluations.pa_eval_utils import LatentRepViaFPVAE, run
from submodules.pairwise_formulation.pa_basics.import_data import kfold_splits
from submodules.pairwise_formulation.pa_basics.all_pairs import pair_by_pair_id_per_feature
from train_vae import load_model


def main(params: ChemVAETrainingParams, n_qsar_test_size=None):
    # Load the data
    data_preprocessor = DataPreprocessor()
    data_preprocessor.vectorize_data(params)
    if n_qsar_test_size is None:
        n_qsar_test_size = params.data_size_for_all_loops
    train_test = data_preprocessor.X_all[:int(n_qsar_test_size)]

    # Load the FP VAE model
    # fp_autoencoder = load_model(params, evaluating=True)
    # pairing_method = LatentRepViaFPVAE(fp_autoencoder, params.hidden_dim).get_latent_rep

    # Prepare pairs
    train_test_splits_dict = kfold_splits(train_test=train_test, fold=3)
    # pairing_method = ChEMBLToDeltaYNN.make_pairs_from_all_data_and_pair_ids
    metrics_per_dataset, all_metrics = run(
        train_test_splits_dict=train_test_splits_dict,
        SA_ML_reg=RandomForestRegressor(random_state=1, n_jobs=-1),  # for predecessor comparison
        # ML_reg=LinearRegression(),  # for debugging purpose only

        # ML_reg=ChEMBLToDeltaYNN(params),
        # pairing_method=pairing_method,

        ML_cls=RandomForestClassifier(random_state=1, n_jobs=-1),

        percentage_of_top_samples=0.1,  # top-performing as in top 10%
    )
    return metrics_per_dataset


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


def run_eval_in_batch(list_of_chembl_file_path: list, eval_size: int = 100):

    #################### CHECK
    fp_siamese_depth = 2
    fp_concat_depth = 2
    fp_n_activity_layers = 2
    fp_activity_layer_size_scalar = 1.0
    fp_hidden_dim_reduction_rate = 0.5
    fp_loss_weight = 10.0
    epochs = 70

    hidden_dim = 392
    save_model_per_epoch = False
    ###################

    metrics_all = pd.DataFrame()
    for data_file_path in list_of_chembl_file_path:
        df = pd.read_csv(data_file_path, index_col="molecule_id")
        chembl_dataset_id = data_file_path.split("CHEMBL")[-1].split(".")[0]
        chembl_name = f"chembl{chembl_dataset_id}"

        ### Get the whole training set from the entire dataset.
        data_width = df.shape[1] - 1
        n_molecules = len(df)
        all_digits = list(str(int(n_molecules * 0.9)))
        first_digit = int(all_digits[0])
        length = len(all_digits) - 1
        if (n_molecules - first_digit * 10 ** length) < int(n_molecules * 0.1):
            first_digit -= 1

        data_size_for_all_loops = first_digit * 10 ** length

        params = {
            "name": chembl_name,
            "data_file": data_file_path,
            "data_width": data_width,
            "save_model_per_epoch": save_model_per_epoch,
            "reload_model": False,
            "batchnorm_conv": True,
            "batchnorm_mid": True,

            "epochs": epochs,
            "lr": 0.0001,
            "RAND_SEED": 42,
            "data_size_for_all_loops": data_size_for_all_loops,
            # "loop_over_fit_batch_size": 500,
            # "loop_over_fit_batch_id": 0,

            "model_fit_batch_size": 100,
            "paired_output": True,
            "if_smiles": False,

            "if_fp_one_way": True,
            "hidden_dim": hidden_dim,
            "fp_siamese_depth": fp_siamese_depth,
            "fp_concat_depth": fp_concat_depth,
            "fp_hidden_dim_reduction_rate": fp_hidden_dim_reduction_rate,
            "fp_dropout_rate": 0.1,
            "fp_activation": "relu",
            "fp_loss_weight": fp_loss_weight,
            "fp_n_activity_layers": fp_n_activity_layers,
            "fp_activity_layer_size_scalar": fp_activity_layer_size_scalar,
            "fp_activity_dropout_rate": 0.1,
        }

        training_params = ChemVAETrainingParams(**params)

        metrics_per_dataset = main(training_params, eval_size)
        # set the name of index of metrics_per_dataset to "metrics"
        metrics_per_dataset = metrics_per_dataset.rename_axis("metrics").reset_index()

        metrics_per_dataset['dataset'] = chembl_name
        metrics_per_dataset['eval_size'] = eval_size
        metrics_all = pd.concat([metrics_all, metrics_per_dataset.reset_index()], ignore_index=True)

        metrics_all.to_csv(f"chembl_pa_metrics_batch_eval_size_{eval_size}_ts.csv", index=False)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.info("Logging started.")
    # run_eval("./trained_models/chembl4016/exp_oneway.json", 50)

    ### to run in batch:
    root_dir = "./trained_models/chembl_data/"

    chembl_info = pd.read_csv(
        root_dir + "chembl_datasets_info.csv"
    ).sort_values(by=["N(sample)"], ascending=False)
    large_chembles = chembl_info[
        (chembl_info["N(sample)"] > 900) & (chembl_info['Repetition Rate'] <= 0.15)]["File name"].tolist()

    list_of_data_file = [root_dir + fr'{file_name}' for file_name in large_chembles]

    run_eval_in_batch(
        list_of_data_file, eval_size=200
    )
