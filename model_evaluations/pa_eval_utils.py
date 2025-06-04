import logging
from functools import partial

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from chemvae_train.fp_models import FPVAEAutoEncoder, FPEncoderToDeltaY
from model_evaluations.vae_utils import get_torch_of_eval_data
from submodules.pairwise_formulation.evaluations.extrapolation_evaluation import ExtrapolationEvaluation
from submodules.pairwise_formulation.pa_basics.all_pairs import pair_by_pair_id_per_feature
from submodules.pairwise_formulation.pa_basics.rating import rating_trueskill
from submodules.pairwise_formulation.pairwise_data import PairwiseDataInfo
from submodules.pairwise_formulation.pairwise_model import PairwiseModel, build_ml_model


def run(
    train_test_splits_dict: dict, ML_cls=None, ML_reg=None, SA_ML_reg=None,
    percentage_of_top_samples=0.1, target_value_col_name='y', n_jobs=None,
    pairing_method=pair_by_pair_id_per_feature,
):
    metrics_per_dataset = []
    if n_jobs is None:
        for fold_id, foldwise_data in train_test_splits_dict.items():
            logging.info(f"Running fold {fold_id}")
            metrics_per_fold = run_per_dataset(
                foldwise_data=foldwise_data,
                ML_cls=ML_cls,
                ML_reg=ML_reg,
                SA_ML_reg=SA_ML_reg,
                paring_method=pairing_method,
                percentage_of_top_samples=percentage_of_top_samples,
                target_value_col_name=target_value_col_name,
            )
            metrics_per_fold["fold_id"] = fold_id
            logging.info(f"Metrics for fold {fold_id}: \n {metrics_per_fold.to_string()}")
            metrics_per_dataset.append(metrics_per_fold)
            metrics_per_fold.to_csv("temp_res_ok_to_delete.csv")

        all_metrics = pd.concat(metrics_per_dataset)
        mean_metrics = all_metrics.drop(columns=["fold_id"]).groupby(all_metrics.index).mean(numeric_only=True)
        print()
        print(mean_metrics.to_string())

        return mean_metrics, all_metrics


def run_per_dataset(
        foldwise_data: dict,
        ML_cls=None,
        ML_reg=None,
        percentage_of_top_samples=0.1,
        paring_method=pair_by_pair_id_per_feature,
        target_value_col_name='y',
        SA_ML_reg=None,
) -> pd.DataFrame:

    train_set = foldwise_data['train_set']
    test_set = foldwise_data['test_set']

    # pairwise approach
    pairwise_data = PairwiseDataInfo(
        train_set, test_set, target_value_col_name=target_value_col_name
    )
    pairwise_model = PairwiseModel(
        pairwise_data_info=pairwise_data,
        ML_cls=ML_cls,
        ML_reg=ML_reg,
        pairing_method=paring_method,
    ).fit()

    metrics_dict = results_of_pairwise_combinations(
        pairwise_model=pairwise_model,
        if_rank_with_dist=False,
        rank_method=rating_trueskill,
        percentage_of_top_samples=percentage_of_top_samples,
    )

    # standard approach
    if SA_ML_reg is not None:
        model = SA_ML_reg
    else:
        model = ML_reg
    _, y_sa_pred = build_ml_model(
        model=model,
        train_data=pairwise_data.train_ary,
        test_data=pairwise_data.test_ary
    )

    y_sa_pred_w_train = np.array(pairwise_data.y_true_all)
    y_sa_pred_w_train[pairwise_data.test_ids] = y_sa_pred

    if ML_cls is not None:
        metrics_sa = ExtrapolationEvaluation(
            percentage_of_top_samples=percentage_of_top_samples,
            y_train_with_predicted_test=y_sa_pred_w_train,
            pairwise_data_info=pairwise_model.pairwise_data_info,
        ).run_extrapolation_evaluation()
        metrics_dict["rank_metrics_sa"] = metrics_sa

    if ML_reg is not None:
        metrics_est_sa = metrics_evaluation(
            pairwise_model.pairwise_data_info.test_ary[:, 0],
            y_sa_pred
        )

        metrics_dict["reg_metrics_sa"] = metrics_est_sa

        metrics_rank_sa = ExtrapolationEvaluation(
            percentage_of_top_samples=percentage_of_top_samples,
            y_train_with_predicted_test=y_sa_pred_w_train,
            pairwise_data_info=pairwise_model.pairwise_data_info,
        ).run_extrapolation_evaluation()
        metrics_dict["reg_rank_metrics_sa"] = metrics_rank_sa

    metrics_per_fold = pd.DataFrame(metrics_dict)
    return metrics_per_fold


def results_of_pairwise_combinations(
        pairwise_model: PairwiseModel,
        if_rank_with_dist: bool,
        rank_method=rating_trueskill,
        percentage_of_top_samples=0.1,
):
    results_df = pd.DataFrame()
    if pairwise_model.ML_cls is not None:
        logging.info("Extrapolation performance evaluation...")

        ####
        # extrap_eval_for_cls = partial(
        #     run_ranking_plus_extrapolation_eval,
        #     predict_method=pairwise_model.predict_rank,
        #     rank_method=rank_method,
        #     if_rank_with_dist=if_rank_with_dist,
        #     percentage_of_top_samples=percentage_of_top_samples,
        #     pairwise_data_info=pairwise_model.pairwise_data_info
        # )
        # metrics_c2 = extrap_eval_for_cls(ranking_input_type="c2")
        # results_df["rank_metrics_c2"] = metrics_c2
        ####

        # Refactor the below using the template above. REMEMBER TO TEST

        y_ranking_c2 = pairwise_model.predict_rank(
            ranking_method=rank_method,
            ranking_input_type="c2",
            if_sbbr_dist=if_rank_with_dist,
        )

        metrics_c2 = ExtrapolationEvaluation(
            percentage_of_top_samples=percentage_of_top_samples,
            y_train_with_predicted_test=y_ranking_c2,
            pairwise_data_info=pairwise_model.pairwise_data_info,
        ).run_extrapolation_evaluation()
        results_df["rank_metrics_c2"] = metrics_c2

        y_ranking_c2_c3 = pairwise_model.predict_rank(
            ranking_method=rank_method,
            ranking_input_type="c2_c3",
            if_sbbr_dist=if_rank_with_dist,
        )

        metrics_c2_c3 = ExtrapolationEvaluation(
            percentage_of_top_samples=percentage_of_top_samples,
            y_train_with_predicted_test=y_ranking_c2_c3,
            pairwise_data_info=pairwise_model.pairwise_data_info,
        ).run_extrapolation_evaluation()
        results_df["rank_metrics_c2_c3"] = metrics_c2_c3

        y_ranking_c1_c2_c3 = pairwise_model.predict_rank(
            ranking_method=rank_method,
            ranking_input_type="c1_c2_c3",
            if_sbbr_dist=if_rank_with_dist,
        )

        metrics_c1_c2_c3 = ExtrapolationEvaluation(
            percentage_of_top_samples=percentage_of_top_samples,
            y_train_with_predicted_test=y_ranking_c1_c2_c3,
            pairwise_data_info=pairwise_model.pairwise_data_info,
        ).run_extrapolation_evaluation()
        results_df["rank_metrics_c1_c2_c3"] = metrics_c1_c2_c3

        # Regressive prediction performance evaluation:
        # if not if_rank_with_dist:
        #     metrics_est = [np.nan for _ in range(6)]
        #
        # results_dict["reg_metrics_from_sbbr"] = metrics_est

    if pairwise_model.ML_reg is not None:
        logging.info("Pairwise regressive performance evaluation...")
        pairwise_model.predict()
        logging.info("Getting predictions for training set for training errors (2nd times)...")
        X_train = pairwise_model.trained_reg_model.Xy_train.tensors[0].numpy()
        Y_train_pred = list(pairwise_model.trained_reg_model.predict(X_train))

        logging.info("Quick evaluation on training and test errors on Y...")
        mse_train = mean_squared_error(pairwise_model.Y_values.Y_pa_c1_true, Y_train_pred)
        mse_train2 = mean_squared_error(pairwise_model.Y_values.Y_pa_c1_true, pairwise_model.Y_values.Y_pa_c1_nume)
        mse_test2 = mean_squared_error(pairwise_model.Y_values.Y_pa_c2_nume_true, pairwise_model.Y_values.Y_pa_c2_nume)
        mse_test3 = mean_squared_error(pairwise_model.Y_values.Y_pa_c3_nume_true, pairwise_model.Y_values.Y_pa_c3_nume)
        logging.info("Training MSE: %.4f (back up: %.4f), Test C2 MSE: %.4f, Test C3 MSE: %.4f" % (
            mse_train, mse_train2, mse_test2, mse_test3
        ))
        metrics_train_mse = pd.Series([mse_train, mse_train2, mse_test2, mse_test3], index=[0, 1, 2, 3],
                                      name='mse_train1_train2_test2_test3')

        y_train_est = estimate_y_from_averaging(
            pairwise_model.Y_values.Y_pa_c1_nume,
            pairwise_model.pairwise_data_info.c1_test_pair_ids,
            pairwise_model.pairwise_data_info.train_ids,
            pairwise_model.pairwise_data_info.y_true_all,
        )
        metrics_train_est = metrics_evaluation(
            pairwise_model.pairwise_data_info.train_ary[:, 0],
            y_train_est
        )
        metrics_train_est.name = "reg_metrics_train"

        y_est = estimate_y_from_averaging(
            pairwise_model.Y_values.Y_pa_c2_nume,
            pairwise_model.pairwise_data_info.c2_test_pair_ids,
            pairwise_model.pairwise_data_info.test_ids,
            pairwise_model.pairwise_data_info.y_true_all,
        )

        extrap_eval_for_reg = partial(
            run_ranking_plus_extrapolation_eval,
            predict_method=pairwise_model.rank,
            rank_method=rank_method,
            if_rank_with_dist=if_rank_with_dist,
            percentage_of_top_samples=percentage_of_top_samples,
            pairwise_data_info=pairwise_model.pairwise_data_info
        )
        metrics_reg_rank_c2 = extrap_eval_for_reg(ranking_input_type="c2")
        results_df["reg_rank_metrics_c2"] = metrics_reg_rank_c2

        metrics_reg_rank_c3 = extrap_eval_for_reg(ranking_input_type="c3")
        results_df["reg_rank_metrics_c3"] = metrics_reg_rank_c3

        metrics_reg_rank_c2_c3 = extrap_eval_for_reg(ranking_input_type="c2_c3")
        results_df["reg_rank_metrics_c2_c3"] = metrics_reg_rank_c2_c3

        metrics_reg_rank_c1_c2_c3 = extrap_eval_for_reg(ranking_input_type="c1_c2_c3")
        results_df["reg_rank_metrics_c1_c2_c3"] = metrics_reg_rank_c1_c2_c3

        metrics_reg_rank_c1_c2_c3 = extrap_eval_for_reg(ranking_input_type="c1_c2_c3")
        results_df["reg_rank_metrics_c1_c2_c3"] = metrics_reg_rank_c1_c2_c3

        metrics_reg_rank_y_est = ExtrapolationEvaluation(
            percentage_of_top_samples=percentage_of_top_samples,
            y_train_with_predicted_test=y_est,
            pairwise_data_info=pairwise_model.pairwise_data_info,
        ).run_extrapolation_evaluation()
        results_df["reg_rank_metrics_y_est"] = metrics_reg_rank_y_est


        metrics_est = metrics_evaluation(
            pairwise_model.pairwise_data_info.test_ary[:, 0],
            y_est
        )
        metrics_est.name = "reg_metrics_c2"
        results_df = pd.concat([results_df, metrics_est], axis=1)
        results_df = pd.concat([results_df, metrics_train_mse], axis=1)
        results_df = pd.concat([results_df, metrics_train_est], axis=1)

    return results_df

def run_ranking_plus_extrapolation_eval(
        predict_method,
        rank_method, ranking_input_type, if_rank_with_dist,
        percentage_of_top_samples, pairwise_data_info
):
    y_ranking = predict_method(
        ranking_method=rank_method,
        ranking_input_type=ranking_input_type,
        if_sbbr_dist=if_rank_with_dist,
    )

    metrics = ExtrapolationEvaluation(
        percentage_of_top_samples=percentage_of_top_samples,
        y_train_with_predicted_test=y_ranking,
        pairwise_data_info=pairwise_data_info,
    ).run_extrapolation_evaluation()
    return metrics


def metrics_evaluation(y_true, y_predict):
    rho = spearmanr(y_true, y_predict, nan_policy="omit")[0]
    mse = mean_squared_error(y_true, y_predict)
    mae = mean_absolute_error(y_true, y_predict)
    r2 = r2_score(y_true, y_predict)
    # return [rho, mse, mae, r2, np.nan, np.nan]
    return pd.Series({"rho": rho, "mse": mse, "mae": mae, "r2": r2})


def estimate_y_from_averaging(Y_pa_c2, c2_test_pair_ids, test_ids, y_true, Y_weighted=None):
    """
    Estimate activity values from C2-type test pairs via arithmetic mean or weighted average, It is calculated by
    estimating y_test from [Y_(test, train)_pred + y_train_true] and [ - Y_(train, test)_pred + y_train_true]

    :param Y_pa_c2: np.array of (predicted) differences in activities for C2-type test pairsc
    :param c2_test_pair_ids: list of tuples, each specifying samples IDs for a c2-type pair.
            * Y_pa_c2 and c2_test_pair_ids should match in position; their length should be the same.
    :param test_ids: list of int for test sample IDs
    :param y_true: np.array of true activity values of all samples
    :param Y_weighted: np.array of weighting of each Y_pred (for example, from model prediction probability)
    :return: np.array of estimated activity values for test set
    """
    if y_true is None:
        y_true = y_true
    if Y_weighted is None:  # linear arithmetic
        Y_weighted = np.ones((len(Y_pa_c2)))

    records = np.zeros((len(y_true)))
    weights = np.zeros((len(y_true)))

    for pair in range(len(Y_pa_c2)):
        ida, idb = c2_test_pair_ids[pair]
        delta_ab = Y_pa_c2[pair]
        weight = Y_weighted[pair]

        if ida in test_ids:
            # (test, train)
            weighted_estimate = (y_true[idb] + delta_ab) * weight
            records[ida] += weighted_estimate
            weights[ida] += weight

        elif idb in test_ids:
            # (train, test)
            weighted_estimate = (y_true[ida] - delta_ab) * weight
            records[idb] += weighted_estimate
            weights[idb] += weight

    return np.divide(records[test_ids], weights[test_ids])


class LatentRepViaFPVAE:
    def __init__(self, fp_autoencoder: FPVAEAutoEncoder, laten_rep_dim: int,latent_rep_csv_path=None):
        self.fp_autoencoder = fp_autoencoder
        self.latent_rep_dim = laten_rep_dim
        self.fp_autoencoder.eval()
        self.latent_rep_csv_path = latent_rep_csv_path

    def load_saved_latent_rep(self, data_all, pair_ids):
        pass

    def get_latent_rep(self, data, pair_ids):
        if self.latent_rep_csv_path is None:
            return self.compute_latent_rep_from_fp_vae(data, pair_ids)
        else:
            return self.load_saved_latent_rep(data, pair_ids)

    def compute_latent_rep_from_fp_vae(self, data_all, pair_ids):
        """
        Create the relevant latent representation for the pair_ids from the data_all.

        :param data_all: np.array of shape (n_samples, 1 + n_features). n_features should be 1024.
        :param pair_ids: list of tuples, each tuple containing two integers linking to the row indices of data_all.
        :return: np.array of latent representations of shape (n_samples, n_latent_features=params.hidden_dim).
        """

        # Chunk the data to avoid memory issues
        chunk_size = 5000
        n_chunks = int(len(pair_ids) / chunk_size) + 1

        y_diff_all = []
        latent_reps = []
        for chunk in range(n_chunks):
            start_idx = chunk * chunk_size
            end_idx = min((chunk + 1) * chunk_size, len(pair_ids))
            if start_idx == end_idx:
                break
            pair_ids_chunk = pair_ids[start_idx:end_idx]

            # For each pair in pair_ids_chunk,
            # 1. get the individual two rows from data_all,
            # 2. keep the differences in Y values (i.e. row_a[0] - row_b[0]) in a new list.
            # 3. stack the two vectors of features (i.e. row_a[1:] and row_b[1:]) into a new
            # numpy array of shape (n_features, 2)
            # 4. append the stacked array to a list so that it can create a numpy array of shape (n_chunk, n_features, 2)
            # 5. Pass the numpy array of shape (n_chunk, n_features, 2) to the autoencoder to get the latent representation
            # 6. Append the latent representation to a list so that it can create a numpy array of shape (n_chunk, n_latent_features)

            y_diff = []
            stacked_features = []
            for pair in pair_ids_chunk:
                row_a_idx = pair[0]
                row_b_idx = pair[1]

                row_a = data_all[row_a_idx]
                row_b = data_all[row_b_idx]

                y_diff.append(row_a[0] - row_b[0])

                features = np.stack([row_a[1:], row_b[1:]], axis=1)
                stacked_features.append(features)

            stacked_features = np.array(stacked_features)
            _, z_mean_log_var_output = self.fp_autoencoder.encoder(get_torch_of_eval_data(stacked_features))
            z_mean = z_mean_log_var_output[:, :self.latent_rep_dim]

            latent_reps += list(z_mean.detach().numpy())
            y_diff_all += y_diff

        latent_reps_all = np.array(latent_reps)
        y_diff_all = np.array(y_diff_all).reshape(-1, 1)

        final_output = np.hstack([y_diff_all, latent_reps_all])
        return final_output





#
#
# class ChemblToDeltaYNN:
#
#     def __init__(self, fp_oneway_model: FPEncoderToDeltaY):
#         self.fp_oneway_model = fp_oneway_model
#         self.fp_oneway_model.eval()
#
#     def pairing_for_oneway_model(self, data_all, pair_ids):
#         """
#          Create the relevant latent representation for the pair_ids from the data_all.
#
#          :param data_all: np.array of shape (n_samples, 1 + n_features). n_features should be 1024.
#          :param pair_ids: list of tuples, each tuple containing two integers linking to the row indices of data_all.
#          :return: np.array of latent representations of shape (n_samples, n_latent_features=params.hidden_dim).
#          """
#
#         # Chunk the data to avoid memory issues
#         chunk_size = 5000
#         n_chunks = int(len(pair_ids) / chunk_size) + 1
#
#         y_diff_all = []
#         latent_reps = []
#         for chunk in range(n_chunks):
#             start_idx = chunk * chunk_size
#             end_idx = min((chunk + 1) * chunk_size, len(pair_ids))
#             if start_idx == end_idx:
#                 break
#             pair_ids_chunk = pair_ids[start_idx:end_idx]
#
#             # For each pair in pair_ids_chunk,
#             # 1. get the individual two rows from data_all,
#             # 2. keep the differences in Y values (i.e. row_a[0] - row_b[0]) in a new list.
#             # 3. stack the two vectors of features (i.e. row_a[1:] and row_b[1:]) into a new
#             # numpy array of shape (n_features, 2)
#             # 4. append the stacked array to a list so that it can create a numpy array of shape (n_chunk, n_features, 2)
#             # 5. Pass the numpy array of shape (n_chunk, n_features, 2) to the autoencoder to get the latent representation
#             # 6. Append the latent representation to a list so that it can create a numpy array of shape (n_chunk, n_latent_features)
#
#             y_diff = []
#             stacked_features = []
#             for pair in pair_ids_chunk:
#                 row_a_idx = pair[0]
#                 row_b_idx = pair[1]
#
#                 row_a = data_all[row_a_idx]
#                 row_b = data_all[row_b_idx]
#
#                 y_diff.append(row_a[0] - row_b[0])
#
#                 features = np.stack([row_a[1:], row_b[1:]], axis=1)
#                 stacked_features.append(features)
#
#             stacked_features = np.array(stacked_features)
#     def compute_delta_y(self, data_all, pair_ids):
#
#
