import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from submodules.pairwise_formulation.pairwise_data import PairwiseDataInfo


class ExtrapolationEvaluation:
    def __init__(self, y_train_with_predicted_test,
                 pairwise_data_info: PairwiseDataInfo, percentage_of_top_samples=0.1):
        self.y_pred_all = y_train_with_predicted_test
        self.y_true_all = np.zeros(pairwise_data_info.train_test.shape[0])
        self.y_true_all[pairwise_data_info.test_ids] = pairwise_data_info.test_ary[:, 0]
        self.y_true_all[pairwise_data_info.train_ids] = pairwise_data_info.train_ary[:, 0]

        self.pairwise_data_info = pairwise_data_info
        self.test_ids = pairwise_data_info.test_ids
        self.train_ids = pairwise_data_info.train_ids
        self.pc = percentage_of_top_samples

    def pairwise_differences_for_standard_approach(self, y_pred_all):
        Y_c2_abs_derived = []
        for pair_id in self.pairwise_data_info.c2_test_pair_ids:
            id_a, id_b = pair_id
            Y_c2_abs_derived.append(abs(y_pred_all[id_a] - y_pred_all[id_b]))
        return np.array(Y_c2_abs_derived)

    def find_top_test_ids(self, y_pred_all):
        # trains == train samples; tests == test samples.

        overall_orders = np.argsort(-y_pred_all)  # a list of sample IDs in the descending order of activity values
        top_trains_and_tests = overall_orders[0: int(self.pc * len(overall_orders))]
        top_tests = [idx for idx in top_trains_and_tests if idx in self.test_ids]

        # Find the ID of top train sample in the overall_order
        top_train_order_position = 0
        while True:
            if overall_orders[top_train_order_position] in self.train_ids: break
            top_train_order_position += 1
        top_train_id = overall_orders[top_train_order_position]

        tests_better_than_top_train = list(overall_orders[:top_train_order_position])

        return top_tests, tests_better_than_top_train

    def estimate_precision_recall(self, top_tests_true, top_tests):
        test_samples_boolean_true = [0 for _ in range(len(self.test_ids))]
        for top_test_id_true in top_tests_true:
            position_in_test_ids = int(np.where(self.test_ids == top_test_id_true)[0])
            test_samples_boolean_true[position_in_test_ids] = 1

        test_samples_boolean_pred = [0 for _ in range(len(self.test_ids))]
        for top_test_id in top_tests:
            position_in_test_ids = int(np.where(self.test_ids == top_test_id)[0])
            test_samples_boolean_pred[position_in_test_ids] = 1

        precision = precision_score(test_samples_boolean_true, test_samples_boolean_pred)
        recall = recall_score(test_samples_boolean_true, test_samples_boolean_pred)
        f1 = f1_score(test_samples_boolean_true, test_samples_boolean_pred)

        return precision, recall, f1

    def run_extrapolation_evaluation(self):
        top_tests_true, tests_better_than_top_train_true = self.find_top_test_ids(self.y_true_all)
        top_tests, tests_better_than_top_train = \
            self.find_top_test_ids(self.y_pred_all)

        # print("Number of tops: ", top_tests_true,", ", tests_better_than_top_train_true)
        # print("Predicted")
        # print("Number of tops: ", top_tests, ", ", tests_better_than_top_train)

        if len(top_tests_true) > 0:
            # precision & recall:
            precision_top, recall_top, f1_top = self.estimate_precision_recall(
                top_tests_true, top_tests
            )
            precision_better, recall_better, f1_better = self.estimate_precision_recall(
                tests_better_than_top_train_true, tests_better_than_top_train
            )

            return pd.Series({
                "precision_top": precision_top,
                "recall_top": recall_top,
                "f1_top": f1_top,
                "precision_better": precision_better,
                "recall_better": recall_better,
                "f1_better": f1_better
            })
        else:
            return pd.Series({
                "precision_top": np.nan,
                "recall_top": np.nan,
                "f1_top": np.nan,
                "precision_better": np.nan,
                "recall_better": np.nan,
                "f1_better": np.nan
            })
