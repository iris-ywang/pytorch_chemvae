import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from itertools import permutations, product


class PairwiseDataInfo():
    """Stores the necessary data and pairwise information.
    Test column is the first one."""

    def __init__(self, train_data, test_data, target_value_col_name='y'):
        self.target_value_col_name = target_value_col_name

        # Placeholder
        self.train_df = None
        self.test_df = None
        self.train_test = None
        self.y_true_all = None
        self.train_ids = None
        self.test_ids = None
        self.c1_test_pair_ids = None  # not acutally test pairs but for the ease of naming nad using
        self.c2_test_pair_ids = None
        self.c3_test_pair_ids = None

        # Process inputs
        self.check_inputs_and_transform_into_df(train_data, test_data)
        self.process_train_and_test_data()

        logging.info(f"Training set size: {self.train_df.shape}, test set size: {self.test_df.shape}.")

        self.train_company_index = self.train_df.index
        self.test_company_index = self.test_df.index
        self.train_ary = self.train_df.to_numpy()
        self.test_ary = self.test_df.to_numpy()

        self.get_pair_test_pair_ids()

    def check_inputs_and_transform_into_df(self, train_data, test_data):
        if isinstance(train_data, pd.DataFrame):
            assert self.target_value_col_name in train_data.columns
            assert self.target_value_col_name in test_data.columns

            train_df = pd.DataFrame(train_data)
            test_df = pd.DataFrame(test_data)

        if isinstance(train_data, np.ndarray):
            train_df = pd.DataFrame(train_data).rename(columns={0: 'y'})

        if isinstance(train_data, np.ndarray):
            test_df = pd.DataFrame(test_data).rename(columns={0: 'y'})

        self.train_df = train_df
        self.test_df = test_df

    def get_pair_test_pair_ids(self):
        self.train_test = np.concatenate((self.train_ary, self.test_ary), axis=0)
        self.y_true_all = self.train_test[:, 0]
        self.train_ids = list(range(len(self.train_ary)))
        self.test_ids = list(range(len(self.train_ids), len(self.train_ids) + len(self.test_ary)))
        self.c1_test_pair_ids = list(permutations(self.train_ids, 2)) + [(a, a) for a in self.train_ids]
        self.c2_test_pair_ids = self.pair_test_with_train_ids_for_c2_pairs(self.train_ids, self.test_ids)
        self.c3_test_pair_ids = list(permutations(self.test_ids, 2)) + [(a, a) for a in self.test_ids]

    def process_train_and_test_data(self):
        self.train_df = self.impute_missing_values_using_simple_imputer(
            self.remove_a_row_from_df_if_the_first_item_is_nan(
                self.train_df
            )
        )
        self.test_df = self.impute_missing_values_using_simple_imputer(
            self.remove_a_row_from_df_if_the_first_item_is_nan(
                self.test_df
            )
        )

    def remove_a_row_from_df_if_the_first_item_is_nan(self, df):
        y_col_filtered = df[self.target_value_col_name].dropna(axis=0, how='any')
        df_filtered = df.loc[y_col_filtered.index]
        return df_filtered

    def impute_missing_values_using_simple_imputer(self, df):
        try:
            df = df.set_index("index")
        except KeyError:
            logging.warning("Dataframe has no index")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.columns = df.columns.astype(str)
        imputer = SimpleImputer(strategy='mean')
        imputer.fit(df)
        # imputer.set_output(transform="pandas")
        df_ary = imputer.transform(df)
        df_imputed = pd.DataFrame(df_ary, columns=df.columns, index=df.index)
        assert (df_imputed[self.target_value_col_name] - df[self.target_value_col_name]).sum() == 0.0

        return df_imputed

    @staticmethod
    def pair_test_with_train_ids_for_c2_pairs(train_ids, test_ids):
        """
        Generate C2-type pairs (test samples pairing with train samples)
        :param train_ids: list of int for training sample IDs
        :param test_ids: list of int for test sample IDs
        :return: list of tuples of sample IDs
        """
        c2test_combs = []
        for comb in product(test_ids, train_ids):
            c2test_combs.append(comb)
            c2test_combs.append(comb[::-1])
        return c2test_combs


@dataclass
class PairwiseValues():
    """Class to hold intermediate pairwise learning values."""
    Y_pa_c1_sign: list = None
    Y_pa_c1_nume: list = None
    Y_pa_c1_true: list = None
    Y_pa_c2_sign_true: list = None
    Y_pa_c2_sign: list = None
    Y_pa_c3_sign_true: list = None
    Y_pa_c3_sign: list = None
    Y_pa_c2_nume_true: list = None
    Y_pa_c2_nume: list = None
    Y_pa_c3_nume_true: list = None
    Y_pa_c3_nume: list = None
