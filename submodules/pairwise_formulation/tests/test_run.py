import pandas as pd

from submodules.pairwise_formulation.pairwise_data import PairwiseDataInfo
from submodules.pairwise_formulation.pairwise_model import PairwiseModel, build_ml_model
from submodules.pairwise_formulation.pa_basics.rating import rating_sbbr

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression

from scipy.stats import spearmanr

if __name__ == '__main__':
    data = pd.read_csv("test_data.csv")
    train_set, test_set = train_test_split(data, test_size = 0.1, random_state=40)

    # pairwise approach
    pairwise_data = PairwiseDataInfo(train_set, test_set)
    pairwise_model = PairwiseModel(
        pairwise_data_info=pairwise_data,
        ML_cls=LogisticRegression(),
        ML_reg=LinearRegression()
    ).fit()

    y_rank_scores_v2 = pairwise_model.predict(
        ranking_method=rating_sbbr,
        ranking_input_type='c2_c3')
    rho_pa_c2_v2 = spearmanr(pairwise_data.test_df['y'], y_rank_scores_v2)[0]

    y_rank_scores_v1 = pairwise_model.predict(
        ranking_method=rating_sbbr,
        ranking_input_type='c2_c3')
    rho_pa_c2_v1 = spearmanr(pairwise_data.test_df['y'], y_rank_scores_v1)[0]

    # standard approach
    _, y_sa_pred = build_ml_model(
        model=LinearRegression(),
        train_data=pairwise_data.train_ary,
        test_data=pairwise_data.test_ary
    )
    rho_sa = spearmanr(pairwise_data.test_df['y'], y_sa_pred)[0]

