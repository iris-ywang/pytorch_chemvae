#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 18:00:02 2022

@author: dangoo
"""

from trueskill import Rating, rate_1vs1 
import numpy as np
from sklearn.metrics import mean_squared_error, ndcg_score
from scipy.stats import spearmanr, kendalltau
from submodules.ScoreBasedTrueSkill.score_based_bayesian_rating import ScoreBasedBayesianRating as sd_rate_1vs1
from submodules.ScoreBasedTrueSkill.rating import Rating as SDRating



def rating_trueskill(Y, test_pair_ids, y_true):
    n_samples = len(y_true)
    n_comparisons = len(Y)
    ranking = [Rating() for _ in range(n_samples)]

    for comb_id in range(n_comparisons):
        id_a, id_b = test_pair_ids[comb_id]
        comp_result = Y[comb_id]
            
        if comp_result > 0:
            # i.e. id_a wins
            ranking[id_a], ranking[id_b] = rate_1vs1(ranking[id_a], ranking[id_b])
        elif comp_result < 0:
            ranking[id_b], ranking[id_a] = rate_1vs1(ranking[id_b], ranking[id_a]) 
        elif comp_result == 0:
            ranking[id_b], ranking[id_a] = rate_1vs1(ranking[id_b], ranking[id_a], drawn=True) 

    final_ranking = np.array([i.mu for i in ranking])
    
    return final_ranking


def rating_sbbr(Y, test_pair_ids, y_true):
    n_samples = len(y_true)
    n_comparisons = len(Y)
    # mean_train = np.mean(y_true[train_ids])
    # dev_train = mean_train / 3
    # beta = mean_train / 6
    ranking = [[SDRating()] for _ in range(n_samples)]

    for comp_id in range(n_comparisons):
        ida, idb = test_pair_ids[comp_id]
        comp_result = Y[comp_id]
        sd_rate_1vs1([ranking[ida], ranking[idb]], [comp_result, 0]).update_skills()

    return np.array([i[0].mean for i in ranking])


def evaluation(dy, combs_lst, train_ids, test_ids, y_true, func = rating_trueskill, params = None):
    y0 = func(dy, combs_lst, y_true, train_ids)
    # print(y_true[test_ids], y0[test_ids])
    rho = spearmanr(y_true[test_ids], y0[test_ids], nan_policy = "omit")[0]
    ndcg = ndcg_score([y_true[test_ids]], [y0[test_ids]])
    mse = mean_squared_error(y_true[test_ids], y0[test_ids])
    tau = kendalltau(y_true[test_ids], y0[test_ids])[0]
    # print(rho, ndcg, tau, mse)

    return y0, (rho, ndcg, tau, mse)
