import numpy as np
import logging


def get_index_of_top_x_item_in_a_array(ary, x):
    return np.argpartition(ary, -x)[-x:]


def mean_of_top_x_in_an_array(ary, x):
    return np.mean(ary[get_index_of_top_x_item_in_a_array(ary, x)])


def mean_of_top_x_in_other_array(ary, x_index):
    return np.mean(ary[x_index])


def calculate_returns(y_test_pred, y_test_true, n_portofolio):
    pred_index_of_top_x = get_index_of_top_x_item_in_a_array(y_test_pred, n_portofolio)
    mean_y_top_x = mean_of_top_x_in_an_array(y_test_pred, n_portofolio)
    mean_y_true_top_x = mean_of_top_x_in_other_array(y_test_true, pred_index_of_top_x)
    mean_y_top_x_true = mean_of_top_x_in_an_array(y_test_true, n_portofolio)
    logging.info(f"Predicted mean return is {mean_y_top_x + 1}, \
                 \n predicted true return is {mean_y_true_top_x + 1}, \
                 \n true return is {mean_y_top_x_true + 1}")
    return mean_y_true_top_x + 1, mean_y_top_x_true + 1
