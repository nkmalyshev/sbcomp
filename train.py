import argparse
import os
import pickle
import time
import numpy as np
from utils_mk2 import load_data, preprocessing
from utils_model import xgb_train_wrapper, xgb_predict_wrapper
from sklearn.metrics import mean_squared_error, roc_auc_score


# use this to stop the algorithm before time limit exceeds
TIME_LIMIT = int(os.environ.get('TIME_LIMIT', 5 * 60))

ONEHOT_MAX_UNIQUE_VALUES = 20

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-csv', required=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--mode', choices=['classification', 'regression'], required=True)
    args = parser.parse_args()

    start_time = time.time()


    overall_params = {
        'prerocessin_ss': 10000,
        'xgb_params_search_ss': 40000,
        'feature_selections_cols': 25
    }
    metric = mean_squared_error if args.mode == 'regression' else roc_auc_score

    x_sample, y_sample, _, header, _ = load_data(args.train_csv, mode='train', input_rows=overall_params['prerocessin_ss'])
    _, _, col_stats, freq_stats = preprocessing(x=x_sample, y=y_sample, max_columns=overall_params['feature_selections_cols'])
    cols_to_use = col_stats['parent_feature'][col_stats['usefull']].unique()
    cols_to_use = cols_to_use[np.isin(cols_to_use, header)]

    x_train, y_train, _, _, _ = load_data(args.train_csv, mode='train', input_cols=np.append(cols_to_use, ['target', 'line_id']))
    x_train_proc, _, _, _ = preprocessing(x=x_train, y=0, col_stats_init=col_stats, cat_freq_init=freq_stats)

    xgb_model = xgb_train_wrapper(x_train_proc, y_train, metric, overall_params['xgb_params_search_ss'])
    p_xgb_train = xgb_predict_wrapper(x_train_proc, xgb_model)

    model_config = dict()
    model_config['model'] = xgb_model
    model_config['cols_to_use'] = cols_to_use
    model_config['col_stats'] = col_stats
    model_config['freq_stats'] = freq_stats

    model_config_filename = os.path.join(args.model_dir, 'model_config.pkl')
    with open(model_config_filename, 'wb') as fout:
        pickle.dump(model_config, fout, protocol=pickle.HIGHEST_PROTOCOL)

    print('Train time: {:0.2f}'.format(time.time() - start_time))
