import time
import numpy as np
import os
from utils_mk2 import load_data, load_test_label, preprocessing
from utils_model import calc_xgb
from sklearn.metrics import mean_squared_error, roc_auc_score
from utils_model_lgb import lgb_train_wrapper, lgb_predict_wrapper


TIME_LIMIT = int(os.environ.get('TIME_LIMIT', 5*60))

_DATA_PATH = 'data/'

data_sets = [
    'check_1_r',
    'check_2_r',
    'check_3_r',
    'check_4_c',
    # 'check_5_c',
    # 'check_6_c',
    # 'check_7_c',
    # 'check_8_c',
]

def run_train_test(ds_name, metric):
    start_time = time.time()
    path = _DATA_PATH + ds_name

    overall_params = {
        'preprocessing_ss': 20000,
        'xgb_params_search_ss': 40000,
        'small_data_rows': 20000,
        'feature_selections_cols': 75
    }

    hyperopt_params = {
        'HYPEROPT_NUM_ITERATIONS': 100,
        'TIME_LIMIT': 150,
        'HYPEROPT_MAX_TRAIN_SIZE': 10 * 1024 * 1024,
        'HYPEROPT_MAX_TRAIN_ROWS': 40000,
    }

    mode = 'regression' if metric.__name__ == 'mean_squared_error' else 'classification'

    x_sample, y_sample, _, header, _ = load_data(f'{path}/train.csv', mode='train', input_rows=overall_params['preprocessing_ss'])
    _, _, col_stats, freq_stats = preprocessing(x=x_sample, y=y_sample, max_columns=overall_params['feature_selections_cols'], metric=metric)
    cols_to_use = col_stats['parent_feature'][col_stats['usefull']].unique()
    cols_to_use = cols_to_use[np.isin(cols_to_use, header)]

    x_train, y_train, _, _, _ = load_data(f'{path}/train.csv', mode='train', input_cols=np.append(cols_to_use, ['target', 'line_id']))
    x_test, _, line_id_test, _, _ = load_data(f'{path}/test.csv', mode='test', input_cols=np.append(cols_to_use, ['line_id']))
    y_test = load_test_label(f'{path}/test-target.csv')

    x_train_proc, _, _, freq_stats = preprocessing(x=x_train, y=0, col_stats_init=col_stats, cat_freq_init=None)
    x_test_proc, _, _, _ = preprocessing(x=x_test, y=0, col_stats_init=col_stats, cat_freq_init=freq_stats)


    import pandas as pd
    train = x_train_proc.copy()
    test = x_test_proc.copy()
    train['target'] = 1
    test['target'] = 0
    x = pd.concat([train, test], axis=0)
    y = x.target
    x = x.drop('target', axis=1)
    p, importance = calc_xgb(x, y, roc_auc_score)
    err = metric(y, p)

    p_norm = abs(p - .5)
    idx_keep = y.index.values[(y == 1) & (p_norm < .07)]

    # x_train_proc_exclude = x_train_proc.loc[idx_keep, :]
    # x = x.drop('number_11', axis=1)
    # p, importance = calc_xgb(x, y, roc_auc_score)
    # err = metric(y, p)

    elapsed = time.time()-start_time
    hyperopt_params['TIME_LIMIT'] = int((TIME_LIMIT-elapsed)*0.7)

    model = lgb_train_wrapper(x_train_proc, y_train, mode, hyperopt_params)
    p_train = lgb_predict_wrapper(x_train_proc, model, mode)
    p_test = lgb_predict_wrapper(x_test_proc, model, mode)

    err = [metric(y_train, p_train), metric(y_test, p_test)]
    print(err)

    model = lgb_train_wrapper(x_train_proc.loc[idx_keep, :], y_train.loc[idx_keep], mode, hyperopt_params)
    p_train = lgb_predict_wrapper(x_train_proc.loc[idx_keep, :], model, mode)
    p_test = lgb_predict_wrapper(x_test_proc, model, mode)

    err = [metric(y_train[idx_keep], p_train), metric(y_test, p_test)]
    print(err)

    if metric.__name__ == 'mean_squared_error':
        err = np.sqrt(err)

    return err


def main():
    start_time0 = time.time()
    for data_path in data_sets:
        mode = data_path[-1]

        start_time = time.time()
        metric = mean_squared_error if mode == 'r' else roc_auc_score
        errors = run_train_test(data_path, metric)

        print('train time: {:0.2f}'.format(time.time() - start_time),
              f'xgb ds={data_path} eval_metric={metric.__name__} train_err={errors[0]:.4f} test_err={errors[1]:.4f}')

    print('train time: {:0.2f}'.format(time.time() - start_time0))


if __name__ == '__main__':
    main()
