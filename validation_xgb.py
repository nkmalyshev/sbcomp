import time
import numpy as np
from utils_mk2 import load_data, load_test_label, preprocessing
from utils_model import xgb_train_wrapper, xgb_predict_wrapper
from sklearn.metrics import mean_squared_error, roc_auc_score

_DATA_PATH = 'data/'

data_sets = [
    'check_1_r',
    'check_2_r',
    'check_3_r',
    'check_4_c',
    'check_5_c',
    'check_6_c',
    'check_7_c',
    'check_8_c',
]

def run_train_test(ds_name, metric):
    path = _DATA_PATH + ds_name

    x_train, y_train, line_id_train, is_test, is_big = load_data(f'{path}/train.csv', mode='train')
    x_test, _, line_id_test, is_test, _ = load_data(f'{path}/test.csv', mode='test')
    y_test = load_test_label(f'{path}/test-target.csv')

    _, _, col_stats, freq_stats = preprocessing(x=x_train, y=y_train, sample_size=20000)
    x_train_proc, _, _, _ = preprocessing(x=x_train, y=0, col_stats_init=col_stats, cat_freq_init=freq_stats)
    x_test_proc, _, _, _ = preprocessing(x=x_test, y=0, col_stats_init=col_stats, cat_freq_init=freq_stats)

    print(x_train.shape)
    print(x_train_proc.shape)

    xgb_model = xgb_train_wrapper(x_train_proc, y_train, metric, 40000)
    p_xgb_train = xgb_predict_wrapper(x_train_proc, xgb_model)
    p_xgb_test = xgb_predict_wrapper(x_test_proc, xgb_model)

    xgb_err = [metric(y_train, p_xgb_train), metric(y_test, p_xgb_test)]

    if metric.__name__ == 'mean_squared_error':
        xgb_err = np.sqrt(xgb_err)

    return xgb_err


def main():
    start_time0 = time.time()
    for data_path in data_sets:
        mode = data_path[-1]

        start_time = time.time()
        metric = mean_squared_error if mode == 'r' else roc_auc_score
        errors = run_train_test(data_path, metric)
        print('train time: {:0.2f}'.format(time.time() - start_time))

        print(
            f'xgb ds={data_path} eval_metric={metric.__name__} train_err={errors[0]:.4f} test_err={errors[1]:.4f}')
    print('train time: {:0.2f}'.format(time.time() - start_time0))


if __name__ == '__main__':
    main()
