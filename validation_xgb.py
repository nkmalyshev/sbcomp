import time
import xgboost as xgb
from utils_mk2 import load_data, load_test_label, preprocessing
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


def run_train_test(ds_name, metric, xgb_params):
    path = _DATA_PATH + ds_name

    x_train, y_train, line_id_train, is_test, is_big = load_data(f'{path}/train.csv', mode='train')
    x_test, _, line_id_test, is_test, _ = load_data(f'{path}/test.csv', mode='test')
    y_test = load_test_label(f'{path}/test-target.csv')

    # _, col_stats = preprocessing(x=x_train, y=y_train)
    _, col_stats, freq_stats = preprocessing(x=x_train, y=y_train, sample_size=30000)
    x_train_proc, _, _ = preprocessing(x=x_train, y=0, col_stats_init=col_stats, cat_freq_init=freq_stats)
    x_test_proc, _, _ = preprocessing(x=x_test, y=0, col_stats_init=col_stats, cat_freq_init=freq_stats)

    # xgb
    dtrain = xgb.DMatrix(x_train_proc, label=y_train)
    dtest = xgb.DMatrix(x_test_proc)
    xgb_model = xgb.train(xgb_params['params'], dtrain, xgb_params['num_rounds'])
    p_train = xgb_model.predict(dtrain)
    p_test = xgb_model.predict(dtest)

    xgb_err = [metric(y_train, p_train), metric(y_test, p_test)]

    return xgb_err

def main():
    start_time0 = time.time()
    for data_path in data_sets:
        mode = data_path[-1]

        xgb_params = {
            'params': {
                'silent': 1,
                'objective': 'reg:linear' if mode == 'r' else 'binary:logistic',
                'max_depth': 13,
                'min_child_weight': 6,
                'eta': .03,
                'lambda': 3,
                'alpha': .03},
            'num_rounds': 300
        }

        start_time = time.time()
        metric = mean_squared_error if mode == 'r' else roc_auc_score
        xgb_error = run_train_test(data_path, metric, xgb_params)
        print('train time: {:0.2f}'.format(time.time() - start_time))

        print(f'xgb ds={data_path} eval_metric={metric.__name__} train_err={xgb_error[0]:.4f} test_err={xgb_error[1]:.4f}')
    print('train time: {:0.2f}'.format(time.time() - start_time0))

if __name__ == '__main__':
    main()