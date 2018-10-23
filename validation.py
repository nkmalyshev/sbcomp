import time
import lightgbm as lgb
from sdsj_feat import load_data, load_test_label
from sklearn.metrics import mean_squared_error, roc_auc_score

_DATA_PATH = 'data/'

data_sets = [
    'check_4_c', 'check_5_c', 'check_6_c',
    'check_7_c',
    'check_8_c'
]


def run_train_test(ds_name, metric, params, sample_train):
    path = _DATA_PATH + ds_name
    x_train, y_train, train_params, _ = load_data(f'{path}/train.csv', mode='train', sample=sample_train)
    x_test, _, test_params, _ = load_data(f'{path}/test.csv', mode='test')
    y_test = load_test_label(f'{path}/test-target.csv')

    start_time = time.time()
    model = lgb.train(
        params,
        lgb.Dataset(x_train, label=y_train),
        600)
    print('train time: {:0.2f}'.format(time.time() - start_time))

    y_train_out = model.predict(x_train)
    y_test_out = model.predict(x_test)

    train_err = metric(y_train, y_train_out)
    test_err = metric(y_test, y_test_out)

    return train_err, test_err


def main():
    start_time = time.time()
    for data_path in data_sets:
        mode = data_path[-1]
        default_params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'regression' if mode == 'r' else 'binary',
            'metric': 'rmse',
            "learning_rate": 0.01,
            "num_leaves": 200,
            "feature_fraction": 0.70,
            "bagging_fraction": 0.70,
            'bagging_freq': 4,
            "max_depth": -1,
            "verbosity": -1,
            "reg_alpha": 0.3,
            "reg_lambda": 0.1,
            "min_child_weight": 10,
            'zero_as_missing': True,
            'num_threads': 4,
            'seed': 1
        }
        metric = roc_auc_score if mode == 'c' else mean_squared_error
        train_err, test_err = run_train_test(data_path, metric, default_params, 10000)

        print(f'ds={data_path} train_err={train_err:.4f} test_err={test_err:.4f}')

    print('total time: {:0.2f}'.format(time.time() - start_time))


if __name__ == '__main__':
    main()

