import lightgbm as lgb
from sdsj_feat import load_data, load_test_label
from sklearn.metrics import mean_squared_error, roc_auc_score

_DATA_PATH = 'data/'

data_sets = [
    # 'check_4_c', 'check_5_c', 'check_6_c',
    'check_7_c',
    'check_8_c'
]

default_params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary', #if args.mode == 'regression' else 'binary',
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
        'num_threads': 8,
        'seed': 1
    }


def run_train_test(ds_name, metric):
    path = _DATA_PATH + ds_name
    x_train, y_train, train_params, _ = load_data(f'{path}/train.csv', mode='train')
    x_test, _, test_params, _ = load_data(f'{path}/test.csv', mode='test')
    y_test = load_test_label(f'{path}/test-target.csv')

    model = lgb.train(
        default_params,
        lgb.Dataset(x_train, label=y_train),
        600)

    y_train_out = model.predict(x_train)
    y_test_out = model.predict(x_test)

    train_err = metric(y_train, y_train_out)
    test_err = metric(y_test, y_test_out)

    return train_err, test_err


def main():
    for data_path in data_sets:
        train_err, test_err = run_train_test(data_path, roc_auc_score)

        print(f'ds={data_path} train_err={train_err:.4f} test_err={test_err:.4f}')


if __name__ == '__main__':
    main()

