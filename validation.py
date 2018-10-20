import lightgbm as lgb
from sdsj_feat import load_data, load_test_label
from sklearn.metrics import mean_squared_error

_DATA_PATH = 'data/check_2_r'
_PATH_TO_TRAIN = 'data/check_2_r'
_PATH_TO_TEST = 'data/check_2_r'

default_params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression', #if args.mode == 'regression' else 'binary',
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


def main():
    x_train, y_train, train_params, _ = load_data(f'{_PATH_TO_TRAIN}/train.csv', mode='train')
    x_test, _, test_params, _ = load_data(f'{_PATH_TO_TEST}/test.csv', mode='test')
    y_test = load_test_label(f'{_PATH_TO_TEST}/test-target.csv')

    model = lgb.train(
        default_params,
        lgb.Dataset(x_train, label=y_train),
        600)

    y_train_out = model.predict(x_train)
    y_test_out = model.predict(x_test)

    train_err = mean_squared_error(y_train, y_train_out)
    test_err = mean_squared_error(y_test, y_test_out)

    print('train=', x_train.shape, 'y train=', y_train.shape, 'err=', train_err)
    print('test=', x_test.shape, 'y test=', y_test.shape, 'err=', test_err)


if __name__ == '__main__':
    main()

