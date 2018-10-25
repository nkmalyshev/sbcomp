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


def run_train_test(ds_name, metric, xgb_params, lgb_params):
    path = _DATA_PATH + ds_name

    x_train, y_train, line_id, is_test, is_big = load_data(f'{path}/train.csv', mode='train')
    # x_test, _, line_id, _ = load_data(f'{path}/test.csv', mode='test')
    # y_test = load_test_label(f'{path}/test-target.csv')

    x_train = preprocessing(x_train, y_train, path)
    # x_test = preprocessing(x_test)

    # xgb
    dtrain = xgb.DMatrix(x_train, label=y_train)
    # dtest = xgb.DMatrix(x_test)

    xgb_model = xgb.train(xgb_params['params'], dtrain, xgb_params['num_rounds'])
    y_train_out = xgb_model.predict(dtrain)
    # y_test_out = xgb_model.predict(dtest)

    train_err = metric(y_train, y_train_out)
    # test_err = metric(y_test, y_test_out)
    test_err = 0

    return train_err, test_err

def main():
    start_time0 = time.time()
    for data_path in data_sets:
        mode = data_path[-1]

        xgb_params = {
            'params': {
                'silent': 1,
                'objective': 'reg:linear' if mode == 'r' else 'binary:logistic',
                'max_depth': 10,
                'min_child_weight': 6,
                'eta': .01,
                'lambda': 1,
                'alpha': 0},
            'num_rounds': 200
        }

        lgb_params = {
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

        start_time = time.time()
        metric = mean_squared_error if mode == 'r' else roc_auc_score
        train_err, test_err = run_train_test(data_path, metric, xgb_params, lgb_params)
        print('train time: {:0.2f}'.format(time.time() - start_time))

        print(f'ds={data_path} eval_metric={metric.__name__} train_err={train_err:.4f} test_err={test_err:.4f}')
    print('train time: {:0.2f}'.format(time.time() - start_time0))

if __name__ == '__main__':
    main()