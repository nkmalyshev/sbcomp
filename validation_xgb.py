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



def run_train_test(ds_name, metric, params):
    path = _DATA_PATH + ds_name

    x_train, y_train, line_id, is_test, is_big = load_data(f'{path}/train.csv', mode='train')
    # x_test, _, line_id, _ = load_data(f'{path}/test.csv', mode='test')
    # y_test = load_test_label(f'{path}/test-target.csv')

    x_train = preprocessing(x_train, y_train, path)
    # x_test = preprocessing(x_test)

    # xgb
    dtrain = xgb.DMatrix(x_train, label=y_train)
    # dtest = xgb.DMatrix(x_test)

    xgb_model = xgb.train(params['params'], dtrain, params['num_rounds'])
    y_train_out = xgb_model.predict(dtrain)
    # y_test_out = xgb_model.predict(dtest)

    train_err = metric(y_train, y_train_out)
    # test_err = metric(y_test, y_test_out)
    test_err = 0

    return train_err, test_err

def main():
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

        start_time = time.time()
        metric = mean_squared_error if mode == 'r' else roc_auc_score
        train_err, test_err = run_train_test(data_path, metric, xgb_params)
        print('train time: {:0.2f}'.format(time.time() - start_time))

        print(f'ds={data_path} eval_metric={metric.__name__} train_err={train_err:.4f} test_err={test_err:.4f}')


if __name__ == '__main__':
    main()