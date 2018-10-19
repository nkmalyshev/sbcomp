from sdsj_feat import load_data
import lightgbm as lgb
from multiple_encoder import MultiColumnLabelEncoder
from sklearn.metrics import mean_squared_error

_DATA_PATH = 'data/check_2_r'
_PATH_TO_TRAIN = 'data/check_2_r'
_PATH_TO_TEST = 'data/check_2_r'

params = {
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
    x_train, y_train, train_params, _ = load_data(_PATH_TO_TRAIN)
    x_test, y_test, test_params, _ = load_data(_PATH_TO_TEST, mode='test')

    cat_features = train_params['categorical_values']
    cat_cols = list(cat_features.keys())
    print('cat features=', cat_cols)
    label_enc = MultiColumnLabelEncoder(cat_cols)
    label_enc.fit(x_train[cat_cols])

    x_train_enc = label_enc.transform(x_train[cat_cols])
    x_test_enc = label_enc.transform(x_test)

    model = lgb.train(params, lgb.Dataset(x_train_enc, label=y_train), 600)

    y_train_out = model.predict(x_train_enc)
    y_test_out = model.predict(x_test_enc)

    train_err = mean_squared_error(y_train, y_train_out)
    test_err = mean_squared_error(y_test, y_test_out)

    print('train=', x_train.shape, 'y train=', y_train.shape, 'err=', train_err)
    print('test=', x_test.shape, 'y test=', y_test.shape, 'err=', test_err)


if __name__ == '__main__':
    main()

