import numpy as np
import pandas as pd
import lightgbm as lgb
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_auc_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from functools import partial


def calc_ols(x, y, reg_l2):
    power = 2
    for feature in x.columns.values:
        x = x.fillna(x[feature].mean())
        x[feature] = x[feature] / max(abs(x[feature]))
    x = pd.concat([x, x ** 2], axis=1)
    x['const'] = 1
    ols_w = np.dot(np.linalg.inv(np.dot(x.T, x) + reg_l2 * np.eye(power + 1, dtype=int)), np.dot(x.T, y))
    p = np.dot(x, ols_w)
    return p


def lgb_model(params, mode):
    if mode == 'regression':
        model = lgb.LGBMRegressor(**params)
    else:
        model = lgb.LGBMClassifier(**params)
    return model


# def feature_importance_lgb(x, y, mode):
#     params = {
#         'n_estimators': 100,
#         'learning_rate': 0.05,
#         'num_leaves': 200,
#         'subsample': 1,
#         'colsample_bytree': 1,
#         'random_state': 42,
#         'n_jobs': -1
#     }
#     model = lgb_model(params, mode)
#     # train model
#     model.fit(x, y)
#     feature_importance = pd.Series(model.booster_.feature_importance('gain'),
#         index=x.columns).fillna(0).sort_values(ascending=False)
#     # print(feature_importance.head(50))
#     # print(feature_importance.tail(10))
#
#     # remove totally unimportant features
#     best_features = feature_importance[feature_importance>0]
#
#     # leave most relevant features for big dataset
#     if df_size > BIG_DATASET_SIZE:
#         new_feature_count = min(df.shape[1], int(coef * df.shape[1] / (df_size / BIG_DATASET_SIZE)))
#         best_features = best_features.head(new_feature_count)
#
#     # select features
#     used_columns = best_features.index.tolist()
#     df = df[used_columns]
#
#     print('feature selection done')
#     print('number of selected features {}'.format(len(used_columns)))
#
#     dtrain = xgb.DMatrix(x, label=y)
#     xgb_model = xgb.train(params['params'], dtrain, params['num_rounds'])
#     f_score = pd.DataFrame.from_dict(data=xgb_model.get_score(importance_type='gain'), orient='index', columns=['xgb_score'])
#     f_score.sort_values('xgb_score').reset_index(drop=True)
#     f_score = f_score.sort_values('xgb_score', ascending=False)
#     f_score['xgb_score'] = f_score['xgb_score']/sum(f_score['xgb_score'])
#     p = xgb_model.predict(dtrain)
#     return p, f_score


def lgb_train_wrapper(x, y, mode, hyperopt_params):
    params = hyperopt_lgb(x, y, mode=mode, N=hyperopt_params['HYPEROPT_NUM_ITERATIONS'],
                          time_limit=hyperopt_params['TIME_LIMIT'],
                          max_train_size=hyperopt_params['HYPEROPT_MAX_TRAIN_SIZE'],
                          max_train_rows=hyperopt_params['HYPEROPT_MAX_TRAIN_ROWS'])
    # training
    model = lgb_model(params, mode)
    model.fit(x, y)
    return model


def lgb_predict_wrapper(x, model, mode):
    if mode == 'regression':
        p = model.predict(x)
    elif mode == 'classification':
        p = model.predict_proba(x)[:, 1]
    return p


def hyperopt_lgb(X, y, mode, N, time_limit, max_train_size=None, max_train_rows=None):
    """hyperparameters optimization with hyperopt"""

    print('hyperopt..')

    start_time = time.time()

    # train-test split
    train_size = 0.7
    # restrict size of train set to be not greater than max_train_size
    if max_train_size is not None:
        size_factor = max(1, 0.7*X.memory_usage(deep=True).sum()/max_train_size)
    # restrict number of rows in train set to be not greater than max_train_rows
    if max_train_rows is not None:
        rows_factor = max(1, 0.7*X.shape[0]/max_train_rows)
    train_size = train_size/max(size_factor, rows_factor)
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size=train_size, random_state=42)
    print('train shape {}, size {}'.format(Xtrain.shape, Xtrain.memory_usage(deep=True).sum()/1024/1024))

    # search space to pass to hyperopt
    fspace = {
        'num_leaves': hp.choice('num_leaves', [5,10,20,30,50,70,100]),
        'subsample': hp.choice('subsample', [0.7,0.8,0.9,1]),
        'colsample_bytree': hp.choice('colsample_bytree', [0.5,0.6,0.7,0.8,0.9,1]),
        'min_child_weight': hp.choice('min_child_weight', [5,10,15,20,30,50]),
        'learning_rate': hp.choice('learning_rate', [0.02,0.03,0.05,0.07,0.1,0.2]),
    }

    # objective function to pass to hyperopt
    def objective(params):

        iteration_start = time.time()

        # print(params)
        params.update({'n_estimators': 500, 'random_state': 0, 'n_jobs': -1})

        model = lgb_model(params, mode)
        model.fit(Xtrain, ytrain)

        if mode == 'regression':
            pred = model.predict(Xtest)
            loss = np.sqrt(mean_squared_error(ytest, pred))
        elif mode == 'classification':
            pred = model.predict_proba(Xtest)[:, 1]
            loss = -roc_auc_score(ytest, pred)

        iteration_time = time.time()-iteration_start
        # print('iteration time %.1f, loss %.5f' % (iteration_time, loss))

        return {'loss': loss, 'status': STATUS_OK,
                'runtime': iteration_time,
                'params': params}


    # object with history of iterations to pass to hyperopt
    trials = Trials()

    # loop over iterations of hyperopt
    for t in range(N):
        # run hyperopt, n_startup_jobs - number of first iterations with random search
        best = fmin(fn=objective, space=fspace, algo=partial(tpe.suggest, n_startup_jobs=10),
                    max_evals=t+1, trials=trials)
        # check if time limit exceeded, then interrupt search
        elapsed = time.time()-start_time
        if elapsed >= time_limit:
            print('time limit exceeded')
            break

    print('best parameters', trials.best_trial['result']['params'])

    return trials.best_trial['result']['params']
