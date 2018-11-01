import pandas as pd
import numpy as np
import xgboost as xgb


def simple_feature_selector(cols, x, y, max_columns=50):
    df_out = pd.DataFrame(
        {'col_names': cols, 'ols_error': -1})
    df_out.set_index('col_names', inplace=True)

    for col in df_out.index.values:
        xi = pd.DataFrame({'feature': x[col]})
        p_ols = calc_ols(xi, y, .3)
        ols_error = np.sqrt(np.dot(p_ols - y, p_ols - y) / y.shape[0])
        df_out.loc[col, 'ols_error'] = ols_error

    df_out = df_out.sort_values('ols_error')
    df_out['ols_rank'] = df_out.sort_values('ols_error').reset_index().index

    ##############################################################
    xgb_cols = df_out.index.values[df_out['ols_rank'] < max_columns]
    _, xgb_score = calc_xgb(x[xgb_cols], y)

    df_out = pd.concat([df_out, xgb_score], axis=1, sort=True)
    df_out = df_out.fillna(0)

    df_out = df_out.sort_values(by=['xgb_score'], ascending=False)
    df_out['xgb_rank'] = df_out.sort_values('xgb_score').reset_index().index
    ##############################################################

    df_out['usefull'] = (df_out['ols_rank'] < max_columns) & (df_out['xgb_rank'] < max_columns) & (df_out['xgb_score'] > 0)
    return df_out


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


def calc_xgb(x, y):
    params = {
        'params': {
            'silent': 1,
            'objective': 'reg:linear'},
        'num_rounds': 100}

    dtrain = xgb.DMatrix(x, label=y)
    xgb_model = xgb.train(params['params'], dtrain, params['num_rounds'])
    f_score = pd.DataFrame.from_dict(data=xgb_model.get_score(importance_type='gain'), orient='index', columns=['xgb_score'])
    f_score.sort_values('xgb_score').reset_index(drop=True)
    f_score = f_score.sort_values('xgb_score', ascending=False)
    f_score['xgb_score'] = f_score['xgb_score']/sum(f_score['xgb_score'])
    p = xgb_model.predict(dtrain)
    return p, f_score


def xgb_cv(xgb_params, cv_params, dtrain):
    cv = xgb.cv(xgb_params, dtrain, num_boost_round=xgb_params['num_rounds'], nfold=cv_params['nfold'], seed=0)
    cv['diff'] = cv.iloc[:, 2] - cv.iloc[:, 0]
    # test_error = (cv.iloc[:, 2] - min(cv.iloc[:, 2]))
    # opt_rounds = min(test_error.index.values[test_error == 0])

    if cv.iloc[:, 2].nunique() == 1:
        opt_rounds = 1
    else:
        test_error = (cv.iloc[:, 2] - min(cv.iloc[:, 2])) / (
                max(cv.iloc[:, 2]) - min(cv.iloc[:, 2]))
        test_error = test_error[test_error.index.values <= max(test_error[test_error == 0].index.values)]
        if test_error.index.values[test_error >= cv_params['threshold']].shape[0] == 0:
            opt_rounds = 1
        else:
            # opt_rounds = max(test_error.index.values[test_error >= cv_params['threshold']])
            cv_best = cv.loc[(test_error.index.values[test_error <= cv_params['threshold']]), :]
            cv_best = cv_best.sort_values('diff')
            opt_rounds = cv_best.index.values[0]




    test_error = cv.iloc[:, 2][opt_rounds]
    train_error = cv.iloc[:, 0][opt_rounds]
    return test_error, train_error, opt_rounds


def xgb_gs(params_out, params_search, dtrain):

    params_out['num_rounds'] = 300
    cv_params = {'nfold': 2, 'threshold': .01}
    t, tr, opt_rounds = xgb_cv(params_out, cv_params, dtrain)
    rounds_coeff = opt_rounds / 50
    params_out['eta'] = round(params_out['eta'] * rounds_coeff, 2)
    params_out['num_rounds'] = int(opt_rounds/rounds_coeff)+1

    print(params_out['num_rounds'], params_out['eta'])

    key0 = [*params_search][0]
    key1 = [*params_search][1]
    out_df = pd.DataFrame(columns=[key0, key1, 'num_rounds', 'test_error', 'train_error'])
    iter = 0

    for i in params_search[key0]:
        for j in params_search[key1]:
            params_out[key0] = i
            params_out[key1] = j

            cv_params = {'nfold': 2, 'threshold': .1}
            test_error, train_error, opt_rounds = xgb_cv(params_out, cv_params, dtrain)

            out_df.loc[iter, :] = [i, j, opt_rounds, test_error, train_error]
            iter = iter + 1

    out_df = out_df.sort_values('test_error').reset_index(drop=True)
    out_df['diff'] = out_df['test_error']-out_df['train_error']
    out_df['test_error_norm'] = out_df['test_error']/min(out_df['test_error'])
    out_df['train_error_norm'] = out_df['train_error']/out_df.loc[0, 'train_error']

    out_df = out_df.loc[out_df['test_error_norm'] < 1.05, :]

    best_params = out_df.sort_values('diff').reset_index(drop=True).loc[0, :]

    params_out[key0] = best_params[key0]
    params_out[key1] = best_params[key1]
    params_out['num_rounds'] = best_params['num_rounds']+1

    return params_out


def xgb_train_wrapper(x, y, metric, sample_size=None):

    dtrain = xgb.DMatrix(x, label=y)
    dsample = dtrain

    if (sample_size is not None):
        if (sample_size < x.shape[0]):
            ids = x.index.values
            sample_ids = ids[np.random.randint(0, ids.shape[0], sample_size)]
            dsample = xgb.DMatrix(x.loc[sample_ids], label=y.loc[sample_ids])

    init_params = {
        'silent': 1,
        'objective': 'reg:linear' if metric.__name__ == 'mean_squared_error' else 'binary:logistic',
        'eta': .3,
        'lambda': 3,
        'alpha': .1,
        'num_rounds': 0}

    params_search = {
        'max_depth': [2, 5, 7, 13],
        'min_child_weight': [1, 2, 6],
    }
    params_out = xgb_gs(init_params, params_search, dsample)
    # params_search = {
    #     'lambda': [0, 3],
    #     'alpha': [0, .3],
    # }
    # params_out = xgb_gs(params_out, params_search, dsample)
    print(params_out)

    model = xgb.train(params_out, dtrain, params_out['num_rounds'])
    return model


def xgb_predict_wrapper(x, model):
    dmatrix = xgb.DMatrix(x)
    p = model.predict(dmatrix)
    return p
