import pandas as pd
import numpy as np
import xgboost as xgb
from sdsj_feat import cat_frequencies

ONEHOT_MAX_UNIQUE_VALUES = 20
BIG_DATASET_SIZE = 500 * 1024 * 1024


def get_mem(df):
    mem = df.memory_usage().sum() / 1000000
    return f'{mem:.2f}Mb'


def transform_datetime_features(df):

    if df.shape[1] == 0:
        return df
    else:
        datetime_columns = df.columns.values

        res_date_cols = []
        for col_name in datetime_columns:
            df[col_name] = pd.to_datetime(df[col_name])

            year = f'date_year_{col_name}'
            month = f'date_month_{col_name}'
            weekday = f'date_weekday_{col_name}'
            day = f'date_day_{col_name}'
            hour = f'date_hour_{col_name}'

            df[year] = df[col_name].dt.year
            df[month] = df[col_name].dt.month
            df[weekday] = df[col_name].dt.weekday
            df[day] = df[col_name].dt.day
            df[hour] = df[col_name].dt.hour

            # df = df.drop(col_name, axis=1)

            res_date_cols += [year, month, weekday, day, hour]

        df[res_date_cols] = df[res_date_cols].fillna(-1)

        for col in res_date_cols:
            if 'year' in col:
                df[col] = df[col].astype(np.int16)
            else:
                df[col] = df[col].astype(np.int8)
        # df = df.drop(df.columns[df.nunique() == 1], axis=1)
        return df


def transform_categorical_features(df):
    categorical_values = {}
    # categorical encoding
    for col_name in list(df.columns):
        if col_name.startswith('id') or col_name.startswith('string'):
            categorical_values[col_name] = True

    return categorical_values


def old_transform_categorical_features(df, categorical_values={}):
    # categorical encoding
    for col_name in list(df.columns):
        if col_name not in categorical_values:
            if col_name.startswith('id') or col_name.startswith('string'):
                categorical_values[col_name] = df[col_name].value_counts().to_dict()

        if col_name in categorical_values:
            col_unique_values = df[col_name].unique()
            for unique_value in col_unique_values:
                df.loc[df[col_name] == unique_value, col_name] = categorical_values[col_name].get(unique_value, -1)

    return df, categorical_values


def check_column_name(name):
    if name == 'line_id':
        return False
    if name.startswith('datetime'):
        return False
    if name.startswith('string'):
        return False
    if name.startswith('id'):
        return False
    return True


def load_test_label(path):
    y = pd.read_csv(path, low_memory=False).target
    return y


def load_data(path, mode='train'):
    is_big = False
    if mode == 'train':
        df = pd.read_csv(path, low_memory=False)
        df.set_index('line_id', inplace=True)
        line_id = pd.DataFrame(df.index)
        y = df.target
        df = df.drop('target', axis=1)
        df['is_test'] = 0
        is_test = df.is_test
        df = df.drop('is_test', axis=1)
        if df.memory_usage().sum() > BIG_DATASET_SIZE:
            is_big = True
    else:
        df = pd.read_csv(path, low_memory=False)
        df.set_index('line_id', inplace=True)
        line_id = pd.DataFrame(df.index)
        df['is_test'] = 0
        is_test = df.is_test
        df = df.drop('is_test', axis=1)
        y = None
    return df, y, line_id, is_test, is_big

def simple_feature_selector(cols, X, Y, min_columns=3, max_columns=50, max_rel_error=.9):
    df_out = pd.DataFrame(
        {'col_names': cols, 'ols_error': -1, 'xgb_error': -1})
    df_out.set_index('col_names', inplace=True)

    for col in df_out.index.values:
        Xi = pd.DataFrame({'feature': X[col]})

        p_ols = calc_ols(Xi, Y, .3)
        ols_error = np.sqrt(np.dot(p_ols - Y, p_ols - Y) / Y.shape[0])

        p_xgb = calc_xgb(Xi, Y)
        xgb_error = np.sqrt(np.dot(p_xgb - Y, p_xgb - Y) / Y.shape[0])

        df_out.loc[col,'ols_error'] = ols_error
        df_out.loc[col,'xgb_error'] = xgb_error

    df_out['ols_related_error'] = df_out.ols_error / max(df_out.ols_error)
    df_out = df_out.sort_values('ols_related_error')
    df_out['ols_rank'] = df_out.sort_values('ols_related_error').reset_index().index

    df_out['xgb_related_error'] = df_out.xgb_error / max(df_out.xgb_error)
    df_out = df_out.sort_values('xgb_related_error')
    df_out['xgb_rank'] = df_out.sort_values('xgb_related_error').reset_index().index

    df_out['ols_usefull'] = (df_out['ols_rank'] < min_columns) | (df_out['ols_related_error'] < max_rel_error) | (df_out['ols_rank'] < max_columns)
    df_out['xgb_usefull'] = (df_out['xgb_rank'] < min_columns) | (df_out['xgb_related_error'] < max_rel_error) | (df_out['xgb_rank'] < max_columns)

    df_out['usefull'] = (df_out['ols_rank'] < max_columns) | (df_out['xgb_rank'] < max_columns)

    return df_out

def calc_ols(X, Y, reg_l2):
    power = 2
    for feature in X.columns.values:
        X = X.fillna(X[feature].mean())
        X[feature] = X[feature] / max(abs(X[feature]))
    X = pd.concat([X, X ** 2], axis=1)
    X['const'] = 1

    ols_w = np.dot(np.linalg.inv(np.dot(X.T, X) + reg_l2 * np.eye(power + 1, dtype=int)), np.dot(X.T, Y))
    p = np.dot(X, ols_w)
    return p


def calc_xgb(X,Y):

    params = {
        'params': {
            'silent': 1,
            'objective': 'reg:linear',
            'max_depth': 10,
            'min_child_weight': 1,
            'eta': .1},
        'num_rounds': 5
    }

    dtrain = xgb.DMatrix(X, label=Y)
    xgb_model = xgb.train(params['params'], dtrain, params['num_rounds'])
    p = xgb_model.predict(dtrain)

    return p



def collect_col_stats(df):
    col_stats = df.describe(include='all').T
    col_stats['nunique'] = df.nunique()
    col_stats['is_numeric'] = pd.isna([col_stats['mean']])[0]==False
    col_stats['prefix'] = [val[0:val.index('_')] for val in df.columns.values]
    return col_stats


def preprocessing(X, Y, mode='train'):

    # check main columns stats
    col_stats = collect_col_stats(X)

    # remove useless columns
    # X = X.drop(X.columns[X.nunique() == 1], axis=1)

    cols_date = col_stats.index.values[col_stats['prefix'] == 'datetime']
    cols_number = col_stats.index.values[col_stats['prefix'] == 'number']
    # cols_category = col_stats.index.values[col_stats['prefix'] == 'datetime']

    X_date = transform_datetime_features(X[cols_date].copy())
    X_number = X[cols_number]

    X_out = pd.concat([X_number, X_date], axis=1)

    col_stats = collect_col_stats(X_out)

    cols = col_stats.index.values[col_stats.is_numeric & (col_stats['nunique'] > 1)]
    fs_results = simple_feature_selector(cols, X_out, Y, min_columns=3, max_columns=50, max_rel_error=.9)

    return X_out[fs_results.index.values[fs_results.usefull]]