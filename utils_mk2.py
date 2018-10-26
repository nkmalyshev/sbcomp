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
        return df


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
    p_xgb, xgb_score = calc_xgb(x[xgb_cols], y)
    ##############################################################

    df_out = pd.concat([df_out, xgb_score], axis=1, sort=True)
    df_out = df_out.fillna(0)

    df_out = df_out.sort_values(by=['xgb_score'], ascending=False)
    df_out['xgb_rank'] = df_out.sort_values('xgb_score').reset_index().index

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
            'objective': 'reg:linear',
            'max_depth': 10,
            'min_child_weight': 1,
            'eta': .03},
        'num_rounds': 50}

    dtrain = xgb.DMatrix(x, label=y)
    xgb_model = xgb.train(params['params'], dtrain, params['num_rounds'])
    f_score = pd.DataFrame.from_dict(data=xgb_model.get_score(importance_type='gain'), orient='index', columns=['xgb_score'])
    f_score['xgb_score'] = f_score['xgb_score']/max(f_score['xgb_score'])
    p = xgb_model.predict(dtrain)
    return p, f_score


def collect_col_stats(df):
    col_stats = df.describe(include='all').T
    col_stats['nunique'] = df.nunique()
    col_stats['is_numeric'] = pd.isna([col_stats['mean']])[0]==False

    col_stats['default_type'] = ''
    for col_name in list(col_stats.index.values):
        if col_name.startswith('id') or col_name.startswith('string'):
            col_stats.loc[col_name, 'default_type'] = 'category'
        elif col_name.startswith('number'):
            col_stats.loc[col_name, 'default_type'] = 'number'
        elif col_name.startswith('datetime'):
            col_stats.loc[col_name, 'default_type'] = 'datetime'

    col_stats['parent_feature'] = ''
    for col_name in list(col_stats.index.values):
        for orig_name in list(col_stats.index.values[col_stats['default_type'] != '']):
            if col_name.endswith(orig_name):
                col_stats.loc[col_name, 'parent_feature'] = orig_name

    col_stats['usefull'] = False
    return col_stats


def preprocessing(x, y, col_stats_init=None, sample_size=None):

    if (sample_size is not None):
        if (sample_size < x.shape[0]):
            ids = x.index.values
            sample_ids = ids[np.random.randint(0, ids.shape[0], sample_size)]
            x = x.loc[sample_ids]
            y = y.loc[sample_ids]

    col_stats = collect_col_stats(x)
    if col_stats_init is None:
        col_stats['usefull'] = True
    else:
        parent_features = col_stats_init['parent_feature'][col_stats_init['usefull']].unique()
        col_stats.loc[parent_features, 'usefull'] = True

    cols_date = col_stats.index.values[(col_stats['default_type'].values == 'datetime') & (col_stats['usefull'].values)]
    cols_number = col_stats.index.values[(col_stats['default_type'].values == 'number') & (col_stats['usefull'].values)]
    cols_category = col_stats.index.values[(col_stats['default_type'].values == 'category') & (col_stats['usefull'].values)]

    x_date = transform_datetime_features(x[cols_date].copy())
    x_number = x[cols_number]
    x_sum = pd.concat([x_number, x_date], axis=1)

    if col_stats_init is None:
        col_stats = collect_col_stats(x_sum)
        cols = col_stats.index.values[col_stats.is_numeric & (col_stats['nunique'] > 1)]
        fs_results = simple_feature_selector(cols, x_sum, y, max_columns=75)
        col_stats.update(fs_results)
        col_stats['usefull'] = col_stats['usefull'].astype('bool')
        col_stats_out = col_stats
    else:
        col_stats_out = col_stats_init

    cols_to_use = col_stats_out.index.values[col_stats_out.usefull]
    x_out = x_sum[cols_to_use]

    return x_out, col_stats_out
