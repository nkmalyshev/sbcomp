import pandas as pd
import numpy as np
from utils_model import calc_xgb, calc_ols

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

            res_date_cols += [year, month, weekday, day, hour]
        return df


def transform_categorigical_features(df, freq=None):
    if df.shape[1] == 0:
        return df, None
    else:
        cat_columns = df.columns.values
        if freq is None:
            out_freq = {col: (df[col].value_counts()/df.shape[0]).to_dict() for col in cat_columns}
        else:
            out_freq = freq

        for col in cat_columns:
            col_name = f'category_{col}'
            df[col_name] = df[col].map(out_freq[col])

        return df, out_freq



def load_test_label(path):
    y = pd.read_csv(path, low_memory=False).target
    return y


def load_data(path, mode='train', input_rows=None, input_cols=None):
    is_big = False
    if mode == 'train':
        df = pd.read_csv(path, low_memory=False, nrows=input_rows, usecols=input_cols, header=0)
        header = df.columns.values
        df.set_index('line_id', inplace=True)
        line_id = pd.DataFrame(df.index)
        y = df.target
        df = df.drop('target', axis=1)
        if df.memory_usage().sum() > BIG_DATASET_SIZE:
            is_big = True
    else:
        df = pd.read_csv(path, low_memory=False, nrows=input_rows, usecols=input_cols, header=0)
        header = df.columns.values
        df.set_index('line_id', inplace=True)
        line_id = pd.DataFrame(df.index)
        y = None
    return df, y, line_id, header, is_big


def simple_feature_selector(cols, x, y, max_columns=50):
    # df_out = pd.DataFrame(
    #     {'col_names': cols, 'ols_error': -1})
    # df_out.set_index('col_names', inplace=True)
    #
    # for col in df_out.index.values:
    #     xi = pd.DataFrame({'feature': x[col]})
    #     p_ols = calc_ols(xi, y, .3)
    #     ols_error = np.sqrt(np.dot(p_ols - y, p_ols - y) / y.shape[0])
    #     df_out.loc[col, 'ols_error'] = ols_error
    #
    # df_out = df_out.sort_values('ols_error')
    # df_out['ols_rank'] = df_out.sort_values('ols_error').reset_index().index
    #
    # ##############################################################
    # xgb_cols = df_out.index.values[df_out['ols_rank'] < max_columns]
    # _, xgb_score = calc_xgb(x[xgb_cols], y)
    #
    # df_out = pd.concat([df_out, xgb_score], axis=1, sort=True)
    # df_out = df_out.fillna(0)
    #
    # df_out = df_out.sort_values(by=['xgb_score'], ascending=False)
    # df_out['xgb_rank'] = df_out.sort_values('xgb_score').reset_index().index
    # ##############################################################
    #
    # df_out['usefull'] = (df_out['ols_rank'] < max_columns) & (df_out['xgb_rank'] < max_columns) & (df_out['xgb_score'] > 0)

    _, df_out = calc_xgb(x[cols], y)
    df_out['xgb_rank'] = df_out.sort_values('xgb_score').reset_index().index
    df_out['usefull'] = (df_out['xgb_rank'] < max_columns) & (df_out['xgb_score'] > 0)


    return df_out


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


def preprocessing(x, y, col_stats_init=None, cat_freq_init=None, sample_size=None):

    x.loc[:, 'number_nulls'] = x.isnull().sum(axis=1)

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
    x_cat, cat_freq_out = transform_categorigical_features(x[cols_category].copy(), cat_freq_init)
    x_agg = pd.concat([x_number, x_date, x_cat], axis=1)

    # features selection
    if col_stats_init is None:
        col_stats = collect_col_stats(x_agg)
        cols = col_stats.index.values[col_stats.is_numeric & (col_stats['nunique'] > 1)]
        fs_results = simple_feature_selector(cols, x_agg, y, max_columns=75)
        col_stats.update(fs_results)
        col_stats['usefull'] = col_stats['usefull'].astype('bool')
        col_stats_out = col_stats
    else:
        col_stats_out = col_stats_init

    cols_to_use = col_stats_out.index.values[col_stats_out.usefull]
    x_out = x_agg[cols_to_use].copy()


    col_norm = col_stats_out.loc[cols_to_use]
    col_norm = col_norm[['mean', 'std', 'nunique', '50%', 'max', 'min']]
    col_norm.columns = ['mean', 'std', 'nunique', 'median', 'max', 'min']
    col_norm = col_norm.sort_values('nunique')
    col_norm = col_norm.loc[col_norm['nunique'] > 2]
    x_out.loc[:, col_norm.index.values] = (x_out.loc[:, col_norm.index.values] - col_norm['mean'])/(col_norm['std']*3)

    # fillna
    x_out = x_out.fillna(-1)
    # norm


    return x_out, y, col_stats_out, cat_freq_out