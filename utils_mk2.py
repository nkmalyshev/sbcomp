import pandas as pd
import numpy as np
from utils_model import simple_feature_selector

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

            # df[year] = (df[col_name].dt.year-2000)/20
            # df[month] = df[col_name].dt.month/12
            # df[weekday] = df[col_name].dt.weekday/7
            # df[day] = df[col_name].dt.day/31
            # df[hour] = df[col_name].dt.hour/24

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


def collect_col_stats(df):
    col_stats = df.describe(include='all').T
    col_stats = col_stats.drop(['25%', '75%'], axis=1)
    col_stats.rename(index=str, columns={"50%": "median"}, inplace=True)

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


def preprocessing(x, y, col_stats_init=None, cat_freq_init=None, sample_size=None, max_columns=None, metric=None):

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

    x_na = x_number.copy()
    x_na = x_na.isnull().astype(int)
    x_na.columns = 'na_' + x_na.columns

    x_agg = pd.concat([x_number, x_date, x_cat, x_na], axis=1)
    # x_agg = pd.concat([x_number, x_date, x_cat], axis=1)

    # features selection
    if col_stats_init is None:
        col_stats = collect_col_stats(x_agg)
        cols = col_stats.index.values[col_stats.is_numeric & (col_stats['nunique'] > 2)]
        fs_results = simple_feature_selector(cols, x_agg, y, metric, max_columns=max_columns)
        col_stats.update(fs_results)
        col_stats['usefull'] = col_stats['usefull'].astype('bool')
        col_stats_out = col_stats
    else:
        col_stats_out = col_stats_init

    cols_to_use = col_stats_out.index.values[col_stats_out.usefull]
    x_out = x_agg[cols_to_use].copy()

    x_out = x_out.fillna(-1)
    x_out.loc[:, 'na_nulls'] = x_na.sum(axis=1)/x_na.shape[1]
    return x_out, y, col_stats_out, cat_freq_out

    # col_norm = col_stats_out.loc[cols_to_use]
    # col_norm = col_norm.loc[col_norm['nunique'] > 2]
    # col_norm['diff'] = col_norm['max'] - col_norm['min']
    # col_norm = col_norm.loc[col_norm['diff']>1]
    #
    # x_norm = x_out.copy()
    # x_norm.loc[:, col_norm.index.values] = (x_norm.loc[:, col_norm.index.values] - col_norm['min']) / col_norm['diff']
    # x_norm = x_norm.astype(float)
    #
    # x_norm = x_norm.fillna(-1)
    # x_norm.loc[:, 'na_nulls'] = x_na.sum(axis=1)/x_na.shape[1]
    #
    # return x_norm, y, col_stats_out, cat_freq_out