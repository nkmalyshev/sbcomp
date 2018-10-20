import pandas as pd
import numpy as np
from utils import transform_datetime_features

ONEHOT_MAX_UNIQUE_VALUES = 20
BIG_DATASET_SIZE = 500 * 1024 * 1024


def transform_categorical_features(df_orig, categorical_values=None):
    if categorical_values is None:
        categorical_values = {}

    df = df_orig.copy(deep=True)
    # categorical encoding
    for col_name in list(df.columns):
        if col_name not in categorical_values:
            if col_name.startswith('id') or col_name.startswith('string'):
                categorical_values[col_name] = df[col_name].value_counts().to_dict()

        # if col_name in categorical_values:
        #     col_unique_values = df[col_name].unique()
        #     for unique_value in col_unique_values:
        #         df.loc[df[col_name] == unique_value, col_name] = categorical_values[col_name].get(unique_value, -1)

    return categorical_values


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
    # model_config = dict()
    # model_config['missing'] = True

    # read dataset
    is_big = False
    if mode == 'train':
        df = pd.read_csv(path, low_memory=False)
        y = df.target
        df = df.drop('target', axis=1)
        if df.memory_usage().sum() > BIG_DATASET_SIZE:
            is_big = True
    else:
        df = pd.read_csv(path, low_memory=False)
        y = None

    print('Dataset read, shape {}'.format(df.shape))

    # features from datetime
    df = transform_datetime_features(df)
    print('Transform datetime done, shape {}'.format(df.shape))

    # categorical encoding
    categorical_columns = transform_categorical_features(df)
    cat_cols = list(categorical_columns.keys())
    for col in cat_cols:
        df[col] = df[col].astype('category')

    # filter columns
    used_columns = [c for c in df.columns if check_column_name(c) or c in categorical_columns]
    print('Used {} columns'.format(len(used_columns)))

    line_id = df[['line_id', ]]
    df = df[used_columns]
    numeric_cols = df.select_dtypes(include=np.number).columns.values

    if is_big:
        df[numeric_cols] = df[numeric_cols].astype(np.float16)

    model_config = dict(
        used_columns=used_columns,
        categorical_values=categorical_columns,
        is_big=is_big
    )
    return df, y, model_config, line_id
