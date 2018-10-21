import pandas as pd
import numpy as np
from utils import transform_datetime_features

ONEHOT_MAX_UNIQUE_VALUES = 20
BIG_DATASET_SIZE = 500 * 1024 * 1024


def get_mem(df):
    mem = df.memory_usage().sum() / 1000000
    return f'{mem:.2f}Mb'


def transform_categorical_features(df):
    categorical_values = {}

    # categorical encoding
    for col_name in list(df.columns):
        if col_name.startswith('id') or col_name.startswith('string'):
            # categorical_values[col_name] = df[col_name].value_counts().to_dict()
            categorical_values[col_name] = True

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


def load_data(path, mode='train', model_config={}):
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

    print(f'Dataset read, shape: {df.shape}, memory: {get_mem(df)}')

    # features from datetime
    df, date_cols = transform_datetime_features(df)

    # # old cat encoding
    # if mode == 'train':
    #     df, categorical_columns = old_transform_categorical_features(df)
    #     model_config['categorical_values'] = categorical_columns
    # else:
    #     df, categorical_columns = old_transform_categorical_features(df, model_config['categorical_values'])
    # cat_cols = list(categorical_columns.keys())
    # # fill nan so column becomes numeric
    # df[cat_cols] = df[cat_cols].fillna(-1)

    # categorical encoding
    categorical_columns = transform_categorical_features(df)
    cat_cols = list(categorical_columns.keys())
    for col in cat_cols:
        df[col] = df[col].astype('category')

    # filter columns
    used_columns = [c for c in df.columns if check_column_name(c) or c in categorical_columns]

    line_id = df[['line_id', ]]
    df = df[used_columns]
    numeric_cols = df.select_dtypes(include=np.float).columns.values

    print(f'Cat: {len(cat_cols)}, numeric: {len(numeric_cols)}, date: {len(date_cols)}')
    print(f'Used: {len(used_columns)}, memory: {get_mem(df)}')

    if is_big:
        df[numeric_cols] = df[numeric_cols].astype(np.float16)

    model_config = dict(
        used_columns=used_columns,
        categorical_values=categorical_columns,
        numeric_cols=numeric_cols,
        is_big=is_big
    )
    return df, y, model_config, line_id


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
