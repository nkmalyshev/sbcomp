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


def load_data(path, mode='train'):
    # model_config = dict()
    # model_config['missing'] = True

    # read dataset
    is_big = False
    if mode == 'train':
        df = pd.read_csv(f'{path}/train.csv', low_memory=False)
        y = df.target
        df = df.drop('target', axis=1)
        if df.memory_usage().sum() > BIG_DATASET_SIZE:
            is_big = True
    else:
        df = pd.read_csv(f'{path}/test.csv', low_memory=False)
        y = pd.read_csv(f'{path}/test-target.csv', low_memory=False).target

    print('Dataset read, shape {}'.format(df.shape))

    # features from datetime
    df = transform_datetime_features(df)
    print('Transform datetime done, shape {}'.format(df.shape))

    # categorical encoding
    # if datatype == 'train':
    categorical_values = transform_categorical_features(df)
    # model_config['categorical_values'] = categorical_values
    # else:
    #     df, categorical_values = transform_categorical_features(df, model_config['categorical_values'])
    # print('Transform categorical done, shape {}'.format(df.shape))

    # drop constant features
    # if datatype == 'train':
    #     constant_columns = [
    #         col_name
    #         for col_name in df.columns
    #         if df[col_name].nunique() == 1
    #     ]
    #     df.drop(constant_columns, axis=1, inplace=True)

    # filter columns
    # if mode == 'train':
    used_columns = [c for c in df.columns if check_column_name(c) or c in categorical_values]
    # used_columns = model_config['used_columns']
    print('Used {} columns'.format(len(used_columns)))

    line_id = df[['line_id', ]]
    df = df[used_columns]

    # # missing values
    # if model_config['missing']:
    #     df.fillna(-1, inplace=True)

    if is_big:
        df.values = df.values.astype(np.float16)

    model_config = dict(
        used_columns=used_columns,
        categorical_values=categorical_values,
        is_big=is_big
    )
    return df, y, model_config, line_id
