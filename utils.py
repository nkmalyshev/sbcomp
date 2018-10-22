import datetime
import numpy as np
import pandas as pd

start_date = datetime.datetime.strptime('2010-01-01', '%Y-%m-%d')


def transform_datetime_features(df):
    datetime_columns = [
        col_name
        for col_name in df.columns
        if col_name.startswith('datetime')
    ]

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

    df[res_date_cols] = df[res_date_cols].fillna(-1)

    for col in res_date_cols:
        if 'year' in col:
            df[col] = df[col].astype(np.int16)
        else:
            df[col] = df[col].astype(np.int8)
    return df, res_date_cols, datetime_columns

