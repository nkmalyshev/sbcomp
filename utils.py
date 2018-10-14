import datetime
import pandas as pd

start_date = datetime.datetime.strptime('2010-01-01', '%Y-%m-%d')


def transform_datetime_features(df):
    datetime_columns = [
        col_name
        for col_name in df.columns
        if col_name.startswith('datetime')
    ]
    for col_name in datetime_columns:
        df[col_name] = pd.to_datetime(df[col_name])

        df['number_year_{}'.format(col_name)] = df[col_name].dt.year
        df['number_weekday_{}'.format(col_name)] = df[col_name].dt.weekday
        df['number_month_{}'.format(col_name)] = df[col_name].dt.month
        df['number_day_{}'.format(col_name)] = df[col_name].dt.day
        df['number_hour_{}'.format(col_name)] = df[col_name].dt.hour
    return df
