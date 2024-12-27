# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 19:01:44 2024

@author: TEO
"""
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame
from sklearn.preprocessing import RobustScaler

def index_multivariate_data_by_date(data_path):
    """
    Processes a CSV file containing multivariate time series data in a specific format.

    Parameters:
    - data_path (str): The path to the CSV file.

    Returns:
    - pd.DataFrame: A DataFrame with dates as the index and measurements as columns.
    """
    data = pd.read_csv(data_path)
    
    data['date'] = pd.to_datetime(data['date'])
    data.set_index('date', inplace=True)

    pivoted_data = data.pivot(columns='cols', values='data')
    # convert non numeric values to Nan
    pivoted_data = pivoted_data.apply(pd.to_numeric, errors='coerce')
    # handle missing values
    pivoted_data.ffill(inplace=True)
    # remove column if all values are Nan
    pivoted_data = pivoted_data.dropna(axis=1, how='all')            
    return pivoted_data


def get_data_for_chronos_bolt(data_path):
    df = pd.read_csv(data_path, parse_dates=['date'], index_col='date')

    if 'data' in df.columns:
        df = df.rename(columns={'data': 'target'})

    if 'target' not in df.columns:
        raise KeyError(f"Expected 'target' column, but found {df.columns}")

    # Apply robust scaling
    scaler = RobustScaler()
    df['target'] = scaler.fit_transform(df[['target']].values)

    # Apply rolling mean for smoothing
    df['target'] = df['target'].rolling(window=2, min_periods=1).mean()

    df = df.reset_index()
    df['item_id'] = 'series_1'

    ts_df = TimeSeriesDataFrame.from_data_frame(df, id_column='item_id', timestamp_column='date')

    train_size = int(len(ts_df) * 0.8)
    train_data = ts_df[:train_size]
    test_data = ts_df[train_size:]

    return train_data, test_data, scaler
