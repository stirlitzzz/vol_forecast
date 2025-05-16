# vol_utils.py

import numpy as np
import pandas as pd
from pathlib import Path
from zoneinfo import ZoneInfo  # Built-in from Python 3.9+


import sys
sys.path.append('./src')  # only needed if you're not packaging yet

def compute_log_returns(price_series):
    return np.log(price_series / price_series.shift(1)).dropna()

def annualized_volatility_std(log_returns, interval_minutes):
    scale = np.sqrt(252 * 24 * 60 / interval_minutes)
    return log_returns.std() * scale

def annualized_volatility_mean_square(log_returns, interval_minutes):
    scale = np.sqrt(252 * 24 * 60 / interval_minutes)
    return np.sqrt((log_returns ** 2).mean()) * scale

def annualized_volatility_parkinson(high, low, interval_minutes):
    log_hl = np.log(high / low)
    realized_var = (log_hl ** 2).mean() / (4 * np.log(2))
    scale = np.sqrt(252 * 24 * 60 / interval_minutes)
    return np.sqrt(realized_var) * scale

def get_midpoint(df):
    return (df['bid'] + df['ask']) / 2

def clean_timestamp(df):
    ts = pd.to_datetime(df['window_start'], unit='ns')
    return ts.dt.tz_localize('UTC').dt.tz_convert('America/New_York')

def compute_realized_volatility(input_file, intervals=['5min', '15min'], estimator='mean_square'):
    df = pd.read_csv(input_file)
    date = input_file.stem.split("_")[0]

    if 'window_start' not in df.columns:
        raise ValueError("Missing 'window_start' column in file.")

    # Convert nanoseconds to datetime
    df['timestamp'] = pd.to_datetime(df['window_start'], unit='ns')
    df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert(ZoneInfo('America/New_York'))
    df = df.sort_values('timestamp')

    if 'close' not in df.columns and 'price' in df.columns:
        df['close'] = df['price']
    elif 'close' not in df.columns:
        raise ValueError("No 'close' or 'price' column found.")

    vol_data = []

    for ticker, group in df.groupby('ticker', sort=False):
        group = group.set_index('timestamp')
        group = group.between_time('09:30', '15:59')

        if group.empty:
            continue

        row = {
            'ticker': ticker,
            'date': date,
            'open': group['close'].iloc[0],
            'close': group['close'].iloc[-1],
        }
        for interval in intervals:
            interval_minutes = int(interval.replace('min', ''))

            price_series = group['close'].resample(interval).last().dropna()
            log_returns = compute_log_returns(price_series)

            if len(log_returns) < 2:
                row[f'annualized_vol_{interval}'] = np.nan
            else:
                if estimator == 'std':
                    vol = annualized_volatility_std(log_returns, interval_minutes)
                elif estimator == 'parkinson':
                    high = group['high'].resample(interval).max().dropna()
                    low = group['low'].resample(interval).min().dropna()
                    vol = annualized_volatility_parkinson(high, low, interval_minutes)
                else:
                    vol = annualized_volatility_mean_square(log_returns, interval_minutes)

                row[f'annualized_vol_{interval}'] = vol
        vol_data.append(row)

    return pd.DataFrame(vol_data)