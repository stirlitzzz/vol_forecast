import argparse
import pandas as pd
import ivolatility as ivol
import os
import numpy as np
import pandas_market_calendars as mcal
import time
import logging
from tqdm import tqdm
from dotenv import load_dotenv
from pathlib import Path

# ------------------------------- Logging Setup -------------------------------
logging.basicConfig(
    filename='ivol_scraper.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ------------------------------- Load API Key -------------------------------
load_dotenv(dotenv_path=Path(__file__).parent / '.env')
api_key = os.getenv("IVOL_API_KEY")
if not api_key:
    raise RuntimeError("❌ IVOL_API_KEY not found in .env file or environment.")

ivol.setLoginParams(apiKey=api_key)

# ------------------------------- Core Functions -------------------------------
def get_option_data(ticker, date):
    getMarketData = ivol.setMethod('/equities/eod/stock-opts-by-param')
    calls = getMarketData(symbol=ticker, tradeDate=date, dteFrom=0, dteTo=120, deltaFrom=0.2, deltaTo=0.8, cp='C')
    puts  = getMarketData(symbol=ticker, tradeDate=date, dteFrom=0, dteTo=120, deltaFrom=-0.8, deltaTo=-0.2, cp='P')
    return calls, puts

def find_closest_by_expiry(df, delta_col='delta_call', targets=[0.25, 0.75], n=2):
    result = []
    for expiry, group in df.groupby('expiration_date'):
        for target in targets:
            closest = (
                group
                .assign(delta_diff=(group[delta_col] - target).abs())
                .nsmallest(n, 'delta_diff')
            )
            result.append(closest)
    return pd.concat(result).drop_duplicates(subset=['price_strike', 'expiration_date'])

def find_nearest_strikes_around_spot(df, spot_col='underlying_price_call'):
    result = []
    for expiry, group in df.groupby('expiration_date'):
        spot = group[spot_col].iloc[0]
        below = group[group['price_strike'] <= spot].sort_values('price_strike').tail(1)
        above = group[group['price_strike'] >= spot].sort_values('price_strike').head(1)
        result.extend([below, above])
    return pd.concat(result)

def load_and_filter_day_ticker(ticker, date):
    calls, puts = get_option_data(ticker, date)
    if calls.empty or puts.empty:
        return None
    merged = pd.merge(calls, puts, on=['price_strike', 'expiration_date'], suffixes=('_call', '_put'))
    merged['vol_weighted'] = (
        merged['delta_call'] * merged['iv_call'] +
        (1 - merged['delta_call']) * merged['iv_put']
    )
    delta_filtered = find_closest_by_expiry(merged)
    atm_pair = find_nearest_strikes_around_spot(merged)
    keep = pd.concat([delta_filtered, atm_pair]).drop_duplicates(subset=['price_strike', 'expiration_date'])
    return keep

def extract_vol_features(df):
    all_expiry = df['expiration_date'].unique()
    val_date_str = df.iloc[0]['c_date_call']
    features = []

    for expiry in all_expiry:
        data = df[df['expiration_date'] == expiry]
        spot = data['underlying_price_call'].iloc[0]
        below = data[data['price_strike'] <= spot].sort_values('price_strike').tail(1)
        above = data[data['price_strike'] >= spot].sort_values('price_strike').head(1)
        if below.empty or above.empty:
            continue
        K_dn = below['price_strike'].values[0]
        K_up = above['price_strike'].values[0]
        iv_dn = below['vol_weighted'].values[0]
        iv_up = above['vol_weighted'].values[0]
        atm_iv = iv_up * (spot - K_dn) / (K_up - K_dn) + iv_dn * (K_up - spot) / (K_up - K_dn) if K_up != K_dn else iv_up
        k_hi = data.sort_values('price_strike').tail(1)
        k_lo = data.sort_values('price_strike').head(1)
        vol_hi = k_hi['vol_weighted'].values[0]
        vol_lo = k_lo['vol_weighted'].values[0]
        texp = np.busday_count(pd.to_datetime(val_date_str).date(), pd.to_datetime(expiry).date()) / 252.0
        if (texp <= 0):
            continue
        scale = np.sqrt(texp) / 10
        slope_up = (vol_hi - atm_iv) / np.log(k_hi['price_strike'].values[0] / spot) * scale
        slope_down = (vol_lo - atm_iv) / np.log(k_lo['price_strike'].values[0] / spot) * scale

        features.append({
            'c_date': val_date_str, 'expiry': expiry, 'spot': spot, 'atm_iv': atm_iv,
            'slope_up': slope_up, 'slope_down': slope_down,
            'K_dn': K_dn, 'K_up': K_up, 'iv_dn': iv_dn, 'iv_up': iv_up,
            'K_hi': k_hi['price_strike'].values[0], 'K_lo': k_lo['price_strike'].values[0],
            'vol_hi': vol_hi, 'vol_lo': vol_lo, 'texp': texp, 'scale_factor': scale
        })
    return pd.DataFrame(features)

# ------------------------------- Main Script -------------------------------

def main(args):
    with open(args.ticker_file, 'r') as f:
        tickers = [line.strip() for line in f if line.strip()]

    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date=args.start_date, end_date=args.end_date)
    dates = schedule.index.strftime('%Y-%m-%d').tolist()

    completed = set()
    if os.path.exists(args.completed_file):
        completed_df = pd.read_csv(args.completed_file, names=["ticker", "date"])
        completed = set(zip(completed_df["ticker"], completed_df["date"]))

    for ticker in tqdm(tickers):
        for date in tqdm(dates, leave=False):
            key = (ticker, date)
            if key in completed:
                continue
            try:
                time.sleep(args.delay)
                filtered_df = load_and_filter_day_ticker(ticker, date)
                if filtered_df is None or filtered_df.empty:
                    continue
                features_df = extract_vol_features(filtered_df)
                features_df['ticker'] = ticker

                filtered_df.to_csv(args.filtered_file, mode='a', header=not os.path.exists(args.filtered_file), index=False)
                features_df.to_csv(args.features_file, mode='a', header=not os.path.exists(args.features_file), index=False)

                with open(args.completed_file, "a") as f:
                    #print(f"✅ {ticker} {date} processed")
                    #print(f"writing to completed file {args.completed_file}")
                    f.write(f"{ticker},{date}\n")
        

                logging.info(f"✅ {ticker} {date} processed")
            except Exception as e:
                logging.error(f"❌ {ticker} {date} failed: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker_file', type=str, required=True, help='Path to file containing ticker list (one per line)')
    parser.add_argument('--start_date', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--filtered_file', type=str, default='filtered_data.csv')
    parser.add_argument('--features_file', type=str, default='features_data.csv')
    parser.add_argument('--completed_file', type=str, default='completed.csv')
    parser.add_argument('--delay', type=float, default=1.5, help='Delay between requests in seconds')

    args = parser.parse_args()
    main(args)
