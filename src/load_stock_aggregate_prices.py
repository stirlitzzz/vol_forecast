import argparse
import boto3
from botocore.config import Config
import pandas as pd
from io import BytesIO
from pathlib import Path
from tqdm import tqdm
import gzip
import pandas_market_calendars as mcal
import os
from dotenv import load_dotenv

def get_spx_tickers(tickers_file):
    df_spx_constituents = pd.read_csv(tickers_file)
    df_spx_constituents = df_spx_constituents.rename(columns={"Symbol": "ticker"})
    tickers = set(df_spx_constituents["ticker"].unique())
    return tickers
"""
🧠 Minor Bonus Ideas (Optional later)
    •	Output directory as a parameter (--output_dir)
    •	Save filtered files compressed (.csv.gz)
    •	Add retries if S3 fails (network hiccups)
    •	Validate ticker file format at load time

"""
def main(start_date, end_date, tickers_file):
    load_dotenv()  # Load environment variables from .env file
    aws_key = os.getenv("AWS_KEY")
    aws_secret = os.getenv("AWS_SECRET")
    # Setup AWS session
    session = boto3.Session(
        aws_access_key_id=aws_key,
        aws_secret_access_key=aws_secret,
    )

    s3 = session.client(
        's3',
        endpoint_url='https://files.polygon.io',
        config=Config(signature_version='s3v4'),
    )

    bucket_name = 'flatfiles'
    prefix = 'us_stocks_sip/minute_aggs_v1'

    # Setup output directory
    output_dir = Path("./data/filtered_data")
    output_dir.mkdir(exist_ok=True)

    # Load tickers
    my_tickers = get_spx_tickers(tickers_file)

    # Get valid trading dates (NYSE calendar)
    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)
    dates = schedule.index

    missing_files = []

    # Process each date
    for date in tqdm(dates):
        date_str = date.strftime("%Y-%m-%d")
        year = date.strftime("%Y")
        month = date.strftime("%m")
        
        object_key = f"{prefix}/{year}/{month}/{date_str}.csv.gz"
        
        try:
            response = s3.get_object(Bucket=bucket_name, Key=object_key)
            compressed_data = response['Body'].read()
            with gzip.GzipFile(fileobj=BytesIO(compressed_data)) as gz:
                df = pd.read_csv(gz)
        except s3.exceptions.NoSuchKey:
            missing_files.append(date_str)
            continue
        except Exception as e:
            print(f"Failed to load {object_key}: {e}")
            continue

        if 'ticker' not in df.columns:
            print(f"No 'ticker' column in {object_key}")
            continue

        filtered = df[df['ticker'].isin(my_tickers)]

        if filtered.empty:
            print(f"No tickers found for {date_str}")
            continue

        output_file = output_dir / f"{date_str}_filtered.csv"
        filtered.to_csv(output_file, index=False)

    print(f"Missing files: {missing_files}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and filter Polygon daily files for S&P tickers.")
    parser.add_argument("--start_date", required=True, help="Start date in format YYYY-MM-DD")
    parser.add_argument("--end_date", required=True, help="End date in format YYYY-MM-DD")
    parser.add_argument("--tickers_file", required=True, help="CSV file with tickers (expects 'Symbol' column)")
    args = parser.parse_args()

    main(
        start_date=args.start_date,
        end_date=args.end_date,
        tickers_file=args.tickers_file
    )