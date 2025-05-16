import argparse
import pandas as pd
from pathlib import Path
import sys
sys.path.append('./src')  # only needed if you're not packaging yet

from vol_utils import compute_realized_volatility

def parse_args():
    parser = argparse.ArgumentParser(description="Compute realized vol for multiple tickers and dates.")
    parser.add_argument('--input_dir', type=str, required=True, help="Path to directory with filtered input files")
    parser.add_argument('--output_file', type=str, required=True, help="Path to output CSV file")
    parser.add_argument('--intervals', nargs='+', default=['5min'], help="List of intervals (e.g. 1min 5min 15min)")
    parser.add_argument('--estimator', type=str, default='mean_square',
                        choices=['mean_square', 'std', 'parkinson'],
                        help="Volatility estimator to use")
    return parser.parse_args()

def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)
    intervals = args.intervals
    estimator = args.estimator

    all_results = []

    for input_file in input_dir.glob("*_filtered.csv"):
        result_df = compute_realized_volatility(
            input_file,
            intervals=intervals,
            estimator=estimator
        )
        if not result_df.empty:
            all_results.append(result_df)

    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_df.to_csv(output_file, index=False)
        print(f"Saved volatility data to {output_file}")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()