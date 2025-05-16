#!/bin/bash
# Download SPX minute data from Polygon S3
python src/load_stock_aggregate_prices.py --start_date 2023-01-01\
        --end_date 2023-01-10\
        --tickers_file data/SPX_constituents_2023_filled.csv
