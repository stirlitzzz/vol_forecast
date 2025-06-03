import pandas as pd
def filter_earnings(earnings, min_date, max_date):
    """Filter earnings data to only include dates within the range of realized volatility."""
    return earnings[(earnings['date'] >= min_date) & (earnings['date'] <= max_date)]


def filter_earnings_by_ticker(earnings, tickers):
    """Filter earnings data to only include specified tickers."""
    return earnings[earnings['act_symbol'].isin(tickers)]


def create_earnings_mask(realized, earnings_subset):
    """Create a mask for earnings dates in the realized volatility DataFrame."""
    # Initialize mask with False
    earnings_mask = pd.DataFrame(False, index=realized.index, columns=realized.columns)
    
    # Set True on earnings days
    for idx, row in earnings_subset.iterrows():
        date, ticker = row['date'], row['act_symbol']
        if date in earnings_mask.index and ticker in earnings_mask.columns:
            earnings_mask.at[date, ticker] = True
            
    return earnings_mask




def align_multiple(*dfs, align_columns=False):
    # Intersect indices (rows/dates) explicitly
    common_idx = dfs[0].index
    for df in dfs[1:]:
        common_idx = common_idx.intersection(df.index)

    if align_columns:
        # Optional column alignment
        common_cols = dfs[0].columns
        for df in dfs[1:]:
            common_cols = common_cols.intersection(df.columns)
        return [df.loc[common_idx, common_cols] for df in dfs]
    else:
        # Only align rows
        return [df.loc[common_idx] for df in dfs]