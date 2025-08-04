# ===================== src/load_data.py =======================
"""Data loading and alignment utilities."""
from pathlib import Path
import pandas as pd
import pandas_market_calendars as mcal
import numpy as np
from src.analytic_utils import (
    filter_earnings,
    create_earnings_mask,
    filter_earnings_by_ticker,
)

DATA_DIR = Path("../data")
OUTPUT_DIR = Path("../output")
# Ensure directories exist\


# ---------------- Realized Vol ---------------- #


def load_realized_vol(filepath: str | Path = DATA_DIR / "all_vols.csv") -> pd.DataFrame:
    """Return pivoted realized vol matrix [date x ticker] for 30‑min realized vol."""
    df = pd.read_csv(filepath, parse_dates=["date"])
    df.sort_values(["ticker", "date"], inplace=True)

    nyse = mcal.get_calendar("NYSE")
    global_dates = (
        nyse.valid_days(df["date"].min(), df["date"].max()).tz_convert(None).normalize()
    )
    vol_matrix = df.pivot(
        index="date", columns="ticker", values="annualized_vol_30min"
    ).reindex(global_dates)
    return vol_matrix


# ---------------- Earnings Calendar ------------- #


def load_earnings(
    filepath: str | Path = DATA_DIR / "earnings_calendar.csv",
) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df = df.dropna(subset=["act_symbol"])
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


# Build mask: set((ticker, excluded_date))
def load_pivoted_field(
    field: str,
    filepath: str | Path = DATA_DIR / "all_vols.csv"
) -> pd.DataFrame:
    """
    Return a pivoted matrix [date x ticker] for the specified field.
    Example: field='close' or field='annualized_vol_30min'
    """
    df = pd.read_csv(filepath, parse_dates=["date"])
    df.sort_values(["ticker", "date"], inplace=True)

    nyse = mcal.get_calendar("NYSE")
    global_dates = (
        nyse.valid_days(df["date"].min(), df["date"].max()).tz_convert(None).normalize()
    )

    matrix = df.pivot(index="date", columns="ticker", values=field)
    return matrix.reindex(global_dates)

def build_earnings_mask(
    earnings_df: pd.DataFrame, days_buffer: int = 1
) -> set[tuple[str, pd.Timestamp]]:
    nyse = mcal.get_calendar("NYSE")
    min_d, max_d = earnings_df["date"].min(), earnings_df["date"].max()
    trade_days = nyse.valid_days(min_d, max_d).tz_convert(None).normalize()
    mask = set()
    for sym, d in zip(earnings_df["act_symbol"], earnings_df["date"]):
        if d not in trade_days:
            continue
        idx = trade_days.get_loc(d)
        for off in range(-days_buffer, days_buffer + 1):
            j = idx + off
            if 0 <= j < len(trade_days):
                mask.add((sym, trade_days[j]))
    return mask


# -------------- Implied Vol ---------------------- #


def load_implied_vol(
    filepath: str | Path = DATA_DIR / "features_data.csv",
    earnings_df: pd.DataFrame = None,
    max_dte_days: int = 40,
    mode: str = "avg2",
) -> pd.DataFrame:
    """Return implied vol matrix [date x ticker] (ATM IV) after earnings exclusion."""

    iv_raw = pd.read_csv(filepath)
    iv_raw["c_date"] = pd.to_datetime(iv_raw["c_date"]).dt.normalize()
    iv_raw["expiry"] = pd.to_datetime(iv_raw["expiry"]).dt.normalize()
    iv_raw = iv_raw[iv_raw["texp"] <= max_dte_days / 365]

    earnings_lookup = earnings_df.groupby("act_symbol")["date"].apply(set).to_dict()

    def earns_between(row):
        sy = row["ticker"]
        if sy not in earnings_lookup:
            return False
        return any(row["c_date"] <= e <= row["expiry"] for e in earnings_lookup[sy])

    iv_clean = iv_raw[~iv_raw.apply(earns_between, axis=1)].copy()

    if mode == "min":
        iv_summary = (
            iv_clean.sort_values("texp")
            .groupby(["c_date", "ticker"])
            .first()["atm_iv"]
            .unstack()
            .sort_index()
        )
    elif mode == "avg2":

        def avg_two(g):
            return g.nsmallest(2, "texp")["atm_iv"].mean()

        iv_summary = (
            iv_clean.groupby(["c_date", "ticker"]).apply(avg_two).unstack().sort_index()
        )
    else:
        raise ValueError("mode must be 'min' or 'avg2'")

    return iv_summary


def load_close_to_close_realized_volatility(csv_path, realized_vol_term=5):
    """
    Load prices from CSV, compute realized volatility and future realized volatility.

    Parameters:
        csv_path (str): Path to the CSV file.
        realized_vol_term (int): Window length for realized volatility and forecast shift.

    Returns:
        realized_vol (DataFrame): Current realized volatility.
        future_realized_vol (DataFrame): Future realized volatility shifted by realized_vol_term.
    """
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df_prices = df.pivot(index="date", columns="ticker", values="close")

    df_returns = df_prices.pct_change()

    realized_vol = np.sqrt(
        (df_returns**2).rolling(realized_vol_term, min_periods=realized_vol_term).mean()
    ) * np.sqrt(252)

    future_realized_vol = realized_vol.shift(-realized_vol_term)

    return realized_vol, future_realized_vol


def prepare_volatility_data(config):
    realized = load_realized_vol()
    earnings = load_earnings()

    # from src.analytic_utils only keeps earnings that are within the realized volatility data
    earnings_subset = filter_earnings(
        earnings, realized.index.min(), realized.index.max()
    )
    # from src.analytic_utils only keeps earnings for the tickers that are in the realized volatility data
    earnings_subset = filter_earnings_by_ticker(earnings_subset, realized.columns)

    # from src.analytic_utils creates a mask for the earnings that are within the realized volatility data
    earnings_mask = create_earnings_mask(realized, earnings_subset)

    close_realized, future_realized = load_close_to_close_realized_volatility(
        "../data/all_vols.csv", realized_vol_term=config["forecast_horizon"]
    )

    implied = load_implied_vol("../output/features_data.csv", earnings_subset)

    return {
        "realized": realized,
        "earnings_mask": earnings_mask,
        "close_realized": close_realized,
        "future_realized": future_realized,
        "implied": implied,
    }
