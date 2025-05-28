# ===================== src/load_data.py =======================
"""Data loading and alignment utilities."""
from pathlib import Path
import pandas as pd
import pandas_market_calendars as mcal

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
        nyse.valid_days(df["date"].min(), df["date"].max())
        .tz_convert(None)
        .normalize()
    )
    vol_matrix = (
        df.pivot(index="date", columns="ticker", values="annualized_vol_30min")
        .reindex(global_dates)
    )
    return vol_matrix

# ---------------- Earnings Calendar ------------- #

def load_earnings(filepath: str | Path = DATA_DIR / "earnings_calendar.csv") -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df = df.dropna(subset=["act_symbol"])
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df

# Build mask: set((ticker, excluded_date))

def build_earnings_mask(earnings_df: pd.DataFrame, days_buffer: int = 1) -> set[tuple[str, pd.Timestamp]]:
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
    import numpy as np

    iv_raw = pd.read_csv(filepath)
    iv_raw["c_date"] = pd.to_datetime(iv_raw["c_date"]).dt.normalize()
    iv_raw["expiry"] = pd.to_datetime(iv_raw["expiry"]).dt.normalize()
    iv_raw = iv_raw[iv_raw["texp"] <= max_dte_days / 365]

    earnings_lookup = (
        earnings_df.groupby("act_symbol")["date"].apply(set).to_dict()
    )

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
