# expiry.py
import numpy as np, pandas as pd
from datetime import date, time
from typing import Callable
from .time_utils import session_close_utc
from .pricing import compute_payoff_vec
from .portfolio import PortfolioView  # if you split it; otherwise inline query
from .ledger import TradeLedger
# ... your preview_expiries + expire_trades_vectorized ...



# =========================
# Expiry (vectorized close)
# =========================






def expire_trades_vectorized_v1(
    ledger: TradeLedger,
    as_of_date: date,
    price_lookup_fn: Callable[[np.ndarray, date], np.ndarray],
    close_time: time = time(16, 30)
):
    """
    Close all options with expiry <= as_of_date at NY close on as_of_date.
    price_lookup_fn(symbols: np.ndarray, d: date) -> np.ndarray of spots (float)
    """
    close_ts_utc = session_close_utc(as_of_date, close=close_time)
    opens = PortfolioView(ledger).positions_at(close_ts_utc)
    if opens.empty:
        return

    to_close = opens[opens["expiry"] <= as_of_date].copy()
    if to_close.empty:
        return

    symbols = to_close["underlying"].to_numpy()
    spots = price_lookup_fn(symbols, as_of_date)

    if spots is None or len(spots) != len(symbols):
        raise ValueError(f"price_lookup_fn returned invalid length: {len(spots) if spots is not None else None} vs {len(symbols)}")

    payoffs = compute_payoff_vec(
        np.asarray(spots, dtype=float),
        to_close["strike"].to_numpy(dtype=float),
        to_close["option_type"].to_numpy(dtype=str)
    )

    actions = np.where(to_close["signed_quantity"].to_numpy(dtype=float) > 0, "sell", "buy")
    qtys    = np.abs(to_close["signed_quantity"].to_numpy(dtype=float))

    batch = pd.DataFrame({
        "timestamp":       close_ts_utc,
        "action":          actions,
        "instrument_type": "option",
        "underlying":      to_close["underlying"].to_numpy(dtype=str),
        "option_type":     to_close["option_type"].to_numpy(dtype=str),
        "strike":          to_close["strike"].to_numpy(dtype=float),
        "expiry":          to_close["expiry"].to_numpy(dtype=object),  # keep as date
        "quantity":        qtys,
        "price":           payoffs,
    })
    ledger.record_trades(batch)

def expire_trades_vectorized(
    ledger: TradeLedger,
    as_of_date: date,
    price_lookup_fn: Callable[[np.ndarray, date], np.ndarray],
    close_time: time = time(16,30),
    return_audit: bool = False,   # <— new
):
    close_ts_utc = session_close_utc(as_of_date, close=close_time)
    opens = PortfolioView(ledger).positions_at(close_ts_utc)
    if opens.empty:
        return pd.DataFrame() if return_audit else None

    to_close = opens[opens["expiry"] <= as_of_date].copy()
    if to_close.empty:
        return pd.DataFrame() if return_audit else None

    symbols = to_close["underlying"].to_numpy()
    spots   = price_lookup_fn(symbols, as_of_date)

    payoffs = compute_payoff_vec(
        np.asarray(spots, dtype=float),
        to_close["strike"].to_numpy(dtype=float),
        to_close["option_type"].to_numpy(dtype=str)
    )
    actions = np.where(to_close["signed_quantity"].to_numpy(dtype=float) > 0, "sell", "buy")
    qtys    = np.abs(to_close["signed_quantity"].to_numpy(dtype=float))

    batch = pd.DataFrame({
        "timestamp":       close_ts_utc,
        "action":          actions,
        "instrument_type": "option",
        "underlying":      to_close["underlying"].to_numpy(),
        "option_type":     to_close["option_type"].to_numpy(),
        "strike":          to_close["strike"].to_numpy(dtype=float),
        "expiry":          to_close["expiry"].to_numpy(object),
        "quantity":        qtys,
        "price":           payoffs,
    })

    # Build audit before writing
    audit = pd.DataFrame({
        "close_timestamp_utc": close_ts_utc,
        "underlying":          batch["underlying"],
        "option_type":         batch["option_type"],
        "strike":              batch["strike"],
        "expiry":              batch["expiry"],
        "action":              batch["action"],
        "qty_to_close":        batch["quantity"],
        "spot":                np.asarray(spots, dtype=float),
        "payoff":              payoffs,
    })

    ledger.record_trades(batch)
    return audit if return_audit else None


# =========================
# Quick audits (manual checks)
# =========================
def audit_ledger_quick(ledger: TradeLedger, year_lo=2000, year_hi=2100) -> pd.DataFrame:
    """Lightweight scan for NaT/NaN & suspicious expiry years."""
    df = ledger.trades_df
    issues = []
    if df.empty:
        return pd.DataFrame(columns=["row","issue","detail"])

    # timestamps
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    for i in df.index[ts.isna()]:
        issues.append((int(i), "timestamp_invalid", df.loc[i, "timestamp"]))

    # expiry
    exp = pd.to_datetime(df["expiry"], errors="coerce").dt.date
    for i in df.index[exp.isna()]:
        issues.append((int(i), "expiry_invalid", df.loc[i, "expiry"]))
    for i, d in zip(df.index, exp):
        if pd.notna(d) and (d.year < year_lo or d.year > year_hi):
            issues.append((int(i), "expiry_suspicious_year", d))

    # numerics
    for c in ["strike","quantity","price","signed_quantity","total_cost"]:
        vals = pd.to_numeric(df[c], errors="coerce")
        for i in df.index[vals.isna()]:
            issues.append((int(i), f"{c}_non_numeric", df.loc[i, c]))

    return pd.DataFrame(issues, columns=["row","issue","detail"])

def assert_sane_ranges(ledger: TradeLedger, year_lo=2000, year_hi=2100):
    """Raise on suspicious expiry years; quick tripwire."""
    df = ledger.trades_df
    if df.empty:
        return
    exp = pd.to_datetime(df["expiry"], errors="coerce").dt.date
    bad = [d for d in exp if pd.isna(d) or d.year < year_lo or d.year > year_hi]
    if bad:
        raise AssertionError(f"Suspicious expiry dates: {bad[:5]}{'…' if len(bad)>5 else ''}")


def preview_expiries(
    ledger: TradeLedger,
    as_of_date: date,
    price_lookup_fn: Callable[[np.ndarray, date], np.ndarray],
    close_time: time = time(16,30),
) -> pd.DataFrame:
    """
    Compute what WOULD be expired on as_of_date at NY close, with full details.
    Does NOT write to the ledger. Returns a DataFrame you can inspect/assert.
    """
    close_ts_utc = session_close_utc(as_of_date, close=close_time)
    opens = PortfolioView(ledger).positions_at(close_ts_utc)
    if opens.empty:
        return pd.DataFrame(columns=[
            "close_timestamp_utc","underlying","option_type","strike","expiry",
            "open_signed_qty","action","qty_to_close","spot","payoff"
        ])

    to_close = opens[opens["expiry"] <= as_of_date].copy()
    if to_close.empty:
        return pd.DataFrame(columns=[
            "close_timestamp_utc","underlying","option_type","strike","expiry",
            "open_signed_qty","action","qty_to_close","spot","payoff"
        ])

    symbols = to_close["underlying"].to_numpy()
    spots   = price_lookup_fn(symbols, as_of_date)

    payoffs = compute_payoff_vec(
        np.asarray(spots, dtype=float),
        to_close["strike"].to_numpy(dtype=float),
        to_close["option_type"].to_numpy(dtype=str)
    )
    actions = np.where(to_close["signed_quantity"].to_numpy(dtype=float) > 0, "sell", "buy")
    qtys    = np.abs(to_close["signed_quantity"].to_numpy(dtype=float))

    out = pd.DataFrame({
        "close_timestamp_utc": close_ts_utc,
        "underlying":          to_close["underlying"].to_numpy(),
        "option_type":         to_close["option_type"].to_numpy(),
        "strike":              to_close["strike"].to_numpy(dtype=float),
        "expiry":              to_close["expiry"].to_numpy(object),   # date
        "open_signed_qty":     to_close["signed_quantity"].to_numpy(dtype=float),
        "action":              actions,
        "qty_to_close":        qtys,
        "spot":                np.asarray(spots, dtype=float),
        "payoff":              payoffs,
    })
    # Convenience: what ledger would record as price/total_cost for the close
    out["expected_close_price"] = out["payoff"]
    out["expected_signed_qty"]  = np.where(out["action"]=="buy", out["qty_to_close"], -out["qty_to_close"])
    out["expected_total_cost"]  = out["expected_signed_qty"] * out["expected_close_price"]
    return out