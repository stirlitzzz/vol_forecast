# simkit/strategies.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Iterable, Sequence, Optional, Tuple
from .datetime_utils import to_date_only
from .calendar_utils import add_nyse_days  # default; you can swap for calendar-days later
from .pricing import compute_option_price
from .ledger import TradeLedger
from .datetime_utils import to_date_only, add_calendar_days

def _vec1(x, n, name):
    import numpy as np
    a = np.asarray(x)
    if a.ndim == 0:
        a = np.repeat(a, n)
    elif a.ndim == 2 and 1 in a.shape:
        a = a.reshape(-1)
    elif a.ndim != 1:
        raise ValueError(f"{name} must be 1-D of length {n}, got shape {a.shape}")
    if len(a) != n:
        raise ValueError(f"{name} length {len(a)} != expected {n}")
    return a



def generate_straddle_trades(
    trades: Iterable[Tuple[object, str, str]],   # (timestamp, ticker, action['buy'|'sell'])
    close_matrix: pd.DataFrame,                  # wide: index=date, cols=tickers -> spot
    vol_matrix: pd.DataFrame,                    # wide: index=date, cols=tickers -> IV
    ledger: TradeLedger,
    ttm_days: int = 5,
    r: float = 0.0,
    q: float = 0.0,
    expiry_fn = None,                            # inject calendar rule; defaults to add_nyse_days
    return_df: bool = False,                     # set True to inspect/audit
    ttm_basis: str = "ACT/365",   
) -> Optional[pd.DataFrame]:
    """
    Builds a *straddle* (call+put) per input trade at ATM strike=spot.
    Quantity = 1/spot scaled by action sign (buy=+, sell=-).
    Records both legs via ledger; optionally returns the batch DataFrame.
    """
    expiry_fn = expiry_fn or add_nyse_days

    # Coerce input to DataFrame
    tdf = pd.DataFrame(trades, columns=["timestamp","underlying","action"])
    tdf["timestamp"] = pd.to_datetime(tdf["timestamp"], errors="coerce")
    if tdf["timestamp"].isna().any():
        bad = tdf.loc[tdf["timestamp"].isna()].index.tolist()
        raise ValueError(f"Unparsable timestamps at rows {bad}")

    # Normalize action early
    tdf["action"] = tdf["action"].astype(str).str.lower()
    if not tdf["action"].isin({"buy","sell"}).all():
        bad = tdf.loc[~tdf["action"].isin({"buy","sell"}), "action"].unique().tolist()
        raise ValueError(f"Invalid action(s): {bad}")

    # Date keys for lookups
    tdf["trade_date"] = to_date_only(tdf["timestamp"])

    # Vectorized lookups via MultiIndex (avoid per-row .loc)
    spot_long = close_matrix.stack()  # (date, ticker) -> price
    vol_long  = vol_matrix.stack()

    key = pd.MultiIndex.from_frame(tdf[["trade_date","underlying"]])
    try:
        spots = spot_long.reindex(key).to_numpy()
        vols  = vol_long.reindex(key).to_numpy()
    except Exception as e:
        raise KeyError(f"Lookup failed; ensure matrices are wide with date index & ticker columns: {e}")

    if np.isnan(spots).any():
        missing = key[np.isnan(spots)]
        raise KeyError(f"Missing spot for {list(dict.fromkeys(missing.tolist()))}")
    if np.isnan(vols).any():
        missing = key[np.isnan(vols)]
        raise KeyError(f"Missing vol for {list(dict.fromkeys(missing.tolist()))}")

    tdf["spot"]  = spots
    tdf["sigma"] = vols

    # Expiry dates (trading-day rule by default; swap expiry_fn later for calendar-days)
    tdf["expiry"] = expiry_fn(tdf["trade_date"].tolist(), ttm_days)

    #exact_days = (pd.to_datetime(tdf["expiry"]) - pd.to_datetime(tdf["trade_date"])).dt.days
    delta = pd.to_datetime(tdf["expiry"]) - pd.to_datetime(tdf["trade_date"])
    exact_days = delta.dt.days.to_numpy()              # <-- .dt.days, then to 1-D numpy

    if ttm_basis.upper() == "ACT/365":
        ttm = exact_days / 365.0
    elif ttm_basis.upper() in {"ACT/365.25", "ACT/36525"}:
        ttm = exact_days / 365.25
    elif ttm_basis.upper() in {"ACT/252", "TRADING"}:
        ttm = exact_days / 252.0
    else:
        raise ValueError(f"Unknown ttm_basis: {ttm_basis}")
    # Sizing & signs
    qty   = 1.0 / tdf["spot"].to_numpy(dtype=float)
    sign  = np.where(tdf["action"].to_numpy() == "buy", 1.0, -1.0)
    sqty  = qty * sign

    # Black–Scholes inputs (vectorized)
    n = len(tdf)
    ttm   = np.full(n, ttm_days/252.0, dtype=float)
    r_arr = np.full(n, r, dtype=float)
    q_arr = np.full(n, q, dtype=float)
    strike = tdf["spot"].to_numpy(dtype=float)

    strike_v = _vec1(strike, n, "strike")
    qty_v    = _vec1(qty,    n, "quantity")
    sqty_v   = _vec1(sqty,   n, "signed_quantity")

    batches = []
    for opt_type in ("call","put"):
        price = compute_option_price(
            spot=tdf["spot"].to_numpy(dtype=float),
            strike=strike,
            cp=[opt_type]*n,
            ttm=ttm, r=r_arr, q=q_arr, sigma=tdf["sigma"].to_numpy(dtype=float)
        )
        """
        df = pd.DataFrame({
            "timestamp":       tdf["timestamp"].to_numpy(),
            "action":          tdf["action"].to_numpy(),
            "instrument_type": "option",
            "underlying":      tdf["underlying"].to_numpy(),
            "option_type":     opt_type,
            "strike":          strike,
            "expiry":          tdf["expiry"].tolist(),   # python date objects
            "quantity":        qty,
            "price":           price,
            "signed_quantity": sqty,
            "total_cost":      sqty * price,
        })
        """
        price_v=_vec1(price, n, "price")
        df = pd.DataFrame({
            "timestamp":       tdf["timestamp"].to_numpy(),
            "action":          tdf["action"].to_numpy(),
            "instrument_type": ["option"] * n,
            "underlying":      tdf["underlying"].to_numpy(),
            "option_type":     [opt_type] * n,
            "strike":          strike_v,
            "expiry":          tdf["expiry"].tolist(),
            "quantity":        qty_v,
            "price":           price_v,
            "signed_quantity": sqty_v,
            "total_cost":      sqty_v * price_v,
        })
        batches.append(df)

    batch = pd.concat(batches, ignore_index=True)
    ledger.record_trades(batch)
    return batch if return_df else None


"""
def generate_straddle_trades_paste(
    trades,
    close_matrix,
    vol_matrix,
    ledger,
    ttm_days: int = 5,
    r: float = 0.0,
    q: float = 0.0,
    expiry_fn=None,                  # now defaults to calendar days
    return_df: bool = False,
    ttm_basis: str = "ACT/365",      # "ACT/365", "ACT/365.25", "ACT/252"
):
    expiry_fn = expiry_fn or add_calendar_days
    # ... (same up through tdf["spot"], tdf["sigma"])

    # Expiry: calendar days by default
    tdf["expiry"] = expiry_fn(tdf["trade_date"].tolist(), ttm_days)

    # Year fraction (use exact calendar-day distance)
    exact_days = (pd.to_datetime(tdf["expiry"]) - pd.to_datetime(tdf["trade_date"])).days
    if ttm_basis.upper() == "ACT/365":
        ttm = exact_days / 365.0
    elif ttm_basis.upper() in {"ACT/365.25", "ACT/36525"}:
        ttm = exact_days / 365.25
    elif ttm_basis.upper() in {"ACT/252", "TRADING"}:
        ttm = exact_days / 252.0
    else:
        raise ValueError(f"Unknown ttm_basis: {ttm_basis}")

    n = len(tdf)
    ttm   = np.asarray(ttm, dtype=float)           # vector
    r_arr = np.full(n, r, dtype=float)
    q_arr = np.full(n, q, dtype=float)
    strike = tdf["spot"].to_numpy(dtype=float)

    strike_v = _vec1(strike, n, "strike")
    qty_v    = _vec1(1.0 / strike_v, n, "quantity")
    sign     = np.where(tdf["action"].to_numpy() == "buy", 1.0, -1.0)
    sqty_v   = qty_v * sign

    batches = []
    for opt_type in ("call", "put"):
        price = compute_option_price(
            spot=tdf["spot"].to_numpy(dtype=float),
            strike=strike_v,
            cp=[opt_type]*n,
            ttm=ttm, r=r_arr, q=q_arr, sigma=tdf["sigma"].to_numpy(dtype=float),
        )
        price_v = _vec1(price, n, "price")
        df = pd.DataFrame({
            "timestamp":       tdf["timestamp"].to_numpy(),
            "action":          tdf["action"].to_numpy(),
            "instrument_type": ["option"] * n,
            "underlying":      tdf["underlying"].to_numpy(),
            "option_type":     [opt_type] * n,
            "strike":          strike_v,
            "expiry":          tdf["expiry"].tolist(),   # calendar dates
            "quantity":        qty_v,
            "price":           price_v,
            "signed_quantity": sqty_v,
            "total_cost":      sqty_v * price_v,
        })
        batches.append(df)

    batch = pd.concat(batches, ignore_index=True)
    ledger.record_trades(batch)
    return batch if return_df else None
    """