# ledger_portfolio.py

from __future__ import annotations

from typing import Callable
from enum import Enum
from datetime import datetime, date, time
from zoneinfo import ZoneInfo

import hashlib
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

nyse = mcal.get_calendar("XNYS")


# =========================
# Timezone constants
# =========================
TZ_NY  = ZoneInfo("America/New_York")
TZ_UTC = ZoneInfo("UTC")


# =========================
# Small enums for safety
# =========================
class Action(str, Enum):
    BUY  = "buy"
    SELL = "sell"

class OptionType(str, Enum):
    CALL = "call"
    PUT  = "put"


# =========================
# Time helpers (DST-safe)
# =========================
def to_utc(ts, assume_tz=TZ_NY, ambiguous="infer", nonexistent="shift_forward"):
    """
    Normalize to tz-aware UTC (Series/Index or scalar).
    If input is naive, assume `assume_tz` then convert.
    Handles DST via `ambiguous` and `nonexistent` args (see pandas tz_localize docs).
    """
    if isinstance(ts, (pd.Series, pd.Index)):
        out = pd.to_datetime(ts, errors="coerce")
        if out.dt.tz is None:
            out = (out
                   .dt.tz_localize(assume_tz, ambiguous=ambiguous, nonexistent=nonexistent)
                   .dt.tz_convert(TZ_UTC))
        else:
            out = out.dt.tz_convert(TZ_UTC)
        return out
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize(assume_tz, ambiguous=ambiguous, nonexistent=nonexistent).tz_convert(TZ_UTC)
    else:
        t = t.tz_convert(TZ_UTC)
    return t

def to_nyc(ts, assume_tz=TZ_UTC):
    """
    Return tz-aware America/New_York timestamps (Series/Index or scalar).
    If input is naive, assume `assume_tz` then convert.
    """
    if isinstance(ts, (pd.Series, pd.Index)):
        out = pd.to_datetime(ts, errors="coerce")
        if out.dt.tz is None:
            out = out.dt.tz_localize(assume_tz).dt.tz_convert(TZ_NY)
        else:
            out = out.dt.tz_convert(TZ_NY)
        return out
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize(assume_tz).tz_convert(TZ_NY)
    else:
        t = t.tz_convert(TZ_NY)
    return t

def ny_midnight(ts, assume_tz=TZ_UTC):
    """Midnight-in-NY (tz-aware Timestamp/Series)."""
    ny = to_nyc(ts, assume_tz=assume_tz)
    return ny.dt.normalize() if isinstance(ny, (pd.Series, pd.Index)) else ny.normalize()

def ny_date(ts, assume_tz=TZ_UTC):
    """NY calendar date (plain date)."""
    ny = to_nyc(ts, assume_tz=assume_tz)
    return ny.dt.date if isinstance(ny, (pd.Series, pd.Index)) else ny.date()

def session_close_utc(d: date, close: time = time(16, 30)) -> pd.Timestamp:
    """NY close for a given date, expressed in UTC."""
    return pd.Timestamp(datetime.combine(d, close, tzinfo=TZ_NY)).tz_convert(TZ_UTC)

def to_date_only(x):
    """Return python date(s) from datetime-like/strings; keeps Series/Index shape."""
    if isinstance(x, (pd.Series, pd.Index)):
        return pd.to_datetime(x, errors="coerce").dt.tz_localize(None).dt.date
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    return pd.to_datetime(x).date()


# =========================
# Option payoff (vectorized)
# =========================
def compute_payoff_vec(spots: np.ndarray,
                       strikes: np.ndarray,
                       opt_types: np.ndarray) -> np.ndarray:
    is_call    = (opt_types == "call")
    call_pay   = np.maximum(spots - strikes, 0)
    put_pay    = np.maximum(strikes - spots, 0)
    return np.where(is_call, call_pay, put_pay)

#========================
# NYSE calendar helpers
#========================
def add_nyse_days(start_dates, n_days):
    out = []
    print(f"add_nyse_days: start_dates: {start_dates}")
    print(f"add_nyse_days: n_days: {n_days}")
    start_dates=[pd.to_datetime(dt).tz_localize("America/New_York") for dt in start_dates]
    #start_dates=start_dates.to_list()
    print(f"add_nyse_days: start_dates: {start_dates}")
    for dt in start_dates:
        # valid_days includes start date, so we skip +1 for same day expiry
        print(f"add_nyse_days: dt: {dt}")
        valid_days = nyse.valid_days(
            start_date=dt.date(),
            end_date=dt + pd.Timedelta(days=n_days*2),  # overshoot buffer
            tz="America/New_York"
        )
        print(f"add_nyse_days: valid_days: {valid_days}")
        # index of start date in valid_days
        start_idx = valid_days.get_loc(
            pd.Timestamp(dt, tz="America/New_York"), method="pad",tz="America/New_York"
        )
        target_idx = start_idx + n_days
        out.append(valid_days[target_idx].date())
    return out


# =========================
# Schema (dtypes guidance)
# =========================
TRADE_SCHEMA = {
    "timestamp": "datetime64[ns, UTC]",  # checked, not force-cast
    "action": "string",
    "instrument_type": "string",
    "underlying": "string",
    "option_type": "string",
    "strike": "float64",
    "expiry": "object",                  # python date
    "quantity": "float64",
    "price": "float64",
    "signed_quantity": "float64",
    "total_cost": "float64",
    "trade_id": "string",
}


# =========================
# TradeLedger
# =========================
class TradeLedger:
    COLS = list(TRADE_SCHEMA.keys())

    def __init__(self):
        self.trades_df = pd.DataFrame(columns=self.COLS)

    # --------- internal helpers ----------
    @staticmethod
    def _row_trade_id(row: pd.Series) -> str:
        # Deterministic key for idempotency; include all economics + identity fields
        key = (
            f"{pd.Timestamp(row['timestamp']).value}|"
            f"{row['action']}|{row['instrument_type']}|{row['underlying']}|"
            f"{row['option_type']}|{row['strike']}|{row['expiry']}|"
            f"{row['quantity']}|{row['price']}"
        )
        return hashlib.blake2b(key.encode(), digest_size=12).hexdigest()

    def _normalize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # Required columns check
        required = [
            "timestamp","action","instrument_type","underlying",
            "option_type","strike","expiry","quantity","price"
        ]
        missing = [c for c in required if c not in out.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Timestamps -> UTC tz-aware
        out["timestamp"] = to_utc(out["timestamp"])
        if out["timestamp"].isna().any():
            raise ValueError("Some timestamps could not be parsed to valid datetime.")

        # Expiry -> python date
        out["expiry"] = to_date_only(out["expiry"])
        if pd.isna(out["expiry"]).any():
            raise ValueError("Some expiry values could not be parsed to valid date.")

        # Categorical/string normalization
        for c in ["action","instrument_type","option_type","underlying"]:
            if out[c].isna().any():
                raise ValueError(f"Column {c} contains NaNs.")
            out[c] = out[c].astype(str)

        # Validate enumerations
        if not out["action"].isin({Action.BUY, Action.SELL}).all():
            bad = out.loc[~out["action"].isin({Action.BUY, Action.SELL}), "action"].unique().tolist()
            raise ValueError(f"Invalid action(s): {bad}")
        if not out["option_type"].isin({OptionType.CALL, OptionType.PUT}).all():
            bad = out.loc[~out["option_type"].isin({OptionType.CALL, OptionType.PUT}), "option_type"].unique().tolist()
            raise ValueError(f"Invalid option_type(s): {bad}")

        # Numerics
        for c in ["strike","quantity","price"]:
            out[c] = pd.to_numeric(out[c], errors="coerce")
            if out[c].isna().any():
                raise ValueError(f"Column {c} contains non-numeric values.")

        # Value constraints
        if (out["quantity"] <= 0).any():
            raise ValueError("quantity must be > 0")
        if (out["strike"] <= 0).any():
            raise ValueError("strike must be > 0")
        if (out["price"] < 0).any():
            raise ValueError("price must be >= 0")

        # Derived
        if "signed_quantity" not in out.columns:
            out["signed_quantity"] = np.where(out["action"] == Action.BUY, out["quantity"], -out["quantity"])
        if "total_cost" not in out.columns:
            out["total_cost"] = out["signed_quantity"] * out["price"]

        # trade_id for idempotency
        out["trade_id"] = out.apply(self._row_trade_id, axis=1)

        # Final column order & dtypes (light cast; keep expiry as date, timestamp already tz-aware)
        out = out[self.COLS]
        cast_map = {k: v for k, v in TRADE_SCHEMA.items() if k not in ("timestamp", "expiry")}
        out = out.astype(cast_map)

        return out

    # --------- public API ----------
    def record_trades(self, new_trades: pd.DataFrame):
        normalized = self._normalize_df(new_trades)
        if not self.trades_df.empty:
            # Drop exact duplicates by trade_id (idempotent inserts)
            mask_dup = normalized["trade_id"].isin(self.trades_df["trade_id"])
            normalized = normalized.loc[~mask_dup]
            if normalized.empty:
                return
        self.trades_df = pd.concat([self.trades_df, normalized], ignore_index=True)
        self._post_write_asserts()

    def record_trade(self, *,
                     timestamp,
                     action: str,
                     instrument_type: str,
                     underlying: str,
                     option_type: str,
                     strike: float,
                     expiry,
                     quantity: float,
                     price: float):
        row = pd.DataFrame([{
            "timestamp": timestamp,
            "action": action,
            "instrument_type": instrument_type,
            "underlying": underlying,
            "option_type": option_type,
            "strike": strike,
            "expiry": expiry,
            "quantity": quantity,
            "price": price
        }])
        self.record_trades(row)

    @property
    def trades(self) -> pd.DataFrame:
        # Sorted view
        return self.trades_df.sort_values("timestamp").reset_index(drop=True)

    # --------- integrity tripwires ----------
    def _post_write_asserts(self):
        if self.trades_df.empty:
            return
        # timestamp must be tz-aware UTC
        assert str(self.trades_df["timestamp"].dtype).endswith(", UTC]"), "timestamps must be UTC tz-aware"
        # expiry must be python date
        assert self.trades_df["expiry"].map(lambda d: isinstance(d, date)).all(), "expiry must be date objects"


# =========================
# Portfolio snapshot
# =========================
class PortfolioView:
    def __init__(self, ledger: TradeLedger):
        self.ledger = ledger

    def positions_at(self, as_of) -> pd.DataFrame:
        """
        as_of: date or datetime. If date, uses NY close of that date.
        Returns aggregated open positions (UTC compare).
        """
        if isinstance(as_of, date) and not isinstance(as_of, datetime):
            as_of_utc = session_close_utc(as_of)
        else:
            as_of_utc = to_utc(as_of)

        df = self.ledger.trades_df
        if df.empty:
            return df

        df = df[df["timestamp"] <= as_of_utc]
        grouped = (
            df.groupby(["instrument_type","underlying","option_type","strike","expiry"],
                       dropna=False)["signed_quantity"].sum().reset_index()
        )
        return grouped[grouped["signed_quantity"] != 0]


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


# =========================
# Minimal self-test (optional)
# =========================
if __name__ == "__main__":
    ledg = TradeLedger()
    ledg.record_trade(
        timestamp=datetime(2025, 8, 8, 10, 15),  # naive (assume NY) -> UTC internally
        action="buy", instrument_type="option", underlying="SPX",
        option_type="call", strike=5500, expiry=date(2025, 8, 8),
        quantity=1, price=12.5
    )

    def px_lookup(symbols: np.ndarray, d: date) -> np.ndarray:
        return np.full(len(symbols), 5540.0, dtype=float)

    expire_trades_vectorized(ledg, date(2025, 8, 8), px_lookup)

    print(ledg.trades.assign(timestamp_ny=to_nyc(ledg.trades["timestamp"])))
    print(audit_ledger_quick(ledg))