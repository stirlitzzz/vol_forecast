# ledger.py
from __future__ import annotations
import pandas as pd, numpy as np, hashlib
from dataclasses import dataclass
from enum import Enum
from datetime import date
from .datetime_utils import to_date_only
from .time_utils import to_utc


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
