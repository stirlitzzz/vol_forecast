# pnl.py
import pandas as pd
from .time_utils import to_nyc
def realized_pnl_total(ledger):
    df = ledger.trades
    return (-df["total_cost"]).sum()
def daily_pnl_curve(ledger):
    df = ledger.trades.copy()
    df["day_ny"] = to_nyc(df["timestamp"]).dt.date
    df["cash_flow"] = -df["total_cost"]
    out = df.groupby("day_ny")["cash_flow"].sum().to_frame("daily_pnl")
    out["cum_pnl"] = out["daily_pnl"].cumsum()
    return out.reset_index()