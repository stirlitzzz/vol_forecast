from __future__ import annotations
from zoneinfo import ZoneInfo

import pandas as pd
from datetime import date, datetime, time



TZ_NY, TZ_UTC = ZoneInfo("America/New_York"), ZoneInfo("UTC")

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
