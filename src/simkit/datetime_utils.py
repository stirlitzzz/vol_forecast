# src/simkit/datetime_utils.py
from __future__ import annotations
import pandas as pd
from datetime import date, datetime
from typing import Union

def to_date_only(x):
    """
    Convert scalars/Series/Index to Python date(s), stripping any tz.
    - Strings, datetime, pandas Timestamp → date
    - Series/Index → same shape of `date` values
    """
    if isinstance(x, (pd.Series, pd.Index)):
        return pd.to_datetime(x, errors="coerce").dt.tz_localize(None).dt.date

    # scalar path
    t = pd.Timestamp(x)  # handles date/datetime/str/np.datetime64
    # remove tz if present
    if t.tzinfo is not None:
        t = t.tz_localize(None)
    return t.date()



def add_calendar_days(start_dates, n_days: int):
    """
    Return python date(s) that are n *calendar* days after each start date.
    Accepts scalars/iterables of date/datetime/timestamp (naive or tz-aware).
    """
    out = []
    for x in start_dates:
        t = pd.Timestamp(x)                 # parse anything
        d = t.tz_localize(None).normalize() # strip tz; calendar math ignores tz
        out.append((d + pd.Timedelta(days=n_days)).date())
    return out