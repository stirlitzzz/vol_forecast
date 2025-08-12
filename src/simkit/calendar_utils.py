import pandas as pd, numpy as np
import pandas_market_calendars as mcal
NY = "America/New_York"
nyse = mcal.get_calendar("XNYS")


def nyse_days(start_day, end_day):
    try:
        import pandas_market_calendars as mcal
        nyse = mcal.get_calendar("XNYS")
        days = nyse.valid_days(start_date=start_day, end_date=end_day, tz="America/New_York")
        return pd.to_datetime(days).tz_convert("America/New_York").date
    except Exception:
        # Fallback: US business days (not exact NYSE, but close); union with price dates will fix gaps
        from pandas.tseries.holiday import USFederalHolidayCalendar
        cbd = pd.offsets.CustomBusinessDay(calendar=USFederalHolidayCalendar())
        return pd.date_range(start_day, end_day, freq=cbd).date


def add_nyse_days(start_dates, n_days:int):
    sd = []
    for x in start_dates:
        t = pd.Timestamp(x)
        t = t.tz_localize(NY) if t.tzinfo is None else t.tz_convert(NY)
        sd.append(t.normalize())
    start_min = min(sd) - pd.Timedelta(days=5)
    end_max   = max(sd) + pd.Timedelta(days=n_days*5 + 5)
    vd = nyse.valid_days(start_date=start_min, end_date=end_max, tz=NY)
    pos = vd.get_indexer(pd.DatetimeIndex(sd), method="bfill")
    tgt = pos + n_days
    if (tgt >= len(vd)).any(): raise IndexError("calendar window too short")
    return list(vd[tgt].tz_convert(None).date)