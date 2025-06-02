# forecast.py
import pandas as pd

def create_forecast(har_1d, har_1w, har_1m, har_betas, av_period=5):
    averaged_betas = har_betas[["beta_0", "beta_1d", "beta_1w", "beta_1m"]]\
                        .rolling(window=av_period, min_periods=av_period)\
                        .mean().shift(0)

    forecast = (
        averaged_betas["beta_0"].values[:, None]
        + averaged_betas["beta_1d"].values[:, None] * har_1d.loc[averaged_betas.index]
        + averaged_betas["beta_1w"].values[:, None] * har_1w.loc[averaged_betas.index]
        + averaged_betas["beta_1m"].values[:, None] * har_1m.loc[averaged_betas.index]
    )

    forecast.index = averaged_betas.index
    forecast.columns = har_1d.columns
    return forecast