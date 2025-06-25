# forecast.py
import pandas as pd
from src.features import build_har_features, fit_har_regression


def create_forecast(har_1d, har_1w, har_1m, har_betas, av_period=5):
    averaged_betas = (
        har_betas[["beta_0", "beta_1d", "beta_1w", "beta_1m"]]
        .rolling(window=av_period, min_periods=av_period)
        .mean()
        .shift(0)
    )

    forecast = (
        averaged_betas["beta_0"].values[:, None]
        + averaged_betas["beta_1d"].values[:, None] * har_1d.loc[averaged_betas.index]
        + averaged_betas["beta_1w"].values[:, None] * har_1w.loc[averaged_betas.index]
        + averaged_betas["beta_1m"].values[:, None] * har_1m.loc[averaged_betas.index]
    )

    forecast.index = averaged_betas.index
    forecast.columns = har_1d.columns
    return forecast


def build_forecast_pipeline(realized, earnings_mask, config):
    har_windows = config["har_windows"]
    har_features = build_har_features(
        realized,
        term_1d=har_windows["short"],
        term_1w=har_windows["medium"],
        term_1m=har_windows["long"],
    )

    har_factors = fit_har_regression(
        realized, *(feat.shift(1) for feat in har_features), earnings_mask=earnings_mask
    )

    forecast = create_forecast(*har_features, har_factors)

    # Adjust for earnings
    realized_adjusted = realized.copy()
    realized_adjusted[earnings_mask] = forecast[earnings_mask]

    har_features_adj = build_har_features(
        realized_adjusted,
        term_1d=har_windows["short"],
        term_1w=har_windows["medium"],
        term_1m=har_windows["long"],
    )

    har_factors_adj = fit_har_regression(
        realized_adjusted,
        *(feat.shift(1) for feat in har_features_adj),
        earnings_mask=earnings_mask
    )
    forecast_final = create_forecast(*har_features_adj, har_factors_adj)
    return forecast_final
