# ===================== src/features.py ========================
"""Feature engineering utilities (HAR etc.)."""
import numpy as np
import pandas as pd


def build_har_features(
    vol_matrix: pd.DataFrame, term_1d=1, term_1w=5, term_1m=21
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    har_1d = vol_matrix.rolling(term_1d, min_periods=term_1d).mean()
    har_1w = vol_matrix.rolling(term_1w, min_periods=term_1w).mean()
    har_1m = vol_matrix.rolling(term_1m, min_periods=term_1m).mean()
    return har_1d, har_1w, har_1m


# Example future realized vol (n‑day ahead average)


def build_future_realized(vol_matrix: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    return vol_matrix.rolling(n).mean().shift(-n)


def fit_har_regression(
    vol_matrix, har_1d, har_1w, har_1m, earnings_mask=None, min_obs=10
):
    """
    Perform cross-sectional HAR regression per day.

    Parameters:
        vol_matrix (DataFrame): Realized volatilities (dates × tickers).
        har_1d, har_1w, har_1m (DataFrame): HAR features aligned with vol_matrix.
        earnings_mask (DataFrame, optional): Boolean mask for earnings dates (dates × tickers).
        min_obs (int): Minimum observations required per day.

    Returns:
        DataFrame: Daily regression results (betas, rmse, n_obs).
    """
    results = []
    global_dates = vol_matrix.index.intersection(har_1d.index)

    # print(f"Fitting HAR regression for {len(global_dates)} dates...")
    for date in global_dates:
        if (
            date not in vol_matrix.index
            or date not in har_1d.index
            or date not in har_1w.index
            or date not in har_1m.index
            or (earnings_mask is not None and date not in earnings_mask.index)
        ):
            continue  # skip if any data missing

        # Rest of your regression code...
        # print(f"Processing date: {date.strftime('%Y-%m-%d')}")
        y = vol_matrix.loc[date].values
        X = np.vstack(
            [
                np.ones_like(y),
                har_1d.loc[date].values,
                har_1w.loc[date].values,
                har_1m.loc[date].values,
            ]
        ).T

        # Construct valid mask
        valid = (~np.isnan(X).any(axis=1)) & (~np.isnan(y))

        # Include earnings mask if provided
        if earnings_mask is not None:
            valid &= ~earnings_mask.loc[date].values

        if valid.sum() < min_obs:
            continue

        beta, residuals, _, _ = np.linalg.lstsq(X[valid], y[valid], rcond=None)

        results.append(
            {
                "date": date,
                "beta_0": beta[0],
                "beta_1d": beta[1],
                "beta_1w": beta[2],
                "beta_1m": beta[3],
                "rmse": np.sqrt(np.mean((y[valid] - X[valid] @ beta) ** 2)),
                "n_obs": valid.sum(),
            }
        )

    return pd.DataFrame(results).set_index("date")
