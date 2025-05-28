# ===================== src/features.py ========================
"""Feature engineering utilities (HAR etc.)."""
import numpy as np
import pandas as pd

def build_har_features(vol_matrix: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    har_1d = vol_matrix.shift(1)
    har_1w = har_1d.rolling(5, min_periods=5).mean()
    har_1m = har_1d.rolling(21, min_periods=21).mean()
    return har_1d, har_1w, har_1m

# Example future realized vol (n‑day ahead average)

def build_future_realized(vol_matrix: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    return vol_matrix.rolling(n).mean().shift(-n)
