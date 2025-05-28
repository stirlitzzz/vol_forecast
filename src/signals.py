# ===================== src/signals.py =========================
"""Signal construction & sizing."""
import numpy as np
import pandas as pd


def compute_signal_matrix(forecast: pd.DataFrame, implied: pd.DataFrame) -> pd.DataFrame:
    """Core alpha: forecast minus implied."""
    return forecast - implied.reindex_like(forecast)


def size_positions(
    signal: pd.DataFrame,
    implied: pd.DataFrame,
    method: str = "unit",
    threshold: float = 0.0,
) -> pd.DataFrame:
    """Return position size matrix (positive ⇒ long vol)."""
    mask = signal > threshold
    if method == "unit":
        return mask.astype(float)
    elif method == "prop":
        return signal.where(mask, 0.0)
    elif method == "risk":
        return (signal / implied).where(mask, 0.0)
    else:
        raise ValueError("method must be 'unit', 'prop', or 'risk'")
