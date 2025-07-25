import pandas as pd


def format_model_inputs(
    y_matrix: pd.DataFrame,
    x_matrices: dict[str, pd.DataFrame],
    threshold: float = None
):
    """
    Takes:
      - y_matrix: date x ticker label matrix
      - x_matrices: dict of name → date x ticker feature matrices
    Returns:
      - X_df: model-ready flat DataFrame
      - y_series: aligned target
      - common_idx: (date, ticker) MultiIndex
    """

    # Convert y to long-form
    if threshold is not None:
        y_binary = (y_matrix > threshold).stack().astype(int)
    else:
        y_binary = y_matrix.stack()

    X_df_parts = []
    common_idx = y_binary.index.copy()

    for name, mat in x_matrices.items():
        mat_long = mat.stack()
        common_idx = common_idx.intersection(mat_long.index)
        X_df_parts.append(mat_long.rename(name))

    # Truncate everything to common_idx
    X_df = pd.concat([col.loc[common_idx].reset_index(drop=True) for col in X_df_parts], axis=1)
    y_series = y_binary.loc[common_idx].reset_index(drop=True)


    return X_df, y_series, common_idx

def create_classifier_features(config, train_realized, train_implied, train_forecast, train_signal):
    signal = train_signal
    signal_rank = signal.rank(axis=1, pct=True)
    rv_rank = train_realized.rank(axis=1, pct=True)
    implied_rank = train_implied.rank(axis=1, pct=True)
    label_matrix = rv_rank - implied_rank #this is the label

    features = {
        "signal": signal,
        "signal_rank": signal_rank,
        "implied": train_implied,
        "forecast": train_forecast,
    }

    return features, label_matrix