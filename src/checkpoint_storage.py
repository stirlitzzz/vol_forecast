import os
import json
import yaml
import joblib
import pandas as pd
from typing import Dict, Optional, Any
from datetime import datetime


def save_all_checkpoints(
    output_root: str,
    experiment_name: str,
    X_df: pd.DataFrame,
    y: pd.Series,
    label_matrix: pd.DataFrame,
    model: Any,
    metrics: Dict[str, Any],
    config: Dict[str, Any],
    idx: Optional[pd.Index] = None,
    features: Optional[Dict[str, pd.DataFrame]] = None,
    extras: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Saves all outputs from an experiment into a unique timestamped folder.

    Returns:
        str: path to the saved output directory
    """
    run_id = f"{experiment_name}_{datetime.now():%Y%m%d_%H%M%S}"
    output_dir = os.path.join(output_root, run_id)
    os.makedirs(output_dir, exist_ok=True)

    print(f"📦 Saving experiment outputs to: {output_dir}")

    # Save core artifacts
    print(f"Saving X_df to {os.path.join(output_dir, 'X_df.parquet')}")
    X_df.to_parquet(os.path.join(output_dir, "X_df.parquet"))
    print(f"X_df saved to {os.path.join(output_dir, 'X_df.parquet')}")
    y.to_frame("target").to_parquet(os.path.join(output_dir, "y.parquet"))
    label_matrix.to_parquet(os.path.join(output_dir, "label_matrix.parquet"))
    joblib.dump(model, os.path.join(output_dir, "model.pkl"))

    # Save optional
    if idx is not None:
        idx_df = idx.to_frame(index=False) if hasattr(idx, "to_frame") else pd.DataFrame(idx)
        idx_df.to_parquet(os.path.join(output_dir, "idx.parquet"))

    if features is not None:
        joblib.dump(features, os.path.join(output_dir, "features.pkl"))

    if extras:
        for name, obj in extras.items():
            path = os.path.join(output_dir, f"{name}.pkl")
            joblib.dump(obj, path)

    # Save metrics + config
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(config, f)

    return output_dir



def load_all_checkpoints(output_dir: str) -> Dict[str, Any]:
    """
    Loads all standard experiment artifacts from a checkpoint directory.

    Returns:
        Dict[str, Any]: includes X_df, y, label_matrix, model, metrics, config, etc.
    """
    result = {}

    # Core files
    result["X_df"] = pd.read_parquet(os.path.join(output_dir, "X_df.parquet"))
    result["y"] = pd.read_parquet(os.path.join(output_dir, "y.parquet")).squeeze("columns")
    result["label_matrix"] = pd.read_parquet(os.path.join(output_dir, "label_matrix.parquet"))
    result["model"] = joblib.load(os.path.join(output_dir, "model.pkl"))

    # Optional files
    idx_path = os.path.join(output_dir, "idx.parquet")
    if os.path.exists(idx_path):
        result["idx"] = pd.read_parquet(idx_path)#.set_index(["level_0", "level_1"])

    features_path = os.path.join(output_dir, "features.pkl")
    if os.path.exists(features_path):
        result["features"] = joblib.load(features_path)

    # Metrics + config
    with open(os.path.join(output_dir, "metrics.json")) as f:
        result["metrics"] = json.load(f)

    with open(os.path.join(output_dir, "config.yaml")) as f:
        result["config"] = yaml.safe_load(f)

    return result



