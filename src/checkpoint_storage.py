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



def save_all_checkpoints_v2(
    output_root: str,
    experiment_name: str,
    splits: Dict[str,  Dict[str, Any]],
    model: Any,
    metrics: Dict[str, Any],
    config: Dict[str, Any],
    notes: Optional[str] = "",
) -> str:
    from datetime import datetime
    import inspect

    run_id = f"{experiment_name}_{datetime.now():%Y%m%d_%H%M%S}"
    output_dir = os.path.join(output_root, run_id)
    os.makedirs(output_dir, exist_ok=True)

    print(f"📦 Saving to: {output_dir}")

    # Save each split (train/val/test)
    for name, dataset in splits.items():
        prefix = os.path.join(output_dir, name)
        dataset["X_df"].to_parquet(f"{prefix}_X_df.parquet")
        dataset["y_series"].to_frame("target").to_parquet(f"{prefix}_y.parquet")
        dataset["label_matrix"].to_parquet(f"{prefix}_label_matrix.parquet")

        if dataset.get("idx") is not None:
            idx = dataset["idx"]
            idx_df = idx.to_frame(index=False) if hasattr(idx, "to_frame") else pd.DataFrame(idx)
            idx_df.to_parquet(f"{prefix}_idx.parquet")

        if dataset.get("features") is not None:
            joblib.dump(dataset["features"], f"{prefix}_features.pkl")
    # Save model + metadata
    joblib.dump(model, os.path.join(output_dir, "model.pkl"))
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(config, f)

    # Save tracking info
    with open(os.path.join(output_dir, "README.txt"), "w") as f:
        f.write(f"Notebook: {inspect.stack()[-1].filename}\n")
        f.write(f"Saved: {datetime.now()}\n")
        f.write(f"Notes: {notes}\n")

    return output_dir



def load_all_checkpoints_v2(output_dir: str) -> Dict[str, Any]:
    """
    Loads all saved artifacts from a checkpoint directory created by save_all_checkpoints_v2.

    Returns:
        Dict[str, Any]: keys include 'train', 'val', 'test', 'model', 'metrics', 'config', etc.
    """
    result = {}
    splits = ["train", "val", "test"]  # add more if needed

    for name in splits:
        prefix = os.path.join(output_dir, name)
        if not os.path.exists(f"{prefix}_X_df.parquet"):
            continue  # skip missing splits

        result[name] = {
            "X_df": pd.read_parquet(f"{prefix}_X_df.parquet"),
            "y_series": pd.read_parquet(f"{prefix}_y.parquet").squeeze("columns"),
            "label_matrix": pd.read_parquet(f"{prefix}_label_matrix.parquet")
        }

        idx_path = f"{prefix}_idx.parquet"
        if os.path.exists(idx_path):
            result[name]["idx"] = pd.read_parquet(idx_path)

        features_path = f"{prefix}_features.pkl"
        if os.path.exists(features_path):
            result[name]["features"] = joblib.load(features_path)

    # Load global artifacts
    model_path = os.path.join(output_dir, "model.pkl")
    if os.path.exists(model_path):
        result["model"] = joblib.load(model_path)

    metrics_path = os.path.join(output_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            result["metrics"] = json.load(f)

    config_path = os.path.join(output_dir, "config.yaml")
    if os.path.exists(config_path):
        with open(config_path) as f:
            result["config"] = yaml.safe_load(f)

    readme_path = os.path.join(output_dir, "README.txt")
    if os.path.exists(readme_path):
        with open(readme_path) as f:
            result["notes"] = f.read()

    return result