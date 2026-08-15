"""Train multi-component smell mixture deconvolution and ratio estimation model.

Given a sensor recording of a compound smell mixture (e.g. 20% Almond + 80% Orange),
this model predicts the constituent odor components and their relative percentage concentrations.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.preprocess import PreprocessConfig, preprocess_trial
from src.features.extract_features import extract_window_feature_vector, feature_names_for_sensors
from src.models.pipeline_io import save_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multi-output smell mixture deconvolution model.")
    parser.add_argument("--data-root", default="data/raw/SmellNet/mixture_data", help="Mixture data directory")
    parser.add_argument("--output-dir", default="models/mixture_regressor", help="Output directory")
    parser.add_argument("--target-points", type=int, default=300, help="Resample length")
    parser.add_argument("--warmup-ratio", type=float, default=0.05, help="Warmup trim ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-files", type=int, default=0, help="Cap for fast debugging")
    return parser.parse_args()


def locate_local_csv(raw_filepath: str, mixture_dir: Path, filename_map: dict[str, Path]) -> Path | None:
    fname = Path(raw_filepath).name
    if fname in filename_map:
        return filename_map[fname]
    # Fallback search
    candidates = list(mixture_dir.rglob(fname))
    if candidates:
        return candidates[0]
    return None


def load_mixture_split(
    index_csv: Path,
    mixture_dir: Path,
    cfg: PreprocessConfig,
    filename_map: dict[str, Path],
    sensor_cols_ref: list[str] | None = None,
    max_rows: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    df_index = pd.read_csv(index_csv)
    if max_rows > 0:
        df_index = df_index.head(max_rows)

    target_cols = [c for c in df_index.columns if c.startswith("label_")]
    clean_target_names = [c.replace("label_", "") for c in target_cols]

    features_list: list[np.ndarray] = []
    targets_list: list[np.ndarray] = []

    for _, row in df_index.iterrows():
        raw_fp = str(row["filepath"])
        local_path = locate_local_csv(raw_fp, mixture_dir, filename_map)
        if local_path is None or not local_path.exists():
            continue

        try:
            raw_df = pd.read_csv(local_path)
            proc_df, _, cols = preprocess_trial(raw_df, sensor_columns=sensor_cols_ref, config=cfg)
            if sensor_cols_ref is None:
                sensor_cols_ref = cols

            values = proc_df[sensor_cols_ref].to_numpy(dtype=float)
            feat = extract_window_feature_vector(values, sensor_cols_ref)
            target_vec = row[target_cols].to_numpy(dtype=float)

            features_list.append(feat)
            targets_list.append(target_vec)
        except Exception:
            continue

    if not features_list or sensor_cols_ref is None:
        raise RuntimeError(f"Could not load any usable trials from {index_csv}")

    X = np.vstack(features_list)
    Y = np.vstack(targets_list)
    return X, Y, clean_target_names, sensor_cols_ref


def evaluate_mixture_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    # Project predictions to simplex (non-negative and sum to 1)
    y_pred_proj = np.maximum(y_pred, 0.0)
    row_sums = y_pred_proj.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    y_pred_proj = y_pred_proj / row_sums

    mae = float(mean_absolute_error(y_true, y_pred_proj))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred_proj)))
    r2 = float(r2_score(y_true, y_pred_proj))

    # Cosine similarity per sample
    dot = np.sum(y_true * y_pred_proj, axis=1)
    norm_true = np.linalg.norm(y_true, axis=1)
    norm_pred = np.linalg.norm(y_pred_proj, axis=1)
    valid = (norm_true > 0) & (norm_pred > 0)
    cos_sim = float(np.mean(dot[valid] / (norm_true[valid] * norm_pred[valid]))) if np.any(valid) else 0.0

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "cosine_similarity": cos_sim,
    }


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    mixture_dir = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Index all local CSV files for fast lookup
    print("Indexing local mixture CSV files...")
    filename_map = {p.name: p for p in mixture_dir.rglob("*.csv") if not p.name.endswith(".index.csv")}

    train_index = mixture_dir / "train_index_seen.csv"
    test_index = mixture_dir / "test_index_seen.csv"

    if not train_index.exists() or not test_index.exists():
        raise FileNotFoundError(f"Missing train_index_seen.csv or test_index_seen.csv in {mixture_dir}")

    prep_cfg = PreprocessConfig(target_points=args.target_points, warmup_ratio=args.warmup_ratio)

    print("Building training feature matrix...")
    X_train, Y_train, target_names, sensor_cols = load_mixture_split(
        train_index,
        mixture_dir,
        prep_cfg,
        filename_map,
        sensor_cols_ref=None,
        max_rows=args.max_files,
    )

    print("Building testing feature matrix...")
    X_test, Y_test, _, _ = load_mixture_split(
        test_index,
        mixture_dir,
        prep_cfg,
        filename_map,
        sensor_cols_ref=sensor_cols,
        max_rows=args.max_files,
    )

    print(f"Loaded Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples, {len(target_names)} target odorants.")

    # Candidate models
    candidates: dict[str, Any] = {
        "RandomForestRegressor": make_pipeline(
            StandardScaler(),
            MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=args.seed, n_jobs=-1)),
        ),
        "ExtraTreesRegressor": make_pipeline(
            StandardScaler(),
            MultiOutputRegressor(ExtraTreesRegressor(n_estimators=100, random_state=args.seed, n_jobs=-1)),
        ),
        "RidgeRegressor": make_pipeline(
            StandardScaler(),
            MultiOutputRegressor(Ridge(alpha=1.0)),
        ),
    }

    results: dict[str, dict[str, float]] = {}
    best_name = ""
    best_cos = -1.0
    best_pipeline = None

    for name, model in candidates.items():
        print(f"Training {name}...")
        model.fit(X_train, Y_train)
        y_pred = model.predict(X_test)
        eval_metrics = evaluate_mixture_predictions(Y_test, y_pred)
        results[name] = eval_metrics
        print(f"  -> MAE: {eval_metrics['mae']:.4f} | RMSE: {eval_metrics['rmse']:.4f} | Cosine Sim: {eval_metrics['cosine_similarity']:.4f}")

        if eval_metrics["cosine_similarity"] > best_cos:
            best_cos = eval_metrics["cosine_similarity"]
            best_name = name
            best_pipeline = model

    print(f"\nBest Mixture Model: {best_name} (Cosine Similarity: {best_cos:.4f})")

    # Save full inference bundle
    feature_names = feature_names_for_sensors(sensor_cols)
    bundle = {
        "framework": "sklearn",
        "task": "mixture_deconvolution",
        "best_model_name": best_name,
        "model": best_pipeline,
        "target_names": target_names,
        "sensor_columns": sensor_cols,
        "feature_names": feature_names,
        "preprocess_config": asdict(prep_cfg),
        "target_points": args.target_points,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    save_pipeline(bundle, output_dir / "model.joblib")

    # Save metrics JSON
    metrics = {
        "best_model_name": best_name,
        "target_odorants": target_names,
        "n_samples_train": int(X_train.shape[0]),
        "n_samples_test": int(X_test.shape[0]),
        "candidate_results": results,
        "best_metrics": results[best_name],
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Mixture model artifact and metrics saved to: {output_dir}")


if __name__ == "__main__":
    main()
