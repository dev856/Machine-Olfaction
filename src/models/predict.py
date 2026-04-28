from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.preprocess import PreprocessConfig, identify_time_column, infer_sensor_columns, preprocess_trial, window_array
from src.features.extract_features import extract_contextual_window_feature_vector, extract_window_feature_vector
from src.models.pipeline_io import load_pipeline


def validate_uploaded_schema(df: pd.DataFrame, trained_sensor_cols: list[str]) -> tuple[str | None, list[str], list[str]]:
    time_col = identify_time_column(df)
    detected_sensor_cols = infer_sensor_columns(df, time_column=time_col)
    missing = [col for col in trained_sensor_cols if col not in df.columns]
    return time_col, detected_sensor_cols, missing


def windows_with_starts(values: np.ndarray, window_size: int, stride: int) -> tuple[list[np.ndarray], list[int]]:
    windows = window_array(values, window_size=window_size, stride=stride)
    if window_size <= 0 or window_size >= len(values):
        return windows, [0] * len(windows)
    return windows, list(range(0, len(values) - window_size + 1, stride))


def build_feature_matrix(uploaded_df: pd.DataFrame, bundle: dict[str, Any]) -> tuple[np.ndarray, pd.DataFrame, str | None, list[str]]:
    preprocess_cfg = PreprocessConfig(**bundle["preprocess_config"])
    trained_sensor_cols: list[str] = list(bundle["sensor_columns"])
    time_col, detected_sensor_cols, missing = validate_uploaded_schema(uploaded_df, trained_sensor_cols)

    if missing:
        raise ValueError(f"Uploaded CSV is missing expected sensor columns: {missing}")
    if len(detected_sensor_cols) == 0:
        raise ValueError("No candidate numeric sensor columns were detected in the uploaded file.")

    proc_df, _, _ = preprocess_trial(
        uploaded_df,
        sensor_columns=trained_sensor_cols,
        time_column=time_col,
        config=preprocess_cfg,
    )

    values = proc_df[trained_sensor_cols].to_numpy(dtype=float)
    window_size = int(bundle.get("window_size", 0))
    window_stride = int(bundle.get("window_stride", 50))
    windows, starts = windows_with_starts(values, window_size=window_size, stride=window_stride)
    if not windows:
        raise ValueError("Preprocessing produced no usable time-series windows.")

    feature_mode = bundle.get("feature_mode", "window")
    if feature_mode == "contextual_window":
        features = [
            extract_contextual_window_feature_vector(
                window,
                values,
                trained_sensor_cols,
                start=start,
                trial_length=len(values),
            )
            for window, start in zip(windows, starts)
        ]
    else:
        features = [extract_window_feature_vector(window, trained_sensor_cols) for window in windows]

    feature_matrix = np.vstack(features)
    expected_feature_names = bundle.get("feature_names")
    if expected_feature_names is not None and feature_matrix.shape[1] != len(expected_feature_names):
        raise ValueError(
            "Model artifact feature count does not match the current feature extractor. "
            "Retrain the model before running prediction."
        )
    return feature_matrix, proc_df, time_col, detected_sensor_cols


def aggregate_window_probabilities(window_proba: np.ndarray, method: str = "mean") -> np.ndarray:
    if method == "mean":
        proba = np.mean(window_proba, axis=0)
    elif method == "max":
        proba = np.max(window_proba, axis=0)
    elif method == "median":
        proba = np.median(window_proba, axis=0)
    else:
        raise ValueError(f"Unknown probability aggregation method: {method}")

    total = float(np.sum(proba))
    if total > 0:
        proba = proba / total
    return proba


def predict_dataframe(uploaded_df: pd.DataFrame, bundle: dict[str, Any], aggregation: str | None = None) -> dict[str, Any]:
    model = bundle["model"]
    classes = bundle["label_encoder"].classes_
    features, proc_df, time_col, detected_sensor_cols = build_feature_matrix(uploaded_df, bundle)

    window_proba = model.predict_proba(features)
    aggregation_method = aggregation or bundle.get("trial_aggregation", "mean")
    proba = aggregate_window_probabilities(window_proba, method=aggregation_method)
    sorted_idx = np.argsort(proba)[::-1]

    return {
        "class_names": classes,
        "probabilities": proba,
        "window_probabilities": window_proba,
        "top_indices": sorted_idx,
        "predicted_class": str(classes[sorted_idx[0]]),
        "confidence": float(proba[sorted_idx[0]]),
        "processed_df": proc_df,
        "time_column": time_col,
        "detected_sensor_columns": detected_sensor_cols,
        "n_windows": int(len(window_proba)),
        "aggregation": aggregation_method,
    }


def predict_csv(csv_path: Path, model_path: Path) -> dict[str, Any]:
    bundle = load_pipeline(model_path)
    df = pd.read_csv(csv_path)
    return predict_dataframe(df, bundle)
