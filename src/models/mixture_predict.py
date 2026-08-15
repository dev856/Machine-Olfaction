"""Inference utilities for smell mixture deconvolution and ratio estimation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.preprocess import PreprocessConfig, identify_time_column, infer_sensor_columns, preprocess_trial
from src.features.extract_features import extract_window_feature_vector
from src.models.pipeline_io import load_pipeline


def deconvolve_mixture_dataframe(
    df: pd.DataFrame,
    bundle: dict[str, Any],
    threshold: float = 0.05,
) -> dict[str, Any]:
    """Deconvolve a sensor CSV of a smell mixture into its constituent odor components and percentages."""
    model = bundle["model"]
    target_names: list[str] = list(bundle["target_names"])
    trained_sensor_cols: list[str] = list(bundle["sensor_columns"])

    time_col = identify_time_column(df)
    detected_sensor_cols = infer_sensor_columns(df, time_column=time_col)
    missing = [c for c in trained_sensor_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Mixture CSV is missing expected sensor columns: {missing}")

    prep_cfg = PreprocessConfig(**bundle["preprocess_config"])
    proc_df, _, _ = preprocess_trial(
        df,
        sensor_columns=trained_sensor_cols,
        time_column=time_col,
        config=prep_cfg,
    )

    values = proc_df[trained_sensor_cols].to_numpy(dtype=float)
    feat = extract_window_feature_vector(values, trained_sensor_cols).reshape(1, -1)

    raw_pred = model.predict(feat)[0]
    # Project to non-negative simplex
    pos_pred = np.maximum(raw_pred, 0.0)
    total = float(pos_pred.sum())
    ratios = pos_pred / total if total > 0 else np.ones_like(pos_pred) / len(pos_pred)

    # Sort components by percentage
    sorted_idx = np.argsort(ratios)[::-1]
    sorted_targets = [target_names[i] for i in sorted_idx]
    sorted_ratios = ratios[sorted_idx]

    # Filter active components above threshold
    active_components = [
        {"odor": target_names[i], "percentage": float(ratios[i] * 100), "ratio": float(ratios[i])}
        for i in sorted_idx
        if ratios[i] >= threshold
    ]

    return {
        "target_names": target_names,
        "ratios": ratios,
        "sorted_odorants": sorted_targets,
        "sorted_ratios": sorted_ratios,
        "active_components": active_components,
        "primary_odor": sorted_targets[0] if len(sorted_targets) > 0 else "Unknown",
        "primary_percentage": float(sorted_ratios[0] * 100) if len(sorted_ratios) > 0 else 0.0,
        "processed_df": proc_df,
        "time_column": time_col,
        "detected_sensor_columns": detected_sensor_cols,
    }


def deconvolve_mixture_csv(csv_path: Path, model_path: Path, threshold: float = 0.05) -> dict[str, Any]:
    bundle = load_pipeline(model_path)
    df = pd.read_csv(csv_path)
    return deconvolve_mixture_dataframe(df, bundle, threshold=threshold)
