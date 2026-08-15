"""Sensor channel importance and hardware pruning analysis.

Evaluates how individual gas sensors contribute to smell classification accuracy,
helping hardware designers determine the minimum viable sensor array.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.pipeline_io import load_pipeline


def compute_sensor_importance_scores(bundle: dict[str, Any]) -> dict[str, float]:
    """Compute aggregate importance score per physical sensor channel."""
    model = bundle.get("model")
    sensor_cols: list[str] = list(bundle.get("sensor_columns", []))
    feature_names: list[str] = list(bundle.get("feature_names", []))

    scores: dict[str, float] = {col: 0.0 for col in sensor_cols}
    if not sensor_cols:
        return scores

    # Extract feature importances if tree model or pipeline
    estimator = model
    if hasattr(model, "named_steps"):
        estimator = model.named_steps.get("classifier", model.steps[-1][1])

    if hasattr(estimator, "feature_importances_") and len(feature_names) == len(estimator.feature_importances_):
        importances = estimator.feature_importances_
        for feat_name, imp in zip(feature_names, importances):
            for col in sensor_cols:
                if feat_name.startswith(col):
                    scores[col] += float(imp)
                    break
    elif hasattr(estimator, "coef_") and len(feature_names) == estimator.coef_.shape[1]:
        # Average absolute weights across classes for linear models
        mean_abs_weights = np.mean(np.abs(estimator.coef_), axis=0)
        for feat_name, weight in zip(feature_names, mean_abs_weights):
            for col in sensor_cols:
                if feat_name.startswith(col):
                    scores[col] += float(weight)
                    break
    else:
        # Uniform fallback
        for col in sensor_cols:
            scores[col] = 1.0 / len(sensor_cols)

    # Normalize to sum to 100%
    total = sum(scores.values())
    if total > 0:
        scores = {k: float(v / total * 100) for k, v in scores.items()}

    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))


def compute_sensor_pruning_curve(
    eval_npz_path: Path,
    bundle: dict[str, Any],
) -> list[dict[str, Any]]:
    """Simulate dropping least-important sensors and measure classification accuracy."""
    if not eval_npz_path.exists():
        return []

    data = np.load(eval_npz_path, allow_pickle=True)
    X_test = data["X_test"]
    y_test = data["y_test"]
    model = bundle["model"]
    sensor_cols: list[str] = list(bundle["sensor_columns"])
    feature_names: list[str] = list(bundle.get("feature_names", []))

    if len(feature_names) != X_test.shape[1]:
        return []

    importance = compute_sensor_importance_scores(bundle)
    ranked_sensors = list(importance.keys())  # Most to least important

    # Feature indices belonging to each sensor
    sensor_feat_indices: dict[str, list[int]] = {s: [] for s in sensor_cols}
    for idx, fname in enumerate(feature_names):
        for s in sensor_cols:
            if fname.startswith(s):
                sensor_feat_indices[s].append(idx)
                break

    curve = []
    # Test using top-1, top-2, ..., all sensors
    for k in range(1, len(ranked_sensors) + 1):
        active_sensors = ranked_sensors[:k]
        # Zero out inactive sensor features (simulating missing / pruned sensors)
        X_pruned = X_test.copy()
        inactive_sensors = ranked_sensors[k:]
        for inact in inactive_sensors:
            for feat_idx in sensor_feat_indices.get(inact, []):
                X_pruned[:, feat_idx] = 0.0

        try:
            preds = model.predict(X_pruned)
            acc = float(accuracy_score(y_test, preds))
        except Exception:
            acc = 0.0

        curve.append({
            "num_sensors": k,
            "active_sensors": active_sensors,
            "pruned_sensors": inactive_sensors,
            "accuracy": acc,
        })

    return curve
