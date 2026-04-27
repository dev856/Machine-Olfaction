from __future__ import annotations

from typing import Sequence

import numpy as np

FEATURE_SUFFIXES: tuple[str, ...] = (
    "mean",
    "std",
    "median",
    "q25",
    "q75",
    "min",
    "max",
    "range",
    "slope",
    "auc",
    "final",
    "delta",
    "abs_delta",
    "first_half_mean",
    "second_half_mean",
    "half_delta",
    "time_to_max",
    "time_to_min",
    "energy",
)



def _basic_signal_features(signal: np.ndarray) -> dict[str, float]:
    x = np.asarray(signal, dtype=float)
    t = np.linspace(0.0, 1.0, num=len(x), dtype=float)

    if len(x) < 2:
        slope = 0.0
        auc = float(x[0]) if len(x) == 1 else 0.0
        delta = 0.0
    else:
        slope = float(np.polyfit(t, x, deg=1)[0])
        auc = float(np.trapezoid(x, t))
        delta = float(x[-1] - x[0])

    if len(x):
        mid = max(1, len(x) // 2)
        first_half_mean = float(np.mean(x[:mid]))
        second_half_mean = float(np.mean(x[mid:])) if mid < len(x) else first_half_mean
        min_value = float(np.min(x))
        max_value = float(np.max(x))
        time_to_max = float(np.argmax(x) / max(1, len(x) - 1))
        time_to_min = float(np.argmin(x) / max(1, len(x) - 1))
    else:
        first_half_mean = 0.0
        second_half_mean = 0.0
        min_value = 0.0
        max_value = 0.0
        time_to_max = 0.0
        time_to_min = 0.0

    return {
        "mean": float(np.mean(x)) if len(x) else 0.0,
        "std": float(np.std(x)) if len(x) else 0.0,
        "median": float(np.median(x)) if len(x) else 0.0,
        "q25": float(np.percentile(x, 25)) if len(x) else 0.0,
        "q75": float(np.percentile(x, 75)) if len(x) else 0.0,
        "min": min_value,
        "max": max_value,
        "range": max_value - min_value,
        "slope": slope,
        "auc": auc,
        "final": float(x[-1]) if len(x) else 0.0,
        "delta": delta,
        "abs_delta": abs(delta),
        "first_half_mean": first_half_mean,
        "second_half_mean": second_half_mean,
        "half_delta": second_half_mean - first_half_mean,
        "time_to_max": time_to_max,
        "time_to_min": time_to_min,
        "energy": float(np.mean(np.square(x))) if len(x) else 0.0,
    }



def feature_names_for_sensors(sensor_names: Sequence[str]) -> list[str]:
    names: list[str] = []
    for sensor in sensor_names:
        for suffix in FEATURE_SUFFIXES:
            names.append(f"{sensor}__{suffix}")
    return names



def extract_window_feature_dict(window: np.ndarray, sensor_names: Sequence[str]) -> dict[str, float]:
    if window.ndim != 2:
        raise ValueError("Expected window shape [time, sensors].")

    if window.shape[1] != len(sensor_names):
        raise ValueError("sensor_names length must match number of sensor columns in window.")

    out: dict[str, float] = {}
    for idx, sensor in enumerate(sensor_names):
        feats = _basic_signal_features(window[:, idx])
        for suffix in FEATURE_SUFFIXES:
            out[f"{sensor}__{suffix}"] = feats[suffix]
    return out



def extract_window_feature_vector(window: np.ndarray, sensor_names: Sequence[str]) -> np.ndarray:
    feat_dict = extract_window_feature_dict(window, sensor_names)
    ordered_names = feature_names_for_sensors(sensor_names)
    return np.array([feat_dict[name] for name in ordered_names], dtype=float)
