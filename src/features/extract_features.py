from __future__ import annotations

from typing import Sequence

import numpy as np

FEATURE_SUFFIXES: tuple[str, ...] = (
    "mean",
    "std",
    "skew",
    "kurtosis",
    "median",
    "q25",
    "q75",
    "iqr",
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
    "diff_mean",
    "diff_std",
    "diff_max_abs",
    "fft_low_power",
    "fft_high_power",
    "fft_balance",
)

WINDOW_CONTEXT_FEATURES: tuple[str, ...] = (
    "window_start_ratio",
    "window_center_ratio",
    "window_stop_ratio",
    "window_width_ratio",
)


def _safe_moment_features(x: np.ndarray) -> tuple[float, float]:
    if len(x) < 2:
        return 0.0, 0.0

    centered = x - np.mean(x)
    std = float(np.std(centered))
    if std < 1e-12:
        return 0.0, 0.0

    z = centered / std
    return float(np.mean(z**3)), float(np.mean(z**4) - 3.0)


def _fft_band_features(x: np.ndarray) -> tuple[float, float, float]:
    if len(x) < 4:
        return 0.0, 0.0, 0.0

    spectrum = np.abs(np.fft.rfft(x - np.mean(x))) ** 2
    if len(spectrum) <= 1:
        return 0.0, 0.0, 0.0

    spectrum = spectrum[1:]
    split = max(1, len(spectrum) // 3)
    low_power = float(np.mean(spectrum[:split]))
    high_power = float(np.mean(spectrum[split:])) if split < len(spectrum) else 0.0
    balance = float(low_power / (low_power + high_power + 1e-12))
    return low_power, high_power, balance


def _basic_signal_features(signal: np.ndarray) -> dict[str, float]:
    x = np.asarray(signal, dtype=float)
    t = np.linspace(0.0, 1.0, num=len(x), dtype=float)
    skew, kurtosis = _safe_moment_features(x)
    fft_low_power, fft_high_power, fft_balance = _fft_band_features(x)

    if len(x) < 2:
        slope = 0.0
        auc = float(x[0]) if len(x) == 1 else 0.0
        delta = 0.0
        diff_mean = 0.0
        diff_std = 0.0
        diff_max_abs = 0.0
    else:
        slope = float(np.polyfit(t, x, deg=1)[0])
        auc = float(np.trapezoid(x, t))
        delta = float(x[-1] - x[0])
        diff = np.diff(x)
        diff_mean = float(np.mean(diff))
        diff_std = float(np.std(diff))
        diff_max_abs = float(np.max(np.abs(diff)))

    if len(x):
        mid = max(1, len(x) // 2)
        first_half_mean = float(np.mean(x[:mid]))
        second_half_mean = float(np.mean(x[mid:])) if mid < len(x) else first_half_mean
        min_value = float(np.min(x))
        max_value = float(np.max(x))
        q25 = float(np.percentile(x, 25))
        q75 = float(np.percentile(x, 75))
        time_to_max = float(np.argmax(x) / max(1, len(x) - 1))
        time_to_min = float(np.argmin(x) / max(1, len(x) - 1))
    else:
        first_half_mean = 0.0
        second_half_mean = 0.0
        min_value = 0.0
        max_value = 0.0
        q25 = 0.0
        q75 = 0.0
        time_to_max = 0.0
        time_to_min = 0.0

    return {
        "mean": float(np.mean(x)) if len(x) else 0.0,
        "std": float(np.std(x)) if len(x) else 0.0,
        "skew": skew,
        "kurtosis": kurtosis,
        "median": float(np.median(x)) if len(x) else 0.0,
        "q25": q25,
        "q75": q75,
        "iqr": q75 - q25,
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
        "diff_mean": diff_mean,
        "diff_std": diff_std,
        "diff_max_abs": diff_max_abs,
        "fft_low_power": fft_low_power,
        "fft_high_power": fft_high_power,
        "fft_balance": fft_balance,
    }


def cross_sensor_feature_names(sensor_names: Sequence[str]) -> list[str]:
    names: list[str] = []
    for left_idx, left in enumerate(sensor_names):
        for right in sensor_names[left_idx + 1 :]:
            names.append(f"{left}__x__{right}__corr")
            names.append(f"{left}__x__{right}__mean_diff")
            names.append(f"{left}__x__{right}__delta_diff")
    return names


def feature_names_for_sensors(sensor_names: Sequence[str]) -> list[str]:
    names: list[str] = []
    for sensor in sensor_names:
        for suffix in FEATURE_SUFFIXES:
            names.append(f"{sensor}__{suffix}")
    names.extend(cross_sensor_feature_names(sensor_names))
    return names


def contextual_feature_names_for_sensors(sensor_names: Sequence[str]) -> list[str]:
    base_names = feature_names_for_sensors(sensor_names)
    return (
        [f"window__{name}" for name in base_names]
        + [f"trial__{name}" for name in base_names]
        + list(WINDOW_CONTEXT_FEATURES)
    )


def _cross_sensor_features(window: np.ndarray, sensor_names: Sequence[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for left_idx, left in enumerate(sensor_names):
        left_signal = window[:, left_idx]
        for right_idx, right in enumerate(sensor_names[left_idx + 1 :], start=left_idx + 1):
            right_signal = window[:, right_idx]
            if len(left_signal) < 2 or np.std(left_signal) < 1e-12 or np.std(right_signal) < 1e-12:
                corr = 0.0
            else:
                corr = float(np.corrcoef(left_signal, right_signal)[0, 1])
            out[f"{left}__x__{right}__corr"] = corr
            out[f"{left}__x__{right}__mean_diff"] = float(np.mean(left_signal) - np.mean(right_signal))
            out[f"{left}__x__{right}__delta_diff"] = float(
                (left_signal[-1] - left_signal[0]) - (right_signal[-1] - right_signal[0])
            )
    return out


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
    out.update(_cross_sensor_features(window, sensor_names))
    return out


def extract_window_feature_vector(window: np.ndarray, sensor_names: Sequence[str]) -> np.ndarray:
    feat_dict = extract_window_feature_dict(window, sensor_names)
    ordered_names = feature_names_for_sensors(sensor_names)
    return np.array([feat_dict[name] for name in ordered_names], dtype=float)


def window_position_features(start: int, window_length: int, trial_length: int) -> np.ndarray:
    if trial_length <= 1:
        return np.array([0.0, 0.5, 1.0, 1.0], dtype=float)

    stop = min(start + window_length, trial_length)
    denom = float(max(1, trial_length - 1))
    start_ratio = float(start / denom)
    stop_ratio = float((stop - 1) / denom)
    center_ratio = float((start_ratio + stop_ratio) / 2.0)
    width_ratio = float(window_length / max(1, trial_length))
    return np.array([start_ratio, center_ratio, stop_ratio, width_ratio], dtype=float)


def extract_contextual_window_feature_vector(
    window: np.ndarray,
    full_trial: np.ndarray,
    sensor_names: Sequence[str],
    start: int,
    trial_length: int,
) -> np.ndarray:
    window_features = extract_window_feature_vector(window, sensor_names)
    trial_features = extract_window_feature_vector(full_trial, sensor_names)
    position_features = window_position_features(start, len(window), trial_length)
    return np.concatenate([window_features, trial_features, position_features])
