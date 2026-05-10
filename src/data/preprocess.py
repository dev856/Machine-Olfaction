"""Preprocessing utilities for one gas-sensor trial.

The project treats one CSV file as one sensor trial: rows are time steps and
numeric sensor columns are the gas sensor channels. These helpers keep the
same preprocessing rules in training, evaluation, and Streamlit inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

TIME_COLUMN_CANDIDATES: tuple[str, ...] = (
    "time",
    "timestamp",
    "timestamp_ms",
    "ts",
    "t",
    "second",
    "seconds",
    "sec",
    "sample_idx",
    "index",
)

DEFAULT_METADATA_HINTS: tuple[str, ...] = (
    "label",
    "class",
    "target",
    "id",
    "trial",
    "sample",
    "filepath",
    "file_path",
)


@dataclass(slots=True)
class PreprocessConfig:
    """Small set of preprocessing choices saved inside each model artifact."""

    target_points: int = 300
    warmup_ratio: float = 0.05
    fill_strategy: str = "ffill_bfill"
    normalize: bool = True



def _norm(name: str) -> str:
    return name.strip().lower()



def identify_time_column(df: pd.DataFrame, candidates: Iterable[str] = TIME_COLUMN_CANDIDATES) -> str | None:
    """Return the most likely time column, or None when row order is the timeline."""

    normalized_map = {_norm(col): col for col in df.columns}

    for candidate in candidates:
        if candidate in normalized_map:
            return normalized_map[candidate]

    for normalized, original in normalized_map.items():
        if "time" in normalized or normalized.endswith("_ms"):
            return original

    return None



def infer_sensor_columns(
    df: pd.DataFrame,
    time_column: str | None = None,
    extra_exclude_hints: Sequence[str] | None = None,
    min_unique_values: int = 3,
) -> list[str]:
    """Find numeric columns that look like sensor channels instead of labels or IDs."""

    exclude_hints = list(DEFAULT_METADATA_HINTS)
    if extra_exclude_hints:
        exclude_hints.extend(_norm(item) for item in extra_exclude_hints)

    sensors: list[str] = []
    for col in df.columns:
        if col == time_column:
            continue

        ncol = _norm(col)
        if any(hint in ncol for hint in exclude_hints):
            continue

        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        if df[col].nunique(dropna=True) < min_unique_values:
            continue

        sensors.append(col)

    return sensors



def fill_missing_values(df: pd.DataFrame, columns: Sequence[str], strategy: str = "ffill_bfill") -> pd.DataFrame:
    out = df.copy()
    if not columns:
        return out

    if strategy == "ffill_bfill":
        out[list(columns)] = out[list(columns)].ffill().bfill()
    elif strategy == "zero":
        out[list(columns)] = out[list(columns)].fillna(0.0)
    elif strategy == "drop":
        out = out.dropna(subset=list(columns)).reset_index(drop=True)
    else:
        raise ValueError(f"Unknown fill strategy: {strategy}")

    return out



def trim_warmup(df: pd.DataFrame, warmup_ratio: float = 0.05) -> pd.DataFrame:
    if df.empty:
        return df

    warmup_rows = int(len(df) * warmup_ratio)
    if warmup_rows <= 0:
        return df
    if warmup_rows >= len(df):
        return df.iloc[0:0].copy()
    return df.iloc[warmup_rows:].reset_index(drop=True)



def _to_time_array(df: pd.DataFrame, time_column: str | None) -> np.ndarray:
    if time_column is None or time_column not in df.columns:
        return np.arange(len(df), dtype=float)

    raw = df[time_column]
    if pd.api.types.is_numeric_dtype(raw):
        arr = raw.to_numpy(dtype=float)
    else:
        as_dt = pd.to_datetime(raw, errors="coerce")
        if as_dt.notna().sum() > 0:
            arr = (as_dt.astype("int64") / 1e9).to_numpy(dtype=float)
        else:
            arr = np.arange(len(df), dtype=float)

    if np.isnan(arr).any():
        fallback = np.arange(len(df), dtype=float)
        mask = np.isnan(arr)
        arr[mask] = fallback[mask]

    return arr



def resample_dataframe(
    df: pd.DataFrame,
    sensor_columns: Sequence[str],
    time_column: str | None,
    target_points: int,
) -> pd.DataFrame:
    if target_points <= 0 or len(df) == 0:
        return df.reset_index(drop=True)

    time_arr = _to_time_array(df, time_column)
    order = np.argsort(time_arr)
    x = time_arr[order]

    unique_x, unique_idx = np.unique(x, return_index=True)
    if len(unique_x) < 2:
        return df.iloc[order].reset_index(drop=True)

    x_new = np.linspace(unique_x.min(), unique_x.max(), target_points)
    out = {"time": x_new}

    for col in sensor_columns:
        y = df[col].to_numpy(dtype=float)[order]
        y_unique = y[unique_idx]
        out[col] = np.interp(x_new, unique_x, y_unique)

    return pd.DataFrame(out)



def zscore_normalize(df: pd.DataFrame, columns: Sequence[str], eps: float = 1e-8) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        series = out[col].to_numpy(dtype=float)
        mean = float(np.mean(series))
        std = float(np.std(series))
        out[col] = (series - mean) / (std + eps)
    return out



def window_array(values: np.ndarray, window_size: int, stride: int) -> list[np.ndarray]:
    if values.ndim != 2:
        raise ValueError("Expected values with shape [time, sensors].")

    if len(values) == 0:
        return []

    if window_size <= 0 or window_size >= len(values):
        return [values]

    if stride <= 0:
        raise ValueError("stride must be > 0")

    windows: list[np.ndarray] = []
    for start in range(0, len(values) - window_size + 1, stride):
        stop = start + window_size
        windows.append(values[start:stop])

    return windows



def preprocess_trial(
    df: pd.DataFrame,
    sensor_columns: Sequence[str] | None = None,
    time_column: str | None = None,
    config: PreprocessConfig | None = None,
) -> tuple[pd.DataFrame, str | None, list[str]]:
    """Clean and standardize one trial before feature extraction.

    Steps:
    1. Choose time and sensor columns.
    2. Fill missing sensor readings.
    3. Remove the early warm-up segment.
    4. Interpolate to a fixed number of time points.
    5. Z-score each sensor within the trial.
    """

    cfg = config or PreprocessConfig()

    chosen_time = time_column or identify_time_column(df)
    chosen_sensors = list(sensor_columns) if sensor_columns else infer_sensor_columns(df, time_column=chosen_time)

    if not chosen_sensors:
        raise ValueError("No numeric sensor columns were detected.")

    working_cols = [col for col in chosen_sensors]
    if chosen_time and chosen_time in df.columns:
        working_cols = [chosen_time] + working_cols

    work_df = df[working_cols].copy()
    work_df = fill_missing_values(work_df, chosen_sensors, strategy=cfg.fill_strategy)
    work_df = trim_warmup(work_df, warmup_ratio=cfg.warmup_ratio)
    work_df = resample_dataframe(work_df, chosen_sensors, chosen_time, target_points=cfg.target_points)

    if cfg.normalize:
        work_df = zscore_normalize(work_df, chosen_sensors)

    return work_df, chosen_time, chosen_sensors
