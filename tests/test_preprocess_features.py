from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.preprocess import PreprocessConfig, identify_time_column, infer_sensor_columns, preprocess_trial, window_array
from src.features.extract_features import (
    contextual_feature_names_for_sensors,
    extract_contextual_window_feature_vector,
    extract_window_feature_vector,
    feature_names_for_sensors,
)


def test_preprocess_detects_time_and_sensors() -> None:
    df = pd.DataFrame(
        {
            "timestamp": [0, 1, 2, 3],
            "NO2": [1.0, np.nan, 3.0, 4.0],
            "VOC": [4.0, 5.0, 6.0, 7.0],
            "label": ["x", "x", "x", "x"],
        }
    )

    assert identify_time_column(df) == "timestamp"
    assert infer_sensor_columns(df, time_column="timestamp") == ["NO2", "VOC"]

    processed, time_col, sensors = preprocess_trial(
        df,
        config=PreprocessConfig(target_points=6, warmup_ratio=0.0, normalize=True),
    )

    assert time_col == "timestamp"
    assert sensors == ["NO2", "VOC"]
    assert processed.shape == (6, 3)
    assert not processed[sensors].isna().any().any()


def test_window_array_returns_expected_windows() -> None:
    values = np.arange(20).reshape(10, 2)
    windows = window_array(values, window_size=4, stride=3)

    assert len(windows) == 3
    assert windows[0].shape == (4, 2)
    assert np.array_equal(windows[1], values[3:7])


def test_feature_vector_matches_feature_names() -> None:
    window = np.array([[0.0, 1.0], [1.0, 3.0], [2.0, 5.0], [3.0, 7.0]])
    sensors = ["NO2", "VOC"]

    vector = extract_window_feature_vector(window, sensors)
    names = feature_names_for_sensors(sensors)

    assert len(vector) == len(names)
    assert "NO2__time_to_max" in names
    assert "VOC__energy" in names
    assert np.isfinite(vector).all()


def test_contextual_feature_vector_matches_feature_names() -> None:
    full_trial = np.array(
        [
            [0.0, 1.0],
            [1.0, 3.0],
            [2.0, 5.0],
            [3.0, 7.0],
            [4.0, 8.0],
        ]
    )
    window = full_trial[1:4]
    sensors = ["NO2", "VOC"]

    vector = extract_contextual_window_feature_vector(window, full_trial, sensors, start=1, trial_length=len(full_trial))
    names = contextual_feature_names_for_sensors(sensors)

    assert len(vector) == len(names)
    assert "trial__NO2__mean" in names
    assert "window_center_ratio" in names
    assert np.isfinite(vector).all()
