from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import asdict
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import LabelEncoder

from src.data.preprocess import PreprocessConfig
from src.features.extract_features import feature_names_for_sensors
from src.models.predict import aggregate_window_probabilities, build_feature_matrix, predict_dataframe


def test_prediction_averages_window_probabilities() -> None:
    sensors = ["NO2", "VOC"]
    df = pd.DataFrame(
        {
            "time": np.arange(20),
            "NO2": np.linspace(0.0, 1.0, 20),
            "VOC": np.linspace(1.0, 0.0, 20),
        }
    )
    encoder = LabelEncoder().fit(["almond", "garlic"])
    feature_names = feature_names_for_sensors(sensors)

    features, _, _, _ = build_feature_matrix(
        df,
        {
            "sensor_columns": sensors,
            "preprocess_config": asdict(PreprocessConfig(target_points=20, warmup_ratio=0.0, normalize=True)),
            "window_size": 5,
            "window_stride": 5,
        },
    )

    model = DummyClassifier(strategy="prior")
    model.fit(features, np.array([0, 1, 1, 1]))
    bundle = {
        "model": model,
        "label_encoder": encoder,
        "sensor_columns": sensors,
        "feature_names": feature_names,
        "preprocess_config": asdict(PreprocessConfig(target_points=20, warmup_ratio=0.0, normalize=True)),
        "window_size": 5,
        "window_stride": 5,
    }

    result = predict_dataframe(df, bundle)

    assert result["n_windows"] == 4
    assert result["predicted_class"] == "garlic"
    assert result["feature_matrix"].shape[0] == 4
    assert np.isclose(result["probabilities"].sum(), 1.0)


def test_prediction_builds_contextual_features() -> None:
    sensors = ["NO2", "VOC"]
    df = pd.DataFrame(
        {
            "time": np.arange(20),
            "NO2": np.linspace(0.0, 1.0, 20),
            "VOC": np.linspace(1.0, 0.0, 20),
        }
    )

    base_features, _, _, _ = build_feature_matrix(
        df,
        {
            "sensor_columns": sensors,
            "preprocess_config": asdict(PreprocessConfig(target_points=20, warmup_ratio=0.0, normalize=True)),
            "window_size": 5,
            "window_stride": 5,
        },
    )
    context_features, _, _, _ = build_feature_matrix(
        df,
        {
            "sensor_columns": sensors,
            "preprocess_config": asdict(PreprocessConfig(target_points=20, warmup_ratio=0.0, normalize=True)),
            "window_size": 5,
            "window_stride": 5,
            "feature_mode": "contextual_window",
        },
    )

    assert context_features.shape[0] == base_features.shape[0]
    assert context_features.shape[1] == base_features.shape[1] * 2 + 4


def test_probability_aggregation_methods_normalize_output() -> None:
    window_proba = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
        ]
    )

    mean_proba = aggregate_window_probabilities(window_proba, method="mean")
    max_proba = aggregate_window_probabilities(window_proba, method="max")

    assert np.isclose(mean_proba.sum(), 1.0)
    assert np.isclose(max_proba.sum(), 1.0)
    assert max_proba[0] > mean_proba[0]
