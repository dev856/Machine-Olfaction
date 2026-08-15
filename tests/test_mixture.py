"""Unit tests for mixture odor deconvolution and ratio estimation."""

import numpy as np
import pandas as pd
import pytest

from src.models.mixture_predict import deconvolve_mixture_dataframe
from src.models.pipeline_io import load_pipeline


@pytest.fixture
def dummy_mixture_bundle():
    class DummyRegressor:
        def predict(self, X):
            # Returns fixed 12-dim vector simulating 80% orange, 20% almond
            preds = np.zeros((X.shape[0], 12), dtype=float)
            preds[:, 3] = 0.8  # orange
            preds[:, 7] = 0.2  # almond
            return preds

    target_names = [
        "peach", "apple", "banana", "orange", "pear",
        "strawberry", "mango", "almond", "clove", "coriander", "cumin", "garlic"
    ]
    return {
        "framework": "sklearn",
        "task": "mixture_deconvolution",
        "model": DummyRegressor(),
        "target_names": target_names,
        "sensor_columns": ["NO2", "C2H5OH", "VOC", "CO", "Alcohol", "LPG"],
        "preprocess_config": {"target_points": 300, "warmup_ratio": 0.05, "fill_strategy": "ffill_bfill", "normalize": True},
    }


def test_deconvolve_mixture_dataframe(dummy_mixture_bundle):
    # Create dummy dataframe matching sensor columns
    data = {col: np.random.randn(300) for col in dummy_mixture_bundle["sensor_columns"]}
    data["time"] = np.arange(300)
    df = pd.DataFrame(data)

    res = deconvolve_mixture_dataframe(df, dummy_mixture_bundle, threshold=0.05)

    assert res["primary_odor"] == "orange"
    assert np.isclose(res["primary_percentage"], 80.0, atol=1.0)
    assert len(res["active_components"]) == 2

    comp_odors = [c["odor"] for c in res["active_components"]]
    assert "orange" in comp_odors
    assert "almond" in comp_odors
