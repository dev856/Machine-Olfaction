"""Unit tests for sensor signal augmentation methods."""

import numpy as np
import pytest

from src.data.augmentation import (
    AugmentationConfig,
    SensorAugmenter,
    add_baseline_drift,
    add_gaussian_jitter,
    apply_magnitude_scaling,
    apply_sensor_dropout,
    apply_time_warp,
)


@pytest.fixture
def dummy_sensor_trial():
    # 100 time points, 6 sensors
    t = np.linspace(0, 1, 100)
    arr = np.column_stack([np.sin(2 * np.pi * t * (i + 1)) + 2.0 for i in range(6)])
    return arr.astype(np.float32)


def test_add_gaussian_jitter(dummy_sensor_trial):
    noisy = add_gaussian_jitter(dummy_sensor_trial, sigma=0.05)
    assert noisy.shape == dummy_sensor_trial.shape
    assert not np.allclose(noisy, dummy_sensor_trial)
    assert np.abs(np.mean(noisy - dummy_sensor_trial)) < 0.05


def test_add_baseline_drift(dummy_sensor_trial):
    drifted_linear = add_baseline_drift(dummy_sensor_trial, max_slope=0.3, drift_type="linear")
    assert drifted_linear.shape == dummy_sensor_trial.shape
    assert not np.allclose(drifted_linear, dummy_sensor_trial)

    drifted_sin = add_baseline_drift(dummy_sensor_trial, max_slope=0.3, drift_type="sinusoidal")
    assert drifted_sin.shape == dummy_sensor_trial.shape


def test_apply_sensor_dropout(dummy_sensor_trial):
    dropped = apply_sensor_dropout(dummy_sensor_trial, max_drop=2, drop_value=0.0)
    assert dropped.shape == dummy_sensor_trial.shape
    # Check that at least one column is all zeros
    zero_cols = np.where(np.all(dropped == 0.0, axis=0))[0]
    assert len(zero_cols) >= 1


def test_apply_magnitude_scaling(dummy_sensor_trial):
    scaled = apply_magnitude_scaling(dummy_sensor_trial, scale_range=(1.2, 1.2), per_sensor=False)
    assert scaled.shape == dummy_sensor_trial.shape
    np.testing.assert_allclose(scaled, dummy_sensor_trial * 1.2, rtol=1e-5)


def test_apply_time_warp(dummy_sensor_trial):
    warped = apply_time_warp(dummy_sensor_trial, max_warp_ratio=0.15)
    assert warped.shape == dummy_sensor_trial.shape
    assert not np.allclose(warped, dummy_sensor_trial)


def test_sensor_augmenter_composition(dummy_sensor_trial):
    cfg = AugmentationConfig(
        jitter_prob=1.0,
        drift_prob=1.0,
        dropout_prob=1.0,
        scale_prob=1.0,
        time_warp_prob=1.0,
    )
    augmenter = SensorAugmenter(config=cfg, seed=42)
    aug = augmenter.augment(dummy_sensor_trial)
    assert aug.shape == dummy_sensor_trial.shape
    assert not np.allclose(aug, dummy_sensor_trial)
