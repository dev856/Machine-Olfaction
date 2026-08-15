"""Gas sensor signal augmentation module for machine olfaction.

Provides physically motivated augmentations for electronic nose time-series:
- Baseline thermal drift injection (linear, exponential, or sinusoidal)
- Sensor channel dropout (simulating physical sensor failures / disconnection)
- Gaussian jitter / electrical noise
- Magnitude scaling (simulating gas concentration and airflow variance)
- Temporal warping / time stretching
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np


@dataclass
class AugmentationConfig:
    """Configuration parameters for sensor signal augmentation."""
    # Probability of applying each augmentation
    jitter_prob: float = 0.5
    jitter_sigma: float = 0.05
    drift_prob: float = 0.5
    drift_max_slope: float = 0.20
    dropout_prob: float = 0.3
    max_drop_sensors: int = 2
    scale_prob: float = 0.5
    scale_range: tuple[float, float] = (0.85, 1.15)
    time_warp_prob: float = 0.3
    time_warp_max_ratio: float = 0.15


def add_gaussian_jitter(arr: np.ndarray, sigma: float = 0.05, rng: np.random.Generator | None = None) -> np.ndarray:
    """Add Gaussian noise to simulate sensor electrical and thermal jitter."""
    gen = rng if rng is not None else np.random.default_rng()
    noise = gen.normal(loc=0.0, scale=sigma, size=arr.shape)
    return arr + noise


def add_baseline_drift(
    arr: np.ndarray,
    max_slope: float = 0.20,
    drift_type: str = "linear",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Inject baseline drift to simulate sensor aging and temperature changes.

    Args:
        arr: 2D array of shape (time_steps, n_sensors)
        max_slope: Maximum drift amplitude relative to signal spread
        drift_type: 'linear', 'exponential', or 'sinusoidal'
        rng: Random generator
    """
    gen = rng if rng is not None else np.random.default_rng()
    n_steps, n_sensors = arr.shape
    t = np.linspace(0.0, 1.0, num=n_steps)

    out = arr.copy()
    for s in range(n_sensors):
        slope = gen.uniform(-max_slope, max_slope)
        if drift_type == "linear":
            drift = slope * t
        elif drift_type == "exponential":
            drift = slope * (np.exp(t) - 1.0) / (np.e - 1.0)
        elif drift_type == "sinusoidal":
            phase = gen.uniform(0, np.pi)
            drift = slope * np.sin(np.pi * t + phase)
        else:
            drift = slope * t
        out[:, s] += drift

    return out


def apply_sensor_dropout(
    arr: np.ndarray,
    max_drop: int = 2,
    drop_value: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate complete or partial failure of specific gas sensor channels."""
    gen = rng if rng is not None else np.random.default_rng()
    n_steps, n_sensors = arr.shape
    if n_sensors <= 1:
        return arr.copy()

    k = gen.integers(1, min(max_drop + 1, n_sensors))
    dropped_sensors = gen.choice(n_sensors, size=k, replace=False)

    out = arr.copy()
    for s in dropped_sensors:
        out[:, s] = drop_value

    return out


def apply_magnitude_scaling(
    arr: np.ndarray,
    scale_range: tuple[float, float] = (0.85, 1.15),
    per_sensor: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Scale sensor response amplitudes to simulate odor concentration variations."""
    gen = rng if rng is not None else np.random.default_rng()
    n_steps, n_sensors = arr.shape
    out = arr.copy()

    if per_sensor:
        scales = gen.uniform(scale_range[0], scale_range[1], size=n_sensors)
        out = out * scales[np.newaxis, :]
    else:
        scale = gen.uniform(scale_range[0], scale_range[1])
        out = out * scale

    return out


def apply_time_warp(
    arr: np.ndarray,
    max_warp_ratio: float = 0.15,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Non-linearly stretch or compress the time timeline to simulate volatile flow changes."""
    gen = rng if rng is not None else np.random.default_rng()
    n_steps, n_sensors = arr.shape
    if n_steps < 10:
        return arr.copy()

    # Create non-linear time warping grid
    orig_t = np.linspace(0.0, 1.0, num=n_steps)
    warp_center = gen.uniform(0.2, 0.8)
    warp_amount = gen.uniform(-max_warp_ratio, max_warp_ratio)

    warped_t = orig_t + warp_amount * np.sin(np.pi * orig_t)
    warped_t = (warped_t - warped_t[0]) / (warped_t[-1] - warped_t[0])

    out = np.zeros_like(arr)
    for s in range(n_sensors):
        out[:, s] = np.interp(orig_t, warped_t, arr[:, s])

    return out


class SensorAugmenter:
    """Composes multiple signal augmentations for sensor time-series data."""

    def __init__(self, config: AugmentationConfig | None = None, seed: int | None = None) -> None:
        self.config = config or AugmentationConfig()
        self.rng = np.random.default_rng(seed)

    def augment(self, arr: np.ndarray) -> np.ndarray:
        """Apply active augmentations stochastically to a 2D sensor trial array."""
        out = np.asarray(arr, dtype=np.float32).copy()

        if self.rng.random() < self.config.scale_prob:
            out = apply_magnitude_scaling(out, self.config.scale_range, per_sensor=True, rng=self.rng)

        if self.rng.random() < self.config.drift_prob:
            out = add_baseline_drift(out, self.config.drift_max_slope, rng=self.rng)

        if self.rng.random() < self.config.jitter_prob:
            out = add_gaussian_jitter(out, self.config.jitter_sigma, rng=self.rng)

        if self.rng.random() < self.config.time_warp_prob:
            out = apply_time_warp(out, self.config.time_warp_max_ratio, rng=self.rng)

        if self.rng.random() < self.config.dropout_prob:
            out = apply_sensor_dropout(out, self.config.max_drop_sensors, rng=self.rng)

        return out
