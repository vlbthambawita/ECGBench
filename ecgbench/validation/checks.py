"""
Individual quality check functions for ECG signal validation.

Every check has the signature:
    def check_<name>(signal: np.ndarray, config: DatasetConfig) -> list[str]

Args:
    signal: numpy array of shape (leads, samples)
    config: the dataset's DatasetConfig

Returns:
    List of issue descriptions. Empty list means check passed.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ecgbench.config import DatasetConfig


def check_missing_leads(signal: np.ndarray, config: DatasetConfig) -> list[str]:
    """Detect leads where ALL values are NaN or ALL values are exactly 0.0."""
    issues = []
    for i in range(signal.shape[0]):
        lead = signal[i]
        if np.all(np.isnan(lead)) or np.all(lead == 0.0):
            issues.append(f"missing_lead_{i}")
    return issues


def check_nan_values(signal: np.ndarray, config: DatasetConfig) -> list[str]:
    """Detect any NaN values anywhere in the signal."""
    count = int(np.sum(np.isnan(signal)))
    if count > 0:
        return [f"nan_values:{count}_NaN_samples"]
    return []


def check_truncated_signal(
    signal: np.ndarray, config: DatasetConfig, sampling_rate: int | None = None,
) -> list[str]:
    """Detect if signal has fewer samples than expected."""
    if config.validation is None:
        return []
    rate = sampling_rate or config.default_sampling_rate
    expected = config.validation.expected_samples.get(rate)
    if expected is None:
        return []
    actual = signal.shape[1]
    if actual < expected:
        return [f"truncated:{actual}_vs_{expected}"]
    return []


def check_flat_line(signal: np.ndarray, config: DatasetConfig) -> list[str]:
    """Detect leads with near-zero variance (not already caught by missing_leads)."""
    issues = []
    for i in range(signal.shape[0]):
        lead = signal[i]
        # Skip leads that are all NaN or all zero (caught by missing_leads)
        if np.all(np.isnan(lead)) or np.all(lead == 0.0):
            continue
        if np.nanvar(lead) < 1e-6:
            issues.append(f"flat_line_lead_{i}")
    return issues


def check_amplitude_outlier(signal: np.ndarray, config: DatasetConfig) -> list[str]:
    """Detect samples outside the physiological amplitude range."""
    if config.validation is None:
        return []
    low, high = config.validation.amplitude_range_mv
    issues = []
    for i in range(signal.shape[0]):
        lead = signal[i]
        valid = lead[~np.isnan(lead)]
        if len(valid) == 0:
            continue
        lead_min = float(np.min(valid))
        lead_max = float(np.max(valid))
        if lead_min < low or lead_max > high:
            issues.append(f"amplitude_outlier:lead_{i}_min_{lead_min:.2f}_max_{lead_max:.2f}")
    return issues


# Registry of all checks. corrupt_header is handled specially in the engine.
CHECK_REGISTRY: dict[str, Callable] = {
    "missing_leads": check_missing_leads,
    "nan_values": check_nan_values,
    "truncated_signal": check_truncated_signal,
    "flat_line": check_flat_line,
    "amplitude_outlier": check_amplitude_outlier,
}
