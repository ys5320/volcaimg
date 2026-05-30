"""
Basic tests for volcaimg.preprocessing and volcaimg.correlation.
Run with: pytest tests/
"""

import numpy as np
import pytest
from volcaimg.preprocessing import (
    apply_mean_centering,
    apply_linear_detrend_after_centering,
    apply_minmax_normalisation,
    apply_highpass_filter,
    ensure_numeric_data,
)
from volcaimg.correlation import compute_pearson_per_cell


def test_mean_centering_removes_offset():
    """After mean centering, each channel should have mean close to zero."""
    rng = np.random.default_rng(42)
    data = rng.normal(loc=5.0, scale=1.0, size=(10, 200))
    centered = apply_mean_centering(data)
    assert np.allclose(centered.mean(axis=1), 0, atol=1e-10)


def test_mean_centering_shape_preserved():
    """Mean centering should not change array shape."""
    data = np.random.default_rng(0).normal(size=(5, 300))
    assert apply_mean_centering(data).shape == data.shape


def test_detrend_shape_preserved():
    """Detrending should not change array shape."""
    data = np.random.default_rng(0).normal(size=(5, 300))
    assert apply_linear_detrend_after_centering(data).shape == data.shape


def test_minmax_normalisation_range():
    """After normalisation, all values should be within [-1, 1]."""
    rng = np.random.default_rng(7)
    data = rng.normal(loc=10.0, scale=3.0, size=(8, 500))
    normalised = apply_minmax_normalisation(data)
    assert np.all(normalised >= -1.0 - 1e-10)
    assert np.all(normalised <= 1.0 + 1e-10)


def test_highpass_filter_shape_preserved():
    """High-pass filtering should not change array shape."""
    data = np.random.default_rng(1).normal(size=(4, 400))
    filtered = apply_highpass_filter(data, sampling_rate_hz=5, cutoff_freq=0.01)
    assert filtered.shape == data.shape


def test_highpass_filter_removes_dc():
    """High-pass filter should remove constant offset (DC component)."""
    n_timepoints = 1000
    data = np.ones((3, n_timepoints)) * 10.0
    data += np.random.default_rng(2).normal(scale=0.01, size=(3, n_timepoints))
    filtered = apply_highpass_filter(data, sampling_rate_hz=5, cutoff_freq=0.01)
    assert np.allclose(filtered.mean(axis=1), 0, atol=0.1)


def test_pearson_anticorrelated_signal():
    """Perfectly anticorrelated signals should return r close to -1."""
    rng = np.random.default_rng(99)
    voltage = rng.normal(size=(5, 300))
    calcium = -voltage
    r_values = compute_pearson_per_cell(voltage, calcium, flip_voltage=False)
    assert all(r < -0.99 for r in r_values)


def test_pearson_shape_mismatch_raises():
    """Mismatched array shapes should raise ValueError."""
    voltage = np.random.default_rng(0).normal(size=(5, 300))
    calcium = np.random.default_rng(1).normal(size=(6, 300))
    with pytest.raises(ValueError):
        compute_pearson_per_cell(voltage, calcium)


def test_pearson_returns_one_value_per_cell():
    """Should return exactly one correlation value per cell."""
    rng = np.random.default_rng(3)
    n_cells = 7
    voltage = rng.normal(size=(n_cells, 200))
    calcium = rng.normal(size=(n_cells, 200))
    r_values = compute_pearson_per_cell(voltage, calcium)
    assert len(r_values) == n_cells


def test_ensure_numeric_data_converts_correctly():
    """ensure_numeric_data should return a float64 array."""
    data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    result = ensure_numeric_data(data)
    assert result.dtype == np.float64
    assert result.shape == (2, 3)
