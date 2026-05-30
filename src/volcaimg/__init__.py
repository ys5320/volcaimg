"""
volcaimg
--------
Python package for analysing simultaneous voltage and calcium
fluorescence imaging timeseries in cancer cells.

Developed during doctoral research at Imperial College London.

Basic usage
-----------
>>> import volcaimg as vi
>>> import numpy as np
>>> voltage = np.load("voltage_traces.npy")   # shape (n_cells, n_timepoints)
>>> calcium = np.load("calcium_traces.npy")
>>> voltage_filtered = vi.apply_highpass_filter(voltage, sampling_rate_hz=5)
>>> calcium_centered = vi.apply_mean_centering(calcium)
>>> correlations = vi.compute_pearson_per_cell(voltage_filtered, calcium_centered)
"""

__version__ = "0.1.0"
__author__ = "Yilin Sun"

from .preprocessing import (
    apply_mean_centering,
    apply_linear_detrend_after_centering,
    apply_minmax_normalisation,
    apply_highpass_filter,
    ensure_numeric_data,
)

from .correlation import (
    compute_pearson_per_cell,
    compute_pearson_from_directory,
)
