"""
preprocessing.py
----------------
Normalisation, filtering, and detrending of multi-channel
fluorescence imaging timeseries (voltage and calcium).
"""

import numpy as np
from scipy.signal import butter, filtfilt, detrend


def apply_mean_centering(data_array, offset_value=0):
    """
    Mean-centre each channel independently.

    Parameters
    ----------
    data_array : np.ndarray, shape (n_cells, n_timepoints)
    offset_value : float, optional
        Constant to add after centring (default 0)

    Returns
    -------
    np.ndarray, same shape as input
    """
    centered = np.zeros_like(data_array)
    for i in range(data_array.shape[0]):
        channel_mean = np.mean(data_array[i, :])
        centered[i, :] = data_array[i, :] - channel_mean + offset_value
    return centered


def apply_linear_detrend_after_centering(data_array):
    """
    Apply linear detrending to each channel independently.

    Parameters
    ----------
    data_array : np.ndarray, shape (n_cells, n_timepoints)

    Returns
    -------
    np.ndarray, same shape as input
    """
    detrended = np.zeros_like(data_array)
    for i in range(data_array.shape[0]):
        detrended[i, :] = detrend(data_array[i, :], type='linear')
    return detrended


def apply_minmax_normalisation(data_array):
    """
    Normalise each channel to the range [-1, 1] using min-max scaling.

    Parameters
    ----------
    data_array : np.ndarray, shape (n_cells, n_timepoints)

    Returns
    -------
    np.ndarray, same shape as input
    """
    normalised = np.zeros_like(data_array, dtype=float)
    for i in range(data_array.shape[0]):
        channel = data_array[i, :]
        ch_min, ch_max = np.min(channel), np.max(channel)
        ch_range = ch_max - ch_min
        if ch_range > 0:
            normalised[i, :] = 2 * (channel - ch_min) / ch_range - 1
        else:
            normalised[i, :] = 0.0
    return normalised


def apply_highpass_filter(data_array, sampling_rate_hz=5, cutoff_freq=0.01):
    """
    Apply a Butterworth high-pass filter to each channel independently.

    Parameters
    ----------
    data_array : np.ndarray, shape (n_cells, n_timepoints)
    sampling_rate_hz : float
        Imaging frame rate in Hz (default 5)
    cutoff_freq : float
        High-pass cutoff frequency in Hz (default 0.01)

    Returns
    -------
    np.ndarray, same shape as input
    """
    nyquist = sampling_rate_hz / 2.0
    normalised_cutoff = cutoff_freq / nyquist
    b, a = butter(2, normalised_cutoff, btype='high', analog=False)

    filtered = np.zeros_like(data_array, dtype=float)
    for i in range(data_array.shape[0]):
        filtered[i, :] = filtfilt(b, a, data_array[i, :])
    return filtered


def ensure_numeric_data(data_matrix):
    """
    Convert input to a float64 numpy array, replacing non-numeric
    values with zero.

    Parameters
    ----------
    data_matrix : array-like

    Returns
    -------
    np.ndarray, dtype float64
    """
    arr = np.array(data_matrix, dtype=object)
    numeric = np.zeros(arr.shape, dtype=np.float64)
    for idx in np.ndindex(arr.shape):
        try:
            numeric[idx] = float(arr[idx])
        except (TypeError, ValueError):
            numeric[idx] = 0.0
    return numeric
