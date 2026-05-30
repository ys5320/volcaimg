"""
correlation.py
--------------
Pearson cross-correlation between simultaneous voltage and
calcium fluorescence timeseries, computed per cell.
"""

import numpy as np
import warnings
from pathlib import Path
from scipy.stats import pearsonr


def compute_pearson_per_cell(voltage_array, calcium_array, flip_voltage=True):
    """
    Compute Pearson correlation between voltage and calcium traces
    for each cell independently.

    Parameters
    ----------
    voltage_array : np.ndarray, shape (n_cells, n_timepoints)
        Voltage fluorescence timeseries, one row per cell.
    calcium_array : np.ndarray, shape (n_cells, n_timepoints)
        Calcium fluorescence timeseries, one row per cell.
    flip_voltage : bool, optional
        If True, negate voltage traces before correlating (default True).
        This reflects the sign convention used in JEDI-1P voltage imaging
        where depolarisation produces a decrease in fluorescence.

    Returns
    -------
    correlations : list of float
        Pearson r values, one per cell. Cells with NaN values are skipped.

    Examples
    --------
    >>> import numpy as np
    >>> from volcaimg.correlation import compute_pearson_per_cell
    >>> rng = np.random.default_rng(42)
    >>> voltage = rng.normal(size=(5, 200))
    >>> calcium = -voltage + rng.normal(scale=0.1, size=(5, 200))
    >>> r = compute_pearson_per_cell(voltage, calcium)
    >>> print([round(x, 2) for x in r])
    """
    if voltage_array.shape != calcium_array.shape:
        raise ValueError(
            f"voltage_array and calcium_array must have the same shape, "
            f"got {voltage_array.shape} and {calcium_array.shape}"
        )

    correlations = []
    for i in range(voltage_array.shape[0]):
        v = -voltage_array[i] if flip_voltage else voltage_array[i]
        c = calcium_array[i]

        if np.isnan(v).any() or np.isnan(c).any():
            warnings.warn(f"Cell {i} contains NaN values and will be skipped.")
            continue

        r, _ = pearsonr(v, c)
        correlations.append(r)

    return correlations


def compute_pearson_from_directory(results_dir, flip_voltage=True):
    """
    Compute per-cell Pearson correlations from a directory of paired
    voltage and calcium CSV files.

    Expects files named:
        pre_voltage_*_<folder_name>.csv
        pre_calcium_*_<folder_name>.csv

    Each CSV should have cells as rows and timepoints as columns.

    Parameters
    ----------
    results_dir : str or Path
        Path to the results directory containing per-trial subfolders.
    flip_voltage : bool, optional
        Passed to compute_pearson_per_cell (default True).

    Returns
    -------
    list of float
        All per-cell Pearson r values across all trials.
    """
    import pandas as pd

    results_dir = Path(results_dir)
    all_correlations = []

    for folder in sorted(results_dir.iterdir()):
        if not folder.is_dir():
            continue

        folder_name = folder.name
        voltage_files = sorted(folder.glob(f'pre_voltage_*_{folder_name}.csv'))
        calcium_files = sorted(folder.glob(f'pre_calcium_*_{folder_name}.csv'))

        if len(voltage_files) != len(calcium_files):
            warnings.warn(
                f"Mismatched pairs in {folder_name}: "
                f"{len(voltage_files)} voltage, {len(calcium_files)} calcium files."
            )
            continue

        if len(voltage_files) == 0:
            continue

        for v_file, c_file in zip(voltage_files, calcium_files):
            df_v = pd.read_csv(v_file, index_col=0)
            df_c = pd.read_csv(c_file, index_col=0)

            if df_v.shape != df_c.shape:
                warnings.warn(
                    f"Shape mismatch in {folder_name}: "
                    f"voltage {df_v.shape}, calcium {df_c.shape}."
                )
                continue

            r_values = compute_pearson_per_cell(
                df_v.values, df_c.values, flip_voltage=flip_voltage
            )
            all_correlations.extend(r_values)

    return all_correlations
