# volcaimg

Python package for analysing simultaneous voltage and calcium fluorescence
imaging timeseries in cancer cells.

Developed during doctoral research at Imperial College London on the
bioelectrical dynamics of human breast cancer cells.

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Background

Cancer cells exhibit spontaneous membrane voltage fluctuations whose
relationship to intracellular calcium dynamics remains poorly understood.
This package provides tools to quantify the temporal relationship between
simultaneous voltage (JEDI-1P) and calcium (Calbryte-630) fluorescence
imaging timeseries at single-cell resolution, including preprocessing,
event detection, and cross-channel correlation analysis.

The core correlation analysis underpins the findings reported in:

> Sun Y, Eneva B, Djamgoz MBA, Bakal C, Foust AJ.
> *"Anticorrelated voltage and calcium dynamics in human breast cancer cells."*
> Manuscript in preparation.

> Quicke P\*, Sun Y\* et al.
> *"Voltage imaging reveals the dynamic electrical signatures of human breast cancer cells."*
> Communications Biology, 2022. (\*co-first authors)

---

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/ys5320/volcaimg.git
```

---

## Quickstart

```python
import numpy as np
import volcaimg as vi

# Load your timeseries data: shape (n_cells, n_timepoints)
voltage = np.load("voltage_traces.npy")
calcium = np.load("calcium_traces.npy")

# Step 1: Preprocess
voltage_filtered = vi.apply_highpass_filter(voltage, sampling_rate_hz=5)
calcium_centered = vi.apply_mean_centering(calcium)

# Step 2: Compute per-cell Pearson correlation between voltage and calcium
r_values = vi.compute_pearson_per_cell(voltage_filtered, calcium_centered)

print(f"Mean correlation across {len(r_values)} cells: {np.mean(r_values):.3f}")
```

To analyse a full directory of paired CSV files from the imaging pipeline:

```python
from volcaimg.correlation import compute_pearson_from_directory

r_values = compute_pearson_from_directory("path/to/results/")
```

---

## Module Overview

| Module | Description |
|---|---|
| `volcaimg.preprocessing` | Normalisation, high-pass filtering, detrending |
| `volcaimg.correlation` | Per-cell Pearson correlation between voltage and calcium channels |
| `volcaimg.detection` | Voltage and calcium event detection |
| `volcaimg.qc` | Quality control tools for event curation |
| `volcaimg.summary` | Combining results across experimental trials |
| `volcaimg.visualisation` | AVI video export and timeseries plotting |

---

## Key Functions

### Preprocessing

```python
# Mean-centre each cell independently
centered = vi.apply_mean_centering(data)

# High-pass filter to remove slow drift
filtered = vi.apply_highpass_filter(data, sampling_rate_hz=5, cutoff_freq=0.01)

# Min-max normalise to [-1, 1]
normalised = vi.apply_minmax_normalisation(data)

# Linear detrend after centring
detrended = vi.apply_linear_detrend_after_centering(data)
```

### Correlation

```python
# Per-cell Pearson r between voltage and calcium arrays
r_values = vi.compute_pearson_per_cell(voltage, calcium, flip_voltage=True)

# From a directory of paired CSV files
r_values = vi.compute_pearson_from_directory("results/")
```

---

## Running Tests

```bash
git clone https://github.com/ys5320/volcaimg.git
cd volcaimg
pip install pytest numpy scipy
PYTHONPATH=src pytest tests/ -v
```

---

## Experimental Context

Data were acquired using:
- **Voltage indicator:** JEDI-1P (genetically encoded)
- **Calcium indicator:** Calbryte-630 (chemical dye)
- **Cell lines:** MDA-MB-231, MDA-MB-468, and other human breast cancer lines
- **Imaging:** Simultaneous dual-channel widefield fluorescence microscopy
- **Sampling rate:** 5 Hz
- **Analysis:** HPC-compatible pipeline (Imperial College London CX3 cluster)

---

## Citation

If you use this package in your research, please cite:
Sun Y, Eneva B, Djamgoz MBA, Bakal C, Foust AJ.
"Anticorrelated voltage and calcium dynamics in human breast cancer cells."
Manuscript in preparation.
Quicke P*, Sun Y*, Arias-Garcia M, Beykou M, Acker CD, Djamgoz MBA, Bakal C, Foust AJ.
"Voltage imaging reveals the dynamic electrical signatures of human breast cancer cells."
Communications Biology, 2022.
---

## Author

**Yilin Sun**
PhD Candidate, Bioengineering, Imperial College London
yilin.sun20@imperial.ac.uk
