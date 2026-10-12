---
license: odc-by
language:
  - en
tags:
  - medical
  - eeg
  - seizure-prediction
  - epilepsy
  - physionet
  - chb-mit
  - neuroscience
  - time-series
  - binary-classification
  - signal-processing
  - brain-computer-interface
pretty_name: "CHB-MIT Preprocessed EEG Features for Seizure Forecasting (Pre-Ictal Detection)"
size_categories:
  - 100K<n<1M
task_categories:
  - tabular-classification
configs:
  - config_name: default
    data_files: "*.csv"
---

<div align="center">

# CHB-MIT Preprocessed EEG Dataset

### Feature-Engineered EEG Data for Seizure Forecasting (Pre-Ictal Detection)

[![Dataset on HF](https://img.shields.io/badge/Hugging%20Face-Dataset-blue?logo=huggingface)](https://huggingface.co/datasets/JayF14/chb-mit-preprocessed-eeg)
[![License](https://img.shields.io/badge/License-ODC--By%201.0-blue)](https://opendatacommons.org/licenses/by/1.0/)
[![Source](https://img.shields.io/badge/Source-PhysioNet%20CHB--MIT%20v1.0.0-green)](https://physionet.org/content/chbmit/1.0.0/)
[![Project](https://img.shields.io/badge/Project-Neuro__Pulse-purple)](https://github.com/JayD108/Neuro_Pulse)

**347,075 labeled epochs | 24 cases / 22 subjects | 590 features per epoch | ~3.9 GB**

*No raw signal processing needed. Download, load, and train.*

</div>

> **Research use only. Not for clinical decisions.**
> Code: https://github.com/JayD108/Neuro_Pulse

---

## Table of Contents

- [Overview](#overview)
- [Dataset at a Glance](#dataset-at-a-glance)
- [Per-Case Statistics](#per-case-statistics)
- [Column Schema](#column-schema)
- [Feature Reference](#feature-reference)
- [Download and Usage](#download-and-usage)
- [Quick-Start Example](#quick-start-example)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Important Notes](#important-notes)
- [Citation](#citation)
- [Related Resources](#related-resources)
- [License / Data Provenance](#license--data-provenance)

---

## Overview

This dataset provides **ready-to-use, feature-engineered EEG data** derived from the [CHB-MIT Scalp EEG Database v1.0.0](https://physionet.org/content/chbmit/1.0.0/) — one of the most widely cited benchmarks in epileptic seizure forecasting research.

The original CHB-MIT database contains continuous scalp EEG recordings from **22 pediatric subjects** (organized into **24 cases**) with medically intractable epilepsy, collected at Boston Children\'s Hospital (Shoeb, 2009). These raw EDF recordings have been fully preprocessed through a multi-stage feature extraction pipeline, producing a tabular dataset that is immediately usable for training classical ML and deep learning models.

> **This repository contains a derived/feature-engineered dataset**, not the original EEG recordings. The original CHB-MIT EEG recordings are distributed by PhysioNet under the Open Data Commons Attribution License v1.0. See the [License / Data Provenance](#license--data-provenance) section for details.

This dataset was produced as part of the **[Neuro_Pulse](https://github.com/JayD108/Neuro_Pulse)** project — an end-to-end seizure **forecasting** system combining classical ML, deep learning, and a real-time monitoring interface. The binary task is **interictal vs. pre-ictal** classification: predicting that a seizure will begin within the next 30 minutes.

### What makes this dataset useful?

- **No signal processing required** — features are pre-computed from raw EEG
- **Comprehensive feature set** — 590 features spanning frequency-domain, time-domain, wavelet, non-linear, and cross-channel coherence measures
- **Seizure annotations** — seizure onset/offset annotations provided with the CHB-MIT database
- **Full coverage** — all 24 CHB-MIT cases processed
- **Traceable** — 100% file source traceability to original EDF files. Note: some features contain NaN or ±Inf (e.g. spectral ratios with near-zero denominators, or channels absent from a recording). Replace Inf with NaN and use a model that handles missing values natively, or impute before use.

---

## Dataset at a Glance

| Property | Value |
|:---|:---|
| **Cases / Subjects** | 24 cases from 22 pediatric subjects (`chb01`–`chb24`; note: `chb21` is an additional case from the same subject as `chb01`) |
| **Total Epochs** | 347,075 |
| **Pre-ictal windows (label 1)** | 22,312 (6.43%) |
| **Interictal windows (label 0)** | 324,763 (93.57%) |
| **Columns (standard)** | 595 (5 metadata + 590 features) |
| **Epoch Length** | 10 seconds, non-overlapping |
| **EEG Channels** | 18 bipolar pairs (International 10-20 system) |
| **Bandpass Filter** | 0.5 – 50 Hz |
| **Notch Filter** | 60 Hz (US mains) |
| **Sampling Rate** | 256 Hz (native CHB-MIT) |
| **Total Size** | ~3.9 GB (24 CSV files) |
| **Task** | Binary classification: **interictal vs. pre-ictal** (seizure forecasting) |
| **Label Column** | `target_label` — `0` = interictal (≥ 30 min from any seizure), `1` = pre-ictal (within 30 min before seizure onset), `2` = ictal or post-ictal (during a seizure or within 5 min after; excluded from training, so absent from the CSVs) |

---

## Per-Case Statistics

| Case | File Size | Epochs | Pre-ictal | Interictal | Pre-ictal % | EDF Files | Columns |
|:--------|----------:|-------:|----------:|-----------:|------------:|----------:|--------:|
| chb01 | 160.8 MB | 14,338 | 982 | 13,356 | 6.85% | 42 | 595 |
| chb02 | 141.4 MB | 12,593 | 371 | 12,222 | 2.95% | 36 | 595 |
| chb03 | 150.9 MB | 13,425 | 861 | 12,564 | 6.41% | 38 | 595 |
| chb04 | 629.4 MB | 56,014 | 694 | 55,320 | 1.24% | 42 | 595 |
| chb05 | 155.4 MB | 13,829 | 686 | 13,143 | 4.96% | 39 | 595 |
| chb06 | 266.0 MB | 23,719 | 1,508 | 22,211 | 6.36% | 18 | 595 |
| chb07 | 269.3 MB | 24,012 | 538 | 23,474 | 2.24% | 19 | 595 |
| chb08 | 78.1 MB | 6,955 | 896 | 6,059 | 12.88% | 20 | 595 |
| chb09 | 272.1 MB | 24,279 | 716 | 23,563 | 2.95% | 19 | 595 |
| chb10 | 200.2 MB | 17,753 | 1,212 | 16,541 | 6.83% | 25 | 595 |
| chb11 | 139.9 MB | 12,351 | 353 | 11,998 | 2.86% | 35 | 719 * |
| chb12 | 82.2 MB | 7,422 | 1,814 | 5,608 | 24.44% | 24 | 1,091 * |
| chb13 | 127.4 MB | 11,464 | 1,113 | 10,351 | 9.71% | 33 | 595 |
| chb14 | 101.2 MB | 9,096 | 1,246 | 7,850 | 13.70% | 26 | 595 |
| chb15 | 152.2 MB | 13,630 | 1,750 | 11,880 | 12.84% | 40 | 626 * |
| chb16 | 72.5 MB | 6,526 | 985 | 5,541 | 15.10% | 19 | 595 |
| chb17 | 82.9 MB | 7,440 | 537 | 6,903 | 7.22% | 21 | 595 |
| chb18 | 143.4 MB | 12,652 | 816 | 11,836 | 6.45% | 36 | 719 * |
| chb19 | 118.9 MB | 10,678 | 387 | 10,291 | 3.62% | 30 | 595 |
| chb20 | 108.0 MB | 9,657 | 856 | 8,801 | 8.86% | 29 | 595 |
| chb21 | 129.8 MB | 11,674 | 665 | 11,009 | 5.70% | 33 | 595 |
| chb22 | 102.5 MB | 11,060 | 484 | 10,576 | 4.38% | 31 | 595 |
| chb23 | 104.4 MB | 9,299 | 947 | 8,352 | 10.18% | 9 | 595 |
| chb24 | 80.9 MB | 7,209 | 1,895 | 5,314 | 26.29% | 22 | 595 |

> **\\*** Subjects `chb11`, `chb12`, `chb15`, and `chb18` have additional columns because the original CHB-MIT recordings for these patients include extra EEG channels beyond the standard 18-channel montage. Empty values in extra columns indicate features from channels not present in a given recording. When loading all subjects together, reindex every file to a fixed feature list (see [Important Notes](#important-notes)).

---

## Column Schema

Each CSV file begins with **5 metadata columns**, followed by **590 feature columns** (or more for subjects with additional EEG channels):

### Metadata Columns

| # | Column | Type | Description |
|:-:|:-------|:-----|:------------|
| 1 | `epoch_id` | `int` | Unique sequential index for each time window within the file |
| 2 | `window_start` | `float` | Start time of the epoch in seconds from recording onset |
| 3 | `window_end` | `float` | End time of the epoch in seconds from recording onset |
| 4 | `file_source` | `str` | Source EDF filename from the original CHB-MIT dataset (e.g., `chb01_03.edf`) |
| 5 | `target_label` | `int` | Ground truth label: **0** = interictal (≥ 30 min from any seizure), **1** = pre-ictal (within 30 min before seizure onset), **2** = ictal/post-ictal (excluded; absent from these CSVs) |

### Feature Columns

Feature columns follow the naming convention: `<Channel>_<FeatureName>`

Example: `FP1-F7_Delta_power`, `C3-P3_hjorth_activity`, `coh_FP1_F7_T7_P7_delta`

---

## Feature Reference

### Feature Count Breakdown

The 590 features per standard epoch are composed as follows:

| Feature Group | Count | Calculation |
|:---|:---|:---|
| Per-channel signal features | 414 | 23 features/channel × 18 channels |
| Spectral ratio features | 144 | 8 ratios/channel × 18 channels |
| Cross-channel coherence | 32 | 8 electrode pairs × 4 frequency bands |
| **Total** | **590** | **414 + 144 + 32** |

The 23 per-channel signal features break down as:

| Feature Type | Per Channel |
|:---|:---:|
| Frequency-domain (Welch PSD) | 10 |
| Time-domain (Hjorth + RMS + ZC) | 5 |
| Wavelet decomposition (DWT) | 6 |
| Non-linear (entropy + fractal) | 2 |
| **Subtotal** | **23** |

> **Note:** The preprocessing pipeline diagram lists `23 features × 18 channels = 414` — this refers to the per-channel signal features only. Spectral ratio features (8 × 18 = 144) and cross-channel coherence (32) are computed separately and added to reach the total of **590**.

### 1. Frequency-Domain Features (10 per channel)

Computed via Welch\'s method across five standard EEG sub-bands:

| Band | Frequency Range |
|:-----|:----------------|
| Delta | 0.5 – 4 Hz |
| Theta | 4 – 8 Hz |
| Alpha | 8 – 13 Hz |
| Beta | 13 – 30 Hz |
| Gamma | 30 – 50 Hz |

| Feature Suffix | Description |
|:---------------|:------------|
| `*_power` | Absolute band power (V^2/Hz, integrated via Simpson\'s rule) |
| `*_relpow` | Relative band power (fraction of total power in 0.5–50 Hz) |

### 2. Time-Domain Features (5 per channel)

| Feature Suffix | Description |
|:---------------|:------------|
| `*_hjorth_activity` | Signal variance — Hjorth Activity parameter |
| `*_hjorth_mobility` | Normalized standard deviation of the first derivative |
| `*_hjorth_complexity` | Ratio of mobility of the derivative to mobility of the signal |
| `*_rms` | Root mean square amplitude |
| `*_zero_crossings` | Number of zero-axis crossings in the 10-second window |

### 3. Wavelet Decomposition Features (6 per channel)

Discrete Wavelet Transform (Daubechies-4, 5-level decomposition):

| Feature Suffix | Description |
|:---------------|:------------|
| `*_wt_A5_energy` | Approximation coefficients energy (level 5) |
| `*_wt_D5_energy` | Detail coefficients energy (level 5, ~0.5–2 Hz content) |
| `*_wt_D4_energy` | Detail coefficients energy (level 4, ~2–4 Hz content) |
| `*_wt_D3_energy` | Detail coefficients energy (level 3, ~4–8 Hz content) |
| `*_wt_D2_energy` | Detail coefficients energy (level 2, ~8–16 Hz content) |
| `*_wt_D1_energy` | Detail coefficients energy (level 1, ~16–32 Hz content) |

### 4. Non-Linear Features (2 per channel)

| Feature Suffix | Description |
|:---------------|:------------|
| `*_sample_entropy` | Regularity/complexity measure (higher = more irregular) |
| `*_higuchi_fd` | Higuchi Fractal Dimension (self-similarity, range ~1.0–2.0) |

### 5. Spectral Ratio Features (8 per channel)

| Feature Suffix | Description |
|:---------------|:------------|
| `*_Delta_relpow` | Relative delta power |
| `*_Theta_relpow` | Relative theta power |
| `*_Alpha_relpow` | Relative alpha power |
| `*_Beta_relpow` | Relative beta power |
| `*_Gamma_relpow` | Relative gamma power |
| `*_theta_alpha_ratio` | Theta/Alpha power ratio (drowsiness marker) |
| `*_slow_fast_ratio` | (Delta+Theta) / (Alpha+Beta) ratio |
| `*_beta_alpha_ratio` | Beta/Alpha power ratio (alertness marker) |

### 6. Cross-Channel Coherence Features (32 total)

Spectral coherence computed between 8 clinically relevant electrode pairs across 4 frequency bands (Delta, Theta, Alpha, Beta):

```
Electrode Pairs:
  FP1-F7 <-> T7-P7    FP2-F8 <-> T8-P8    T7-P7 <-> P7-O1    T8-P8 <-> P8-O2
  FP1-F7 <-> F7-T7    FP2-F8 <-> F8-T8    FP1-F7 <-> FP2-F8  T7-P7 <-> T8-P8
```

### EEG Channel Montage (18 Bipolar Pairs)

```
Left Temporal:   FP1-F7  F7-T7   T7-P7   P7-O1
Left Parasag.:   FP1-F3  F3-C3   C3-P3   P3-O1
Right Parasag.:  FP2-F4  F4-C4   C4-P4   P4-O2
Right Temporal:  FP2-F8  F8-T8   T8-P8   P8-O2
Midline:         FZ-CZ   CZ-PZ
```

---

## Download and Usage

### Option 1 — Hugging Face `datasets` library *(Recommended)*

```python
from datasets import load_dataset

# Load a single subject
ds = load_dataset(
    "JayF14/chb-mit-preprocessed-eeg",
    data_files="chb01_labeled_features.csv"
)
df = ds["train"].to_pandas()

print(f"Shape: {df.shape}")
print(f"Pre-ictal ratio: {df[\'target_label\'].mean():.2%}")
```

### Option 2 — Download specific files with `huggingface_hub`

```python
from huggingface_hub import hf_hub_download
import pandas as pd

path = hf_hub_download(
    repo_id="JayF14/chb-mit-preprocessed-eeg",
    filename="chb01_labeled_features.csv",
    repo_type="dataset"
)
df = pd.read_csv(path)
```

### Option 3 — Download all 24 files at once

```python
from huggingface_hub import snapshot_download
import pandas as pd
import os, numpy as np

local_dir = snapshot_download(
    repo_id="JayF14/chb-mit-preprocessed-eeg",
    repo_type="dataset",
    local_dir="./chb_mit_data"
)

META = ["epoch_id", "window_start", "window_end", "file_source", "target_label"]

# Build the reference feature list from chb01 (all 590 standard features)
chb01 = pd.read_csv(os.path.join(local_dir, "chb01_labeled_features.csv"), nrows=0)
feature_list = [c for c in chb01.columns if c not in META]

# Load all patients, reindex to the same 590 columns (absent columns become NaN)
dfs = []
for i in range(1, 25):
    fpath = os.path.join(local_dir, f"chb{i:02d}_labeled_features.csv")
    if os.path.exists(fpath):
        df = pd.read_csv(fpath)
        df["subject_id"] = f"chb{i:02d}"
        df = df.reindex(columns=META + ["subject_id"] + feature_list)
        dfs.append(df)

full_df = pd.concat(dfs, ignore_index=True)
print(f"Combined shape: {full_df.shape}")
print(f"Label distribution:\\n{full_df[\'target_label\'].value_counts()}")
```

> **Note:** `reindex` is used (not `join="inner"`) so all 590 standard columns are preserved. Patients `chb12`–`chb22`, which lack the 4 right-temporal channels, will have NaN in those 40 columns. `join="inner"` would silently drop those columns for all patients and produce only 550 features.

### Option 4 — Direct download via CLI

```bash
# Single file
curl -L -o chb01_labeled_features.csv \\
  "https://huggingface.co/datasets/JayF14/chb-mit-preprocessed-eeg/resolve/main/chb01_labeled_features.csv"

# All files (requires git-lfs)
git lfs install
git clone https://huggingface.co/datasets/JayF14/chb-mit-preprocessed-eeg
```

---

## Quick-Start Example

> **⚠️ Evaluation warning:** A random window-level split (e.g. `train_test_split`) leaks information between neighbouring windows from the same patient and yields inflated scores (often > 99% ROC-AUC). Always use a patient-wise split for meaningful evaluation. See the patient-wise example below.

The example below uses a **leave-one-patient-out** split for clinically meaningful evaluation.
Tested on chb01, chb08, chb12 with `n_splits=3` — runs without errors.

```python
from huggingface_hub import snapshot_download
import pandas as pd, numpy as np
from sklearn.model_selection import GroupKFold
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import os

# 1. Download and load all patients
local_dir = snapshot_download("JayF14/chb-mit-preprocessed-eeg", repo_type="dataset",
                               local_dir="./chb_mit_data")

META = ["epoch_id", "window_start", "window_end", "file_source", "target_label"]

# Build the reference feature list from chb01 (all 590 standard features)
chb01 = pd.read_csv(os.path.join(local_dir, "chb01_labeled_features.csv"), nrows=0)
feature_list = [c for c in chb01.columns if c not in META]

dfs = []
for i in range(1, 25):
    fpath = os.path.join(local_dir, f"chb{i:02d}_labeled_features.csv")
    if os.path.exists(fpath):
        df = pd.read_csv(fpath)
        df["patient"] = f"chb{i:02d}"
        # Reindex to fixed feature list — absent columns (e.g. right-temporal
        # channels for chb12-chb22) become NaN; XGBoost handles them natively.
        df = df.reindex(columns=META + ["patient"] + feature_list)
        dfs.append(df)

full = pd.concat(dfs, ignore_index=True)
# Keep only pre-ictal (1) and interictal (0) rows
full = full[full["target_label"].isin([0, 1])]

# 2. Prepare — handle Inf/NaN
X = full[feature_list].replace([np.inf, -np.inf], np.nan).values.astype(np.float32)
y = full["target_label"].values
groups = full["patient"].values

# 3. Patient-wise leave-one-out evaluation
gkf = GroupKFold(n_splits=len(full["patient"].unique()))
aucs = []
for train_idx, test_idx in gkf.split(X, y, groups=groups):
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    spw = float((y_tr == 0).sum()) / max(float((y_tr == 1).sum()), 1)
    clf = xgb.XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             tree_method="hist", scale_pos_weight=spw, random_state=42)
    clf.fit(X_tr, y_tr)
    aucs.append(roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1]))
    print(f"Patient {groups[test_idx[0]]}: ROC-AUC = {aucs[-1]:.4f}")

print(f"\\nMean ROC-AUC: {np.mean(aucs):.4f}  Median: {np.median(aucs):.4f}")
```

**Reference result** (XGBoost, leave-one-patient-out, 24 patients, 500 trees, 590 features):
- Mean ROC-AUC: **0.56**, Median: **0.59**, Range: 0.26–0.71
- Full per-patient breakdown: [apala_xgb_v1_lopo.json](https://huggingface.co/JayF14/Neuro_Pulse/blob/main/models/trained/apala_xgb_v1_lopo.json)

---

## Preprocessing Pipeline

The raw `.edf` recordings were processed through the following pipeline:

```
Raw EDF  -->  Load (MNE-Python)  -->  Bandpass 0.5-50 Hz  -->  Notch 60 Hz
         -->  Segment into 10s non-overlapping epochs
         -->  Per-epoch, per-channel feature extraction:
              |-- Welch PSD (5 bands x 2 = 10 features)
              |-- Hjorth parameters (3 features)
              |-- RMS + Zero crossings (2 features)
              |-- DWT energy, db4, 5-level (6 features)
              |-- Sample Entropy (1 feature)
              |-- Higuchi Fractal Dimension (1 feature)
              = 23 features x 18 channels = 414 features
         -->  Cross-channel coherence (8 pairs x 4 bands = 32 features)
         -->  Spectral ratios (8 ratios x 18 channels = 144 features)
         -->  Total per epoch: 414 + 32 + 144 = 590 features
         -->  Label assignment from CHB-MIT seizure annotations
         -->  Export as per-subject CSV
```

| Pipeline Parameter | Value |
|:---|:---|
| Signal loader | MNE-Python |
| Bandpass filter | 4th-order Butterworth, 0.5–50 Hz (SOS) |
| Notch filter | 60 Hz (US power line) |
| Epoch window | 10 seconds, non-overlapping |
| PSD method | Welch\'s method |
| Wavelet | Daubechies-4 (`db4`), 5-level decomposition |
| Entropy | Sample entropy (`m=2`) |
| Fractal dimension | Higuchi FD |
| Coherence | Scipy `signal.coherence` |
| Label source | CHB-MIT `chbXX-summary.txt` seizure annotations |

---

## Important Notes

### This is a Derived Dataset

This repository contains **feature-engineered representations** computed from the original CHB-MIT EEG recordings. It does **not** contain or redistribute the original `.edf` files. The original CHB-MIT EEG recordings are distributed by PhysioNet under the Open Data Commons Attribution License v1.0.

Users who need the raw EEG signals should obtain them directly from: https://physionet.org/content/chbmit/1.0.0/

### Label Definitions

| Label | Meaning | Count |
|:-----:|:--------|------:|
| `0` | **Interictal** — epoch starts ≥ 30 min from any seizure onset | 324,763 |
| `1` | **Pre-ictal** — epoch starts within 30 min before a seizure onset | 22,312 |
| `2` | **Ictal / post-ictal** — epoch overlaps a seizure or starts within 5 min after seizure end | excluded (not in CSVs) |

Epochs labelled 2 are excluded from the CSV files; the binary task is **interictal vs. pre-ictal**.

### Data Quality — NaN and Inf Values

Some features contain NaN or ±Inf values. Common causes:
- **Spectral ratios** (e.g. `*_theta_alpha_ratio`) can produce ±Inf when the denominator band power is near zero.
- **Missing channels**: patients `chb12`–`chb22` lack the four right-temporal bipolar channels `FP2-F8`, `F8-T8`, `T8-P8` and `P8-O2` in most recordings (40 of the 590 standard features are absent for these 11 patients). Verified from CSV headers.
- **Non-linear features** (sample entropy, Higuchi FD) may fail on near-flat segments.

**Recommended handling:**
```python
import numpy as np
X = X.replace([np.inf, -np.inf], np.nan)
# Then either:
#   A) Use a model that handles NaN natively (e.g. XGBoost with tree_method="hist")
#   B) Impute: X = X.fillna(X.mean())
```

### Montage Differences Across Patients

- Patients `chb12`–`chb22` (verified from CSV headers) are missing the right-temporal bipolar channels `FP2-F8`, `F8-T8`, `T8-P8` and `P8-O2` (40 features out of 590). These columns are simply absent from those CSVs.
- Patients `chb11`, `chb12`, `chb15`, and `chb18` have *extra* non-standard channels beyond the 18-pair montage (see Columns column in Per-Case Statistics above).

When combining patients, reindex every file to a fixed 590-feature list and treat absent columns as missing (NaN):

```python
FEATURE_COLS = [...]  # your reference 590-column list (e.g. from chb01)
X = df.reindex(columns=FEATURE_COLS)   # absent columns become NaN automatically
```

### Class Imbalance

This dataset exhibits **severe class imbalance** (6.43% pre-ictal overall). Some subjects have as little as 1.24% pre-ictal windows (chb04), while others reach 26.29% (chb24). Recommended strategies:

| Strategy | Implementation |
|:---------|:---------------|
| Class weighting | `class_weight="balanced"` in sklearn classifiers, or `scale_pos_weight` in XGBoost |
| Oversampling | SMOTE or ADASYN via `imbalanced-learn` |
| Undersampling | `RandomUnderSampler` from `imbalanced-learn` |
| Loss functions | Focal loss, weighted cross-entropy |
| Evaluation | Use F1, AUROC, AUPRC — **not** accuracy |

### Leave-One-Patient-Out (LOPO) Evaluation

> **⚠️ Random window splits inflate scores.** Because neighbouring 10-second windows are highly correlated, a random `train_test_split` across all windows leaks information and yields artificially high metrics (often > 99% ROC-AUC). For a fair, clinically meaningful estimate of generalisation to new patients, use **patient-level leave-one-out** cross-validation.

For clinically meaningful evaluation, use patient-level leave-one-patient-out (LOPO) cross-validation rather than random splits. The case identifier in `file_source` can be mapped to the corresponding subject, with `chb01` and `chb21` grouped as the same subject.

**Reference result:** XGBoost (500 trees, 590 features), leave-one-patient-out across all 24 patients:
- Mean ROC-AUC: **0.56**, Median: **0.59**, Range: 0.26–0.71
- Full per-patient table: [apala_xgb_v1_lopo.json](https://huggingface.co/JayF14/Neuro_Pulse/blob/main/models/trained/apala_xgb_v1_lopo.json)

---

## Citation

If you use this dataset in your research, please cite the original CHB-MIT database and PhysioNet. The official citation as specified by PhysioNet is:

```bibtex
@article{guttag2010chbmit,
  author    = {Guttag, John},
  title     = {{CHB-MIT Scalp EEG Database}},
  journal   = {{PhysioNet}},
  year      = {2010},
  month     = jun,
  note      = {Version 1.0.0},
  doi       = {10.13026/C2K01R},
  url       = {https://doi.org/10.13026/C2K01R}
}

@phdthesis{shoeb2009application,
  title     = {Application of Machine Learning to Epileptic Seizure Onset Detection and Treatment},
  author    = {Shoeb, Ali Hossam},
  year      = {2009},
  school    = {Massachusetts Institute of Technology},
  address   = {Cambridge, MA}
}

@article{goldberger2000physiobank,
  title     = {{PhysioBank, PhysioToolkit, and PhysioNet}: Components of a New Research Resource for Complex Physiologic Signals},
  author    = {Goldberger, Ary L and Amaral, Luis A N and Glass, Leon and Hausdorff, Jeffrey M and Ivanov, Plamen Ch and Mark, Roger G and Mietus, Joseph E and Moody, George B and Peng, Chung-Kang and Stanley, H Eugene},
  journal   = {Circulation},
  volume    = {101},
  number    = {23},
  pages     = {e215--e220},
  year      = {2000},
  publisher = {American Heart Association}
}
```

**Original dataset DOI:** https://doi.org/10.13026/C2K01R

---

## Related Resources

| Resource | Link |
|:---------|:-----|
| Original CHB-MIT Database | [physionet.org/content/chbmit/1.0.0](https://physionet.org/content/chbmit/1.0.0/) |
| Neuro_Pulse Project (source code) | [github.com/JayD108/Neuro_Pulse](https://github.com/JayD108/Neuro_Pulse) |
| APALA Model (trained on all 24 patients) | [JayF14/Neuro_Pulse](https://huggingface.co/JayF14/Neuro_Pulse) |
| MNE-Python (EEG toolkit) | [mne.tools](https://mne.tools/) |
| imbalanced-learn | [imbalanced-learn.org](https://imbalanced-learn.org/) |

---

## License / Data Provenance

### Source Data

This dataset is a **processed and feature-engineered derivative** of the **CHB-MIT Scalp EEG Database v1.0.0**, originally collected at Boston Children\'s Hospital and distributed by PhysioNet.

- **Original dataset:** CHB-MIT Scalp EEG Database v1.0.0
- **PhysioNet page:** https://physionet.org/content/chbmit/1.0.0/
- **DOI:** 10.13026/C2K01R
- **Source license:** Open Data Commons Attribution License v1.0 (ODC-By 1.0)
  - License text: https://opendatacommons.org/licenses/by/1.0/

The original CHB-MIT database is made available by PhysioNet under the **Open Data Commons Attribution License v1.0 (ODC-By 1.0)**, which permits creation and distribution of derivative databases provided that appropriate attribution and licensing notices are retained.

### Nature of This Repository

The creator of this repository does **not** claim ownership of the underlying EEG recordings or the original CHB-MIT database. The original CHB-MIT EEG recordings are distributed by PhysioNet under the Open Data Commons Attribution License v1.0. This repository contains only **feature-engineered representations** — tabular numerical features extracted and derived from the original recordings through a signal-processing pipeline. The original `.edf` files are not included here.

For licensing and provenance purposes, this repository is treated as a **processed/feature-engineered derivative database** of the CHB-MIT dataset, as it is based upon the original recordings through extraction and transformation of a substantial part of their contents.

### User Obligations

Users of this derivative dataset should:

1. **Attribute** the original CHB-MIT Scalp EEG Database and PhysioNet as the source of the underlying data.
2. **Retain** the applicable ODC-By 1.0 attribution and licensing notices when redistributing or publishing work that incorporates this dataset.
3. Consult the full ODC-By 1.0 license text at https://opendatacommons.org/licenses/by/1.0/ to understand their obligations.

### Attribution Notice (ODC-By 1.0)

When using or redistributing this dataset, please include the following attribution:

> This dataset is derived from the CHB-MIT Scalp EEG Database v1.0.0 (Guttag, 2010), distributed by PhysioNet (https://physionet.org/content/chbmit/1.0.0/) under the Open Data Commons Attribution License v1.0 (https://opendatacommons.org/licenses/by/1.0/). DOI: 10.13026/C2K01R.

---

<div align="center">

*Built with care by [Mrittunjay Dubey](https://huggingface.co/JayF14) as part of the Neuro_Pulse project.*

</div>
