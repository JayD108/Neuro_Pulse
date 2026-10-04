"""
NeuroPulse Feature Extractor (v2 — Full Feature Set)
======================================================
Multi-threaded EEG feature extraction pipeline.

Per-window features (per channel):
  - Welch PSD: absolute + relative power for 5 frequency bands (10 cols)
  - Hjorth parameters: Activity, Mobility, Complexity (3 cols)
  - RMS, Zero-Crossing Rate (2 cols)
  - Wavelet DWT energy: 6 levels (6 cols)
  - Sample Entropy (1 col)
  - Higuchi Fractal Dimension (1 col)
  Total per channel: 23 features x 18 channels = 414 channel features

Global features (whole window):
  - Cross-channel coherence: 8 pairs x 4 bands = 32 features
  - Relative band power ratios: 8 ratios x 18 channels = 144 features
  Grand total: ~590 features per window

Usage:
    python -m ml.preprocessing.feature_extractor --patient chb01 --workers 8
    OR imported as a module from ingest_dataset.py
"""

import os
import re
import warnings
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Dict, Optional

import mne
import numpy as np
import pandas as pd
from scipy.integrate import simpson as simps
from scipy.signal import welch, butter, sosfiltfilt, iirnotch, tf2sos

from ml.preprocessing.parser import parse_patient_summary, assign_window_label
from ml.preprocessing.advanced_features import (
    wavelet_energy,
    fast_sample_entropy,
    higuchi_fd,
    cross_channel_coherence,
    relative_band_power,
)

# Suppress MNE's verbose output for cleaner logs
mne.set_log_level("WARNING")
warnings.filterwarnings("ignore")

# -------------------------------------------------------------------
# Global Configuration
# -------------------------------------------------------------------
RAW_DIR = "./ml/data/raw"
PROCESSED_DIR = "./ml/data/processed"

WINDOW_SIZE_SEC = 10        # Epoch window length in seconds
OVERLAP_FRACTION = 0.0      # Overlap between windows (0 = non-overlapping)
BANDPASS_LOW = 0.5          # Hz
BANDPASS_HIGH = 50.0        # Hz
NOTCH_FREQ = 60.0           # Hz (US power line; change to 50 for EU)
MAX_CHANNELS = 18           # Standardize to 18 channels (10-20 system)
PREICTAL_WINDOW_MIN = 30    # Minutes before seizure to label as Preictal

FREQUENCY_BANDS: Dict[str, Tuple[float, float]] = {
    "Delta": (0.5, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 13.0),
    "Beta":  (13.0, 30.0),
    "Gamma": (30.0, 50.0),
}


# -------------------------------------------------------------------
# Per-window scipy filter (memory-safe alternative to MNE preload filter)
# -------------------------------------------------------------------
def _make_filters(sfreq: float):
    """
    Pre-compute SOS filter coefficients for bandpass (0.5-50 Hz)
    and notch (60 Hz) given a sampling frequency.
    Returns (sos_bp, sos_notch).
    """
    nyq = sfreq / 2.0
    # Bandpass: 0.5 - 50 Hz  (4th-order Butterworth)
    sos_bp = butter(4, [BANDPASS_LOW / nyq, BANDPASS_HIGH / nyq],
                    btype="band", output="sos")
    # Notch: 60 Hz — iirnotch returns (b, a), convert to SOS with tf2sos
    b_notch, a_notch = iirnotch(NOTCH_FREQ / nyq, Q=30)
    sos_notch = tf2sos(b_notch, a_notch)
    return sos_bp, sos_notch


def _apply_filters(epoch: np.ndarray, sos_bp, sos_notch) -> np.ndarray:
    """
    Apply bandpass then notch filter to a (n_channels, n_samples) epoch.
    Works entirely in-place on a small chunk — no full-file allocation.
    """
    filtered = sosfiltfilt(sos_bp, epoch, axis=1)
    filtered = sosfiltfilt(sos_notch, filtered, axis=1)
    return filtered


# -------------------------------------------------------------------
# Hjorth Parameters
# -------------------------------------------------------------------
def compute_hjorth_parameters(signal: np.ndarray) -> Tuple[float, float, float]:
    """
    Computes the three Hjorth parameters for a 1D signal array.

    - Activity:   Variance of the signal (overall power).
    - Mobility:   Ratio of std of the first derivative to std of the signal.
                  Approximates the mean frequency.
    - Complexity: Ratio of mobility of the first derivative to mobility of signal.
                  Approximates the bandwidth of the signal.

    Args:
        signal: 1D numpy array of EEG samples.

    Returns:
        Tuple of (activity, mobility, complexity).
    """
    activity = np.var(signal)
    if activity == 0:
        return 0.0, 0.0, 0.0

    d1 = np.diff(signal)
    d2 = np.diff(d1)

    var_d1 = np.var(d1)
    var_d2 = np.var(d2)

    mobility = np.sqrt(var_d1 / activity) if activity > 0 else 0.0
    mob_d1 = np.sqrt(var_d2 / var_d1) if var_d1 > 0 else 0.0
    complexity = mob_d1 / mobility if mobility > 0 else 0.0

    return float(activity), float(mobility), float(complexity)


# -------------------------------------------------------------------
# Per-file EDF Processing Worker
# -------------------------------------------------------------------
def process_single_edf(args: Tuple) -> pd.DataFrame:
    """
    Worker function: loads a single .edf file, preprocesses it,
    extracts features per epoch, and labels each window.

    Designed to be called by a ProcessPoolExecutor.

    Args:
        args: Tuple of (edf_path, seizure_times, window_size_sec)

    Returns:
        DataFrame of extracted features with target_label column.
        Empty DataFrame on error.
    """
    edf_path, seizure_times, window_size_sec = args
    filename = os.path.basename(edf_path)

    try:
        # --- Load entire EDF into RAM (fast: one sequential disk read per file) ---
        # With 8 workers: 8 x ~500MB = ~4GB peak — safe on 32GB.
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

        # --- Channel Selection ---
        valid_channels = [ch for ch in raw.ch_names if "-" in ch and ch.upper() != "-"]
        if not valid_channels:
            print(f"  [WARN] No valid differential channels in {filename}, skipping.")
            return pd.DataFrame()

        selected = valid_channels[:MAX_CHANNELS]
        raw.pick_channels(selected)

        sfreq            = raw.info["sfreq"]
        n_samples_window = int(window_size_sec * sfreq)

        # --- Get all data as numpy array (n_channels, n_total_samples) ---
        ch_names    = list(selected)        # save before del raw
        data_raw    = raw.get_data()        # float64, already in RAM
        n_total_samples = data_raw.shape[1]
        del raw   # release MNE object — ch_names + data_raw are all we need

        if n_total_samples < n_samples_window:
            print(f"  [WARN] File too short to epoch: {filename}")
            return pd.DataFrame()

        # --- Filter entire recording ONCE (bandpass + notch) ---
        # Vastly faster than filtering each window separately.
        sos_bp, sos_notch = _make_filters(sfreq)
        data = _apply_filters(data_raw, sos_bp, sos_notch)
        del data_raw   # free unfiltered copy

        # --- Sliding Window Segmentation (pure in-memory slicing) ---
        step    = int(n_samples_window * (1 - OVERLAP_FRACTION))
        windows = range(0, n_total_samples - n_samples_window + 1, step)

        features_list = []
        for epoch_idx, w_start_sample in enumerate(windows):
            w_end_sample = w_start_sample + n_samples_window
            w_start_sec  = w_start_sample / sfreq
            w_end_sec    = w_end_sample   / sfreq

            # --- Label Assignment ---
            label = assign_window_label(
                w_start_sec, w_end_sec, seizure_times,
                preictal_window_min=PREICTAL_WINDOW_MIN
            )

            # Skip ictal windows for the binary Preictal vs Interictal task
            if label == 2:
                continue

            # Slice window from in-memory filtered array (nanosecond fast)
            epoch_data = data[:, w_start_sample:w_end_sample]  # (n_ch, n_samples)

            row: Dict = {
                "epoch_id":     epoch_idx,
                "window_start": round(w_start_sec, 2),
                "window_end":   round(w_end_sec, 2),
                "file_source":  filename,
                "target_label": label,   # 0 = Interictal, 1 = Preictal
            }

            # --- Feature Extraction per channel ---
            psd_features: Dict[str, float] = {}  # collect for relative band power

            for ch_idx, ch_name in enumerate(ch_names):
                ch_signal = epoch_data[ch_idx]
                ch_key = ch_name.replace("-", "_").upper()

                # --- Welch PSD ---
                freqs, psd = welch(
                    ch_signal,
                    fs=sfreq,
                    nperseg=min(256, len(ch_signal)),
                )
                freq_res   = freqs[1] - freqs[0]
                total_power = max(simps(psd, dx=freq_res), 1e-10)

                for band_name, (fmin, fmax) in FREQUENCY_BANDS.items():
                    idx_band   = np.logical_and(freqs >= fmin, freqs <= fmax)
                    band_power = simps(psd[idx_band], dx=freq_res)
                    # Absolute power (new)
                    row[f"{ch_name}_{band_name}_power"] = float(band_power)
                    psd_features[f"{ch_key}_{band_name}_power"] = float(band_power)
                    # Relative power (existing)
                    row[f"{ch_name}_{band_name}_relpow"] = float(band_power / total_power)

                # --- Hjorth Parameters ---
                hjorth_act, hjorth_mob, hjorth_cmp = compute_hjorth_parameters(ch_signal)
                row[f"{ch_name}_hjorth_activity"]   = hjorth_act
                row[f"{ch_name}_hjorth_mobility"]   = hjorth_mob
                row[f"{ch_name}_hjorth_complexity"] = hjorth_cmp

                # --- Basic Time-Domain ---
                row[f"{ch_name}_rms"]            = float(np.sqrt(np.mean(ch_signal ** 2)))
                row[f"{ch_name}_zero_crossings"] = int(
                    np.where(np.diff(np.sign(ch_signal)))[0].shape[0]
                )

                # --- Advanced: Wavelet DWT Energy ---
                for wt_key, wt_val in wavelet_energy(ch_signal).items():
                    row[f"{ch_name}_{wt_key}"] = wt_val

                # --- Advanced: Sample Entropy ---
                row[f"{ch_name}_sample_entropy"] = fast_sample_entropy(ch_signal)

                # --- Advanced: Higuchi Fractal Dimension ---
                row[f"{ch_name}_higuchi_fd"] = higuchi_fd(ch_signal)

            # --- Advanced: Cross-Channel Coherence (whole window) ---
            coh_feats = cross_channel_coherence(epoch_data, ch_names, sfreq)
            row.update(coh_feats)

            # --- Advanced: Relative Band Power Ratios (whole window) ---
            relbp_feats = relative_band_power(psd_features, ch_names)
            row.update(relbp_feats)

            # Sanitize any NaN values
            for k, v in row.items():
                if isinstance(v, float) and np.isnan(v):
                    row[k] = 0.0

            features_list.append(row)

        if not features_list:
            return pd.DataFrame()

        df = pd.DataFrame(features_list)
        print(
            f"  [OK] {filename}: {len(df)} windows "
            f"[Interictal={int((df.target_label == 0).sum())}, "
            f"Preictal={int((df.target_label == 1).sum())}]"
        )
        return df

    except Exception as exc:
        print(f"  [FAIL] Error processing {filename}: {exc}")
        return pd.DataFrame()


# -------------------------------------------------------------------
# Per-Patient Ingestion Entry Point
# -------------------------------------------------------------------
def ingest_patient(patient_id: str, num_workers: int = 8) -> Optional[str]:
    """
    Orchestrates the full ingestion pipeline for one CHB-MIT patient:
    1. Parses the patient's summary file for seizure windows.
    2. Collects all .edf files for that patient.
    3. Dispatches per-EDF processing to a process pool.
    4. Concatenates results and saves as a labeled feature CSV.

    Args:
        patient_id: Patient folder name, e.g., "chb01".
        num_workers: Number of parallel CPU workers.

    Returns:
        Path to the saved CSV file, or None if ingestion failed.
    """
    patient_dir = os.path.join(RAW_DIR, patient_id)
    summary_path = os.path.join(patient_dir, f"{patient_id}-summary.txt")

    if not os.path.exists(patient_dir):
        print(f"[ERROR] Patient directory not found: {patient_dir}")
        return None

    # --- Parse seizure times from the summary file ---
    print(f"\n{'='*60}")
    print(f"[BRAIN] Processing patient: {patient_id}")
    print(f"{'='*60}")
    seizure_map = parse_patient_summary(summary_path)

    total_seizures = sum(len(v) for v in seizure_map.values())
    print(f"  Seizure map loaded: {len(seizure_map)} files, {total_seizures} total seizures")

    # --- Collect EDF files ---
    edf_files = sorted([f for f in os.listdir(patient_dir) if f.endswith(".edf")])
    if not edf_files:
        print(f"[ERROR] No .edf files found in {patient_dir}")
        return None

    print(f"  Found {len(edf_files)} EDF files")

    # --- Build task list ---
    tasks = [
        (
            os.path.join(patient_dir, edf),
            seizure_map.get(edf, []),
            WINDOW_SIZE_SEC,
        )
        for edf in edf_files
    ]

    # --- Parallel Processing ---
    print(f"  [*] Dispatching {len(tasks)} files across {num_workers} workers...\n")
    results = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_file = {executor.submit(process_single_edf, t): t[0] for t in tasks}
        for future in as_completed(future_to_file):
            df = future.result()
            if df is not None and not df.empty:
                results.append(df)

    if not results:
        print(f"[ERROR] No features extracted for {patient_id}. Check EDF files.")
        return None

    # --- Save labeled feature matrix ---
    final_df = pd.concat(results, ignore_index=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    save_path = os.path.join(PROCESSED_DIR, f"{patient_id}_labeled_features.csv")
    final_df.to_csv(save_path, index=False)

    n_interictal = int((final_df["target_label"] == 0).sum())
    n_preictal   = int((final_df["target_label"] == 1).sum())
    imbalance_ratio = n_interictal / max(n_preictal, 1)

    print(f"\n{'='*60}")
    print(f"[DONE] Ingestion complete for {patient_id}!")
    print(f"   Total windows  : {len(final_df)}")
    print(f"   Interictal (0) : {n_interictal}")
    print(f"   Preictal   (1) : {n_preictal}")
    print(f"   Imbalance ratio: {imbalance_ratio:.1f}:1")
    print(f"   Feature columns: {final_df.shape[1]}")
    print(f"   Saved to       : {save_path}")
    print(f"{'='*60}\n")

    return save_path


# -------------------------------------------------------------------
# CLI Entry Point
# -------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuroPulse Feature Extractor")
    parser.add_argument(
        "--patient", type=str, default="chb01",
        help="Patient ID to process (e.g., chb01, chb02). Default: chb01"
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Number of parallel CPU workers. Default: 8"
    )
    args = parser.parse_args()

    ingest_patient(args.patient, num_workers=args.workers)
