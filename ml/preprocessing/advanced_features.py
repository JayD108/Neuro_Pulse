"""
NeuroPulse Advanced Feature Engineering
========================================
Extends the base feature set with advanced classical signal processing features:

1. Wavelet Decomposition (DWT)     - Multi-scale time-frequency energy
2. Sample Entropy                  - Signal complexity / irregularity
3. Higuchi Fractal Dimension       - Self-similarity (non-linearity biomarker)
4. Cross-Channel Coherence         - Inter-electrode synchronization
5. Relative Band Power Ratios      - Robust normalized power features

Usage:
    from ml.preprocessing.advanced_features import extract_advanced_features
    advanced_df = extract_advanced_features(epoch_array, sfreq, ch_names)
"""

import warnings
import numpy as np
from typing import List, Dict, Tuple
from scipy.signal import coherence

try:
    import numba
    from numba import njit
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Wavelet Decomposition (DWT)
# ---------------------------------------------------------------------------

def wavelet_energy(signal: np.ndarray, wavelet: str = "db4", level: int = 5) -> Dict[str, float]:
    """
    Compute energy at each level of the Discrete Wavelet Transform.
    Captures transient bursts at multiple time scales -- superior to Fourier
    for non-stationary EEG signals.
    """
    try:
        import pywt
        coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level)
        features = {}
        for i, coeff in enumerate(coeffs):
            key = f"wt_A{level}_energy" if i == 0 else f"wt_D{level - i + 1}_energy"
            features[key] = float(np.sum(coeff ** 2))
        return features
    except ImportError:
        keys = [f"wt_A{level}_energy"] + [f"wt_D{level - i + 1}_energy" for i in range(1, level + 1)]
        return {k: 0.0 for k in keys}


# ---------------------------------------------------------------------------
# Sample Entropy — Numba JIT compiled (100-150x faster than pure Python)
# ---------------------------------------------------------------------------

if _NUMBA_AVAILABLE:
    @njit(cache=True)
    def _sampen_core(sig: np.ndarray, m: int, r: float) -> tuple:
        """
        Inner loop compiled to machine code by numba.
        Counts template matches for SampEn using explicit O(N^2) loop
        that avoids Python overhead and large broadcast arrays.
        cache=True saves the compiled binary — only compiles once ever.
        """
        N = len(sig)
        A = 0  # m+1 matches
        B = 0  # m   matches
        for i in range(N - m):
            for j in range(i + 1, N - m):
                # Check m-length template match (Chebyshev distance < r)
                match_m = True
                for k in range(m):
                    if abs(sig[i + k] - sig[j + k]) >= r:
                        match_m = False
                        break
                if match_m:
                    B += 1
                    # Extend by one — check m+1 match
                    if abs(sig[i + m] - sig[j + m]) < r:
                        A += 1
        return A, B
else:
    def _sampen_core(sig, m, r):  # fallback: won't be fast but won't crash
        N = len(sig)
        A = 0; B = 0
        for i in range(N - m):
            for j in range(i + 1, N - m):
                if max(abs(sig[i+k] - sig[j+k]) for k in range(m)) < r:
                    B += 1
                    if abs(sig[i+m] - sig[j+m]) < r:
                        A += 1
        return A, B


def fast_sample_entropy(signal: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    """
    Sample Entropy via numba-JIT compiled core loop.
    Measures signal unpredictability/complexity — preictal EEG becomes
    more synchronised (lower entropy) before a seizure.

    First call triggers JIT compilation (~2s). All subsequent calls: ~1-5ms.
    With cache=True the compiled binary is reused across sessions.
    """
    N = len(signal)
    if N < 20:
        return np.nan
    r = r_factor * float(np.std(signal, ddof=1))
    if r == 0:
        return 0.0
    # Downsample long signals to keep O(N^2) cost bounded at 512 pts
    if N > 512:
        sig = signal[::N // 512].astype(np.float64)
    else:
        sig = signal.astype(np.float64)
    try:
        A, B = _sampen_core(sig, m, r)
        if B <= 0 or A <= 0:
            return 0.0
        return float(-np.log(A / B))
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# Higuchi Fractal Dimension
# ---------------------------------------------------------------------------

def higuchi_fd(signal: np.ndarray, kmax: int = 6) -> float:
    """
    Higuchi Fractal Dimension -- measures self-similarity.
    Known biomarker: HFD rises in the preictal period.
    """
    N = len(signal)
    if N < kmax * 2:
        return np.nan
    x = np.array(signal)
    Lk = []
    for k in range(1, kmax + 1):
        Lmk = []
        for m in range(1, k + 1):
            indices = np.arange(m - 1, N, k)
            if len(indices) < 2:
                continue
            subseq = x[indices]
            Lm = np.sum(np.abs(np.diff(subseq))) * (N - 1) / (k * len(indices))
            Lmk.append(Lm)
        if Lmk:
            Lk.append(np.mean(Lmk))
    if len(Lk) < 2:
        return np.nan
    log_k = np.log(np.arange(1, len(Lk) + 1))
    log_L = np.log(np.array(Lk) + 1e-10)
    try:
        return float(-np.polyfit(log_k, log_L, 1)[0])
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# Cross-Channel Coherence
# ---------------------------------------------------------------------------

_COHERENCE_PAIRS = [
    ("FP1-F7", "T7-P7"),
    ("FP2-F8", "T8-P8"),
    ("T7-P7",  "P7-O1"),
    ("T8-P8",  "P8-O2"),
    ("FP1-F7", "F7-T7"),
    ("FP2-F8", "F8-T8"),
    ("FP1-F7", "FP2-F8"),
    ("T7-P7",  "T8-P8"),
]

def cross_channel_coherence(
    data: np.ndarray,
    ch_names: List[str],
    sfreq: float,
    bands: Dict[str, Tuple[float, float]] = None,
) -> Dict[str, float]:
    """
    Magnitude-squared coherence between key electrode pairs per frequency band.
    Seizures involve synchronization across brain regions -- coherence rises
    in frontal-temporal and interhemispheric pairs during the preictal period.
    """
    if bands is None:
        bands = {
            "delta": (0.5, 4.0),
            "theta": (4.0, 8.0),
            "alpha": (8.0, 13.0),
            "beta":  (13.0, 30.0),
        }
    ch_index = {name.upper(): i for i, name in enumerate(ch_names)}
    features: Dict[str, float] = {}
    for ch1_name, ch2_name in _COHERENCE_PAIRS:
        idx1 = ch_index.get(ch1_name.upper())
        idx2 = ch_index.get(ch2_name.upper())
        for band_name in bands:
            key = f"coh_{ch1_name}_{ch2_name}_{band_name}".replace("-", "_")
            if idx1 is None or idx2 is None:
                features[key] = 0.0
                continue
            try:
                f, Cxy = coherence(data[idx1], data[idx2], fs=sfreq, nperseg=min(256, len(data[idx1]) // 4))
                flo, fhi = bands[band_name]
                mask = (f >= flo) & (f <= fhi)
                features[key] = float(np.mean(Cxy[mask])) if mask.any() else 0.0
            except Exception:
                features[key] = 0.0
    return features


# ---------------------------------------------------------------------------
# Relative Band Power Ratios
# ---------------------------------------------------------------------------

def relative_band_power(
    psd_features: Dict[str, float],
    ch_names: List[str],
    bands: List[str] = None,
) -> Dict[str, float]:
    """
    Relative power = power(band) / total_power. Normalizes for electrode
    impedance and skull thickness -- more robust across patients.
    Also computes clinical ratios: theta/alpha, slow/fast, beta/alpha.
    """
    if bands is None:
        bands = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
    features: Dict[str, float] = {}
    for ch in ch_names:
        ch_key = ch.replace("-", "_").upper()
        band_powers = {b: psd_features.get(f"{ch_key}_{b}_power", 0.0) for b in bands}
        total = sum(band_powers.values()) + 1e-10
        for band, power in band_powers.items():
            features[f"{ch_key}_{band}_relpow"] = power / total
        theta = band_powers.get("Theta", 0.0)
        alpha = band_powers.get("Alpha", 0.0) + 1e-10
        delta = band_powers.get("Delta", 0.0)
        beta  = band_powers.get("Beta",  0.0) + 1e-10
        features[f"{ch_key}_theta_alpha_ratio"] = theta / alpha
        features[f"{ch_key}_slow_fast_ratio"]   = (delta + theta) / (alpha + beta)
        features[f"{ch_key}_beta_alpha_ratio"]  = beta / alpha
    return features


# ---------------------------------------------------------------------------
# Master Entry Point
# ---------------------------------------------------------------------------

def extract_advanced_features(
    data: np.ndarray,
    sfreq: float,
    ch_names: List[str],
    psd_features: Dict[str, float] = None,
    compute_entropy: bool = True,
    compute_wavelet: bool = True,
    compute_hfd: bool = True,
    compute_coherence: bool = True,
    compute_relbp: bool = True,
) -> Dict[str, float]:
    """
    Master function. Returns all advanced features for a single EEG window.

    Args:
        data: EEG data (n_channels, n_samples).
        sfreq: Sampling frequency in Hz.
        ch_names: List of channel names.
        psd_features: Already-computed PSD features dict (for relative power).
        compute_*: Toggle individual feature families.

    Returns:
        Flat dict of all advanced features for this window.
    """
    all_features: Dict[str, float] = {}

    for i, ch in enumerate(ch_names):
        sig = data[i]
        ch_key = ch.replace("-", "_").upper()

        if compute_wavelet:
            for k, v in wavelet_energy(sig).items():
                all_features[f"{ch_key}_{k}"] = v

        if compute_entropy:
            all_features[f"{ch_key}_sample_entropy"] = fast_sample_entropy(sig)

        if compute_hfd:
            all_features[f"{ch_key}_higuchi_fd"] = higuchi_fd(sig)

    if compute_coherence:
        all_features.update(cross_channel_coherence(data, ch_names, sfreq))

    if compute_relbp and psd_features:
        all_features.update(relative_band_power(psd_features, ch_names))

    # Sanitize NaN/None -> 0.0 for ML compatibility
    for k in all_features:
        v = all_features[k]
        if v is None or (isinstance(v, float) and np.isnan(v)):
            all_features[k] = 0.0

    return all_features
