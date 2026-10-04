"""
NeuroPulse — Classical ML Inference Engine (V1)
=================================================
Loads the trained ManualStackingEnsemble (XGBoost + LightGBM + RandomForest +
Logistic Regression meta-learner) from master_model.pkl and provides
real-time risk predictions from pre-extracted CSV feature vectors.

Output schema per prediction:
    risk_score    : float [0.0–1.0] — preictal probability
    label_pred    : int   [0=Interictal, 1=Preictal]
    channel_risk  : Dict[channel_name -> float] — per-channel risk proxy
    band_power    : Dict[band_name -> float]    — average Welch band power
    top_features  : List[{name, value}]         — top contributing features
"""
from __future__ import annotations

import os
import sys
import json
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np

warnings.filterwarnings("ignore")

# sys.path must be set BEFORE importing ManualStackingEnsemble so pickle can find it
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ManualStackingEnsemble stub — must be in scope so pickle can deserialize master_model.pkl.
# We define it inline here to avoid importing train_ensemble.py which pulls in sklearn.svm
# (triggers a DLL block on Windows application-control environments).
import numpy as _np_pkl

class ManualStackingEnsemble:
    """Minimal pickle-compatible stub matching the saved pkl interface."""
    def __init__(self, trained_bases, meta, scaler, feature_cols):
        self.trained_bases = trained_bases
        self.meta          = meta
        self.scaler        = scaler
        self.feature_cols  = feature_cols

    def predict_proba(self, X_s):
        base_probs = _np_pkl.column_stack(
            [m.predict_proba(X_s)[:, 1] for _, m in self.trained_bases]
        ).astype(_np_pkl.float32)
        prob = self.meta.predict_proba(base_probs)[:, 1]
        out = _np_pkl.zeros((len(prob), 2), dtype=_np_pkl.float32)
        out[:, 1] = prob
        out[:, 0] = 1 - prob
        return out

    def predict(self, X_s):
        prob_arr = self.predict_proba(X_s)
        return (prob_arr[:, 1] >= 0.5).astype(int)

    @property
    def named_estimators_(self):
        return {n: m for n, m in self.trained_bases}


# Register the stub under multiple module paths so pickle deserialization works
# regardless of how the pkl was originally saved.
import types as _types
for _mod_name in ("ml.training.train_ensemble", "__main__"):
    if _mod_name not in sys.modules:
        _m = _types.ModuleType(_mod_name)
        _m.ManualStackingEnsemble = ManualStackingEnsemble
        sys.modules[_mod_name] = _m
    else:
        # Module already exists — just inject the class into it
        sys.modules[_mod_name].ManualStackingEnsemble = ManualStackingEnsemble

# ---------------------------------------------------------------------------
# Paths — V1 model artifacts
# NOTE: Using xgboost_model.pkl (single XGBClassifier, 180 features) instead
# of master_model.pkl (ManualStackingEnsemble) because master_model.pkl embeds
# RandomForestClassifier which requires sklearn Cython DLLs that are blocked by
# Windows Application Control on this machine.
# ---------------------------------------------------------------------------
_MODEL_PATH   = os.path.join(_PROJECT_ROOT, "V1_Classical_ML", "models", "trained",  "xgboost_model.pkl")
_SCALER_PATH  = os.path.join(_PROJECT_ROOT, "V1_Classical_ML", "models", "scalers",  "scaler.pkl")
_NAMES_PATH   = os.path.join(_PROJECT_ROOT, "V1_Classical_ML", "models", "metadata", "feature_names.json")

RISK_THRESHOLD = 0.50

# Welch band power feature name fragments for band extraction
_BAND_KEYS = {
    "Delta": "delta",
    "Theta": "theta",
    "Alpha": "alpha",
    "Beta":  "beta",
    "Gamma": "gamma",
}

# Bipolar channel prefixes for per-channel risk proxy
_CHANNEL_PREFIXES = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ",  "CZ-PZ",
]


class ClassicalMLInferenceEngine:
    """
    Singleton inference engine for the V1 ManualStackingEnsemble.

    Handles:
    - Model + scaler + feature-name loading from disk
    - Safe feature alignment (zero-fills any missing columns)
    - Inf/NaN defense (mirrors the training pipeline)
    - Band power and channel risk extraction from feature vector
    - Top-feature extraction for dashboard display
    """

    def __init__(self) -> None:
        self._model        = None
        self._scaler       = None
        self._feature_cols: List[str] = []
        self._ready        = False
        self._load()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load model, scaler, and feature names from disk."""
        missing = [p for p in [_MODEL_PATH, _SCALER_PATH, _NAMES_PATH]
                   if not os.path.exists(p)]
        if missing:
            print(f"[InferenceEngine] WARNING: Missing artifacts: {missing}")
            print("[InferenceEngine] Run 'python -m ml.training.train_ensemble --split-validation' first.")
            self._ready = False
            return

        try:
            with open(_MODEL_PATH, "rb") as f:
                self._model = pickle.load(f)
            with open(_SCALER_PATH, "rb") as f:
                self._scaler = pickle.load(f)
            with open(_NAMES_PATH) as f:
                self._feature_cols = json.load(f)

            self._ready = True
            print(f"[InferenceEngine] V1 Classical ML model loaded.")
            print(f"[InferenceEngine] Features: {len(self._feature_cols)}")
        except Exception as exc:
            print(f"[InferenceEngine] Load error: {exc}")
            self._ready = False

    def reload(self) -> None:
        """Hot-reload model from disk (call after retraining)."""
        print("[InferenceEngine] Reloading model from disk...")
        self._load()

    def is_ready(self) -> bool:
        return self._ready

    # ------------------------------------------------------------------
    # Safe Scaling
    # ------------------------------------------------------------------

    def _safe_scale(self, X: np.ndarray) -> np.ndarray:
        """Apply StandardScaler and strip inf/nan (mirrors training pipeline)."""
        X_s = self._scaler.transform(X)
        X_s = np.nan_to_num(X_s, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(X_s, -1e15, 1e15).astype(np.float32)

    # ------------------------------------------------------------------
    # Feature Alignment
    # ------------------------------------------------------------------

    def _align_features(self, feature_dict: Dict[str, float]) -> np.ndarray:
        """
        Align an arbitrary feature dictionary to the model's expected feature
        column order. Missing features are filled with 0.0.
        """
        row = np.zeros(len(self._feature_cols), dtype=np.float32)
        for i, col in enumerate(self._feature_cols):
            row[i] = feature_dict.get(col, 0.0)
        # Inf/NaN defense
        row = np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)
        return row.reshape(1, -1)

    # ------------------------------------------------------------------
    # Band Power Extraction
    # ------------------------------------------------------------------

    def _extract_band_power(
        self, feature_dict: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Average the Welch PSD power across all channels for each clinical band.
        Feature names follow the pattern: {CHANNEL}_{BandName}
        e.g. FP1-F7_Delta, FP1-F7_Theta, ...
        """
        band_power: Dict[str, float] = {}
        for band_name, key in _BAND_KEYS.items():
            # key is lowercase (e.g. "delta"), feature names use title case (e.g. "_Delta")
            vals = [
                v for k, v in feature_dict.items()
                if key in k.lower() and np.isfinite(v)
            ]
            band_power[band_name] = float(np.mean(vals)) if vals else 0.0
        return band_power

    # ------------------------------------------------------------------
    # Channel Risk Proxy
    # ------------------------------------------------------------------

    def _extract_channel_risk(
        self, feature_dict: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Derive per-channel risk as the average absolute feature value across
        all features belonging to that channel (RMS + Hjorth mobility proxy).
        """
        channel_risk: Dict[str, float] = {}
        for ch in _CHANNEL_PREFIXES:
            ch_vals = [
                abs(feature_dict[k]) for k in feature_dict
                if k.startswith(ch)
            ]
            channel_risk[ch] = float(np.mean(ch_vals)) if ch_vals else 0.0

        # Normalize 0→1
        max_v = max(channel_risk.values()) or 1.0
        return {ch: v / max_v for ch, v in channel_risk.items()}

    # ------------------------------------------------------------------
    # Top Feature Extraction
    # ------------------------------------------------------------------

    def _top_features(
        self, feature_dict: Dict[str, float], n: int = 8
    ) -> List[Dict[str, Any]]:
        """Return the N features with the highest absolute values."""
        items = [(k, abs(v)) for k, v in feature_dict.items() if np.isfinite(v)]
        items.sort(key=lambda x: x[1], reverse=True)
        return [{"name": k, "value": feature_dict[k]} for k, _ in items[:n]]

    # ------------------------------------------------------------------
    # Main Prediction
    # ------------------------------------------------------------------

    def predict(self, feature_dict: Dict[str, float]) -> Dict[str, Any]:
        """
        Run a single-window prediction from a flat feature dictionary.

        Args:
            feature_dict: {feature_name: float_value} for a 10-second window.

        Returns:
            dict with keys: risk_score, label_pred, channel_risk,
                             band_power, top_features
        """
        if not self._ready:
            return {
                "risk_score": 0.0,
                "label_pred": 0,
                "channel_risk": {},
                "band_power": {},
                "top_features": [],
                "error": "Model not loaded",
            }

        try:
            X = self._align_features(feature_dict)
            X_s = self._safe_scale(X)

            prob_arr = self._model.predict_proba(X_s)  # (1, 2)
            risk_score = float(prob_arr[0, 1])
            label_pred = int(risk_score >= RISK_THRESHOLD)

            channel_risk = self._extract_channel_risk(feature_dict)
            band_power   = self._extract_band_power(feature_dict)
            top_feats    = self._top_features(feature_dict)

            return {
                "risk_score":   risk_score,
                "label_pred":   label_pred,
                "channel_risk": channel_risk,
                "band_power":   band_power,
                "top_features": top_feats,
            }

        except Exception as exc:
            print(f"[InferenceEngine] Predict error: {exc}")
            return {
                "risk_score": 0.0,
                "label_pred": 0,
                "channel_risk": {},
                "band_power": {},
                "top_features": [],
                "error": str(exc),
            }
