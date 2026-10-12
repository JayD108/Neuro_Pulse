"""
NeuroPulse — Classical ML Inference Engine (V1)
=================================================
Loads the trained V1 XGBoost model (xgboost_model.pkl, 180 features) and
provides real-time risk predictions from pre-extracted CSV feature vectors.

Output schema per prediction:
    risk_score    : float [0.0–1.0] — preictal probability
    label_pred    : int   [0=Interictal, 1=Preictal]
    channel_risk  : Dict[channel_name -> float] — per-channel risk proxy
    band_power    : Dict[band_name -> float]    — average Welch band power
    top_features  : List[{name, value, score}]  — top contributing features
"""
from __future__ import annotations

import os
import re
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
_MODELS_DIR   = os.path.join(_PROJECT_ROOT, "V1_Classical_ML", "models")

# Preferred: XGBoost trained on all 24 patients, saved in XGBoost's own JSON
# format (no pickle, no scikit-learn needed to load it). Raw features, no scaler;
# missing channels stay NaN, which XGBoost handles natively.
_APALA_MODEL_PATH = os.path.join(_MODELS_DIR, "trained", "apala_xgb_v1.json")
_APALA_NAMES_PATH = os.path.join(_MODELS_DIR, "trained", "apala_xgb_v1_features.json")
_APALA_LOPO_PATH  = os.path.join(_MODELS_DIR, "trained", "apala_xgb_v1_lopo.json")

# Fallback: the original V1 XGBoost (180 features, pickled, with a scaler).
_MODEL_PATH   = os.path.join(_PROJECT_ROOT, "V1_Classical_ML", "models", "trained",  "xgboost_model.pkl")
_SCALER_PATH  = os.path.join(_PROJECT_ROOT, "V1_Classical_ML", "models", "scalers",  "scaler.pkl")
_NAMES_PATH   = os.path.join(_PROJECT_ROOT, "V1_Classical_ML", "models", "metadata", "feature_names.json")

RISK_THRESHOLD = 0.50

# Clinical bands shown on the dashboard
_BAND_KEYS = ("Delta", "Theta", "Alpha", "Beta", "Gamma")

# "{channel}_{Band}_relpow", e.g. "FP1-F7_Delta_relpow" (hyphenated channel names only)
_BAND_RELPOW_RE = re.compile(r"^[A-Z0-9]+-[A-Z0-9-]+_(Delta|Theta|Alpha|Beta|Gamma)_relpow$")

# Older names used when the XGBoost model was trained: "FP1-F7_Delta" is now
# "FP1-F7_Delta_power" in the processed CSVs.
_FEATURE_ALIASES = ("_power",)

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
    Singleton inference engine for the V1 XGBoost model.

    Handles:
    - Model + scaler + feature-name loading from disk
    - Feature alignment, including older feature names (see _FEATURE_ALIASES)
    - Inf/NaN defense (mirrors the training pipeline)
    - Band power, per-channel risk and top-feature extraction for the dashboard
    """

    def __init__(self) -> None:
        self._model        = None
        self._scaler       = None
        self._feature_cols: List[str] = []
        self._importances: Optional[np.ndarray] = None
        self._fill: Optional[np.ndarray] = None
        self._kind         = "none"      # "apala" | "legacy" | "none"
        self.model_name    = "No model loaded"
        self.lopo_auc: Dict[str, float] = {}
        self._resolved: Dict[tuple, List[Optional[str]]] = {}
        self._ready        = False
        self._load()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load the APALA model if present, otherwise the original V1 model."""
        self._ready = False
        self._kind = "none"
        self.model_name = "No model loaded"
        self.lopo_auc = {}
        self._resolved = {}
        if os.path.exists(_APALA_MODEL_PATH) and os.path.exists(_APALA_NAMES_PATH):
            try:
                self._load_apala()
                return
            except Exception as exc:
                print(f"[InferenceEngine] Could not load {_APALA_MODEL_PATH}: {exc}")
                print("[InferenceEngine] Falling back to the original V1 model.")
        self._load_legacy()

    def _load_apala(self) -> None:
        import xgboost as xgb

        model = xgb.XGBClassifier()
        model.load_model(_APALA_MODEL_PATH)
        model.set_params(device="cpu")   # single-window inference: CPU is faster
        with open(_APALA_NAMES_PATH) as f:
            self._feature_cols = json.load(f)

        self._model = model
        self._scaler = None
        self._importances = None
        self._fill = np.full(len(self._feature_cols), np.nan, dtype=np.float64)
        self.lopo_auc = _read_lopo_auc(_APALA_LOPO_PATH)
        self._kind = "apala"
        self.model_name = "XGBoost, 24 patients (APALA v1)"
        self._ready = True
        print(f"[InferenceEngine] APALA model loaded: {len(self._feature_cols)} features, "
              f"held-out AUC for {len(self.lopo_auc)} patients.")

    def _load_legacy(self) -> None:
        """Load the original V1 model, scaler, and feature names from disk."""
        missing = [p for p in [_MODEL_PATH, _SCALER_PATH, _NAMES_PATH]
                   if not os.path.exists(p)]
        if missing:
            print(f"[InferenceEngine] WARNING: Missing artifacts: {missing}")
            print("[InferenceEngine] Download them from https://huggingface.co/JayF14/Neuro_Pulse "
                  "into V1_Classical_ML/models/ (see README).")
            self._ready = False
            return

        try:
            with open(_MODEL_PATH, "rb") as f:
                self._model = pickle.load(f)
            with open(_SCALER_PATH, "rb") as f:
                self._scaler = pickle.load(f)
            with open(_NAMES_PATH) as f:
                self._feature_cols = json.load(f)

            imp = getattr(self._model, "feature_importances_", None)
            if imp is not None and len(imp) == len(self._feature_cols):
                imp = np.asarray(imp, dtype=np.float64)
                self._importances = imp / (imp.sum() or 1.0)
            else:
                self._importances = None

            # Missing or invalid inputs are filled with the training mean, so
            # they scale to z = 0 and do not push the prediction either way.
            mean = getattr(self._scaler, "mean_", None)
            if mean is not None and len(mean) == len(self._feature_cols):
                self._fill = np.asarray(mean, dtype=np.float64)
            else:
                self._fill = np.zeros(len(self._feature_cols), dtype=np.float64)

            self._resolved = {}
            self._kind = "legacy"
            self.model_name = "XGBoost (V1)"
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
    # Feature name resolution
    # ------------------------------------------------------------------

    def _resolve(self, available) -> List[Optional[str]]:
        """
        Map each model feature to a column that exists in the data.

        The XGBoost model was trained when Welch band powers were named
        "{channel}_{Band}" (e.g. "FP1-F7_Delta"). The current feature
        extractor names them "{channel}_{Band}_power". Without this mapping
        half of the model's inputs would silently be zero.
        """
        available = set(available)
        key = tuple(sorted(available))
        cached = self._resolved.get(key)
        if cached is not None:
            return cached

        resolved: List[Optional[str]] = []
        for col in self._feature_cols:
            if col in available:
                resolved.append(col)
                continue
            for suffix in _FEATURE_ALIASES:
                if col + suffix in available:
                    resolved.append(col + suffix)
                    break
            else:
                resolved.append(None)

        n_missing = sum(r is None for r in resolved)
        if n_missing:
            how = ("left as missing (the model handles them)" if self._kind == "apala"
                   else "filled with the training mean")
            print(f"[InferenceEngine] WARNING: {n_missing}/{len(resolved)} model features "
                  f"not found in the data; they will be {how}.")
        self._resolved = {key: resolved}
        return resolved

    def required_columns(self, available) -> List[str]:
        """Columns of a feature CSV that inference and the dashboard need."""
        needed = {c for c in self._resolve(available) if c} if self._ready else set()
        needed.update(c for c in available if _BAND_RELPOW_RE.match(c))
        return [c for c in available if c in needed]

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
        Align a feature dictionary to the model's expected column order.

        Features that are absent (e.g. channels missing from a patient's
        montage) or not finite are filled with the training mean. A raw 0
        would scale to an extreme z-score and bias the prediction.
        """
        resolved = self._resolve(feature_dict.keys())
        row = self._fill.copy()   # training mean (V1) or NaN = missing (APALA)
        for i, col in enumerate(resolved):
            if col is not None:
                v = feature_dict[col]
                if np.isfinite(v):
                    row[i] = v
        return row.reshape(1, -1)

    # ------------------------------------------------------------------
    # Band Power Extraction
    # ------------------------------------------------------------------

    def _extract_band_power(
        self, feature_dict: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Relative power of each clinical band, averaged over channels.
        Uses "{channel}_{Band}_relpow" columns, so the five values are
        fractions of total power (comparable across patients).
        """
        sums = {b: 0.0 for b in _BAND_KEYS}
        counts = {b: 0 for b in _BAND_KEYS}
        for k, v in feature_dict.items():
            m = _BAND_RELPOW_RE.match(k)
            if m and np.isfinite(v):
                sums[m.group(1)] += v
                counts[m.group(1)] += 1
        return {b: (sums[b] / counts[b] if counts[b] else 0.0) for b in _BAND_KEYS}

    # ------------------------------------------------------------------
    # Per-feature contributions for this window
    # ------------------------------------------------------------------

    def _contributions(self, X_model: np.ndarray) -> np.ndarray:
        """
        How much each feature contributes to this window's score.

        APALA model: XGBoost's exact per-feature contributions (SHAP values,
        log-odds; positive pushes towards pre-ictal).
        V1 model: feature importance x |z-score| (always >= 0).
        """
        if self._kind == "apala":
            import xgboost as xgb
            contribs = self._model.get_booster().predict(
                xgb.DMatrix(X_model, missing=np.nan), pred_contribs=True)
            return np.asarray(contribs[0, :-1], dtype=np.float64)   # last column = bias
        z = np.abs(X_model[0]).astype(np.float64)
        return z * self._importances if self._importances is not None else z

    def _extract_channel_risk(self, contrib: np.ndarray) -> Dict[str, float]:
        """Share of this window's evidence coming from each channel, 0-1 (max = 1)."""
        mag = np.abs(contrib)
        channel_risk: Dict[str, float] = {}
        for ch in _CHANNEL_PREFIXES:
            idx = [i for i, c in enumerate(self._feature_cols) if c.startswith(ch + "_") or c.startswith(ch + "-")]
            channel_risk[ch] = float(mag[idx].sum()) if idx else 0.0
        max_v = max(channel_risk.values()) or 1.0
        return {ch: v / max_v for ch, v in channel_risk.items()}

    def _top_features(
        self, feature_dict: Dict[str, float], contrib: np.ndarray, n: int = 8
    ) -> List[Dict[str, Any]]:
        """The N features with the largest contribution to this window's score."""
        resolved = self._resolve(feature_dict.keys())
        order = np.argsort(np.abs(contrib))[::-1]
        out: List[Dict[str, Any]] = []
        for i in order:
            if contrib[i] == 0 or resolved[i] is None:
                continue
            value = feature_dict[resolved[i]]
            out.append({
                "name":      self._feature_cols[i],
                "value":     float(value) if np.isfinite(value) else 0.0,
                "score":     float(abs(contrib[i])),
                "direction": "up" if contrib[i] > 0 or self._kind != "apala" else "down",
            })
            if len(out) == n:
                break
        return out

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
                "band_power": self._extract_band_power(feature_dict),
                "top_features": [],
                "error": "Model not loaded",
            }

        try:
            X = self._align_features(feature_dict)
            X_model = X.astype(np.float32) if self._kind == "apala" else self._safe_scale(X)

            prob_arr = self._model.predict_proba(X_model)  # (1, 2)
            risk_score = float(prob_arr[0, 1])
            label_pred = int(risk_score >= RISK_THRESHOLD)
            contrib = self._contributions(X_model)

            return {
                "risk_score":   risk_score,
                "label_pred":   label_pred,
                "channel_risk": self._extract_channel_risk(contrib),
                "band_power":   self._extract_band_power(feature_dict),
                "top_features": self._top_features(feature_dict, contrib),
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


def _read_lopo_auc(path: str) -> Dict[str, float]:
    """
    Per-patient held-out ROC-AUC from the leave-one-patient-out results file.
    Accepts a list of per-patient records or a {patient: metrics} mapping,
    at the top level or under a key such as "per_patient" / "patients".
    """
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception as exc:
        print(f"[InferenceEngine] Could not read {path}: {exc}")
        return {}

    if isinstance(data, dict):
        for key in ("per_patient", "patients", "folds", "results"):
            if key in data:
                data = data[key]
                break

    out: Dict[str, float] = {}
    if isinstance(data, dict):
        for pid, m in data.items():
            if isinstance(m, dict) and isinstance(m.get("roc_auc"), (int, float)):
                out[str(pid)] = float(m["roc_auc"])
    elif isinstance(data, list):
        for m in data:
            if not isinstance(m, dict):
                continue
            pid = m.get("patient_id") or m.get("patient") or m.get("test_patient")
            if pid and isinstance(m.get("roc_auc"), (int, float)):
                out[str(pid)] = float(m["roc_auc"])
    return out
