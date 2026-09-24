"""
NeuroPulse Inference Engine (v2 — Master Model)
=================================================
Loads the GPU-trained ManualStackingEnsemble (master_model.pkl).
Provides:
  - predict_row(feature_array)   --> risk_score, label, SHAP, channel_risk
  - predict_single(features_dict) --> same, from a dict of feature_name -> value

The master model is a ManualStackingEnsemble containing:
  - XGBoost (CUDA-trained)
  - LightGBM
  - Random Forest
  - Logistic Regression meta-learner
"""

import os
import json
import pickle
import warnings
from typing import Dict, List, Any

import numpy as np

warnings.filterwarnings("ignore")

# Try master first, fall back to old ensemble
_CANDIDATES = [
    ("./models/trained/master_model.pkl",   "./models/scalers/master_scaler.pkl",   "./models/metadata/master_feature_names.json"),
    ("./models/trained/ensemble_model.pkl", "./models/scalers/ensemble_scaler.pkl", "./models/metadata/ensemble_feature_names.json"),
]

RISK_THRESHOLD = 0.50

BANDS = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]

CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ",  "CZ-PZ",
]


def _safe_array(arr: np.ndarray) -> np.ndarray:
    """Clip inf/nan that could come from scaler on zero-variance columns."""
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(arr, -1e15, 1e15).astype(np.float32)


class InferenceEngine:
    """Singleton inference engine for real-time prediction."""

    def __init__(self):
        self._model  = None
        self._scaler = None
        self._names: List[str] = []
        self._is_manual_stack = False
        self._load()

    def _load(self) -> None:
        for model_path, scaler_path, names_path in _CANDIDATES:
            if not os.path.exists(model_path):
                continue
            try:
                # Need to import ManualStackingEnsemble so pickle can find it
                import sys, importlib
                sys.path.insert(0, ".")
                try:
                    from ml.training.train_ensemble import ManualStackingEnsemble  # noqa
                    import __main__
                    __main__.ManualStackingEnsemble = ManualStackingEnsemble
                    self._is_manual_stack = True
                except Exception:
                    self._is_manual_stack = False

                with open(model_path, "rb") as f:
                    self._model = pickle.load(f)
                with open(scaler_path, "rb") as f:
                    self._scaler = pickle.load(f)
                with open(names_path) as f:
                    self._names = json.load(f)

                model_type = type(self._model).__name__
                print(f"[InferenceEngine] Loaded: {model_path}")
                print(f"[InferenceEngine] Model type: {model_type} | Features: {len(self._names)}")
                return
            except Exception as e:
                print(f"[InferenceEngine] Failed to load {model_path}: {e}")
                self._model = None

        print("[InferenceEngine] WARNING: No model loaded.")

    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def feature_names(self) -> List[str]:
        return self._names

    def _build_raw_row(self, features: Dict[str, float]) -> np.ndarray:
        row = np.array([features.get(n, 0.0) for n in self._names], dtype=np.float32)
        return row.reshape(1, -1)

    def _scale(self, raw_row: np.ndarray) -> np.ndarray:
        scaled = self._scaler.transform(raw_row)
        return _safe_array(scaled)

    def _predict_proba(self, scaled: np.ndarray) -> float:
        """Get class-1 probability from whatever model type we have."""
        prob_arr = self._model.predict_proba(scaled)
        return float(prob_arr[0, 1])

    def _compute_channel_risk(self, raw_row: np.ndarray) -> Dict[str, float]:
        channel_risk = {}
        for ch in CHANNELS:
            vals = []
            for band in ["Theta", "Beta", "Gamma"]:
                key = f"{ch}_{band}_relpow"
                if key in self._names:
                    idx = self._names.index(key)
                    vals.append(float(raw_row[0, idx]))
            channel_risk[ch] = float(np.mean(vals)) if vals else 0.0
        mn, mx = min(channel_risk.values()), max(channel_risk.values())
        if mx > mn:
            channel_risk = {k: (v - mn) / (mx - mn) for k, v in channel_risk.items()}
        return channel_risk

    def _compute_band_power(self, raw_row: np.ndarray) -> Dict[str, float]:
        band_power = {}
        for band in BANDS:
            vals = [
                float(raw_row[0, self._names.index(f"{ch}_{band}_relpow")])
                for ch in CHANNELS if f"{ch}_{band}_relpow" in self._names
            ]
            band_power[band] = float(np.mean(vals)) if vals else 0.0
        return band_power

    def _get_shap_features(self, scaled: np.ndarray, top_n: int = 8) -> List[Dict]:
        """Try SHAP on the XGBoost sub-model if available."""
        try:
            import shap
            xgb_model = None
            if self._is_manual_stack and hasattr(self._model, "trained_bases"):
                for name, m in self._model.trained_bases:
                    if name == "xgboost":
                        xgb_model = m
                        break
            elif hasattr(self._model, "named_estimators_"):
                xgb_model = self._model.named_estimators_.get("xgboost")

            if xgb_model is None:
                return []

            explainer = shap.TreeExplainer(xgb_model)
            sv = explainer.shap_values(scaled)
            if isinstance(sv, list):
                sv = sv[1][0]
            else:
                sv = sv[0]
            top_idx = np.argsort(np.abs(sv))[::-1][:top_n]
            return [{"feature": self._names[i], "shap_value": float(sv[i])} for i in top_idx]
        except Exception:
            return []

    def predict_row(self, feature_array: np.ndarray) -> Dict[str, Any]:
        if not self.is_ready():
            return {"risk_score": 0.0, "label_pred": 0, "top_shap": [], "channel_risk": {}, "band_power": {}}

        raw_row = _safe_array(feature_array.reshape(1, -1))
        channel_risk = self._compute_channel_risk(raw_row)
        band_power   = self._compute_band_power(raw_row)
        scaled       = self._scale(raw_row)
        prob         = self._predict_proba(scaled)
        label_pred   = int(prob >= RISK_THRESHOLD)
        top_shap     = self._get_shap_features(scaled)

        return {
            "risk_score":   prob,
            "label_pred":   label_pred,
            "top_shap":     top_shap,
            "channel_risk": channel_risk,
            "band_power":   band_power,
        }

    def predict_single(self, features: Dict[str, float]) -> Dict[str, Any]:
        if not self.is_ready():
            return {"risk_score": 0.0, "label_pred": 0, "top_shap": [], "channel_risk": {}, "band_power": {}}
        raw_row = self._build_raw_row(features)
        return self.predict_row(raw_row[0])
