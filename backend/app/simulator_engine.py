"""
NeuroPulse — CSV Simulator Engine (V1)
=======================================
Streams a patient's pre-processed labeled feature CSV row-by-row through
the ClassicalMLInferenceEngine, producing real-time telemetry frames for
the WebSocket dashboard.

The simulator:
  1. Loads the patient's labeled CSV from ml/data/processed/.
  2. Filters to Interictal (0) and Preictal (1) rows only (excludes Ictal/Postictal).
  3. Streams each row through ClassicalMLInferenceEngine.predict().
  4. Yields a complete telemetry dict every STREAM_INTERVAL_SEC seconds.
  5. Loops back to start when EOF is reached.
"""

from __future__ import annotations

import os
import sys
import time
import asyncio
from pathlib import Path
from typing import AsyncIterator, Dict, Any, Optional

import numpy as np
import pandas as pd

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backend.app.inference import ClassicalMLInferenceEngine

PROCESSED_DIR       = os.path.join(_PROJECT_ROOT, "V1_Classical_ML", "ml", "data", "processed")
STREAM_INTERVAL_SEC = 0.5   # seconds between frames
ALERT_THRESHOLD     = 0.50  # risk_score threshold for alert flag

NON_FEATURE_COLS = {
    "epoch_id", "window_start", "window_end",
    "file_source", "target_label", "patient_id",
    "window_idx", "edf_file",
}


def _load_patient_csv(patient_id: str) -> Optional[pd.DataFrame]:
    """Load the pre-processed labeled CSV for a patient."""
    path = os.path.join(PROCESSED_DIR, f"{patient_id}_labeled_features.csv")
    if not os.path.exists(path):
        print(f"[Simulator] CSV not found: {path}")
        return None
    df = pd.read_csv(path)
    # Keep only interictal (0) and preictal (1) rows
    df = df[df["target_label"].isin([0, 1])].reset_index(drop=True)
    return df


class SimulatorEngine:
    """
    Async generator that streams V1 CSV rows as telemetry frames.

    Args:
        patient_id : CHB-MIT patient ID (e.g. 'chb01')
        inference  : ClassicalMLInferenceEngine instance (shared singleton)
    """

    def __init__(
        self,
        patient_id: str,
        inference: ClassicalMLInferenceEngine,
    ) -> None:
        self.patient_id = patient_id
        self.inference  = inference
        self._df: Optional[pd.DataFrame] = None

    def _ensure_loaded(self) -> bool:
        if self._df is None:
            self._df = _load_patient_csv(self.patient_id)
        return self._df is not None and len(self._df) > 0

    async def stream(self) -> AsyncIterator[Dict[str, Any]]:
        """
        Async generator yielding telemetry dicts every STREAM_INTERVAL_SEC.
        Loops indefinitely when EOF is reached.
        """
        if not self._ensure_loaded():
            # Send a single error frame and stop
            yield {
                "timestamp":    time.time(),
                "patient_id":   self.patient_id,
                "window_idx":   -1,
                "window_start": 0.0,
                "window_end":   0.0,
                "risk_score":   0.0,
                "soft_score":   0.0,
                "label_pred":   0,
                "label_true":   None,
                "alert":        False,
                "top_features": [],
                "channel_risk": {},
                "band_power":   {},
                "error":        f"No CSV data found for patient {self.patient_id}",
            }
            return

        df = self._df
        n  = len(df)

        # Identify feature columns (exclude metadata)
        feature_cols = [
            c for c in df.columns
            if c not in NON_FEATURE_COLS
            and df[c].dtype in [np.float32, np.float64, np.int32, np.int64, float, int]
        ]

        idx = 0
        while True:
            row = df.iloc[idx]

            # Build feature dictionary
            feature_dict = {
                col: float(row[col])
                for col in feature_cols
                if col in df.columns
            }

            # Run inference
            result = self.inference.predict(feature_dict)

            risk_score = result.get("risk_score", 0.0)
            label_pred = result.get("label_pred", 0)
            label_true = int(row["target_label"]) if "target_label" in row.index else None

            frame: Dict[str, Any] = {
                "timestamp":    time.time(),
                "patient_id":   self.patient_id,
                "window_idx":   idx,
                "window_start": float(row["window_start"]) if "window_start" in row.index else float(idx * 10),
                "window_end":   float(row["window_end"])   if "window_end"   in row.index else float(idx * 10 + 10),
                "risk_score":   risk_score,
                "soft_score":   risk_score,   # V1: no separate fusion layer; use raw score
                "label_pred":   label_pred,
                "label_true":   label_true,
                "alert":        risk_score >= ALERT_THRESHOLD,
                "top_features": result.get("top_features", []),
                "channel_risk": result.get("channel_risk", {}),
                "band_power":   result.get("band_power", {}),
            }

            yield frame

            idx = (idx + 1) % n    # Loop back to start at EOF
            await asyncio.sleep(STREAM_INTERVAL_SEC)
