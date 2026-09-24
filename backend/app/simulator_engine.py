"""
NeuroPulse Simulator Engine
============================
Replays the pre-extracted feature CSV (chb01_labeled_features.csv) as if it
were a live EEG stream arriving from a hospital bedside monitor.

The simulator:
  1. Reads the labeled feature CSV row by row.
  2. Passes each row through the InferenceEngine for real-time prediction.
  3. Yields a complete TelemetryFrame JSON dict at a configurable interval.
  4. Loops infinitely (wraps back to start when EOF reached).

This means the frontend gets a WebSocket frame every STREAM_INTERVAL_SEC
seconds, exactly as it would in a live clinical deployment.
"""

import time
import asyncio
from typing import AsyncIterator, Dict, Any

import numpy as np
import pandas as pd

from backend.app.inference import InferenceEngine

STREAM_INTERVAL_SEC = 1.0    # 1 frame per second (matches 10-second window stride)
ALERT_THRESHOLD     = 0.60   # risk_score above this triggers an alert flag

META_COLS = {
    "patient_id", "edf_file", "file_source", "window_idx",
    "epoch_id", "window_start", "window_end", "target_label"
}


class SimulatorEngine:
    """
    Async generator that replays a patient's feature CSV as a live stream.

    Usage:
        simulator = SimulatorEngine(csv_path, inference, patient_id)
        async for frame in simulator.stream():
            await websocket.send_json(frame)
    """

    def __init__(
        self,
        csv_path: str,
        inference: InferenceEngine,
        patient_id: str,
        loop: bool = True,
        interval: float = STREAM_INTERVAL_SEC,
    ):
        self.csv_path   = csv_path
        self.inference  = inference
        self.patient_id = patient_id
        self.loop       = loop
        self.interval   = interval

        print(f"[Simulator] Loading {csv_path} ...")
        self._df = pd.read_csv(csv_path)
        numeric_cols = self._df.select_dtypes(include=[np.number]).columns.tolist()
        self._feature_cols = [c for c in numeric_cols if c not in META_COLS]
        self._n_windows = len(self._df)
        print(f"[Simulator] Ready | Patient={patient_id} | Windows={self._n_windows}")

    async def stream(self) -> AsyncIterator[Dict[str, Any]]:
        """
        Async generator: yields one telemetry frame per `interval` seconds.
        Loops back to start when all windows are exhausted (if loop=True).
        """
        idx = 0
        while True:
            row = self._df.iloc[idx]

            # Build feature array in training order
            feature_array = np.array(
                [row.get(c, 0.0) for c in self.inference.feature_names],
                dtype=np.float32
            )

            # Run inference
            result = self.inference.predict_row(feature_array)

            # Ground truth label (if available)
            label_true = int(row["target_label"]) if "target_label" in row else None

            # Build telemetry frame
            frame: Dict[str, Any] = {
                "timestamp":    time.time(),
                "patient_id":   self.patient_id,
                "window_idx":   idx,
                "window_start": float(row.get("window_start", idx * 10)),
                "window_end":   float(row.get("window_end",   (idx + 1) * 10)),
                "risk_score":   round(result["risk_score"], 4),
                "label_pred":   result["label_pred"],
                "label_true":   label_true,
                "alert":        result["risk_score"] >= ALERT_THRESHOLD,
                "band_power":   result.get("band_power", {}),
                "top_shap":     result.get("top_shap", []),
                "channel_risk": result.get("channel_risk", {}),
            }

            yield frame

            # Advance window index
            idx = (idx + 1) % self._n_windows if self.loop else idx + 1
            if not self.loop and idx >= self._n_windows:
                break

            # Wait before sending next frame
            await asyncio.sleep(self.interval)
