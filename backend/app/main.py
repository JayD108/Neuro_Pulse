"""
APALA — FastAPI Backend (V1 Classical ML)
==========================================
Real-time EEG seizure prediction powered by the V1 classical ML model (XGBoost).

Endpoints:
  GET  /health               -- Liveness probe + model status
  GET  /patients             -- List patients with available CSV data
  GET  /patient/{id}/info    -- Patient metadata (window counts from CSV)
  GET  /dataset/stats        -- Aggregate stats for all 24 patients
  GET  /model/reload         -- Hot-reload the pkl model (after retraining)
  WS   /ws/{patient_id}      -- Real-time telemetry WebSocket stream

WebSocket frame schema: see schemas.TelemetryFrame
"""

import os
import glob
import threading
from contextlib import asynccontextmanager
from typing import Dict, Tuple
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.app.schemas import PatientInfo, HealthResponse
from backend.app.inference import ClassicalMLInferenceEngine
from backend.app.simulator_engine import SimulatorEngine, PROCESSED_DIR, clear_csv_cache
from backend.app.websocket_manager import WebSocketManager

# ---------------------------------------------------------------------------
# App Initialization
# ---------------------------------------------------------------------------
@asynccontextmanager
async def _lifespan(_app):
    # Count each patient's labels in the background so /dataset/stats is instant.
    threading.Thread(target=_warm_label_cache, daemon=True).start()
    yield


app = FastAPI(
    title="APALA API",
    description="Real-time EEG seizure prediction (V1 classical ML, XGBoost).",
    version="1.0.0",
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Singletons (loaded once at startup)
# ---------------------------------------------------------------------------
inference  = ClassicalMLInferenceEngine()
ws_manager = WebSocketManager()


# ---------------------------------------------------------------------------
# Label counts per patient, cached by file modification time. Reading the
# label column of all 24 CSVs takes ~30 s, so it is done once in the
# background at startup and then served from memory.
# ---------------------------------------------------------------------------
_LABEL_CACHE: Dict[str, Tuple[float, Dict[str, int]]] = {}
_LABEL_LOCK = threading.Lock()


def _label_counts(csv_path: str) -> Dict[str, int]:
    mtime = os.path.getmtime(csv_path)
    with _LABEL_LOCK:
        hit = _LABEL_CACHE.get(csv_path)
    if hit and hit[0] == mtime:
        return hit[1]

    import pandas as pd
    labels = pd.read_csv(csv_path, usecols=["target_label"])["target_label"]
    counts = {
        "n_windows":    int(len(labels)),
        "n_preictal":   int((labels == 1).sum()),
        "n_interictal": int((labels == 0).sum()),
    }
    with _LABEL_LOCK:
        _LABEL_CACHE[csv_path] = (mtime, counts)
    return counts


def _patient_csvs():
    return sorted(glob.glob(os.path.join(PROCESSED_DIR, "*_labeled_features.csv")))


def _warm_label_cache() -> None:
    for path in _patient_csvs():
        try:
            _label_counts(path)
        except Exception as exc:
            print(f"[Stats] Could not read {path}: {exc}")
    print(f"[Stats] Label counts cached for {len(_LABEL_CACHE)} patients.")



# ---------------------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["system"])
def root():
    """Root endpoint — verifies the APALA API is online."""
    return {
        "status": "ok",
        "service": "APALA API",
        "version": "1.0.0",
        "model_loaded": inference.is_ready(),
        "model": inference.model_name,
    }


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health():
    """Liveness probe — confirms backend and model are loaded."""
    return HealthResponse(
        status="ok",
        model_loaded=inference.is_ready(),
        version="1.0.0",
    )


@app.get("/model/reload", tags=["system"])
def reload_model():
    """
    Hot-reload the pkl model from disk.
    Call after retraining to activate updated model without restarting the server.
    """
    inference.reload()
    clear_csv_cache()
    return {
        "status": "reloaded",
        "model_loaded": inference.is_ready(),
    }


@app.get("/patients", tags=["data"])
def list_patients():
    """
    Return all patient IDs that have a processed feature CSV available.
    """
    patients = [
        os.path.basename(c).replace("_labeled_features.csv", "")
        for c in _patient_csvs()
    ]
    return {"patients": patients, "count": len(patients)}


@app.get("/patient/{patient_id}/info", response_model=PatientInfo, tags=["data"])
def patient_info(patient_id: str):
    """Return patient metadata from the processed CSV."""
    csv_path = os.path.join(PROCESSED_DIR, f"{patient_id}_labeled_features.csv")
    if not os.path.exists(csv_path):
        raise HTTPException(
            status_code=404,
            detail=f"No processed CSV found for patient: {patient_id}. "
                   f"Download it from the dataset on Hugging Face (see README)."
        )

    counts = _label_counts(csv_path)

    return PatientInfo(
        patient_id=patient_id,
        **counts,
        model_loaded=inference.is_ready(),
        model_type=inference.model_name,
        lopo_auc=inference.lopo_auc.get(patient_id),
    )


@app.get("/dataset/stats", tags=["data"])
def dataset_stats():
    """Aggregate window stats for all patients (served from the label cache)."""
    csvs = _patient_csvs()
    if not csvs:
        return {"error": "No processed CSVs found.", "patients": []}

    rows = []
    for csv_path in csvs:
        pid = os.path.basename(csv_path).replace("_labeled_features.csv", "")
        try:
            rows.append({
                "patient_id": pid,
                **_label_counts(csv_path),
                "lopo_auc":   inference.lopo_auc.get(pid),
            })
        except Exception as e:
            rows.append({"patient_id": pid, "error": str(e)})

    total_windows   = sum(r.get("n_windows", 0)    for r in rows)
    total_preictal  = sum(r.get("n_preictal", 0)   for r in rows)
    total_interictal= sum(r.get("n_interictal", 0) for r in rows)

    return {
        "n_patients":      len(rows),
        "total_windows":   total_windows,
        "total_preictal":  total_preictal,
        "total_interictal":total_interictal,
        "imbalance_ratio": round(total_interictal / max(total_preictal, 1), 2),
        "patients":        rows,
    }


# ---------------------------------------------------------------------------
# WebSocket Endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws/{patient_id}")
async def websocket_stream(websocket: WebSocket, patient_id: str):
    """
    Real-time seizure risk telemetry stream.
    Streams model predictions from pre-processed CSV every 0.5 seconds.
    Each frame matches the TelemetryFrame schema.
    """
    csv_path = os.path.join(PROCESSED_DIR, f"{patient_id}_labeled_features.csv")
    if not os.path.exists(csv_path):
        await websocket.close(
            code=4004,
            reason=f"No processed CSV data for patient: {patient_id}"
        )
        return

    await ws_manager.connect(websocket)

    simulator = SimulatorEngine(
        patient_id=patient_id,
        inference=inference,
    )

    try:
        async for frame in simulator.stream():
            if ws_manager.is_connected(websocket):
                await websocket.send_json(frame)
            else:
                break
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as exc:
        print(f"[WS ERROR] {patient_id}: {exc}")
        ws_manager.disconnect(websocket)
