"""
NeuroPulse FastAPI Backend
==========================
Entry point for the NeuroPulse real-time seizure prediction API.

Endpoints:
  GET  /health                 -- Liveness probe
  GET  /patients               -- List patients with available data
  GET  /patient/{id}/info      -- Patient metadata and model stats
  POST /predict                -- One-shot batch prediction on a feature row
  WS   /ws/{patient_id}        -- Real-time telemetry WebSocket stream
"""

import os
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.app.schemas import PatientInfo, PredictRequest, PredictResponse, HealthResponse
from backend.app.inference import InferenceEngine
from backend.app.simulator_engine import SimulatorEngine
from backend.app.websocket_manager import WebSocketManager

# ---------------------------------------------------------------------------
# App Initialization
# ---------------------------------------------------------------------------
app = FastAPI(
    title="NeuroPulse API",
    description="Real-time EEG seizure prediction backend powered by classical ML ensemble.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Tighten to frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Singletons (loaded once at startup)
# ---------------------------------------------------------------------------
inference = InferenceEngine()
ws_manager = WebSocketManager()

PROCESSED_DIR = "./ml/data/processed"
META_PATH     = "./models/metadata/master_feature_names.json"
METRICS_PATH  = "./reports/model_results/ensemble_metrics.json"


# ---------------------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["system"])
def health():
    """Liveness probe — confirms backend and model are loaded."""
    return HealthResponse(
        status="ok",
        model_loaded=inference.is_ready(),
        version="2.0.0",
    )


@app.get("/dataset/stats", tags=["data"])
def dataset_stats():
    """Aggregate stats across all processed patients for the Dataset page."""
    import glob
    import pandas as pd
    csvs = glob.glob(os.path.join(PROCESSED_DIR, "*_labeled_features.csv"))
    total_windows = total_preictal = total_interictal = 0
    for c in csvs:
        df = pd.read_csv(c, usecols=["target_label"])
        total_windows   += len(df)
        total_preictal  += int((df["target_label"] == 1).sum())
        total_interictal+= int((df["target_label"] == 0).sum())
    return {
        "total_patients":   len(csvs),
        "total_windows":    total_windows,
        "total_preictal":   total_preictal,
        "total_interictal": total_interictal,
        "features_per_window": len(inference.feature_names),
        "model_loaded": inference.is_ready(),
        "model_type": type(inference._model).__name__ if inference.is_ready() else "None",
    }


@app.get("/patients", tags=["data"])
def list_patients():
    """Return all patient IDs that have processed feature CSVs available."""
    import glob
    csvs = glob.glob(os.path.join(PROCESSED_DIR, "*_labeled_features.csv"))
    patients = sorted([
        os.path.basename(c).replace("_labeled_features.csv", "")
        for c in csvs
    ])
    return {"patients": patients, "count": len(patients)}


@app.get("/patient/{patient_id}/info", response_model=PatientInfo, tags=["data"])
def patient_info(patient_id: str):
    """Return patient metadata, feature count, and model performance metrics."""
    csv_path = os.path.join(PROCESSED_DIR, f"{patient_id}_labeled_features.csv")
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail=f"No data found for patient: {patient_id}")

    import pandas as pd
    df = pd.read_csv(csv_path, nrows=1)  # just read header for col count
    full_df = pd.read_csv(csv_path)

    n_windows     = len(full_df)
    n_preictal    = int((full_df["target_label"] == 1).sum())
    n_interictal  = int((full_df["target_label"] == 0).sum())
    n_features    = len([c for c in df.columns if c not in
                         {"patient_id", "edf_file", "file_source", "window_idx",
                          "epoch_id", "window_start", "window_end", "target_label"}])

    metrics = {}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            metrics = json.load(f)

    return PatientInfo(
        patient_id=patient_id,
        n_windows=n_windows,
        n_preictal=n_preictal,
        n_interictal=n_interictal,
        n_features=n_features,
        model_roc_auc=metrics.get("roc_auc", 0.0),
        model_sensitivity=metrics.get("sensitivity", 0.0),
        model_specificity=metrics.get("specificity", 0.0),
    )


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(request: PredictRequest):
    """
    One-shot prediction endpoint.
    Accepts a dict of feature_name -> value and returns
    seizure probability, predicted label, and top SHAP features.
    """
    if not inference.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    result = inference.predict_single(request.features)
    return PredictResponse(**result)


# ---------------------------------------------------------------------------
# WebSocket Endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws/{patient_id}")
async def websocket_stream(websocket: WebSocket, patient_id: str):
    """
    Real-time seizure risk telemetry stream.
    Connects to the simulator engine and broadcasts JSON frames every second.
    """
    csv_path = os.path.join(PROCESSED_DIR, f"{patient_id}_labeled_features.csv")
    if not os.path.exists(csv_path):
        await websocket.close(code=4004, reason=f"No data for patient: {patient_id}")
        return

    await ws_manager.connect(websocket)
    simulator = SimulatorEngine(
        csv_path=csv_path,
        inference=inference,
        patient_id=patient_id,
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
