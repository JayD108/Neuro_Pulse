"""
NeuroPulse Pydantic Schemas
============================
Request / Response models for all API endpoints.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


# ---------------------------------------------------------------------------
# Patient
# ---------------------------------------------------------------------------

class PatientInfo(BaseModel):
    patient_id: str
    n_windows: int
    n_preictal: int
    n_interictal: int
    n_features: int
    model_roc_auc: float
    model_sensitivity: float
    model_specificity: float


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    features: Dict[str, float] = Field(
        ...,
        description="Feature name -> value dict. Must match training feature names."
    )


class SHAPFeature(BaseModel):
    feature: str
    shap_value: float


class ChannelRisk(BaseModel):
    channel: str
    risk: float


class PredictResponse(BaseModel):
    risk_score: float = Field(..., description="Seizure probability (0.0 - 1.0)")
    label_pred: int   = Field(..., description="0=Interictal, 1=Preictal")
    top_shap: List[SHAPFeature] = Field(default_factory=list)
    channel_risk: Dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# WebSocket Telemetry Frame
# ---------------------------------------------------------------------------

class BandPower(BaseModel):
    Delta: float = 0.0
    Theta: float = 0.0
    Alpha: float = 0.0
    Beta:  float = 0.0
    Gamma: float = 0.0


class TelemetryFrame(BaseModel):
    """Shape of each JSON frame sent over the WebSocket every second."""
    timestamp: float
    patient_id: str
    window_idx: int
    risk_score: float
    label_pred: int
    label_true: Optional[int] = None        # ground truth if available
    band_power: Dict[str, float] = Field(default_factory=dict)
    top_shap: List[Dict] = Field(default_factory=list)
    channel_risk: Dict[str, float] = Field(default_factory=dict)
    alert: bool = False                     # True when risk_score > threshold
