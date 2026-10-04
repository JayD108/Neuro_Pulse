"""
NeuroPulse Pydantic Schemas (V1)
==================================
Request / Response models for all API endpoints.
Uses top-feature importance list instead of KAN spline activations.
"""

from typing import Dict, List, Optional, Any
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
    patient_id:   str
    n_windows:    int
    n_preictal:   int
    n_interictal: int
    model_loaded: bool
    model_type:   str = "XGBoost + LightGBM + RF Ensemble"


# ---------------------------------------------------------------------------
# Top Feature (replaces KAN spline activations)
# ---------------------------------------------------------------------------

class TopFeature(BaseModel):
    """A single high-magnitude feature for dashboard display."""
    name:  str   = Field(..., description="Feature name (e.g. 'FP1-F7_gamma_relative')")
    value: float = Field(..., description="Raw feature value for this window")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

class PredictResponse(BaseModel):
    risk_score:   float               = Field(..., description="Preictal probability (0.0–1.0)")
    label_pred:   int                 = Field(..., description="0=Interictal, 1=Preictal")
    top_features: List[TopFeature]    = Field(default_factory=list)
    channel_risk: Dict[str, float]    = Field(default_factory=dict)
    band_power:   Dict[str, float]    = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# WebSocket Telemetry Frame (streamed to Next.js dashboard every 0.5s)
# ---------------------------------------------------------------------------

class TelemetryFrame(BaseModel):
    """Shape of each JSON frame sent over the WebSocket from the simulator."""
    timestamp:    float
    patient_id:   str
    window_idx:   int
    window_start: float
    window_end:   float
    risk_score:   float                         # Raw ensemble probability
    soft_score:   float = 0.0                   # Same as risk_score in V1 (no fusion)
    label_pred:   int
    label_true:   Optional[int] = None          # Ground truth label (for dashboard debug)
    alert:        bool = False                   # True when risk_score >= 0.5
    top_features: List[Dict[str, Any]]  = Field(default_factory=list)
    channel_risk: Dict[str, float]      = Field(default_factory=dict)
    band_power:   Dict[str, float]      = Field(default_factory=dict)
