"""
FastAPI Predictive Maintenance Inference Service.
Scores real-time ESP telemetry to predict downtime probability within 72 hours.
"""

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime, timezone
from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "esp_failure_classifier.joblib"
METRICS_PATH = BASE_DIR / "models" / "model_metrics.json"

app = FastAPI(
    title="Upstream ESP Predictive Maintenance API",
    version="1.0.0",
    description="Machine learning inference API predicting electrical submersible pump failures and degradation states across upstream production assets."
)

class TelemetryInput(BaseModel):
    well_id: str = Field(..., json_schema_extra={"example": "PRD-WELL-ESP-03"})
    intake_pressure_psi: float = Field(..., json_schema_extra={"example": 890.5})
    discharge_pressure_psi: float = Field(..., json_schema_extra={"example": 2980.2})
    motor_temperature_c: float = Field(..., json_schema_extra={"example": 104.2})
    vibration_rms_mms: float = Field(..., json_schema_extra={"example": 4.15})
    current_draw_amps: float = Field(..., json_schema_extra={"example": 68.4})
    drive_frequency_hz: float = Field(..., json_schema_extra={"example": 55.0})
    water_cut_pct: float = Field(..., json_schema_extra={"example": 28.5})

class PredictionResponse(BaseModel):
    well_id: str
    failure_probability_next_72h: float
    risk_level: str # LOW, MEDIUM, CRITICAL
    maintenance_action_required: str
    inference_timestamp: str

_model_artifact = None

def get_model():
    global _model_artifact
    if _model_artifact is None:
        if not MODEL_PATH.exists():
            # If not yet trained on disk, train or return None
            return None
        _model_artifact = joblib.load(MODEL_PATH)
    return _model_artifact

@app.get("/health")
def health_check():
    return {
        "status": "HEALTHY",
        "service": "upstream-predictive-maintenance-ml",
        "model_loaded": MODEL_PATH.exists(),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.post("/api/v1/predict", response_model=PredictionResponse)
def predict_esp_failure(data: TelemetryInput):
    artifact = get_model()
    
    # Feature calculations
    press_diff = data.discharge_pressure_psi - data.intake_pressure_psi
    press_ratio = press_diff / (data.intake_pressure_psi + 1e-5)

    if artifact:
        model = artifact["model"]
        feature_vector = pd.DataFrame([{
            "intake_pressure_psi": data.intake_pressure_psi,
            "discharge_pressure_psi": data.discharge_pressure_psi,
            "pressure_differential_psi": press_diff,
            "motor_temperature_c": data.motor_temperature_c,
            "vibration_rms_mms": data.vibration_rms_mms,
            "current_draw_amps": data.current_draw_amps,
            "drive_frequency_hz": data.drive_frequency_hz,
            "water_cut_pct": data.water_cut_pct,
            "vib_rolling_mean_6h": data.vibration_rms_mms * 0.98,
            "vib_rolling_std_6h": 0.25 if data.vibration_rms_mms > 3.5 else 0.08,
            "vib_delta_6h": 0.45 if data.vibration_rms_mms > 3.5 else 0.05,
            "temp_rolling_mean_6h": data.motor_temperature_c * 0.99,
            "temp_delta_6h": 2.1 if data.motor_temperature_c > 105 else 0.2,
            "press_ratio": press_ratio
        }])
        prob = float(model.predict_proba(feature_vector)[0][1])
    else:
        # Fallback physics heuristic scoring
        prob = 0.05
        if data.vibration_rms_mms > 4.5 or data.motor_temperature_c > 115.0:
            prob = 0.88
        elif data.vibration_rms_mms > 3.2 or data.motor_temperature_c > 105.0:
            prob = 0.52

    prob = round(prob, 4)

    if prob >= 0.70:
        risk = "CRITICAL"
        action = "Schedule urgent workover inspection within 24 hours. Reduce drive frequency to 45Hz."
    elif prob >= 0.40:
        risk = "MEDIUM"
        action = "Increase telemetry polling frequency to 15m. Monitor bearing thermal trend."
    else:
        risk = "LOW"
        action = "Operating within nominal design envelope. Routine quarterly servicing."

    return PredictionResponse(
        well_id=data.well_id,
        failure_probability_next_72h=prob,
        risk_level=risk,
        maintenance_action_required=action,
        inference_timestamp=datetime.now(timezone.utc).isoformat()
    )

@app.get("/api/v1/model/metrics")
def get_model_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    return {"message": "Model has not been trained yet. Execute `python src/model_trainer.py` to generate metrics."}
