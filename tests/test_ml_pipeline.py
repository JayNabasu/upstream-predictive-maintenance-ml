import pytest
import pandas as pd
from src.data_generator import generate_telemetry_stream
from src.feature_engineering import engineer_features, prepare_train_test_data, FEATURE_COLUMNS
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_data_generation():
    df = generate_telemetry_stream(num_wells=3, days=5)
    assert not df.empty
    assert "intake_pressure_psi" in df.columns
    assert "vibration_rms_mms" in df.columns
    assert "failure_next_72h" in df.columns

def test_feature_engineering():
    df = generate_telemetry_stream(num_wells=3, days=5)
    df_feats = engineer_features(df)
    for col in FEATURE_COLUMNS:
        assert col in df_feats.columns
    assert "vib_rolling_mean_6h" in df_feats.columns

def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "HEALTHY"

def test_api_predict_nominal():
    payload = {
        "well_id": "OML-20-ESP-01",
        "intake_pressure_psi": 920.0,
        "discharge_pressure_psi": 2800.0,
        "motor_temperature_c": 95.0,
        "vibration_rms_mms": 1.9,
        "current_draw_amps": 60.0,
        "drive_frequency_hz": 55.0,
        "water_cut_pct": 20.0
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["risk_level"] in ("LOW", "MEDIUM", "CRITICAL")
    assert "failure_probability_next_72h" in data
