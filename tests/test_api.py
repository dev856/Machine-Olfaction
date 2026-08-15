"""Integration tests for FastAPI microservice endpoints."""

import io
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_api_index():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "docs" in data


def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert len(data["sensor_columns"]) > 0


def test_api_models():
    response = client.get("/models")
    assert response.status_code == 200
    models = response.json()
    assert isinstance(models, list)
    assert len(models) > 0


def test_api_semantics_classes():
    response = client.get("/semantics/classes")
    assert response.status_code == 200
    classes = response.json()
    assert len(classes) >= 50
    assert "banana" in classes


def test_api_semantics_profile():
    response = client.get("/semantics/profile/lemon")
    assert response.status_code == 200
    prof = response.json()
    assert prof["category"] == "Fruits & Citrus"
    assert "Limonene" in prof["volatiles"]


def test_api_predict_raw():
    payload = {
        "sensor_data": {
            "NO2": [0.1, 0.2, 0.3, 0.4, 0.5] * 20,
            "C2H5OH": [0.1, 0.3, 0.5, 0.7, 0.9] * 20,
            "VOC": [0.2, 0.3, 0.4, 0.5, 0.6] * 20,
            "CO": [0.05, 0.1, 0.15, 0.2, 0.25] * 20,
            "Alcohol": [0.1, 0.25, 0.4, 0.55, 0.7] * 20,
            "LPG": [0.0, 0.05, 0.1, 0.15, 0.2] * 20,
        }
    }
    response = client.post("/predict/raw", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert "predicted_class" in res
    assert "confidence" in res
    assert len(res["top_5_predictions"]) <= 5
    assert "semantics" in res
