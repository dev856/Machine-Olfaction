"""FastAPI real-time inference microservice for machine olfaction.

Enables edge devices, microcontrollers (ESP32, Raspberry Pi), and web clients
to classify smell sensor recordings, stream time-series, and retrieve semantic chemical profiles.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.semantic_profiles import get_knowledge_base, get_semantic_profile
from src.models.mixture_predict import deconvolve_mixture_dataframe
from src.models.pipeline_io import load_pipeline
from src.models.predict import predict_dataframe

# Create FastAPI app
app = FastAPI(
    title="Machine Olfaction API",
    description="Production-ready REST API for electronic nose gas sensor classification and odor mixture deconvolution.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model cache
MODEL_CACHE: dict[str, Any] = {}
DEFAULT_MODEL_PATH = ROOT / "models" / "baseline_windowed_trialmax" / "model.joblib"
DEFAULT_MIXTURE_PATH = ROOT / "models" / "test_mixture" / "model.joblib"


def get_loaded_pipeline(model_path: Path | None = None) -> dict[str, Any]:
    path = model_path or DEFAULT_MODEL_PATH
    if not path.exists():
        # Fallback to any available model
        candidates = list(ROOT.glob("models/*/model.joblib"))
        if not candidates:
            raise RuntimeError("No trained model artifacts found in models/ directory.")
        path = candidates[0]

    key = str(path.resolve())
    if key not in MODEL_CACHE:
        MODEL_CACHE[key] = load_pipeline(path)
    return MODEL_CACHE[key]


# Pydantic Schemas
class HealthResponse(BaseModel):
    status: str
    active_model: str
    sensor_columns: list[str]
    target_points: int


class SmellProbability(BaseModel):
    smell_class: str
    probability: float


class SemanticCard(BaseModel):
    category: str
    volatiles: list[str]
    sensory_notes: list[str]
    summary: str


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    confidence_percentage: float
    top_5_predictions: list[SmellProbability]
    analyzed_windows: int
    aggregation_method: str
    semantics: SemanticCard


class RawTimeSeriesPayload(BaseModel):
    sensor_data: dict[str, list[float]] = Field(
        ...,
        description="Dictionary mapping sensor column names to lists of float readings over time.",
        json_schema_extra={
            "example": {
                "Alcohol": [0.1, 0.2, 0.5, 0.8, 0.7],
                "C2H5OH": [0.1, 0.3, 0.6, 0.9, 0.8],
                "CO": [0.05, 0.1, 0.2, 0.3, 0.25],
                "LPG": [0.0, 0.05, 0.1, 0.15, 0.1],
                "NO2": [0.2, 0.4, 0.7, 0.85, 0.8],
                "VOC": [0.15, 0.35, 0.65, 0.8, 0.75],
            }
        },
    )


class MixtureComponent(BaseModel):
    odor: str
    percentage: float
    ratio: float


class MixtureResponse(BaseModel):
    primary_odor: str
    primary_percentage: float
    active_components: list[MixtureComponent]
    all_ratios: dict[str, float]


@app.get("/", tags=["General"])
def index() -> dict[str, Any]:
    return {
        "name": "Machine Olfaction API",
        "status": "online",
        "docs": "/docs",
        "description": "Electronic nose smell classification and mixture deconvolution service.",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health() -> HealthResponse:
    bundle = get_loaded_pipeline()
    return HealthResponse(
        status="healthy",
        active_model=bundle.get("best_model_name", "ClassificationPipeline"),
        sensor_columns=list(bundle.get("sensor_columns", [])),
        target_points=int(bundle.get("target_points", 300)),
    )


@app.get("/models", tags=["Models"])
def list_models() -> list[dict[str, Any]]:
    models_dir = ROOT / "models"
    result = []
    for p in models_dir.glob("*/model.joblib"):
        name = p.parent.name
        metrics_file = p.parent / "metrics.json"
        metrics = {}
        if metrics_file.exists():
            try:
                import json
                with open(metrics_file, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
            except Exception:
                pass
        result.append({
            "artifact_folder": name,
            "path": str(p.relative_to(ROOT)),
            "best_model_name": metrics.get("best_model_name", "Unknown"),
            "trial_top1": metrics.get("trial_top1", metrics.get("test_accuracy")),
            "trial_top5": metrics.get("trial_top5", metrics.get("test_top5")),
        })
    return result


@app.get("/semantics/classes", tags=["Semantics"])
def list_classes() -> list[str]:
    kb = get_knowledge_base()
    return kb.list_all_classes()


@app.get("/semantics/profile/{smell_class}", response_model=SemanticCard, tags=["Semantics"])
def get_class_profile(smell_class: str) -> SemanticCard:
    prof = get_semantic_profile(smell_class)
    return SemanticCard(
        category=prof.category,
        volatiles=prof.volatiles,
        sensory_notes=prof.sensory_notes,
        summary=prof.summary,
    )


@app.post("/predict/csv", response_model=PredictionResponse, tags=["Inference"])
async def predict_from_csv(
    file: UploadFile = File(...),
    aggregation: str = Query("mean", enum=["mean", "max", "median"]),
) -> PredictionResponse:
    """Predict smell class from an uploaded gas sensor CSV file."""
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {exc}")

    try:
        bundle = get_loaded_pipeline()
        res = predict_dataframe(df, bundle, aggregation=aggregation)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Prediction error: {exc}")

    classes = res["class_names"]
    proba = res["probabilities"]
    top_indices = res["top_indices"][: min(5, len(classes))]

    top5 = [
        SmellProbability(smell_class=str(classes[idx]), probability=float(proba[idx]))
        for idx in top_indices
    ]

    sem = get_semantic_profile(res["predicted_class"])
    return PredictionResponse(
        predicted_class=res["predicted_class"],
        confidence=res["confidence"],
        confidence_percentage=round(res["confidence"] * 100, 2),
        top_5_predictions=top5,
        analyzed_windows=res["n_windows"],
        aggregation_method=res["aggregation"],
        semantics=SemanticCard(
            category=sem.category,
            volatiles=sem.volatiles,
            sensory_notes=sem.sensory_notes,
            summary=sem.summary,
        ),
    )


@app.post("/predict/raw", response_model=PredictionResponse, tags=["Inference"])
def predict_from_raw_timeseries(
    payload: RawTimeSeriesPayload,
    aggregation: str = Query("mean", enum=["mean", "max", "median"]),
) -> PredictionResponse:
    """Predict smell class from raw sensor readings sent as JSON (for IoT/Edge devices)."""
    try:
        df = pd.DataFrame(payload.sensor_data)
        bundle = get_loaded_pipeline()
        res = predict_dataframe(df, bundle, aggregation=aggregation)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Inference error: {exc}")

    classes = res["class_names"]
    proba = res["probabilities"]
    top_indices = res["top_indices"][: min(5, len(classes))]

    top5 = [
        SmellProbability(smell_class=str(classes[idx]), probability=float(proba[idx]))
        for idx in top_indices
    ]
    sem = get_semantic_profile(res["predicted_class"])

    return PredictionResponse(
        predicted_class=res["predicted_class"],
        confidence=res["confidence"],
        confidence_percentage=round(res["confidence"] * 100, 2),
        top_5_predictions=top5,
        analyzed_windows=res["n_windows"],
        aggregation_method=res["aggregation"],
        semantics=SemanticCard(
            category=sem.category,
            volatiles=sem.volatiles,
            sensory_notes=sem.sensory_notes,
            summary=sem.summary,
        ),
    )


@app.post("/mixture/csv", response_model=MixtureResponse, tags=["Inference"])
async def deconvolve_mixture_from_csv(
    file: UploadFile = File(...),
    threshold: float = Query(0.05, ge=0.0, le=1.0),
) -> MixtureResponse:
    """Deconvolve an uploaded mixture CSV into its constituent odor percentages."""
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {exc}")

    mix_path = ROOT / "models" / "mixture_regressor" / "model.joblib"
    if not mix_path.exists():
        mix_path = DEFAULT_MIXTURE_PATH
    if not mix_path.exists():
        raise HTTPException(status_code=503, detail="Mixture model not trained yet. Run train_mixture.py first.")

    bundle = load_pipeline(mix_path)
    res = deconvolve_mixture_dataframe(df, bundle, threshold=threshold)

    return MixtureResponse(
        primary_odor=res["primary_odor"],
        primary_percentage=round(res["primary_percentage"], 2),
        active_components=[MixtureComponent(**c) for c in res["active_components"]],
        all_ratios={k: round(float(v), 4) for k, v in zip(res["target_names"], res["ratios"])},
    )
