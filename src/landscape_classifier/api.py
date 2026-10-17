"""The classifier's HTTP API: `fastapi run src/landscape_classifier/api.py`."""

import os
import secrets
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated, Any

import mlflow
import torch
from fastapi import Depends, FastAPI, HTTPException, Security, UploadFile
from fastapi.security import APIKeyHeader
from PIL import Image
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from landscape_classifier.data import IMAGE_SIZE, get_transform

MODEL_URI = os.environ.get("MODEL_URI", "models:/dev.ml.landscape-classifier@champion")
models: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    models["classifier"] = mlflow.pytorch.load_model(MODEL_URI)
    models["classes"] = mlflow.models.get_model_info(MODEL_URI).metadata["classes"]  # ty: ignore[not-subscriptable]
    yield


app = FastAPI(lifespan=lifespan)
Instrumentator().instrument(app).expose(app)  # GET /metrics
CONFIDENCE = Histogram(
    "prediction_confidence",
    "Probability of the predicted class",
    buckets=[0.5, 0.7, 0.9, 1.0],
)
LABELS = Counter("predictions", "Predictions per class", ["label"])
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def check_api_key(key: Annotated[str | None, Security(api_key_header)]) -> None:
    expected = os.environ.get("API_KEY")
    if expected and not secrets.compare_digest(key or "", expected):
        raise HTTPException(status_code=401, detail="Invalid API key")


# Kubernetes' probes: the API only answers once `lifespan` has loaded the model.
@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def record(label: str, confidence: float) -> None:
    CONFIDENCE.observe(confidence)
    LABELS.labels(label=label).inc()


class Prediction(BaseModel):
    label: str
    probabilities: dict[str, float]


@app.post("/predict", dependencies=[Depends(check_api_key)])
def predict(file: UploadFile) -> Prediction:
    image = Image.open(file.file).convert("RGB")
    batch = get_transform(IMAGE_SIZE)(image).unsqueeze(0)
    with torch.inference_mode():
        scores = models["classifier"](batch).softmax(dim=1)[0]
    probabilities = dict(zip(models["classes"], scores.tolist(), strict=True))
    label = max(probabilities, key=probabilities.__getitem__)
    record(label, probabilities[label])
    return Prediction(label=label, probabilities=probabilities)
