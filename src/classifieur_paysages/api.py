"""L'API HTTP du classifieur : `fastapi run src/classifieur_paysages/api.py`."""

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

from classifieur_paysages.donnees import TAILLE_IMAGE, cree_transformation

URI_MODELE = os.environ.get(
    "URI_MODELE", "models:/dev.ml.classifieur-paysages@champion"
)
modeles: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    if not os.environ.get("CLE_API"):
        msg = "CLE_API absente : l'API refuse de démarrer"
        raise RuntimeError(msg)
    modeles["classifieur"] = mlflow.pytorch.load_model(URI_MODELE)
    modeles["classes"] = mlflow.models.get_model_info(URI_MODELE).metadata["classes"]  # pyrefly: ignore[unsupported-operation]
    yield


app = FastAPI(lifespan=lifespan)
Instrumentator().instrument(app).expose(app)  # GET /metrics
CONFIANCE = Histogram(
    "confiance_prediction",
    "Probabilité de la classe prédite",
    buckets=[0.5, 0.7, 0.9, 1.0],
)
ETIQUETTES = Counter("predictions", "Prédictions par classe", ["etiquette"])
entete_cle_api = APIKeyHeader(name="X-API-Key", auto_error=False)


def verifie_cle_api(cle: Annotated[str | None, Security(entete_cle_api)]) -> None:
    if not secrets.compare_digest((cle or "").encode(), os.environ["CLE_API"].encode()):
        raise HTTPException(status_code=401, detail="Clé d'API invalide")


# Sondes de Kubernetes : l'API ne répond qu'une fois le modèle chargé par `lifespan`.
@app.get("/sante")
def sante() -> dict[str, str]:
    return {"statut": "ok"}


def enregistre(etiquette: str, confiance: float) -> None:
    CONFIANCE.observe(confiance)
    ETIQUETTES.labels(etiquette=etiquette).inc()


class Prediction(BaseModel):
    etiquette: str
    probabilites: dict[str, float]


@app.post("/prediction", dependencies=[Depends(verifie_cle_api)])
def predit(fichier: UploadFile) -> Prediction:
    image = Image.open(fichier.file).convert("RGB")
    lot = cree_transformation(TAILLE_IMAGE)(image).unsqueeze(0)
    with torch.inference_mode():
        scores = modeles["classifieur"](lot).softmax(dim=1)[0]
    probabilites = dict(zip(modeles["classes"], scores.tolist(), strict=True))
    etiquette = max(probabilites, key=probabilites.__getitem__)
    enregistre(etiquette, probabilites[etiquette])
    return Prediction(etiquette=etiquette, probabilites=probabilites)
