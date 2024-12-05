from typing import Literal, TypedDict, cast

from fastapi import FastAPI, UploadFile
from mlflow import MlflowClient
from mlflow.keras import load_model
from pydantic import BaseModel

from .data import LABEL_NAMES, process_image
from .utils import get_latest_version_and_runid, load_artifact

app = FastAPI()

client = MlflowClient()

name = "dev.ml.lenet-landscape-classifier"
version, run_id = get_latest_version_and_runid(name)
model = load_model(model_uri=f"models:/{name}/{version}")
image_size = load_artifact(run_id, "model-config")["image_size"]


Probabilities = TypedDict("Probabilities", dict.fromkeys(LABEL_NAMES, float))  # type: ignore


class ClassificationResult(BaseModel):
    predicted: Literal[tuple(LABEL_NAMES)]  # type: ignore
    probabilities: Probabilities


class Response(BaseModel):
    result: ClassificationResult


@app.post("/")
async def classify_image(image: UploadFile) -> Response:
    X = process_image(image.file, image_size)
    output = model(X).numpy()[0]
    probabilities = dict(zip(LABEL_NAMES, output.tolist(), strict=True))
    predicted = LABEL_NAMES[output.argmax()]
    return Response(
        result=ClassificationResult(
            predicted=predicted, probabilities=cast(Probabilities, probabilities)
        )
    )
