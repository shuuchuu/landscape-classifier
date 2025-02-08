from typing import Final, Literal, TypedDict

from fastapi import FastAPI, UploadFile
from mlflow import MlflowClient
from mlflow.pyfunc import load_model
from pydantic import BaseModel

from landscape_classifier.data import LABEL_NAMES

app = FastAPI()

client = MlflowClient()

MODEL_NAME: Final = "dev.ml.landscape-classifier"
MODEL_ALIAS: Final = "champion"
model = load_model(model_uri=f"models:/{MODEL_NAME}@{MODEL_ALIAS}")


Probabilities = TypedDict("Probabilities", dict.fromkeys(LABEL_NAMES, float))  # type: ignore


class ClassificationResult(BaseModel):
    predicted: list[Literal(tuple(LABEL_NAMES))]  # type: ignore
    probabilities: list[Probabilities]


@app.post("/")
async def classify_image(images: list[UploadFile]) -> ClassificationResult:
    return model.predict([image.file.read() for image in images])
