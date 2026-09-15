from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Final, Literal, TypedDict, cast

from fastapi import FastAPI, HTTPException, UploadFile
from mlflow.exceptions import MlflowException
from mlflow.pyfunc import PyFuncModel, load_model
from pydantic import BaseModel

from landscape_classifier.data import LABEL_NAMES

MODEL_NAME: Final = "dev.ml.landscape-classifier"
MODEL_ALIAS: Final = "champion"

model: PyFuncModel | None = None


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:  # noqa: RUF029
    global model
    try:
        model = load_model(model_uri=f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
    except MlflowException:
        model = None
    yield


app = FastAPI(lifespan=lifespan)


Probabilities = TypedDict("Probabilities", dict.fromkeys(LABEL_NAMES, float))  # type: ignore


class ClassificationResult(BaseModel):
    predicted: list[Literal[tuple(LABEL_NAMES)]]  # type: ignore
    probabilities: list[Probabilities]


@app.post("/")
async def classify_image(images: list[UploadFile]) -> ClassificationResult:
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    # `PyFuncModel.predict` is typed to return any pyfunc-compatible output,
    # but our own `WrappedModel` always returns a `ClassificationResult`.
    return cast(
        ClassificationResult, model.predict([image.file.read() for image in images])
    )
