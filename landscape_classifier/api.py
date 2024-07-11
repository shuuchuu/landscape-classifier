from pathlib import Path

from fastapi import FastAPI, UploadFile
from mlflow import MlflowClient
from mlflow.keras import load_model
from pydantic import BaseModel
from yaml import safe_load

from landscape_classifier.data import LABEL_NAMES, process_image

app = FastAPI()

client = MlflowClient()


def get_latest_version_uri(name: str) -> str:
    """
    Récupère la dernière version d'un modèle par son nom.

    Args:
        name: Nom du modèle.

    Returns:
        La dernière version et le run_id du modèle.
    """
    result = client.search_registered_models(f"name = '{name}'")
    version = next(iter(result)).latest_versions[0].version
    return f"models:/{name}/{version}"


model = load_model(
    model_uri=get_latest_version_uri("dev.ml.lenet-landscape-classifier")
)


class ClassificationResult(BaseModel):
    predicted: str
    probabilities: dict[str, float]


class Response(BaseModel):
    result: ClassificationResult


@app.post("/")
async def classify_image(image: UploadFile) -> Response:
    with (Path(__file__).parent.parent / "params.yaml").open(encoding="utf8") as fh:
        config = safe_load(fh)
    X = process_image(image.file, config["image_size"])
    output = model(X).numpy()[0]
    probabilities = dict(zip(LABEL_NAMES, output.tolist()))
    predicted = LABEL_NAMES[output.argmax()]
    return Response(
        result=ClassificationResult(predicted=predicted, probabilities=probabilities)
    )
