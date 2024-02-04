import tensorflow as tf
from dagshub import init as dagshub_init
from fastapi import FastAPI, UploadFile
from mlflow.keras import load_model
from PIL import Image
from pydantic import BaseModel

app = FastAPI()

CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

dagshub_init("landscape-classifier", "m09", mlflow=True)


model_name = "lenet-landscape-classifier"
version = 1
model_uri = f"models:/{model_name}/{version}"
model = load_model(model_uri=model_uri)


class ClassificationResult(BaseModel):
    predicted: str
    probabilities: dict[str, float]


class Response(BaseModel):
    result: ClassificationResult


@app.post("/")
async def classify_image(image: UploadFile) -> Response:
    img = Image.open(image.file).resize((100, 100))
    X = tf.keras.preprocessing.image.img_to_array(img)
    X = tf.expand_dims(X, axis=0)
    output = model(X).numpy()[0]
    probabilities = dict(zip(CLASS_NAMES, output.tolist()))
    predicted = CLASS_NAMES[output.argmax()]
    return Response(
        result=ClassificationResult(predicted=predicted, probabilities=probabilities)
    )
