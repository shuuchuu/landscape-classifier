from io import BytesIO
from pathlib import Path
from typing import Any

from keras import Model, Sequential, layers, optimizers
from mlflow import MlflowClient, set_experiment, start_run
from mlflow.keras import autolog
from mlflow.pyfunc import PythonModel, PythonModelContext, log_model
from numpy import array, vstack
from pydantic import BaseModel

from landscape_classifier.data import LABEL_NAMES, get_images, process_image
from landscape_classifier.utils import get_pip_requirements_from_uv


def get_lenet(image_size: tuple[int, int], learning_rate: float) -> Model:
    def conv(filters: int, padding: str) -> layers.Conv2D:
        return layers.Conv2D(
            filters=filters, kernel_size=5, padding=padding, activation="sigmoid"
        )

    def pooling() -> layers.MaxPooling2D:
        return layers.MaxPooling2D()

    def dense(units: int, activation: str = "sigmoid") -> layers.Dense:
        return layers.Dense(units, activation=activation)

    model = Sequential(
        [
            layers.InputLayer(shape=(*image_size, 3)),
            conv(6, "same"),
            pooling(),
            conv(16, "valid"),
            pooling(),
            layers.Flatten(),
            dense(120),
            dense(84),
            dense(6, activation="softmax"),
        ],
        name="le_net",
    )

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


class ClassificationResult(BaseModel):
    predicted: list[str]
    probabilities: list[dict[str, float]]


class WrappedModel(PythonModel):
    def __init__(self, model: Model) -> None:
        self._model = model

    def load_context(self, context: PythonModelContext) -> None:
        self._image_size = context.model_config["image_size"]

    def predict(
        self,
        context: PythonModelContext,
        model_input: list[bytes],
        params: dict[str, Any] | None = None,
    ) -> ClassificationResult:
        arrays = []
        for item in model_input:
            arrays.append(process_image(BytesIO(item), self._image_size))
        output = self._model.predict(vstack(arrays))
        predicted = array(LABEL_NAMES)[output.argmax(axis=-1)].tolist()
        probabilities = [dict(zip(LABEL_NAMES, row, strict=True)) for row in output]
        return ClassificationResult(predicted=predicted, probabilities=probabilities)


def train(
    experiment: str,
    train_dir: str,
    image_size: tuple[int, int],
    learning_rate: float,
    artifact_path: str,
    model_name: str,
    model_alias: str,
    epochs: int,
) -> None:
    set_experiment(experiment)
    autolog(log_models=False)
    with start_run():
        X_train, X_val, y_train, y_val = get_images(Path(train_dir), image_size)
        model = get_lenet(image_size, learning_rate)
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs)
        model_info = log_model(
            artifact_path=artifact_path,
            python_model=WrappedModel(model),
            code_paths=["src/landscape_classifier"],
            model_config={"image_size": image_size, "label_names": LABEL_NAMES},
            registered_model_name=model_name,
            pip_requirements=get_pip_requirements_from_uv(
                extras=["mlflow-models-serve"]
            ),
        )
        client = MlflowClient()
        client.set_registered_model_alias(
            model_name, model_alias, model_info.registered_model_version
        )
