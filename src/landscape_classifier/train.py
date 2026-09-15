import os
from io import BytesIO
from pathlib import Path
from subprocess import run
from typing import Any

import yaml
from keras import Model, Sequential, layers, optimizers
from mlflow import MlflowClient, log_input, set_experiment, start_run
from mlflow.data.numpy_dataset import from_numpy
from mlflow.keras import autolog
from mlflow.pyfunc import PythonModel, PythonModelContext, log_model
from numpy import array, vstack
from pydantic import BaseModel

from landscape_classifier.data import LABEL_NAMES, get_images, process_image


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


def _get_git_commit() -> str | None:
    result = run(["git", "rev-parse", "HEAD"], capture_output=True)
    if result.returncode != 0:
        return None
    return result.stdout.decode("utf8").strip()


def _get_dvc_revision(train_dir: str) -> str | None:
    dvc_file = Path(f"{train_dir}.dvc")
    if not dvc_file.exists():
        return None
    outs = yaml.safe_load(dvc_file.read_text()).get("outs") or []
    return outs[0].get("md5") if outs else None


def _get_input_example(train_dir: str) -> list[bytes]:
    for subdir_path in sorted(Path(train_dir).iterdir()):
        for image_path in sorted(subdir_path.iterdir()):
            return [image_path.read_bytes()]
    return []


def train(
    experiment: str,
    train_dir: str,
    image_size: tuple[int, int],
    learning_rate: float,
    name: str,
    model_name: str,
    model_alias: str,
    epochs: int,
) -> None:
    set_experiment(experiment)
    # On journalise nous-mêmes le jeu de données d'entraînement et le modèle
    # avec les tags de traçabilité ci-dessous, autolog ne doit donc pas
    # journaliser ses propres copies non taguées.
    autolog(log_models=False, log_datasets=False)
    git_commit = _get_git_commit()
    dvc_revision = _get_dvc_revision(train_dir)
    with start_run():
        X_train, X_val, y_train, y_val = get_images(Path(train_dir), image_size)
        log_input(
            from_numpy(X_train, targets=y_train, source=train_dir, name="train-data"),
            context="training",
            tags={"dvc.revision": dvc_revision} if dvc_revision else None,
        )
        model = get_lenet(image_size, learning_rate)
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs)
        # `log_model` ci-dessous exécute `predict` sur l'exemple d'entrée pour
        # inférer le schéma de sortie, ce qui vide `model.history` en effet de
        # bord ; il faut donc lire la précision de validation avant de l'appeler.
        val_accuracy = model.history.history["val_accuracy"][-1]
        version_tags = {
            key: value
            for key, value in {
                "git.commit": git_commit,
                "dvc.revision": dvc_revision,
            }.items()
            if value is not None
        }
        # On laisse MLflow inférer automatiquement les dépendances pip via
        # `uv export` (MLFLOW_UV_AUTO_DETECT vaut true par défaut), mais sans
        # embarquer uv.lock/pyproject.toml comme artefacts : au moment du
        # service, `--env-manager uv` essaierait alors `uv sync` sur notre
        # vrai pyproject.toml, qui échoue car il déclare ce projet comme un
        # paquet installable (readme, arborescence src) absent du dossier de
        # restauration.
        os.environ["MLFLOW_LOG_UV_FILES"] = "false"
        model_info = log_model(
            name=name,
            python_model=WrappedModel(model),
            code_paths=["src/landscape_classifier"],
            model_config={"image_size": image_size, "label_names": LABEL_NAMES},
            input_example=_get_input_example(train_dir),
            registered_model_name=model_name,
            tags=version_tags,
        )
        client = MlflowClient()
        client.set_registered_model_alias(
            model_name, model_alias, model_info.registered_model_version
        )
        client.update_model_version(
            model_name,
            model_info.registered_model_version,
            description=(
                f"LeNet entraîné sur `{train_dir}` pendant {epochs} époque(s), "
                f"val_accuracy={val_accuracy:.4f}."
            ),
        )
        for key, value in version_tags.items():
            client.set_model_version_tag(
                model_name, model_info.registered_model_version, key, value
            )
