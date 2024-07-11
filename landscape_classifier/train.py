from pathlib import Path

import dvc.api
import mlflow
import mlflow.keras
import tensorflow as tf

from landscape_classifier.data import get_images


def get_lenet(
    image_size: tuple[int, int], learning_rate: float
) -> tf.keras.models.Model:
    def conv(filters: int, padding: str) -> tf.keras.layers.Conv2D:
        return tf.keras.layers.Conv2D(
            filters=filters, kernel_size=5, padding=padding, activation="sigmoid"
        )

    def pooling() -> tf.keras.layers.MaxPooling2D:
        return tf.keras.layers.MaxPooling2D()

    def dense(units: int, activation: str = "sigmoid") -> tf.keras.layers.Dense:
        return tf.keras.layers.Dense(units, activation=activation)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(*image_size, 3)),
            conv(6, "same"),
            pooling(),
            conv(16, "valid"),
            pooling(),
            tf.keras.layers.Flatten(),
            dense(120),
            dense(84),
            dense(6, activation="softmax"),
        ],
        name="le_net",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train() -> None:
    config = dvc.api.params_show(stages=["train"])
    mlflow.set_experiment("Landscape Classification")
    with mlflow.start_run():
        mlflow.log_params(config)
        X_train, X_val, y_train, y_val = get_images(
            Path(config["train_dir"]), config["image_size"]
        )
        model = get_lenet(config["image_size"], config["learning_rate"])
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=3)
        mlflow.keras.log_model(
            model=model,
            artifact_path="keras-model",
            registered_model_name="dev.ml.lenet-landscape-classifier",
        )
