import dagshub
import dvc.api
import mlflow
import mlflow.keras
import tensorflow as tf

CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
CLASS_INDICES = {l: i for i, l in enumerate(CLASS_NAMES)}


def get_images(
    dir_path: str, image_size: tuple[int, int], seed: int, shuffle: bool = True
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    return tf.keras.utils.image_dataset_from_directory(
        dir_path,
        class_names=CLASS_NAMES,
        batch_size=128,
        image_size=image_size,
        validation_split=0.3,
        subset="both",
        seed=seed,
    )


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


if __name__ == "__main__":
    dagshub.init("landscape-classifier", "m09", mlflow=True)
    config = dvc.api.params_show(stages=["train"])
    mlflow.start_run()
    mlflow.log_params(config)
    mlflow.keras.autolog()
    train_dataset, val_dataset = get_images(
        config["data_dir"], config["image_size"], config["seed"]
    )
    model = get_lenet(config["image_size"], config["learning_rate"])
    model.fit(train_dataset, validation_data=val_dataset, epochs=3)
    model.save(config["output_path"])
    mlflow.keras.log_model(
        model=model,
        artifact_path=config["output_path"],
        registered_model_name="lenet-landscape-classifier",
    )
    mlflow.end_run()
