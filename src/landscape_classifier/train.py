from pathlib import Path

import keras

from .data import get_images


def get_lenet(image_size: tuple[int, int], learning_rate: float) -> keras.Model:
    def conv(filters: int, padding: str) -> keras.layers.Conv2D:
        return keras.layers.Conv2D(
            filters=filters, kernel_size=5, padding=padding, activation="sigmoid"
        )

    def pooling() -> keras.layers.MaxPooling2D:
        return keras.layers.MaxPooling2D()

    def dense(units: int, activation: str = "sigmoid") -> keras.layers.Dense:
        return keras.layers.Dense(units, activation=activation)

    model = keras.Sequential(
        [
            keras.layers.InputLayer(input_shape=(*image_size, 3)),
            conv(6, "same"),
            pooling(),
            conv(16, "valid"),
            pooling(),
            keras.layers.Flatten(),
            dense(120),
            dense(84),
            dense(6, activation="softmax"),
        ],
        name="le_net",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train(
    train_dir: str,
    image_size: tuple[int, int],
    learning_rate: float,
    epochs: int,
) -> None:
    X_train, X_val, y_train, y_val = get_images(Path(train_dir), image_size)
    model = get_lenet(image_size, learning_rate)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs)
