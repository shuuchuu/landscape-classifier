import pathlib
import typing

import numpy
import PIL
import tensorflow as tf
import tqdm

INPUT_SHAPE = (150, 150)


LABEL_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
LABEL_TO_INDEX = {l: i for i, l in enumerate(LABEL_NAMES)}


def process_image(file: typing.BinaryIO | str | pathlib.Path) -> numpy.ndarray:
    return numpy.array(PIL.Image.open(file).resize(INPUT_SHAPE))


def get_images(
    dir_path: pathlib.Path,
    shuffle: bool = True,
    create_labels: bool = True,
) -> tuple[tf.Tensor, tf.Tensor]:
    images = []
    if create_labels:
        labels = []

    # On itère sur les sous-dossier de la racine : ils correspondent chacun à une
    # classe
    for subdir_path in tqdm.notebook.tqdm(
        list(dir_path.iterdir()), desc="Traitement des dossiers"
    ):

        dir_name = subdir_path.name

        if create_labels:
            # On attribue le bon label en fonction du nom du dossier "labels"
            label = LABEL_TO_INDEX.get(dir_name)

        # On ajoute chaque image du label (dossier) courant à notre dataset
        for image_path in tqdm.notebook.tqdm(
            list(subdir_path.iterdir()), desc=f"Dossier {dir_name}", leave=False
        ):
            # Utilisation de PIL pour charger l'image
            images.append(process_image(image_path))
            if create_labels:
                labels.append(label)

    images_tensor = tf.constant(numpy.array(images))
    if create_labels:
        labels_tensor = tf.constant(numpy.array(labels))

    if shuffle:
        perm = tf.random.shuffle(tf.range(images_tensor.shape[0]))
        images = tf.gather(images_tensor, perm)
        if create_labels:
            labels_tensor = tf.gather(labels_tensor, perm)

    return (images_tensor, labels_tensor) if create_labels else images_tensor
