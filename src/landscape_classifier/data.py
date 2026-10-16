"""The images: their transformation into tensors, and their loading."""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

IMAGE_SIZE = (100, 100)


def get_transform(image_size: tuple[int, int]) -> v2.Compose:
    """The transformation of an image into the model's input.

    Training and inference must transform images the same way.

    Returns:
        A transformation from a PIL image to a float tensor of shape
        `(3, *image_size)`, its pixels scaled to [0, 1].
    """
    return v2.Compose(
        [v2.ToImage(), v2.Resize(image_size), v2.ToDtype(torch.float32, scale=True)]
    )


def get_loader(
    directory: str, image_size: tuple[int, int], shuffle: bool = False
) -> DataLoader:
    """Load the images of `directory`, one subdirectory per class.

    Returns:
        Batches of 64 images and their class indices, the classes being the
        subdirectories' names in alphabetical order (`loader.dataset.classes`).
    """
    dataset = datasets.ImageFolder(directory, transform=get_transform(image_size))
    return DataLoader(dataset, batch_size=64, shuffle=shuffle)
