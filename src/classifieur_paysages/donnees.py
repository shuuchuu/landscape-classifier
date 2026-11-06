"""Les images : leur transformation en tenseurs et leur chargement."""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

TAILLE_IMAGE = (88, 88)


def cree_transformation(taille_image: tuple[int, int]) -> v2.Compose:
    """La transformation d'une image en entrée du modèle.

    L'entraînement et l'inférence doivent transformer les images de la même façon.

    Returns:
        Une transformation d'une image PIL en un tenseur de flottants de forme
        `(3, *taille_image)`, ses pixels ramenés dans [0, 1].
    """
    return v2.Compose(
        [v2.ToImage(), v2.Resize(taille_image), v2.ToDtype(torch.float32, scale=True)]
    )


def cree_chargeur(
    dossier: str, taille_image: tuple[int, int], melange: bool = False
) -> DataLoader:
    """Charge les images de `dossier`, un sous-dossier par classe.

    Returns:
        Des lots de 64 images et les indices de leurs classes, les classes étant les
        noms des sous-dossiers par ordre alphabétique (`chargeur.dataset.classes`).
    """
    jeu = datasets.ImageFolder(dossier, transform=cree_transformation(taille_image))
    return DataLoader(jeu, batch_size=64, shuffle=melange)
