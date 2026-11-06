"""L'entraînement et l'évaluation."""

import torch
from torch import nn
from torch.utils.data import DataLoader

from classifieur_paysages.donnees import TAILLE_IMAGE, cree_chargeur
from classifieur_paysages.modele import LeNet


def _device(modele: nn.Module) -> torch.device:
    return next(modele.parameters()).device


def entraine_epoque(
    modele: nn.Module, chargeur: DataLoader, optimiseur: torch.optim.Optimizer
) -> float:
    """Entraîne `modele` pendant une époque.

    Returns:
        La perte moyenne sur l'époque.
    """
    device = _device(modele)
    critere = nn.CrossEntropyLoss()
    modele.train()
    total, nombre = 0.0, 0
    for images, etiquettes in chargeur:
        images, etiquettes = images.to(device), etiquettes.to(device)
        optimiseur.zero_grad()
        perte = critere(modele(images), etiquettes)
        perte.backward()
        optimiseur.step()
        total += perte.item() * len(etiquettes)
        nombre += len(etiquettes)
    return total / nombre


def evalue(modele: nn.Module, chargeur: DataLoader) -> float:
    """Mesure l'exactitude de `modele` sur `chargeur`.

    Returns:
        La part des images dont `modele` prédit la bonne classe.
    """
    device = _device(modele)
    modele.eval()
    corrects, nombre = 0, 0
    with torch.inference_mode():
        for images, etiquettes in chargeur:
            predites = modele(images.to(device)).argmax(dim=1).cpu()
            corrects += int((predites == etiquettes).sum())
            nombre += len(etiquettes)
    return corrects / nombre


def predit(modele: nn.Module, chargeur: DataLoader) -> torch.Tensor:
    """Prédit la classe de chaque image de `chargeur`.

    Returns:
        Les indices des classes prédites.
    """
    device = _device(modele)
    modele.eval()
    with torch.inference_mode():
        return torch.cat(
            [modele(images.to(device)).argmax(dim=1).cpu() for images, _ in chargeur]
        )


def entraine(
    dossier_ent: str, dossier_test: str, taux_apprentissage: float, nb_epoques: int
) -> None:
    """Entraîne un LeNet sur `dossier_ent` et l'évalue sur `dossier_test`."""
    chargeur_ent = cree_chargeur(dossier_ent, TAILLE_IMAGE, melange=True)
    chargeur_test = cree_chargeur(dossier_test, TAILLE_IMAGE)
    modele = LeNet()
    optimiseur = torch.optim.AdamW(modele.parameters(), lr=taux_apprentissage)

    for epoque in range(nb_epoques):
        perte = entraine_epoque(modele, chargeur_ent, optimiseur)
        print(f"Époque {epoque + 1}/{nb_epoques} : perte {perte:.3f}")
    print(f"Exactitude sur le test : {evalue(modele, chargeur_test):.3f}")
