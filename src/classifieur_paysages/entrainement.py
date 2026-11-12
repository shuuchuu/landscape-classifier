"""L'entraînement et l'évaluation, suivis avec MLflow."""

import mlflow
import torch
from torch import nn
from torch.utils.data import DataLoader

from classifieur_paysages.donnees import TAILLE_IMAGE, cree_chargeur
from classifieur_paysages.modele import LeNet

NOM_MODELE = "dev.ml.classifieur-paysages"


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
    """Entraîne un LeNet sur `dossier_ent` et l'évalue sur `dossier_test`.

    L'exécution est suivie dans l'expérience `classifieur-paysages` de MLflow, et le
    modèle publié dans le registre, comme nouvelle version de `NOM_MODELE`.
    """
    chargeur_ent = cree_chargeur(dossier_ent, TAILLE_IMAGE, melange=True)
    chargeur_test = cree_chargeur(dossier_test, TAILLE_IMAGE)
    modele = LeNet()
    optimiseur = torch.optim.AdamW(modele.parameters(), lr=taux_apprentissage)

    mlflow.set_experiment("classifieur-paysages")
    with mlflow.start_run():
        mlflow.log_params(
            {"taux_apprentissage": taux_apprentissage, "nb_epoques": nb_epoques}
        )
        for epoque in range(nb_epoques):
            perte = entraine_epoque(modele, chargeur_ent, optimiseur)
            mlflow.log_metric("perte_entrainement", perte, step=epoque)
            print(f"Époque {epoque + 1}/{nb_epoques} : perte {perte:.3f}")
        exactitude = evalue(modele, chargeur_test)
        mlflow.log_metric("exactitude_test", exactitude)
        print(f"Exactitude sur le test : {exactitude:.3f}")
        mlflow.pytorch.log_model(
            modele,
            name="modele",
            registered_model_name=NOM_MODELE,
            input_example=torch.zeros(1, 3, *TAILLE_IMAGE).numpy(),
            metadata={"classes": chargeur_ent.dataset.classes},  # pyrefly: ignore[missing-attribute]
            # Une liste courte, indépendante du projet ; PyTorch sans son suffixe local
            pip_requirements=["torch==" + torch.__version__.split("+")[0]],
        )
