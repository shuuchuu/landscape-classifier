"""Fonctions utilitaires pour manipuler MLFlow."""

from pathlib import Path
from pickle import HIGHEST_PROTOCOL, dump, load
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any

from mlflow import log_artifact
from mlflow.artifacts import download_artifacts
from mlflow.client import MlflowClient


def load_artifact(run_id: str, artifact_name: str) -> Any:
    """Charge un artefact sauvé avec pickle.

    Suit la convention utilisée par la fonction log_pickle.

    Args:
        run_id: Identifiant du run depuis lequel charger l'artefact.
        artifact_name: Nom (chemin) de l'artefact à charger.

    Returns:
        Objet chargé depuis pickle.
    """
    # Création d'un dossier temporaire pour le téléchargement de l'artefact
    with TemporaryDirectory() as temp_dir:
        # Téléchargement depuis le dépôt MLFlow
        download_artifacts(
            run_id=run_id, artifact_path=artifact_name, dst_path=temp_dir
        )
        # Récupération du chemin du premier élément du dossier (normalement le seul)
        pickle_path = next((Path(temp_dir) / artifact_name).iterdir())
        # Chargement de l'objet
        with pickle_path.open("rb") as fh:
            return load(fh)


def get_latest_version_and_runid(name: str) -> tuple[str, str]:
    """Récupère la dernière version et le run_id d'un modèle par son nom.

    Args:
        name: Nom du modèle.

    Returns:
        La dernière version et le run_id du modèle.
    """
    result = MlflowClient().search_registered_models(f"name = '{name}'")
    last_version = next(iter(result)).latest_versions[0]
    return last_version.version, last_version.run_id


def log_pickle(obj: Any, name: str) -> None:
    """Sauvegarde d'un artefact en utilisant une sérialisation pickle.

    Args:
        obj: Artefact à sauver.
        name: Nom (chemin) à utiliser pour l'artefact.
    """
    with NamedTemporaryFile("wb") as fh:
        dump(obj, fh, protocol=HIGHEST_PROTOCOL)
        fh.flush()
        log_artifact(fh.name, name)
