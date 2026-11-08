# Classifieur de paysages : registre de modèles MLflow et déploiement

*[English version](https://github.com/shuuchuu/landscape-classifier/tree/main-en)*

Ce TP suit un modèle de bout en bout : un petit réseau convolutif (LeNet, en PyTorch) qui classe des photos de paysages en 6 classes (bâtiments, forêt, glacier, montagne, mer, rue), dont on suit l'entraînement avec MLflow, que l'on publie dans le registre de modèles de MLflow, puis que l'on sert par une API FastAPI empaquetée dans une image Docker. Il n'a pas besoin de GPU.

Ce README en est l'énoncé, pour un Codespace ou votre poste ; le [notebook du TP](https://colab.research.google.com/github/shuuchuu/labs/blob/main/deployment-mlflow-registry-deployment-hands-on-fr.ipynb) donne le même dans Colab. La [branche `solution`](https://github.com/shuuchuu/landscape-classifier/tree/solution) donne une solution. Le serveur MLflow est celui de votre propre dépôt DagsHub, que vous créez à la section « Identifiants pour le serveur MLflow » : chacun y retrouve ses exécutions et ses modèles, sans ceux des autres.

## Mise en place de l'environnement

Ouvrez un Codespace du dépôt (<https://codespaces.new/shuuchuu/landscape-classifier>), qui installe le projet et Docker, ou, sur votre poste, avec [uv](https://docs.astral.sh/uv/) :

    git clone https://github.com/shuuchuu/landscape-classifier.git
    cd landscape-classifier
    uv sync --all-extras

Les commandes qui suivent se lancent à la racine du dépôt.

### Identifiants pour le serveur MLflow

Chaque stagiaire suit ses modèles dans son propre serveur MLflow, celui d'un dépôt DagsHub à son nom :

1. Sur [DagsHub](https://dagshub.com), créez un compte gratuit, puis un dépôt vide nommé `landscape-classifier` (*New repository*). Son serveur MLflow a pour adresse `https://dagshub.com/<votre nom d'utilisateur>/landscape-classifier.mlflow`.
2. Dans les paramètres de votre compte DagsHub, *Tokens*, créez un jeton : c'est lui, et non votre mot de passe, que demande MLflow.

Créez ensuite un fichier `identifiants.env` (que git ignore) avec l'adresse de votre serveur, votre nom d'utilisateur DagsHub et ce jeton :

    MLFLOW_TRACKING_URI=https://dagshub.com/<votre nom d'utilisateur>/landscape-classifier.mlflow
    MLFLOW_TRACKING_USERNAME=<votre nom d'utilisateur>
    MLFLOW_TRACKING_PASSWORD=<votre jeton>

`uv run --env-file identifiants.env <commande>` et `docker run --env-file identifiants.env` les passent aux programmes qui parlent à MLflow.

### Récupération des données

Les images d'entraînement vont dans `entrainement-images/`, celles de test dans `test-images/`, un sous-dossier par classe :

    wget -q https://github.com/shuuchuu/datasets/raw/refs/heads/main/ghsparse.sh -O /tmp/ghsparse.sh && bash /tmp/ghsparse.sh landscape
    mv landscape/seg_train entrainement-images
    mv landscape/seg_test test-images

## Le projet

Le paquet `src/classifieur_paysages/` a trois modules :

- `donnees.py` : la transformation d'une image en tenseur (`cree_transformation`) et le chargement des images d'un dossier par lots (`cree_chargeur`) ;
- `modele.py` : le réseau `LeNet` ;
- `entrainement.py` : une époque d'entraînement (`entraine_epoque`), l'exactitude d'un modèle (`evalue`), et la fonction `entraine`, qui entraîne un modèle puis l'évalue.

La commande `classifieur-paysages` appelle `entraine` avec les paramètres de `params.yaml`, que lit DVC (l'étape `entrainement` de `dvc.yaml`). Lisez `entrainement.py`.

## Suivi de l'entraînement et publication du modèle

Modifiez la fonction `entraine` de `src/classifieur_paysages/entrainement.py` pour qu'un entraînement soit une exécution (*run*) MLflow de l'expérience `classifieur-paysages`, qui journalise :

- les hyperparamètres `taux_apprentissage` et `nb_epoques` ;
- la perte moyenne de chaque époque, `perte_entrainement`, avec le numéro de l'époque comme pas (`step`) ;
- l'exactitude sur les images de test, `exactitude_test`.

En fin d'exécution, publiez le modèle dans le registre de modèles sous le nom `dev.ml.classifieur-paysages`, avec les noms des classes dans ses métadonnées (`classes`), que l'API relira, et la version de PyTorch installée comme dépendance.

Modifiez le fichier dans votre éditeur. La section suivante lance l'entraînement.

<details>
<summary>Conseils supplémentaires</summary>

`mlflow.set_experiment("classifieur-paysages")` choisit l'expérience (et la crée au besoin), puis tout se passe dans un bloc `with mlflow.start_run():`. `mlflow.log_params` prend un dictionnaire, `mlflow.log_metric(nom, valeur, step=epoque)` une valeur. `mlflow.pytorch.log_model(modele, name="modele", registered_model_name=..., input_example=..., metadata=..., pip_requirements=...)` enregistre le modèle comme artefact de l'exécution et en crée une nouvelle version dans le registre. MLflow exporte le graphe du modèle en l'exécutant sur `input_example`, un tableau NumPy de la forme d'un lot d'une image : `torch.zeros(1, 3, *TAILLE_IMAGE).numpy()`. Les noms des classes sont `chargeur_ent.dataset.classes`. `pip_requirements` prend une liste de dépendances au format de pip : `torch.__version__` donne la version de PyTorch, dont on retire le suffixe local (`+cu128` sur Colab), que PyPI ne connaît pas. Ajoutez `import mlflow` en tête du fichier ; le reste ne change pas.

```python
def entraine(
    dossier_ent: str, dossier_test: str, taux_apprentissage: float, nb_epoques: int
) -> None:
    chargeur_ent = cree_chargeur(dossier_ent, TAILLE_IMAGE, melange=True)
    chargeur_test = cree_chargeur(dossier_test, TAILLE_IMAGE)
    modele = LeNet()
    optimiseur = torch.optim.AdamW(modele.parameters(), lr=taux_apprentissage)

    mlflow.set_experiment("classifieur-paysages")
    with mlflow.start_run():
        mlflow.log_params(...)
        for epoque in range(nb_epoques):
            perte = entraine_epoque(modele, chargeur_ent, optimiseur)
            mlflow.log_metric(...)
            print(f"Époque {epoque + 1}/{nb_epoques} : perte {perte:.3f}")
        exactitude = evalue(modele, chargeur_test)
        ...
        mlflow.pytorch.log_model(
            modele,
            name="modele",
            registered_model_name=...,
            input_example=...,
            metadata=...,
            pip_requirements=...,
        )
```

</details>

<details>
<summary>Solution</summary>

```python
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
            # La version de PyTorch sans son suffixe local (+cu128), inconnu de PyPI
            pip_requirements=["torch==" + torch.__version__.split("+")[0]],
        )
```

</details>

## Entraînement

La commande `classifieur-paysages` lance l'entraînement, 3 époques sur les 14 000 images (environ 7 minutes sur 2 cœurs), avec les identifiants de `identifiants.env`. En fin d'exécution, MLflow vérifie que le modèle publié accepte `input_example` en le rechargeant : `MLFLOW_DEFAULT_PREDICTION_DEVICE=cpu` le recharge sur le CPU, où il a été entraîné, même sur une machine qui a un GPU :

    MLFLOW_DEFAULT_PREDICTION_DEVICE=cpu uv run --env-file identifiants.env classifieur-paysages

Ouvrez ensuite l'interface de MLflow, à l'adresse de votre serveur (`https://dagshub.com/<votre nom d'utilisateur>/landscape-classifier.mlflow`) : l'expérience `classifieur-paysages` montre votre exécution, ses paramètres et la courbe de `perte_entrainement`, et la page *Models* la nouvelle version de `dev.ml.classifieur-paysages`.

## Utilisation du registre de modèles MLflow

Avec l'API Python de MLflow, effectuez les tâches suivantes sur le registre de modèles, dans un script `registre.py` que vous complétez à chaque question et lancez avec `uv run --env-file identifiants.env python registre.py` (ou dans un notebook, avec l'environnement `.venv` du projet).

### Création d'un client MLflow

Créez un client MLflow (`MlflowClient`) et affichez son adresse de suivi (*tracking URI*), pour vérifier qu'elle vient bien de la variable d'environnement `MLFLOW_TRACKING_URI`.

**Documentation utile**

- https://mlflow.org/docs/latest/api_reference/python_api/mlflow.client.html#mlflow.client.MlflowClient.tracking_uri

<details>
<summary>Conseils supplémentaires</summary>

`mlflow.MlflowClient()` lit l'adresse du serveur dans `MLFLOW_TRACKING_URI`, et l'attribut `tracking_uri` du client la donne. Gardez le nom du modèle dans une constante : les questions suivantes s'en servent.

```python
import mlflow

NOM_MODELE = "dev.ml.classifieur-paysages"

client = ...
client.tracking_uri
```

</details>

<details>
<summary>Solution</summary>

```python
import mlflow

NOM_MODELE = "dev.ml.classifieur-paysages"

client = mlflow.MlflowClient()
print(client.tracking_uri)
```

</details>

### Recherche de modèles

Affichez tous les modèles du registre et, pour chacun, le numéro de sa version la plus récente. Écrivez pour cela une fonction `derniere_version(nom)`, que les questions suivantes réutilisent.

**Documentation utile**

- https://mlflow.org/docs/latest/api_reference/python_api/mlflow.client.html#mlflow.client.MlflowClient.search_registered_models

<details>
<summary>Conseils supplémentaires</summary>

`client.search_registered_models()` renvoie les modèles enregistrés (attribut `name`), `client.search_model_versions(f"name = '{nom}'")` les versions d'un modèle, dont l'attribut `version` est une chaîne : comparez les versions comme des entiers.

```python
def derniere_version(nom: str) -> str:
    versions = client.search_model_versions(...)
    return str(max(...))


for modele_enregistre in client.search_registered_models():
    print(modele_enregistre.name, ...)
```

</details>

<details>
<summary>Solution</summary>

```python
def derniere_version(nom: str) -> str:
    versions = client.search_model_versions(f"name = '{nom}'")
    return str(max(int(v.version) for v in versions))


for modele_enregistre in client.search_registered_models():
    print(modele_enregistre.name, derniere_version(modele_enregistre.name))
```

</details>

### Récupération d'un modèle depuis le registre

Chargez la dernière version de `dev.ml.classifieur-paysages` avec `mlflow.pytorch.load_model`, puis prédisez la classe de l'image `foret1.jpg` : ouvrez-la avec Pillow, transformez-la avec la fonction `cree_transformation` de `classifieur_paysages.donnees`, celle de l'entraînement, et lisez les noms des classes dans les métadonnées du modèle (`mlflow.models.get_model_info`).

Un modèle MLflow peut avoir plusieurs *flavors*, autant de manières de le charger : affichez celles du vôtre, et chargez-le aussi avec `mlflow.pyfunc.load_model`. Qu'est-ce que cela change ?

**Documentation utile**

- https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.open
- https://mlflow.org/docs/latest/api_reference/python_api/mlflow.pyfunc.html#mlflow.pyfunc.load_model

<details>
<summary>Conseils supplémentaires</summary>

L'URI d'une version est `models:/<nom>/<version>`. `Image.open("foret1.jpg").convert("RGB")` ouvre l'image, `cree_transformation(TAILLE_IMAGE)(image)` en fait un tenseur de forme `(3, 88, 88)`, et `.unsqueeze(0)` un lot d'une image : la seule taille de lot que le modèle chargé accepte, MLflow l'a figée à celle de `input_example`. Le modèle renvoie des scores, que `.softmax(dim=1)` change en probabilités. `get_model_info(uri)` donne les métadonnées (`.metadata["classes"]`) et les flavors (`.flavors`). Le `predict` du modèle `pyfunc` prend un tableau NumPy (`lot.numpy()`). `mlflow.pyfunc.load_model(uri, model_config={"device": "cpu"})` le charge sur le CPU, même sur une machine qui a un GPU.

```python
import torch
from PIL import Image

from classifieur_paysages.donnees import TAILLE_IMAGE, cree_transformation

version = derniere_version(NOM_MODELE)
uri = f"models:/{NOM_MODELE}/{version}"
modele = mlflow.pytorch.load_model(uri)
info = ...
print(list(info.flavors))

image = Image.open("foret1.jpg").convert("RGB")
lot = ...
with torch.inference_mode():
    probabilites = ...
print(dict(zip(info.metadata["classes"], probabilites.tolist(), strict=True)))

modele_pyfunc = ...
print(modele_pyfunc.predict(...))
```

</details>

<details>
<summary>Solution</summary>

```python
import torch
from PIL import Image

from classifieur_paysages.donnees import TAILLE_IMAGE, cree_transformation

version = derniere_version(NOM_MODELE)
uri = f"models:/{NOM_MODELE}/{version}"
modele = mlflow.pytorch.load_model(uri)
info = mlflow.models.get_model_info(uri)
print(list(info.flavors))

image = Image.open("foret1.jpg").convert("RGB")
lot = cree_transformation(TAILLE_IMAGE)(image).unsqueeze(0)
with torch.inference_mode():
    probabilites = modele(lot).softmax(dim=1)[0]
print(dict(zip(info.metadata["classes"], probabilites.tolist(), strict=True)))

modele_pyfunc = mlflow.pyfunc.load_model(uri, model_config={"device": "cpu"})
print(modele_pyfunc.predict(lot.numpy()))
```

Le modèle a deux flavors. `pytorch` donne le réseau PyTorch (`mlflow.pytorch.load_model`), qui prend et renvoie des tenseurs. `python_function` est l'interface commune à tous les modèles MLflow (`mlflow.pyfunc.load_model`), dont `predict` prend et renvoie des tableaux NumPy : c'est elle que `mlflow models serve` sert, sans que l'appelant ait besoin de PyTorch. Aucune des deux ne transforme l'image : le prétraitement de `cree_transformation` reste à faire, à l'identique de l'entraînement. Sur un GPU, `pyfunc` y chargerait le modèle, ce que refuse un modèle au format pt2 de MLflow 3 (`torch.export`) : il garde le device où il a été enregistré, ici le CPU, d'où `model_config={"device": "cpu"}`.

</details>

### Ajout d'alias et d'étiquettes

Les alias et les étiquettes (*tags*) organisent les modèles du registre. Un alias désigne une version précise, et se déplace d'une version à l'autre ; une étiquette se pose sur un modèle ou sur l'une de ses versions.

Donnez l'alias `champion` à la dernière version de votre modèle, posez l'étiquette `domaine` de valeur `vision` sur le modèle et l'étiquette `taille` de valeur `petit` sur sa dernière version. Chargez ensuite le modèle par son alias plutôt que par son numéro de version, et prédisez à nouveau la classe de `foret1.jpg`.

**Documentation utile**

- https://mlflow.org/docs/latest/api_reference/python_api/mlflow.client.html#mlflow.client.MlflowClient.set_registered_model_tag

<details>
<summary>Conseils supplémentaires</summary>

`client.set_registered_model_alias(nom, alias, version)`, `client.set_registered_model_tag(nom, clé, valeur)` et `client.set_model_version_tag(nom, version, clé, valeur)`. L'URI d'un alias est `models:/<nom>@<alias>`.

```python
client.set_registered_model_alias(NOM_MODELE, ..., version)
client.set_registered_model_tag(NOM_MODELE, ...)
client.set_model_version_tag(NOM_MODELE, version, ...)

modele = mlflow.pytorch.load_model(...)
with torch.inference_mode():
    print(modele(lot).softmax(dim=1)[0])
```

</details>

<details>
<summary>Solution</summary>

```python
client.set_registered_model_alias(NOM_MODELE, "champion", version)
client.set_registered_model_tag(NOM_MODELE, "domaine", "vision")
client.set_model_version_tag(NOM_MODELE, version, "taille", "petit")
print(client.get_model_version_by_alias(NOM_MODELE, "champion"))

modele = mlflow.pytorch.load_model(f"models:/{NOM_MODELE}@champion")
with torch.inference_mode():
    print(modele(lot).softmax(dim=1)[0])
```

</details>

### Promotion d'un modèle

Pour promouvoir un modèle d'un environnement au suivant, la méthode recommandée est de copier la version voulue vers le modèle de l'environnement suivant, qui a son propre nom. Copiez la version `champion` de votre modèle vers le modèle `staging.ml.classifieur-paysages`, le nom où le préfixe `dev.` devient `staging.`.

<details>
<summary>Conseils supplémentaires</summary>

`client.copy_model_version(src_model_uri=..., dst_name=...)` crée le modèle de destination s'il n'existe pas encore ; la source est une URI de modèle, ici celle de l'alias.

```python
client.copy_model_version(
    src_model_uri=...,
    dst_name=...,
)
```

</details>

<details>
<summary>Solution</summary>

```python
client.copy_model_version(
    src_model_uri=f"models:/{NOM_MODELE}@champion",
    dst_name="staging.ml.classifieur-paysages",
)
```

</details>

## Création d'une API avec FastAPI

Écrivez l'API d'inférence dans `src/classifieur_paysages/api.py` :

- au démarrage (`lifespan`), elle charge une fois pour toutes le modèle dont l'URI est la variable d'environnement `URI_MODELE`, `models:/dev.ml.classifieur-paysages@champion` par défaut, et les noms de ses classes ;
- `GET /sante` répond `{"statut": "ok"}` : les sondes de Kubernetes l'appellent pour savoir si l'API est prête ;
- `POST /prediction` reçoit une image (`UploadFile`), la transforme comme à l'entraînement, et renvoie un objet Pydantic `Prediction` : la classe la plus probable (`etiquette`) et la probabilité de chaque classe (`probabilites`, un dictionnaire).

À quoi faut-il faire extrêmement attention dans le prétraitement ? Testez ensuite l'API (section suivante).

<details>
<summary>Conseils supplémentaires</summary>

Le modèle chargé et les noms des classes vont dans un dictionnaire global (`modeles`), que `lifespan` remplit avant son `yield`. La fonction de `/prediction` est une `def`, pas une `async def` : l'inférence bloque. `Image.open(fichier.file).convert("RGB")` lit l'image envoyée, et l'inférence se fait sous `torch.inference_mode()`. Les probabilités sont des `float` Python (`.tolist()`), jamais un tenseur.

```python
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import mlflow
import torch
from fastapi import FastAPI, UploadFile
from PIL import Image
from pydantic import BaseModel

from classifieur_paysages.donnees import TAILLE_IMAGE, cree_transformation

URI_MODELE = os.environ.get(
    "URI_MODELE", "models:/dev.ml.classifieur-paysages@champion"
)
modeles: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    modeles["classifieur"] = ...
    modeles["classes"] = ...
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/sante")
def sante() -> dict[str, str]:
    return ...


class Prediction(BaseModel):
    etiquette: str
    probabilites: ...


@app.post("/prediction")
def predit(fichier: UploadFile) -> Prediction:
    image = Image.open(fichier.file).convert("RGB")
    lot = ...
    with torch.inference_mode():
        scores = ...
    probabilites = ...
    etiquette = ...
    return Prediction(etiquette=etiquette, probabilites=probabilites)
```

</details>

<details>
<summary>Solution</summary>

```python
"""L'API HTTP du classifieur : `fastapi run src/classifieur_paysages/api.py`."""

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import mlflow
import torch
from fastapi import FastAPI, UploadFile
from PIL import Image
from pydantic import BaseModel

from classifieur_paysages.donnees import TAILLE_IMAGE, cree_transformation

URI_MODELE = os.environ.get(
    "URI_MODELE", "models:/dev.ml.classifieur-paysages@champion"
)
modeles: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    modeles["classifieur"] = mlflow.pytorch.load_model(URI_MODELE)
    modeles["classes"] = mlflow.models.get_model_info(URI_MODELE).metadata["classes"]
    yield


app = FastAPI(lifespan=lifespan)


# Sondes de Kubernetes : l'API ne répond qu'une fois le modèle chargé par `lifespan`.
@app.get("/sante")
def sante() -> dict[str, str]:
    return {"statut": "ok"}


class Prediction(BaseModel):
    etiquette: str
    probabilites: dict[str, float]


@app.post("/prediction")
def predit(fichier: UploadFile) -> Prediction:
    image = Image.open(fichier.file).convert("RGB")
    lot = cree_transformation(TAILLE_IMAGE)(image).unsqueeze(0)
    with torch.inference_mode():
        scores = modeles["classifieur"](lot).softmax(dim=1)[0]
    probabilites = dict(zip(modeles["classes"], scores.tolist(), strict=True))
    etiquette = max(probabilites, key=probabilites.__getitem__)
    return Prediction(etiquette=etiquette, probabilites=probabilites)
```

Le prétraitement doit être exactement celui de l'entraînement : l'API importe `cree_transformation` et `TAILLE_IMAGE` de `classifieur_paysages.donnees` au lieu de les réécrire. Une différence (taille, mise à l'échelle des pixels, ordre des canaux) ne lève aucune erreur : elle dégrade les prédictions en silence.

</details>

### Test de l'API

Lancez le serveur avec les identifiants de `identifiants.env`, puis appelez-le depuis un autre terminal ; sa documentation interactive, qui sert aussi de client de test, est à l'adresse <http://localhost:8000/docs> :

    uv run --env-file identifiants.env fastapi dev src/classifieur_paysages/api.py
    curl http://localhost:8000/sante
    curl -F fichier=@foret1.jpg http://localhost:8000/prediction

## Protection de l'API par une clé

Protégez `POST /prediction` par une clé d'API, lue dans la variable d'environnement `CLE_API` : sans elle, l'API refuse de démarrer (dans `lifespan`, avant de charger le modèle), plutôt que de rester ouverte à tous ; une requête qui n'a pas d'en-tête `X-API-Key` égal à sa valeur reçoit une erreur 401. `GET /sante` reste ouvert.

<details>
<summary>Conseils supplémentaires</summary>

`APIKeyHeader(name="X-API-Key", auto_error=False)` lit l'en-tête (`None` s'il manque). Une fonction `verifie_cle_api` qui le reçoit par `Security(entete_cle_api)` lève `HTTPException(status_code=401, detail=...)`, et `@app.post("/prediction", dependencies=[Depends(verifie_cle_api)])` l'appelle avant chaque requête. `secrets.compare_digest` compare deux chaînes en un temps qui ne dépend pas de leur contenu. Sans `CLE_API`, `lifespan` lève une `RuntimeError` : une variable oubliée au déploiement arrête l'API au lieu de l'ouvrir. Ajoutez à `api.py` :

```python
import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    if ...:
        msg = ...
        raise RuntimeError(msg)
    modeles["classifieur"] = mlflow.pytorch.load_model(URI_MODELE)
    modeles["classes"] = mlflow.models.get_model_info(URI_MODELE).metadata["classes"]
    yield


entete_cle_api = APIKeyHeader(name="X-API-Key", auto_error=False)


def verifie_cle_api(cle: Annotated[str | None, Security(entete_cle_api)]) -> None:
    if not secrets.compare_digest(...):
        raise HTTPException(...)


@app.post("/prediction", dependencies=[...])
def predit(fichier: UploadFile) -> Prediction: ...
```

</details>

<details>
<summary>Solution</summary>

```python
"""L'API HTTP du classifieur : `fastapi run src/classifieur_paysages/api.py`."""

import os
import secrets
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated, Any

import mlflow
import torch
from fastapi import Depends, FastAPI, HTTPException, Security, UploadFile
from fastapi.security import APIKeyHeader
from PIL import Image
from pydantic import BaseModel

from classifieur_paysages.donnees import TAILLE_IMAGE, cree_transformation

URI_MODELE = os.environ.get(
    "URI_MODELE", "models:/dev.ml.classifieur-paysages@champion"
)
modeles: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    if not os.environ.get("CLE_API"):
        msg = "CLE_API absente : l'API refuse de démarrer"
        raise RuntimeError(msg)
    modeles["classifieur"] = mlflow.pytorch.load_model(URI_MODELE)
    modeles["classes"] = mlflow.models.get_model_info(URI_MODELE).metadata["classes"]
    yield


app = FastAPI(lifespan=lifespan)
entete_cle_api = APIKeyHeader(name="X-API-Key", auto_error=False)


def verifie_cle_api(cle: Annotated[str | None, Security(entete_cle_api)]) -> None:
    if not secrets.compare_digest(cle or "", os.environ["CLE_API"]):
        raise HTTPException(status_code=401, detail="Clé d'API invalide")


# Sondes de Kubernetes : l'API ne répond qu'une fois le modèle chargé par `lifespan`.
@app.get("/sante")
def sante() -> dict[str, str]:
    return {"statut": "ok"}


class Prediction(BaseModel):
    etiquette: str
    probabilites: dict[str, float]


@app.post("/prediction", dependencies=[Depends(verifie_cle_api)])
def predit(fichier: UploadFile) -> Prediction:
    image = Image.open(fichier.file).convert("RGB")
    lot = cree_transformation(TAILLE_IMAGE)(image).unsqueeze(0)
    with torch.inference_mode():
        scores = modeles["classifieur"](lot).softmax(dim=1)[0]
    probabilites = dict(zip(modeles["classes"], scores.tolist(), strict=True))
    etiquette = max(probabilites, key=probabilites.__getitem__)
    return Prediction(etiquette=etiquette, probabilites=probabilites)
```

</details>

### Test de la clé

Relancez le serveur sans clé : il refuse de démarrer. Relancez-le avec une clé, puis appelez-le sans clé, et avec :

    uv run --env-file identifiants.env fastapi dev src/classifieur_paysages/api.py
    CLE_API=une-cle-de-test uv run --env-file identifiants.env fastapi dev src/classifieur_paysages/api.py
    curl -F fichier=@foret1.jpg http://localhost:8000/prediction
    curl -H "X-API-Key: une-cle-de-test" -F fichier=@foret1.jpg http://localhost:8000/prediction

## Création d'une image Docker

Écrivez un `Dockerfile` pour l'API, qui :

- part de l'image `python:3.13-slim` et y copie `uv` ;
- installe les dépendances du projet (celles de `uv.lock`, avec l'extra `fastapi`) avant d'y copier le code, pour que Docker réutilise leur couche quand seul le code change ;
- installe le projet ;
- fait tourner le serveur sous un utilisateur qui n'est pas `root` ;
- lance l'API avec `fastapi run`, sur le port 8000.

Écrivez aussi un `.dockerignore` qui garde hors du contexte de construction les données, l'environnement virtuel et les fichiers d'identifiants (`*.env`).

Construisez puis lancez l'image, les identifiants MLflow dans `identifiants.env`, et appelez l'API :

    docker build -t classifieur-paysages .
    docker run -p 8000:8000 --env-file identifiants.env -e CLE_API=une-cle-de-test classifieur-paysages
    curl -H "X-API-Key: une-cle-de-test" -F fichier=@foret1.jpg http://localhost:8000/prediction

`-p 8000:8000` redirige le port 8000 de l'hôte vers celui du conteneur, `--env-file` passe les identifiants au lancement sans les mettre dans l'image, et `-e CLE_API=...` donne la clé d'API, sans laquelle l'API refuse de démarrer.

**Documentation utile**

- https://docs.docker.com/build/concepts/context/#dockerignore-files

<details>
<summary>Conseils supplémentaires</summary>

`COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /uvx /bin/` copie uv. `uv sync --locked --no-install-project --no-dev --extra fastapi` installe les dépendances seules : il lui faut `pyproject.toml`, `uv.lock` et les fichiers que `pyproject.toml` cite (`README.md`, `LICENSE`). La même commande sans `--no-install-project`, une fois `src/` copié, installe le projet. `useradd --create-home app` crée l'utilisateur, `USER app` le choisit. Chaque ligne d'un `.dockerignore` est un motif de fichiers à exclure.

```dockerfile
FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /uvx /bin/
WORKDIR /app
COPY ...
RUN uv sync ...
COPY src ./src
RUN uv sync ...
RUN ...
USER app
CMD [...]
```

</details>

<details>
<summary>Solution</summary>

```dockerfile
FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /uvx /bin/
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --locked --no-install-project --no-dev --extra fastapi
COPY src ./src
RUN uv sync --locked --no-dev --extra fastapi
RUN useradd --create-home app
USER app
CMD ["uv", "run", "--no-sync", "fastapi", "run", "src/classifieur_paysages/api.py"]
```

`.dockerignore` :

```text
.env
*.env
.git
.venv
mlflow.db
mlruns
test-images
entrainement-images
```

</details>

## Crédits des images

Voir [IMAGES.md](IMAGES.md).
