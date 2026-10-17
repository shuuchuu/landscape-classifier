# Classifieur de paysages : registre de modèles MLflow et déploiement

*[English version](https://github.com/shuuchuu/landscape-classifier/tree/main-en)*

Ce TP suit un modèle de bout en bout : un petit réseau convolutif (LeNet, en PyTorch) qui classe des photos de paysages en 6 classes (bâtiments, forêt, glacier, montagne, mer, rue), dont on suit l'entraînement avec MLflow, que l'on publie dans le registre de modèles de MLflow, puis que l'on sert par une API FastAPI empaquetée dans une image Docker. Il n'a pas besoin de GPU.

Ce README en est l'énoncé, pour un Codespace ou votre poste ; le [notebook du TP](https://colab.research.google.com/github/shuuchuu/labs/blob/main/deployment-mlflow-registry-deployment-hands-on-fr.ipynb) donne le même dans Colab. La [branche `solution`](https://github.com/shuuchuu/landscape-classifier/tree/solution) donne une solution. Le serveur MLflow est celui du dépôt DagsHub [m09/landscape-classifier](https://dagshub.com/m09/landscape-classifier).

## Mise en place de l'environnement

Ouvrez un Codespace du dépôt (<https://codespaces.new/shuuchuu/landscape-classifier>), qui installe le projet et Docker, ou, sur votre poste, avec [uv](https://docs.astral.sh/uv/) :

    git clone https://github.com/shuuchuu/landscape-classifier.git
    cd landscape-classifier
    uv sync --all-extras

Les commandes qui suivent se lancent à la racine du dépôt.

### Identifiants pour le serveur MLflow

Créez un fichier `creds.env` (que git ignore) avec vos identifiants DagsHub, le mot de passe étant un jeton secret, à créer dans les paramètres de votre compte DagsHub, *Tokens*, et non votre mot de passe DagsHub :

    MLFLOW_TRACKING_URI=https://dagshub.com/m09/landscape-classifier.mlflow
    MLFLOW_TRACKING_USERNAME=votre-nom-d-utilisateur
    MLFLOW_TRACKING_PASSWORD=votre-jeton

`uv run --env-file creds.env <commande>` et `docker run --env-file creds.env` les passent aux programmes qui parlent à MLflow.

### Récupération des données

Les images d'entraînement vont dans `train-data/`, celles de test dans `test-data/`, un sous-dossier par classe :

    wget -qO- https://github.com/shuuchuu/datasets/raw/refs/heads/main/ghsparse.sh | bash -s landscape
    mv landscape/seg_train train-data
    mv landscape/seg_test test-data

## Le projet

Le paquet `src/landscape_classifier/` a trois modules :

- `data.py` : la transformation d'une image en tenseur (`get_transform`) et le chargement des images d'un dossier par lots (`get_loader`) ;
- `model.py` : le réseau `LeNet` ;
- `train.py` : une époque d'entraînement (`train_epoch`), l'exactitude d'un modèle (`evaluate`), et la fonction `train`, qui entraîne un modèle puis l'évalue.

La commande `landscape-classifier` appelle `train` avec les paramètres de `params.yaml`, que lit DVC (l'étape `train` de `dvc.yaml`). Lisez `train.py`.

## Suivi de l'entraînement et publication du modèle

Modifiez la fonction `train` de `src/landscape_classifier/train.py` pour qu'un entraînement soit une exécution (*run*) MLflow de l'expérience `landscape-classifier`, qui journalise :

- les hyperparamètres `learning_rate` et `epochs` ;
- la perte moyenne de chaque époque, `train_loss`, avec le numéro de l'époque comme pas (`step`) ;
- l'exactitude sur les images de test, `test_accuracy`.

En fin d'exécution, publiez le modèle dans le registre de modèles sous le nom `dev.ml.landscape-classifier`, avec les noms des classes dans ses métadonnées (`classes`) : l'API les relira.

Modifiez le fichier dans votre éditeur. La section suivante lance l'entraînement.

<details>
<summary>Conseils supplémentaires</summary>

`mlflow.set_experiment("landscape-classifier")` choisit l'expérience (et la crée au besoin), puis tout se passe dans un bloc `with mlflow.start_run():`. `mlflow.log_params` prend un dictionnaire, `mlflow.log_metric(nom, valeur, step=epoch)` une valeur. `mlflow.pytorch.log_model(model, name="model", registered_model_name=..., input_example=..., metadata=...)` enregistre le modèle comme artefact de l'exécution et en crée une nouvelle version dans le registre. MLflow exporte le graphe du modèle en l'exécutant sur `input_example`, un tableau NumPy de la forme d'un lot d'une image : `torch.zeros(1, 3, *IMAGE_SIZE).numpy()`. Les noms des classes sont `train_loader.dataset.classes`. Ajoutez `import mlflow` en tête du fichier ; le reste ne change pas.

```python
def train(train_dir: str, test_dir: str, learning_rate: float, epochs: int) -> None:
    train_loader = get_loader(train_dir, IMAGE_SIZE, shuffle=True)
    test_loader = get_loader(test_dir, IMAGE_SIZE)
    model = LeNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    mlflow.set_experiment("landscape-classifier")
    with mlflow.start_run():
        mlflow.log_params(...)
        for epoch in range(epochs):
            loss = train_epoch(model, train_loader, optimizer)
            mlflow.log_metric(...)
            print(f"Epoch {epoch + 1}/{epochs}: train loss {loss:.3f}")
        accuracy = evaluate(model, test_loader)
        ...
        mlflow.pytorch.log_model(
            model,
            name="model",
            registered_model_name=...,
            input_example=...,
            metadata=...,
        )
```

</details>

<details>
<summary>Solution</summary>

```python
"""Training and evaluation, tracked with MLflow."""

import mlflow
import torch
from torch import nn
from torch.utils.data import DataLoader

from landscape_classifier.data import IMAGE_SIZE, get_loader
from landscape_classifier.model import LeNet

MODEL_NAME = "dev.ml.landscape-classifier"


def _device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def train_epoch(
    model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer
) -> float:
    """Train `model` for one epoch.

    Returns:
        The mean loss over the epoch.
    """
    device = _device(model)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    total, count = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(labels)
        count += len(labels)
    return total / count


def evaluate(model: nn.Module, loader: DataLoader) -> float:
    """Measure `model`'s accuracy on `loader`.

    Returns:
        The share of images whose class `model` predicts right.
    """
    device = _device(model)
    model.eval()
    correct, count = 0, 0
    with torch.inference_mode():
        for images, labels in loader:
            predicted = model(images.to(device)).argmax(dim=1).cpu()
            correct += int((predicted == labels).sum())
            count += len(labels)
    return correct / count


def predict(model: nn.Module, loader: DataLoader) -> torch.Tensor:
    """Predict the class of every image of `loader`.

    Returns:
        The indices of the predicted classes.
    """
    device = _device(model)
    model.eval()
    with torch.inference_mode():
        return torch.cat(
            [model(images.to(device)).argmax(dim=1).cpu() for images, _ in loader]
        )


def train(train_dir: str, test_dir: str, learning_rate: float, epochs: int) -> None:
    """Train a LeNet on `train_dir`'s images and evaluate it on `test_dir`'s.

    The run is tracked in MLflow's `landscape-classifier` experiment, and the
    model published to the registry as a new version of `MODEL_NAME`.
    """
    train_loader = get_loader(train_dir, IMAGE_SIZE, shuffle=True)
    test_loader = get_loader(test_dir, IMAGE_SIZE)
    model = LeNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    mlflow.set_experiment("landscape-classifier")
    with mlflow.start_run():
        mlflow.log_params({"learning_rate": learning_rate, "epochs": epochs})
        for epoch in range(epochs):
            loss = train_epoch(model, train_loader, optimizer)
            mlflow.log_metric("train_loss", loss, step=epoch)
            print(f"Epoch {epoch + 1}/{epochs}: train loss {loss:.3f}")
        accuracy = evaluate(model, test_loader)
        mlflow.log_metric("test_accuracy", accuracy)
        print(f"Test accuracy: {accuracy:.3f}")
        mlflow.pytorch.log_model(
            model,
            name="model",
            registered_model_name=MODEL_NAME,
            input_example=torch.zeros(1, 3, *IMAGE_SIZE).numpy(),
            metadata={"classes": train_loader.dataset.classes},  # ty: ignore[unresolved-attribute]
        )
```

</details>

## Entraînement

La commande `landscape-classifier` lance l'entraînement, 3 époques sur les 14 000 images (environ 7 minutes sur 2 cœurs), avec les identifiants de `creds.env` :

    uv run --env-file creds.env landscape-classifier

Ouvrez ensuite l'interface de MLflow, <https://dagshub.com/m09/landscape-classifier.mlflow> : l'expérience `landscape-classifier` montre votre exécution, ses paramètres et la courbe de `train_loss`, et la page *Models* la nouvelle version de `dev.ml.landscape-classifier`.

## Utilisation du registre de modèles MLflow

Avec l'API Python de MLflow, effectuez les tâches suivantes sur le registre de modèles, dans un script `registry.py` que vous complétez à chaque question et lancez avec `uv run --env-file creds.env python registry.py` (ou dans un notebook, avec l'environnement `.venv` du projet).

### Création d'un client MLflow

Créez un client MLflow (`MlflowClient`) et affichez son adresse de suivi (*tracking URI*), pour vérifier qu'elle vient bien de la variable d'environnement `MLFLOW_TRACKING_URI`.

**Documentation utile**

- https://mlflow.org/docs/latest/api_reference/python_api/mlflow.client.html#mlflow.client.MlflowClient.tracking_uri

<details>
<summary>Conseils supplémentaires</summary>

`mlflow.MlflowClient()` lit l'adresse du serveur dans `MLFLOW_TRACKING_URI`, et l'attribut `tracking_uri` du client la donne. Gardez le nom du modèle dans une constante : les questions suivantes s'en servent.

```python
import mlflow

MODEL_NAME = "dev.ml.landscape-classifier"

client = ...
client.tracking_uri
```

</details>

<details>
<summary>Solution</summary>

```python
import mlflow

MODEL_NAME = "dev.ml.landscape-classifier"

client = mlflow.MlflowClient()
print(client.tracking_uri)
```

</details>

### Recherche de modèles

Affichez tous les modèles du registre et, pour chacun, le numéro de sa version la plus récente. Écrivez pour cela une fonction `latest_version(name)`, que les questions suivantes réutilisent.

**Documentation utile**

- https://mlflow.org/docs/latest/api_reference/python_api/mlflow.client.html#mlflow.client.MlflowClient.search_registered_models

<details>
<summary>Conseils supplémentaires</summary>

`client.search_registered_models()` renvoie les modèles enregistrés (attribut `name`), `client.search_model_versions(f"name = '{name}'")` les versions d'un modèle, dont l'attribut `version` est une chaîne : comparez les versions comme des entiers.

```python
def latest_version(name: str) -> str:
    versions = client.search_model_versions(...)
    return str(max(...))


for registered_model in client.search_registered_models():
    print(registered_model.name, ...)
```

</details>

<details>
<summary>Solution</summary>

```python
def latest_version(name: str) -> str:
    versions = client.search_model_versions(f"name = '{name}'")
    return str(max(int(v.version) for v in versions))


for registered_model in client.search_registered_models():
    print(registered_model.name, latest_version(registered_model.name))
```

</details>

### Récupération d'un modèle depuis le registre

Chargez la dernière version de `dev.ml.landscape-classifier` avec `mlflow.pytorch.load_model`, puis prédisez la classe de l'image `forest1.jpg` : ouvrez-la avec Pillow, transformez-la avec la fonction `get_transform` de `landscape_classifier.data`, celle de l'entraînement, et lisez les noms des classes dans les métadonnées du modèle (`mlflow.models.get_model_info`).

Un modèle MLflow peut avoir plusieurs *flavors*, autant de manières de le charger : affichez celles du vôtre, et chargez-le aussi avec `mlflow.pyfunc.load_model`. Qu'est-ce que cela change ?

**Documentation utile**

- https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.open
- https://mlflow.org/docs/latest/api_reference/python_api/mlflow.pyfunc.html#mlflow.pyfunc.load_model

<details>
<summary>Conseils supplémentaires</summary>

L'URI d'une version est `models:/<nom>/<version>`. `Image.open("forest1.jpg").convert("RGB")` ouvre l'image, `get_transform(IMAGE_SIZE)(image)` en fait un tenseur de forme `(3, 88, 88)`, et `.unsqueeze(0)` un lot d'une image : la seule taille de lot que le modèle chargé accepte, MLflow l'a figée à celle de `input_example`. Le modèle renvoie des scores, que `.softmax(dim=1)` change en probabilités. `get_model_info(uri)` donne les métadonnées (`.metadata["classes"]`) et les flavors (`.flavors`). Le `predict` du modèle `pyfunc` prend un tableau NumPy (`batch.numpy()`).

```python
import torch
from PIL import Image

from landscape_classifier.data import IMAGE_SIZE, get_transform

version = latest_version(MODEL_NAME)
uri = f"models:/{MODEL_NAME}/{version}"
model = mlflow.pytorch.load_model(uri)
info = ...
print(list(info.flavors))

image = Image.open("forest1.jpg").convert("RGB")
batch = ...
with torch.inference_mode():
    probabilities = ...
print(dict(zip(info.metadata["classes"], probabilities.tolist(), strict=True)))

pyfunc_model = ...
print(pyfunc_model.predict(...))
```

</details>

<details>
<summary>Solution</summary>

```python
import torch
from PIL import Image

from landscape_classifier.data import IMAGE_SIZE, get_transform

version = latest_version(MODEL_NAME)
uri = f"models:/{MODEL_NAME}/{version}"
model = mlflow.pytorch.load_model(uri)
info = mlflow.models.get_model_info(uri)
print(list(info.flavors))

image = Image.open("forest1.jpg").convert("RGB")
batch = get_transform(IMAGE_SIZE)(image).unsqueeze(0)
with torch.inference_mode():
    probabilities = model(batch).softmax(dim=1)[0]
print(dict(zip(info.metadata["classes"], probabilities.tolist(), strict=True)))

pyfunc_model = mlflow.pyfunc.load_model(uri)
print(pyfunc_model.predict(batch.numpy()))
```

Le modèle a deux flavors. `pytorch` donne le réseau PyTorch (`mlflow.pytorch.load_model`), qui prend et renvoie des tenseurs. `python_function` est l'interface commune à tous les modèles MLflow (`mlflow.pyfunc.load_model`), dont `predict` prend et renvoie des tableaux NumPy : c'est elle que `mlflow models serve` sert, sans que l'appelant ait besoin de PyTorch. Aucune des deux ne transforme l'image : le prétraitement de `get_transform` reste à faire, à l'identique de l'entraînement.

</details>

### Ajout d'alias et d'étiquettes

Les alias et les étiquettes (*tags*) organisent les modèles du registre. Un alias désigne une version précise, et se déplace d'une version à l'autre ; une étiquette se pose sur un modèle ou sur l'une de ses versions.

Donnez l'alias `champion` à la dernière version de votre modèle, posez l'étiquette `domain` de valeur `cv` sur le modèle et l'étiquette `size` de valeur `small` sur sa dernière version. Chargez ensuite le modèle par son alias plutôt que par son numéro de version, et prédisez à nouveau la classe de `forest1.jpg`.

**Documentation utile**

- https://mlflow.org/docs/latest/api_reference/python_api/mlflow.client.html#mlflow.client.MlflowClient.set_registered_model_tag

<details>
<summary>Conseils supplémentaires</summary>

`client.set_registered_model_alias(nom, alias, version)`, `client.set_registered_model_tag(nom, clé, valeur)` et `client.set_model_version_tag(nom, version, clé, valeur)`. L'URI d'un alias est `models:/<nom>@<alias>`.

```python
client.set_registered_model_alias(MODEL_NAME, ..., version)
client.set_registered_model_tag(MODEL_NAME, ...)
client.set_model_version_tag(MODEL_NAME, version, ...)

model = mlflow.pytorch.load_model(...)
with torch.inference_mode():
    print(model(batch).softmax(dim=1)[0])
```

</details>

<details>
<summary>Solution</summary>

```python
client.set_registered_model_alias(MODEL_NAME, "champion", version)
client.set_registered_model_tag(MODEL_NAME, "domain", "cv")
client.set_model_version_tag(MODEL_NAME, version, "size", "small")
print(client.get_model_version_by_alias(MODEL_NAME, "champion"))

model = mlflow.pytorch.load_model(f"models:/{MODEL_NAME}@champion")
with torch.inference_mode():
    print(model(batch).softmax(dim=1)[0])
```

</details>

### Promotion d'un modèle

Pour promouvoir un modèle d'un environnement au suivant, la méthode recommandée est de copier la version voulue vers le modèle de l'environnement suivant, qui a son propre nom. Copiez la version `champion` de votre modèle vers le modèle `staging.ml.landscape-classifier`, le nom où le préfixe `dev.` devient `staging.`.

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
    src_model_uri=f"models:/{MODEL_NAME}@champion",
    dst_name="staging.ml.landscape-classifier",
)
```

</details>

## Création d'une API avec FastAPI

Écrivez l'API d'inférence dans `src/landscape_classifier/api.py` :

- au démarrage (`lifespan`), elle charge une fois pour toutes le modèle dont l'URI est la variable d'environnement `MODEL_URI`, `models:/dev.ml.landscape-classifier@champion` par défaut, et les noms de ses classes ;
- `GET /health` répond `{"status": "ok"}` : les sondes de Kubernetes l'appellent pour savoir si l'API est prête ;
- `POST /predict` reçoit une image (`UploadFile`), la transforme comme à l'entraînement, et renvoie un objet Pydantic `Prediction` : la classe la plus probable (`label`) et la probabilité de chaque classe (`probabilities`, un dictionnaire).

À quoi faut-il faire extrêmement attention dans le prétraitement ? Testez ensuite l'API (section suivante).

<details>
<summary>Conseils supplémentaires</summary>

Le modèle chargé et les noms des classes vont dans un dictionnaire global (`models`), que `lifespan` remplit avant son `yield`. La fonction de `/predict` est une `def`, pas une `async def` : l'inférence bloque. `Image.open(file.file).convert("RGB")` lit l'image envoyée, et l'inférence se fait sous `torch.inference_mode()`. Les probabilités sont des `float` Python (`.tolist()`), jamais un tenseur.

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

from landscape_classifier.data import IMAGE_SIZE, get_transform

MODEL_URI = os.environ.get("MODEL_URI", "models:/dev.ml.landscape-classifier@champion")
models: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    models["classifier"] = ...
    models["classes"] = ...
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return ...


class Prediction(BaseModel):
    label: str
    probabilities: ...


@app.post("/predict")
def predict(file: UploadFile) -> Prediction:
    image = Image.open(file.file).convert("RGB")
    batch = ...
    with torch.inference_mode():
        scores = ...
    probabilities = ...
    label = ...
    return Prediction(label=label, probabilities=probabilities)
```

</details>

<details>
<summary>Solution</summary>

```python
"""The classifier's HTTP API: `fastapi run src/landscape_classifier/api.py`."""

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import mlflow
import torch
from fastapi import FastAPI, UploadFile
from PIL import Image
from pydantic import BaseModel

from landscape_classifier.data import IMAGE_SIZE, get_transform

MODEL_URI = os.environ.get("MODEL_URI", "models:/dev.ml.landscape-classifier@champion")
models: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    models["classifier"] = mlflow.pytorch.load_model(MODEL_URI)
    models["classes"] = mlflow.models.get_model_info(MODEL_URI).metadata["classes"]
    yield


app = FastAPI(lifespan=lifespan)


# Kubernetes' probes: the API only answers once `lifespan` has loaded the model.
@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


class Prediction(BaseModel):
    label: str
    probabilities: dict[str, float]


@app.post("/predict")
def predict(file: UploadFile) -> Prediction:
    image = Image.open(file.file).convert("RGB")
    batch = get_transform(IMAGE_SIZE)(image).unsqueeze(0)
    with torch.inference_mode():
        scores = models["classifier"](batch).softmax(dim=1)[0]
    probabilities = dict(zip(models["classes"], scores.tolist(), strict=True))
    label = max(probabilities, key=probabilities.__getitem__)
    return Prediction(label=label, probabilities=probabilities)
```

Le prétraitement doit être exactement celui de l'entraînement : l'API importe `get_transform` et `IMAGE_SIZE` de `landscape_classifier.data` au lieu de les réécrire. Une différence (taille, mise à l'échelle des pixels, ordre des canaux) ne lève aucune erreur : elle dégrade les prédictions en silence.

</details>

### Test de l'API

Lancez le serveur avec les identifiants de `creds.env`, puis appelez-le depuis un autre terminal ; sa documentation interactive, qui sert aussi de client de test, est à l'adresse <http://localhost:8000/docs> :

    uv run --env-file creds.env fastapi dev src/landscape_classifier/api.py
    curl http://localhost:8000/health
    curl -F file=@forest1.jpg http://localhost:8000/predict

## Protection de l'API par une clé

Protégez `POST /predict` par une clé d'API : quand la variable d'environnement `API_KEY` est définie, une requête qui n'a pas d'en-tête `X-API-Key` égal à sa valeur reçoit une erreur 401. Sans `API_KEY`, l'API reste ouverte, comme dans la section précédente, et `GET /health` reste toujours ouvert.

<details>
<summary>Conseils supplémentaires</summary>

`APIKeyHeader(name="X-API-Key", auto_error=False)` lit l'en-tête (`None` s'il manque). Une fonction `check_api_key` qui le reçoit par `Security(api_key_header)` lève `HTTPException(status_code=401, detail=...)`, et `@app.post("/predict", dependencies=[Depends(check_api_key)])` l'appelle avant chaque requête. `secrets.compare_digest` compare deux chaînes en un temps qui ne dépend pas de leur contenu. Ajoutez à `api.py` :

```python
import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def check_api_key(key: Annotated[str | None, Security(api_key_header)]) -> None:
    expected = os.environ.get("API_KEY")
    if ...:
        raise HTTPException(...)


@app.post("/predict", dependencies=[...])
def predict(file: UploadFile) -> Prediction: ...
```

</details>

<details>
<summary>Solution</summary>

```python
"""The classifier's HTTP API: `fastapi run src/landscape_classifier/api.py`."""

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

from landscape_classifier.data import IMAGE_SIZE, get_transform

MODEL_URI = os.environ.get("MODEL_URI", "models:/dev.ml.landscape-classifier@champion")
models: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    models["classifier"] = mlflow.pytorch.load_model(MODEL_URI)
    models["classes"] = mlflow.models.get_model_info(MODEL_URI).metadata["classes"]
    yield


app = FastAPI(lifespan=lifespan)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def check_api_key(key: Annotated[str | None, Security(api_key_header)]) -> None:
    expected = os.environ.get("API_KEY")
    if expected and not secrets.compare_digest(key or "", expected):
        raise HTTPException(status_code=401, detail="Invalid API key")


# Kubernetes' probes: the API only answers once `lifespan` has loaded the model.
@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


class Prediction(BaseModel):
    label: str
    probabilities: dict[str, float]


@app.post("/predict", dependencies=[Depends(check_api_key)])
def predict(file: UploadFile) -> Prediction:
    image = Image.open(file.file).convert("RGB")
    batch = get_transform(IMAGE_SIZE)(image).unsqueeze(0)
    with torch.inference_mode():
        scores = models["classifier"](batch).softmax(dim=1)[0]
    probabilities = dict(zip(models["classes"], scores.tolist(), strict=True))
    label = max(probabilities, key=probabilities.__getitem__)
    return Prediction(label=label, probabilities=probabilities)
```

</details>

### Test de la clé

Relancez le serveur avec une clé, puis appelez-le sans clé, et avec :

    API_KEY=une-cle-de-test uv run --env-file creds.env fastapi dev src/landscape_classifier/api.py
    curl -F file=@forest1.jpg http://localhost:8000/predict
    curl -H "X-API-Key: une-cle-de-test" -F file=@forest1.jpg http://localhost:8000/predict

## Création d'une image Docker

Écrivez un `Dockerfile` pour l'API, qui :

- part de l'image `python:3.13-slim` et y copie `uv` ;
- installe les dépendances du projet (celles de `uv.lock`, avec l'extra `fastapi`) avant d'y copier le code, pour que Docker réutilise leur couche quand seul le code change ;
- installe le projet ;
- fait tourner le serveur sous un utilisateur qui n'est pas `root` ;
- lance l'API avec `fastapi run`, sur le port 8000.

Écrivez aussi un `.dockerignore` qui garde hors du contexte de construction les données, l'environnement virtuel et les fichiers d'identifiants (`*.env`).

Construisez puis lancez l'image, les identifiants MLflow dans `creds.env`, et appelez l'API :

    docker build -t landscape-classifier .
    docker run -p 8000:8000 --env-file creds.env -e API_KEY=une-cle-de-test landscape-classifier
    curl -H "X-API-Key: une-cle-de-test" -F file=@forest1.jpg http://localhost:8000/predict

`-p 8000:8000` redirige le port 8000 de l'hôte vers celui du conteneur, `--env-file` passe les identifiants au lancement sans les mettre dans l'image, et `-e API_KEY=...` active la clé d'API.

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
CMD ["uv", "run", "--no-sync", "fastapi", "run", "src/landscape_classifier/api.py"]
```

`.dockerignore` :

```text
.env
*.env
.git
.venv
mlflow.db
mlruns
test-data
train-data
```

</details>

## Crédits des images

Voir [IMAGES.md](IMAGES.md).
