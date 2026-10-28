# Landscape classifier: MLflow model registry & deployment

*[Version française](https://github.com/shuuchuu/landscape-classifier)*

This lab follows a model end to end: a small convolutional network (LeNet, in PyTorch) that classifies landscape pictures into 6 classes (buildings, forest, glacier, mountain, sea, street), whose training is tracked with MLflow, which is published to MLflow's model registry, then served by a FastAPI API packaged in a Docker image. It needs no GPU.

This README is its statement, for a Codespace or your computer; the [lab's notebook](https://colab.research.google.com/github/shuuchuu/labs/blob/main/deployment-mlflow-registry-deployment-hands-on-en.ipynb) gives the same in Colab. The [`solution-en` branch](https://github.com/shuuchuu/landscape-classifier/tree/solution-en) gives a solution. The MLflow server is that of the [m09/landscape-classifier](https://dagshub.com/m09/landscape-classifier) DagsHub repository.

## Setting up the environment

Open a Codespace of the repository on the `main-en` branch (<https://codespaces.new/shuuchuu/landscape-classifier/tree/main-en>), which installs the project and Docker, or, on your computer, with [uv](https://docs.astral.sh/uv/):

    git clone -b main-en https://github.com/shuuchuu/landscape-classifier.git
    cd landscape-classifier
    uv sync --all-extras

The commands below run from the repository's root.

### Credentials for the MLflow server

Create a `creds.env` file (which git ignores) with your DagsHub credentials, the password being a secret token, created in your DagsHub account's settings, *Tokens*, not your DagsHub password:

    MLFLOW_TRACKING_URI=https://dagshub.com/m09/landscape-classifier.mlflow
    MLFLOW_TRACKING_USERNAME=your-username
    MLFLOW_TRACKING_PASSWORD=your-token

`uv run --env-file creds.env <command>` and `docker run --env-file creds.env` pass them to the programs that talk to MLflow.

### Getting the data

The training images go to `train-data/`, the test images to `test-data/`, one subdirectory per class:

    wget -q https://github.com/shuuchuu/datasets/raw/refs/heads/main/ghsparse.sh -O /tmp/ghsparse.sh && bash /tmp/ghsparse.sh landscape
    mv landscape/seg_train train-data
    mv landscape/seg_test test-data

## The project

The `src/landscape_classifier/` package has three modules:

- `data.py`: the transformation of an image into a tensor (`get_transform`) and the loading of a directory's images in batches (`get_loader`);
- `model.py`: the `LeNet` network;
- `train.py`: one training epoch (`train_epoch`), a model's accuracy (`evaluate`), and the `train` function, which trains a model then evaluates it.

The `landscape-classifier` command calls `train` with the parameters of `params.yaml`, which DVC reads (the `train` stage of `dvc.yaml`). Read `train.py`.

## Tracking the training and publishing the model

Change the `train` function of `src/landscape_classifier/train.py` so that a training is an MLflow run of the `landscape-classifier` experiment, which logs:

- the `learning_rate` and `epochs` hyperparameters;
- the mean loss of each epoch, `train_loss`, with the epoch's number as its `step`;
- the accuracy on the test images, `test_accuracy`.

At the end of the run, publish the model to the model registry under the name `dev.ml.landscape-classifier`, with the class names in its metadata (`classes`), which the API will read back, and the installed PyTorch version as its dependency.

Edit the file in your editor. The next section runs the training.

<details>
<summary>Additional hints</summary>

`mlflow.set_experiment("landscape-classifier")` picks the experiment (and creates it if needed), then everything happens in a `with mlflow.start_run():` block. `mlflow.log_params` takes a dictionary, `mlflow.log_metric(name, value, step=epoch)` one value. `mlflow.pytorch.log_model(model, name="model", registered_model_name=..., input_example=..., metadata=..., pip_requirements=...)` logs the model as an artifact of the run and creates a new version of it in the registry. MLflow exports the model's graph by running it on `input_example`, a NumPy array shaped like a batch of one image: `torch.zeros(1, 3, *IMAGE_SIZE).numpy()`. The class names are `train_loader.dataset.classes`. `pip_requirements` takes a list of dependencies in pip's format: `torch.__version__` gives PyTorch's version: remove its local label (`+cu128` on Colab), which PyPI doesn't know. Add `import mlflow` at the top of the file; the rest doesn't change.

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
            pip_requirements=...,
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
            metadata={"classes": train_loader.dataset.classes},  # pyrefly: ignore[missing-attribute]
            # PyTorch's version without its local label (+cu128 on Colab), not on PyPI
            pip_requirements=["torch==" + torch.__version__.split("+")[0]],
        )
```

</details>

## Training

The `landscape-classifier` command runs the training, 3 epochs over the 14,000 images (about 7 minutes on 2 cores), with the credentials of `creds.env`. At the end of the run, MLflow checks that the published model accepts `input_example` by loading it again: `MLFLOW_DEFAULT_PREDICTION_DEVICE=cpu` loads it on the CPU, where it was trained, even on a machine with a GPU:

    MLFLOW_DEFAULT_PREDICTION_DEVICE=cpu uv run --env-file creds.env landscape-classifier

Then open MLflow's interface, <https://dagshub.com/m09/landscape-classifier.mlflow>: the `landscape-classifier` experiment shows your run, its parameters and the `train_loss` curve, and the *Models* page the new version of `dev.ml.landscape-classifier`.

## Using the MLflow model registry

With MLflow's Python API, carry out the following tasks on the model registry, in a `registry.py` script that you extend at each question and run with `uv run --env-file creds.env python registry.py` (or in a notebook, with the project's `.venv` environment).

### Creating an MLflow client

Create an MLflow client (`MlflowClient`) and print its tracking URI, to check that it comes from the `MLFLOW_TRACKING_URI` environment variable.

**Useful documentation**

- https://mlflow.org/docs/latest/api_reference/python_api/mlflow.client.html#mlflow.client.MlflowClient.tracking_uri

<details>
<summary>Additional hints</summary>

`mlflow.MlflowClient()` reads the server's address from `MLFLOW_TRACKING_URI`, and the client's `tracking_uri` attribute gives it. Keep the model's name in a constant: the next questions use it.

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

### Searching models

Print every model of the registry and, for each one, the number of its most recent version. Write a `latest_version(name)` function for it, which the next questions reuse.

**Useful documentation**

- https://mlflow.org/docs/latest/api_reference/python_api/mlflow.client.html#mlflow.client.MlflowClient.search_registered_models

<details>
<summary>Additional hints</summary>

`client.search_registered_models()` returns the registered models (`name` attribute), `client.search_model_versions(f"name = '{name}'")` a model's versions, whose `version` attribute is a string: compare the versions as integers.

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

### Loading a model from the registry

Load the latest version of `dev.ml.landscape-classifier` with `mlflow.pytorch.load_model`, then predict the class of the `forest1.jpg` image: open it with Pillow, transform it with the `get_transform` function of `landscape_classifier.data`, the training's own, and read the class names from the model's metadata (`mlflow.models.get_model_info`).

An MLflow model can have several *flavors*, as many ways to load it: print those of yours, and load it with `mlflow.pyfunc.load_model` too. What does that change?

**Useful documentation**

- https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.open
- https://mlflow.org/docs/latest/api_reference/python_api/mlflow.pyfunc.html#mlflow.pyfunc.load_model

<details>
<summary>Additional hints</summary>

A version's URI is `models:/<name>/<version>`. `Image.open("forest1.jpg").convert("RGB")` opens the image, `get_transform(IMAGE_SIZE)(image)` turns it into a tensor of shape `(3, 88, 88)`, and `.unsqueeze(0)` into a batch of one image: the only batch size the loaded model accepts, MLflow fixed it to that of `input_example`. The model returns scores, which `.softmax(dim=1)` turns into probabilities. `get_model_info(uri)` gives the metadata (`.metadata["classes"]`) and the flavors (`.flavors`). The `pyfunc` model's `predict` takes a NumPy array (`batch.numpy()`). `mlflow.pyfunc.load_model(uri, model_config={"device": "cpu"})` loads it on the CPU, even on a machine with a GPU.

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

pyfunc_model = mlflow.pyfunc.load_model(uri, model_config={"device": "cpu"})
print(pyfunc_model.predict(batch.numpy()))
```

The model has two flavors. `pytorch` gives the PyTorch network (`mlflow.pytorch.load_model`), which takes and returns tensors. `python_function` is the interface common to every MLflow model (`mlflow.pyfunc.load_model`), whose `predict` takes and returns NumPy arrays: it is what `mlflow models serve` serves, without the caller needing PyTorch. Neither transforms the image: `get_transform`'s preprocessing is still to do, exactly as in training. On a GPU, `pyfunc` would load the model there, which a model in MLflow 3's pt2 format (`torch.export`) refuses: it keeps the device it was saved on, here the CPU, hence `model_config={"device": "cpu"}`.

</details>

### Adding aliases and tags

Aliases and tags organize the registry's models. An alias names one precise version, and moves from one version to another; a tag goes on a model or on one of its versions.

Give the `champion` alias to your model's latest version, put the `domain` tag with value `cv` on the model and the `size` tag with value `small` on its latest version. Then load the model by its alias rather than by its version number, and predict the class of `forest1.jpg` again.

**Useful documentation**

- https://mlflow.org/docs/latest/api_reference/python_api/mlflow.client.html#mlflow.client.MlflowClient.set_registered_model_tag

<details>
<summary>Additional hints</summary>

`client.set_registered_model_alias(name, alias, version)`, `client.set_registered_model_tag(name, key, value)` and `client.set_model_version_tag(name, version, key, value)`. An alias's URI is `models:/<name>@<alias>`.

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

### Promoting a model

To promote a model from one environment to the next, the recommended way is to copy the wanted version to the next environment's model, which has its own name. Copy your model's `champion` version to the `staging.ml.landscape-classifier` model, the name where the `dev.` prefix becomes `staging.`.

<details>
<summary>Additional hints</summary>

`client.copy_model_version(src_model_uri=..., dst_name=...)` creates the destination model if it doesn't exist yet; the source is a model URI, here the alias's.

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

## Creating an API with FastAPI

Write the inference API in `src/landscape_classifier/api.py`:

- at startup (`lifespan`), it loads once and for all the model whose URI is the `MODEL_URI` environment variable, `models:/dev.ml.landscape-classifier@champion` by default, and its class names;
- `GET /health` answers `{"status": "ok"}`: Kubernetes' probes call it to know whether the API is ready;
- `POST /predict` receives an image (`UploadFile`), transforms it as in training, and returns a Pydantic `Prediction` object: the most likely class (`label`) and each class's probability (`probabilities`, a dictionary).

What must you be extremely careful about in the preprocessing? Then test the API (next section).

<details>
<summary>Additional hints</summary>

The loaded model and the class names go in a global dictionary (`models`), which `lifespan` fills before its `yield`. The `/predict` function is a `def`, not an `async def`: inference blocks. `Image.open(file.file).convert("RGB")` reads the uploaded image, and inference runs under `torch.inference_mode()`. The probabilities are Python `float`s (`.tolist()`), never a tensor.

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

The preprocessing must be exactly the training's: the API imports `get_transform` and `IMAGE_SIZE` from `landscape_classifier.data` instead of rewriting them. A difference (size, pixel scaling, channel order) raises no error: it silently degrades the predictions.

</details>

### Testing the API

Start the server with the credentials of `creds.env`, then call it from another terminal; its interactive documentation, which is also a test client, is at <http://localhost:8000/docs>:

    uv run --env-file creds.env fastapi dev src/landscape_classifier/api.py
    curl http://localhost:8000/health
    curl -F file=@forest1.jpg http://localhost:8000/predict

## Protecting the API with a key

Protect `POST /predict` with an API key: when the `API_KEY` environment variable is set, a request without an `X-API-Key` header equal to its value gets a 401 error. Without `API_KEY`, the API stays open, as in the previous section, and `GET /health` always stays open.

<details>
<summary>Additional hints</summary>

`APIKeyHeader(name="X-API-Key", auto_error=False)` reads the header (`None` when it's missing). A `check_api_key` function receiving it through `Security(api_key_header)` raises `HTTPException(status_code=401, detail=...)`, and `@app.post("/predict", dependencies=[Depends(check_api_key)])` calls it before each request. `secrets.compare_digest` compares two strings in a time that doesn't depend on their content. Add to `api.py`:

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

### Testing the key

Restart the server with a key, then call it without the key, and with it:

    API_KEY=a-test-key uv run --env-file creds.env fastapi dev src/landscape_classifier/api.py
    curl -F file=@forest1.jpg http://localhost:8000/predict
    curl -H "X-API-Key: a-test-key" -F file=@forest1.jpg http://localhost:8000/predict

## Creating a Docker image

Write a `Dockerfile` for the API, which:

- starts from the `python:3.13-slim` image and copies `uv` into it;
- installs the project's dependencies (those of `uv.lock`, with the `fastapi` extra) before copying the code, so that Docker reuses their layer when only the code changes;
- installs the project;
- runs the server as a user who isn't `root`;
- starts the API with `fastapi run`, on port 8000.

Also write a `.dockerignore` that keeps the data, the virtual environment and the credential files (`*.env`) out of the build context.

Build then run the image, with the MLflow credentials in `creds.env`, and call the API:

    docker build -t landscape-classifier .
    docker run -p 8000:8000 --env-file creds.env -e API_KEY=a-test-key landscape-classifier
    curl -H "X-API-Key: a-test-key" -F file=@forest1.jpg http://localhost:8000/predict

`-p 8000:8000` forwards the host's port 8000 to the container's, `--env-file` passes the credentials at startup without putting them in the image, and `-e API_KEY=...` turns the API key on.

**Useful documentation**

- https://docs.docker.com/build/concepts/context/#dockerignore-files

<details>
<summary>Additional hints</summary>

`COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /uvx /bin/` copies uv. `uv sync --locked --no-install-project --no-dev --extra fastapi` installs the dependencies alone: it needs `pyproject.toml`, `uv.lock` and the files `pyproject.toml` names (`README.md`, `LICENSE`). The same command without `--no-install-project`, once `src/` is copied, installs the project. `useradd --create-home app` creates the user, `USER app` picks it. Each line of a `.dockerignore` is a pattern of files to exclude.

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

`.dockerignore`:

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

## Image credits

See [IMAGES.md](IMAGES.md).
