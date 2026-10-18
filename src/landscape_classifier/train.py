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
            # PyTorch's version without its local label (+cu128 on Colab), not on PyPI
            pip_requirements=["torch==" + torch.__version__.split("+")[0]],
        )
