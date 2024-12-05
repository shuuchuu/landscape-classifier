import dvc.api

from .train import train


def main() -> None:
    train(**dvc.api.params_show(stages=["train"]))
