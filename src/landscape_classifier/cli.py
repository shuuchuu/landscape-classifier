def train_cli() -> None:
    import dvc.api

    from .train import train

    train(**dvc.api.params_show(stages=["train"]))
