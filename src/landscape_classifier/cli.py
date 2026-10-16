def train_cli() -> None:
    import dvc.api

    from landscape_classifier.train import train

    train(**dvc.api.params_show(stages=["train"]))
