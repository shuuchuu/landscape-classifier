def train_cli() -> None:
    import dvc.api

    from landscape_classifier.train import train

    train(**dvc.api.params_show(stages=["train"]))


def test_api_cli() -> None:
    import sys
    from base64 import b64encode
    from pathlib import Path

    from niquests import HTTPError, post

    def prepare(image_name: str) -> str:
        return b64encode(
            (Path(__file__).parent.parent.parent / image_name).read_bytes()
        ).decode()

    response = post(
        "http://127.0.0.1:5000/invocations",
        headers={"Content-Type": "application/json"},
        json={"inputs": [prepare("forest1.jpg"), prepare("forest2.jpg")]},
    )
    try:
        response.raise_for_status()
    except HTTPError as e:
        print(e)
        print(response.content)
        sys.exit(1)
    print(response.json())
