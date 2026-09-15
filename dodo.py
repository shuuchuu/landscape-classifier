DOIT_CONFIG = {
    "backend": "json",
    "default_tasks": ["check"],
    "dep_file": ".doit.json",
    "verbosity": 2,
}


def task_check():
    return {
        "actions": [
            "uv run ruff check src/landscape_classifier",
            "uv run ruff format --check src/landscape_classifier",
            "uv run ty check src/landscape_classifier",
        ],
    }


def task_fastapi():
    return {
        "actions": [
            "uv run uvicorn landscape_classifier.api:app",
        ],
    }


def task_mlflow():
    return {
        "actions": [
            'uv run mlflow models serve -m "models:/dev.ml.landscape-classifier@champion" --env-manager uv',
        ],
    }


def task_docker():
    return {
        "actions": [
            "docker build -t shuuchuu/landscape-classifier .",
            "docker run --rm -it -e MLFLOW_TRACKING_URI -e MLFLOW_TRACKING_USERNAME "
            "-e MLFLOW_TRACKING_PASSWORD -p 8000:80 "
            "shuuchuu/landscape-classifier:latest",
        ]
    }
