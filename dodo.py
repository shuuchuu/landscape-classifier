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
