DOIT_CONFIG = {
    "backend": "json",
    "default_tasks": ["check"],
    "dep_file": ".doit.json",
    "verbosity": 2,
}


def task_check():
    return {
        "actions": [
            "uv run ruff check src/classifieur_paysages",
            "uv run ruff format --check src/classifieur_paysages",
            "uv run pyrefly check src/classifieur_paysages",
        ],
    }
