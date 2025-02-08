"""Fonctions utilitaires."""

from collections.abc import Iterable


def get_pip_requirements_from_uv(extras: Iterable[str] | None = None) -> list[str]:
    from subprocess import run

    command = [
        "uv",
        "export",
        "--no-dev",
        "--no-emit-project",
        "--no-header",
        "--no-hashes",
    ]
    if extras is not None:
        for extra in extras:
            command.extend(["--extra", extra])
    result = run(command, capture_output=True)
    return [x for x in result.stdout.decode("utf8").split("\n") if x]
