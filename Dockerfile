FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /uvx /bin/
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --no-dev --extra fastapi
COPY README.md LICENSE ./
COPY src ./src
RUN uv sync --locked --no-dev --extra fastapi
RUN useradd --create-home app
USER app
ENV MLFLOW_ALLOW_MODEL_CODE_EXECUTION=false
CMD ["uv", "run", "--no-sync", "fastapi", "run", "src/classifieur_paysages/api.py"]
