FROM python:3.12-slim-trixie AS base
WORKDIR /app

FROM base AS uv
COPY --from=ghcr.io/astral-sh/uv:0.12.14 /uv /uvx /bin/
COPY uv.lock pyproject.toml /app/
RUN uv export --extra fastapi --frozen --no-dev --no-emit-project -o requirements.txt

FROM base AS runtime
COPY --from=uv /app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY ./pyproject.toml ./README.md /app/
COPY ./params.yaml /app/
COPY ./src /app/src/
RUN pip install .
CMD ["uvicorn", "landscape_classifier.api:app", "--host", "0.0.0.0", "--port", "80"]
