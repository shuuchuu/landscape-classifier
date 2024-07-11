FROM python:3.10 AS base

WORKDIR /app

FROM base AS poetry

RUN pip install --no-cache-dir 'poetry == 1.8.2'

COPY poetry.lock pyproject.toml /app/

RUN poetry export -o requirements.txt

FROM base AS runtime

COPY --from=poetry /app/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY ./pyproject.toml ./README.md /app/

COPY ./params.yaml /app/

COPY ./landscape_classifier /app/landscape_classifier

RUN pip install .

CMD ["uvicorn", "landscape_classifier.api:app", "--host", "0.0.0.0", "--port", "80"]
