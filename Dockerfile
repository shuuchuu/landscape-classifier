FROM python:3.10

WORKDIR /api

COPY ./requirements.txt /api/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /api/requirements.txt

COPY ./landscape_classifier/api.py api.py

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "80"]
