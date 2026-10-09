# Landscape classifier: MLflow model registry & deployment

The project of the *MLflow model registry & deployment* lab, which trains a small image
classifier, publishes it to an MLflow model registry, serves it with FastAPI and packages
it with Docker. The lab's instructions are in its notebook, in
[French](https://colab.research.google.com/github/shuuchuu/labs/blob/main/deployment-mlflow-registry-deployment-hands-on-fr.ipynb)
or in
[English](https://colab.research.google.com/github/shuuchuu/labs/blob/main/deployment-mlflow-registry-deployment-hands-on-en.ipynb).

## MLflow server credentials

Create a `creds.env` file with the following lines:

    export MLFLOW_TRACKING_URI=https://dagshub.com/m09/landscape-classifier.mlflow
    export MLFLOW_TRACKING_USERNAME=username
    export MLFLOW_TRACKING_PASSWORD=password

where `username` and `password` are the values DagsHub gives you, as when setting up a
DVC remote. Run `source creds.env` before running anything that talks to MLflow.

## Data

    dvc import https://github.com/shuuchuu/datasets.git landscape/seg_train -o train-data

## Solution

The [`solution-en` branch](https://github.com/shuuchuu/landscape-classifier/tree/solution-en)
holds a solution to every question.
