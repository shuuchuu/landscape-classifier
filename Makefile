fastapi:
	uvicorn landscape_classifier.api:app

mlflow:
	mlflow models serve -m "models:/dev.ml.landscape-classifier@champion"

.PHONY: fastapi mlflow
