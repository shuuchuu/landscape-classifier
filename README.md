# MLFlow, registre de modèles & déploiement

## Identifiants pour le serveur MLFlow

Créez un fichier `creds.env` qui contient les lignes suivantes&nbsp;:

    export MLFLOW_TRACKING_URI=https://dagshub.com/m09/landscape-classifier.mlflow
    export MLFLOW_TRACKING_USERNAME=username
    export MLFLOW_TRACKING_PASSWORD=password

où `username` et `password` sont les valeurs récupérées sur le site de DagsHub comme pour configurer une remote DVC.

Il faudra utiliser `source creds.env` avant de lancer les commandes et le code qui intéragit avec MLFlow.

## Récupération des données

Utilisez le code suivant pour récupérer les données&nbsp;:

    dvc import https://github.com/m09/dataset-landscape.git seg_train -o train-data

## Publication d'un modèle sur un registre de modèle

Adaptez le fichier `train.py` pour entraîner un modèle en enregistrant les métriques avec [MLFlow Tracking](https://mlflow.org/docs/latest/tracking.html). Les pages sur [Keras](https://mlflow.org/docs/latest/python_api/mlflow.keras.html) de l'API Python MLFlow & de la [flavor Keras](https://mlflow.org/docs/latest/models.html#keras-keras) des modèles MLFlow pourront être utiles.

Publiez ensuite le modèle appris dans le registre de modèles MLFlow en fin d'exécution. [Cette page](https://mlflow.org/docs/latest/models.html#keras-keras) en parle ainsi que [celle-ci](https://mlflow.org/docs/latest/model-registry.html#adding-an-mlflow-model-to-the-model-registry).

Vous pourrez partir du code suivant&nbsp;:

## Création d'une API avec FastAPI

Toutes les questions de cette partie sont à coder dans le fichier `landscape_classifier/api.py`.

### Création d'une fonction de prétraitement adaptée à l'inférence

Créez une fonction pour charger une image dans le même format que celui utilisé pendant l'entraînement. Quel est le point auquel il faut faire extrêmement attention à ce stade&nbsp;?

### Création d'une fonction de chargement de modèle

Créez une fonction qui récupère le modèle entraîné au préalable depuis le registre de modèle MLFlow. Cette fonction prendra en entrée l'URI du modèle à récupérer.

### Création d'une classe de retour FastAPI

FastAPI utilise l'excellente bibliothèque [Pydantic](https://docs.pydantic.dev/latest/) pour gérer les types d'entrée et de sortie des requêtes HTTP.

Créez une classe pour modéliser le type de retour avec la librairie Pydantic. La classe de retour devra contenir a minima la classe prédite (celle avec la probabilité maximale), et un dictionnaire des différentes classes et leur probabilité selon le modèle

### Création de l'API FastAPI

Implémentez l'API d'inférence à l'aide d'une méthode POST qui acceptera un fichier d'image. Vous pourrez vous aider de [cette page de documentation](https://fastapi.tiangolo.com/tutorial/request-files/).

## Création d'un Dockerfile

Écrivez un Dockerfile qui&nbsp;:

- Utilise l'image de base Python 3.12
- Installe `uv` comme montré en cours
- Installe les dépendances du projet
- Copie les fichiers du projet
- Met en place un point d'entrée qui lance le serveur FastAPI

### Construction d'image docker

Pour construire l'image (hors de CodeSpace), on utilise la commande `docker build` et son option `-t` pour préciser un nom et on donne le dossier courant en argument (`.` si on est dans le dossier racine du projet) :

```console
docker build -t shuuchuu/landscape-classifier .
```

Pour exécuter l'image, on utilise `docker run`. Ici avec les options :

- `--rm` pour supprimer le conteneur en fin d'exécution
- `-p 8000:80` pour rediriger le port 8000 de la machine hôte vers le port 80 du conteneur

```console
docker run --rm -p 8000:80 landscape-classifier
```

On peut alors naviguer à [l'adresse par défaut de la documentation FastAPI](https://localhost:8000/docs) pour lire la documentation de l'API et l'utiliser comme client de test.

## Amélioration de l'API

Comment modifier l'API pour qu'elle soit plus adaptée à la mise en production par image Docker ?

*Votre réponse ici.*

## Solution

Pour toutes les questions qui n'ont pas de solution, consulter [la branche solution du dépôt GitHub](https://github.com/shuuchuu/landscape-classifier/tree/solution) de ces travaux pratiques.
