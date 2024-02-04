# Landscape Classifier

Mise en place d'un réseau de reconnaissance de paysages.

## Utilisation de docker

Pour construire l'image, on utilise l'option `-t` pour préciser un nom et on donne le dossier courant en argument (`.` si on est dans le dossier racine du projet) :

```console
docker build -t landscape-classifier .
```

Pour exécuter l'image, on utilise `docker run`. Ici avec les options :

- `--rm` pour supprimer le conteneur en fin d'exécution
- `-p 8000:80` pour rediriger le port 8000 de la machine hôte vers le port 80 du conteneur

```console
docker run --rm -p 8000:80 landscape-classifier
```

On peut alors naviguer à [l'adresse par défaut de la documentation FastAPI](https://localhost:8000/docs) pour lire la documentation de l'API et l'utiliser comme client de test.
