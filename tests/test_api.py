"""La clé d'API de l'API, testée sans modèle ni serveur MLflow."""

import pytest
from fastapi.testclient import TestClient

from classifieur_paysages.api import app


@pytest.fixture
def client_test(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("CLE_API", "une-cle-de-test")
    # Sans bloc `with`, TestClient n'exécute pas `lifespan` : aucun modèle à charger
    return TestClient(app)


@pytest.mark.parametrize(
    "entetes",
    [
        {},
        {"X-API-Key": "une-autre-cle"},
        {"X-API-Key": "une-clé-accentuée".encode()},
    ],
)
def test_cle_refusee(client_test: TestClient, entetes: dict) -> None:
    reponse = client_test.post("/prediction", headers=entetes)
    assert reponse.status_code == 401
