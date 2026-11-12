"""The API's key, tested without a model nor an MLflow server."""

import pytest
from fastapi.testclient import TestClient

from landscape_classifier.api import app


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("API_KEY", "a-test-key")
    # Without a `with` block, TestClient doesn't run `lifespan`: no model to load
    return TestClient(app)


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"X-API-Key": "another-key"},
        {"X-API-Key": "a-kéy-with-àccents".encode()},
    ],
)
def test_key_refused(client: TestClient, headers: dict) -> None:
    response = client.post("/predict", headers=headers)
    assert response.status_code == 401
