"""Tests for the health and readiness endpoints."""
from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(app)


def test_health_returns_200_and_healthy():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"


def test_readiness_returns_200_and_ready():
    response = client.get("/readiness")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
