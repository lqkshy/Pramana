"""API tests for the Pramana fact-checking service."""

from fastapi.testclient import TestClient

from main import app


def test_health_returns_ok():
    """GET /health should return 200 with {"status": "ok", ...}."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"