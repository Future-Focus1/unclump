"""Tests for the Unstuck FastAPI application."""

import pytest
from fastapi.testclient import TestClient
from main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "unstuck"


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Unstuck API"


def test_breakdown_endpoint_uses_fallback(client):
    """Without OPENAI_API_KEY, should use fallback templates."""
    response = client.post("/api/breakdown", json={"task": "clean my room"})
    assert response.status_code == 200
    data = response.json()
    assert data["original_task"] == "clean my room"
    assert len(data["micro_steps"]) > 0
    assert data["micro_steps"][0]["is_entry_point"] is True
    assert isinstance(data["total_estimated_minutes"], (int, float))
    assert len(data["entry_hook"]) > 0


def test_breakdown_rejects_empty_task(client):
    response = client.post("/api/breakdown", json={"task": ""})
    assert response.status_code == 422  # validation error


def test_breakdown_rejects_missing_task(client):
    response = client.post("/api/breakdown", json={})
    assert response.status_code == 422


def test_breakdown_trims_whitespace(client):
    response = client.post("/api/breakdown", json={"task": "  do the thing  "})
    assert response.status_code == 200
    data = response.json()
    assert data["original_task"] == "do the thing"


def test_breakdown_handles_various_tasks(client):
    tasks = [
        "send an email to my boss about the project timeline",
        "I need to work out but I can't get off the couch",
        "something completely random and unknown",
    ]
    for task in tasks:
        response = client.post("/api/breakdown", json={"task": task})
        assert response.status_code == 200
        data = response.json()
        assert len(data["micro_steps"]) > 0
        assert all(s["step"] > 0 for s in data["micro_steps"])
        assert all(10 <= s["estimated_seconds"] <= 120 for s in data["micro_steps"])
