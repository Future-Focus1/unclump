"""Tests for the Unclump FastAPI application."""

import pytest
from fastapi.testclient import TestClient
import main
from main import app


@pytest.fixture(autouse=True)
def isolate_session_store(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "SESSIONS_FILE", tmp_path / "sessions.json")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)


@pytest.fixture
def client():
    return TestClient(app)


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "unclump"


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Unclump API"


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


def test_session_start_uses_fallback(client):
    response = client.post(
        "/api/session/start",
        json={"task": "I am too exhausted to clean my room", "energy_level": "low"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"]
    assert data["status"] == "active"
    assert data["block_type"] == "low_energy"
    assert data["current_step"]["is_entry_point"] is True
    assert len(data["feedback_options"]) > 0


def test_session_start_finance_includes_safety_note(client):
    response = client.post(
        "/api/session/start",
        json={"task": "invest in the stock market"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["task_domain"] == "finance"
    assert data["task_intent"] == "finance_setup"
    assert data["safety_note"]
    assert "Not financial advice" in data["safety_note"]


def test_session_feedback_adapts_current_step(client):
    start = client.post("/api/session/start", json={"task": "reply to an email"}).json()
    response = client.post(
        f"/api/session/{start['session_id']}/feedback",
        json={"feedback": "too_hard"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "active"
    assert data["current_step_index"] == 0
    assert data["current_step"]["adapted_for"] == "too_hard"
    assert data["nudge"]


def test_session_feedback_done_advances(client):
    start = client.post("/api/session/start", json={"task": "send an email"}).json()
    response = client.post(
        f"/api/session/{start['session_id']}/feedback",
        json={"feedback": "done"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["completed_steps"] == 1
    assert data["current_step_index"] == 1


def test_support_endpoint_uses_fallback(client):
    response = client.post(
        "/api/support",
        json={
            "task": "reply to an email",
            "moment": "nudge",
            "user_state": "overwhelmed",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["message"]
    assert data["suggested_action"]
    assert data["reminder_after_minutes"] > 0


def test_coworking_chat_endpoint_uses_fallback(client):
    response = client.post(
        "/api/coworking/chat",
        json={
            "task": "reply to an email",
            "mode": "reply",
            "session_minutes": 30,
            "coworkers": [
                {"name": "Maya", "task": "sorting receipts", "quirk": "uses_2"},
                {"name": "Sam", "task": "opening a draft", "quirk": "lol"},
            ],
            "recent_messages": [],
            "user_message": "I'm stuck, what should I do?",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert 1 <= len(data["messages"]) <= 2
    assert all(message["sender"] for message in data["messages"])
    assert all(len(message["text"].split()) <= 30 for message in data["messages"])
