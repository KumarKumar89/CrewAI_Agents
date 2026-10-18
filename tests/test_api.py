"""FastAPI endpoint checks that do not require a running Ollama service."""

from types import SimpleNamespace

import httpx
from fastapi.testclient import TestClient

import app as api


client = TestClient(api.app)


def test_root_endpoint_reports_service_and_routes():
    response = client.get("/")
    assert response.status_code == 200
    payload = response.json()
    assert payload["service"] == "CrewAI Local Agents API"
    assert "POST /research" in payload["endpoints"]
    assert "POST /summarize-youtube" in payload["endpoints"]


def test_health_is_200_when_ollama_is_reachable(monkeypatch):
    monkeypatch.setattr(
        httpx,
        "get",
        lambda *args, **kwargs: SimpleNamespace(status_code=200),
    )
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["ollama"] == "reachable"


def test_health_is_503_when_ollama_is_unreachable(monkeypatch):
    def fail_request(*args, **kwargs):
        raise httpx.ConnectError("simulated offline Ollama")

    monkeypatch.setattr(httpx, "get", fail_request)
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["status"] == "degraded"
    assert response.json()["api"] == "up"


def test_youtube_endpoint_passes_configured_model_and_base_url(monkeypatch):
    import local_tools

    captured = {}

    class FakeTranscriptTool:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self, video_url):
            return "Mock summary"

    monkeypatch.setattr(local_tools, "YouTubeTranscriptTool", FakeTranscriptTool)
    response = client.post(
        "/summarize-youtube",
        json={"video_url": "https://youtu.be/hBEAF-pCkz4"},
    )

    assert response.status_code == 200
    assert response.json()["summary"] == "Mock summary"
    assert captured == {
        "model": api.OLLAMA_MODEL,
        "base_url": api.OLLAMA_BASE_URL,
    }


def test_invalid_research_request_is_rejected_before_work_starts():
    response = client.post("/research", json={"topic": ""})
    assert response.status_code == 422
