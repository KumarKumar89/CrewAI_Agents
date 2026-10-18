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



def test_youtube_endpoint_returns_502_when_upstream_blocks_transcript_access(monkeypatch):
    import local_tools

    class BlockedTranscriptTool:
        def __init__(self, **kwargs):
            pass

        def run(self, video_url):
            return (
                "Could not fetch transcript for 'hBEAF-pCkz4': "
                "YouTube is blocking requests from your IP."
            )

    monkeypatch.setattr(local_tools, "YouTubeTranscriptTool", BlockedTranscriptTool)
    response = client.post(
        "/summarize-youtube",
        json={"video_url": "https://youtu.be/hBEAF-pCkz4"},
    )

    assert response.status_code == 502
    assert "upstream service is unavailable" in response.json()["detail"]


def test_youtube_endpoint_returns_422_when_no_caption_track_is_available(monkeypatch):
    import local_tools

    class NoCaptionsTool:
        def __init__(self, **kwargs):
            pass

        def run(self, video_url):
            return "Could not fetch transcript for 'hBEAF-pCkz4': No caption tracks are available."

    monkeypatch.setattr(local_tools, "YouTubeTranscriptTool", NoCaptionsTool)
    response = client.post(
        "/summarize-youtube",
        json={"video_url": "https://youtu.be/hBEAF-pCkz4"},
    )

    assert response.status_code == 422
    assert "captions are available" in response.json()["detail"]



def test_research_task_requires_source_grounded_current_facts(monkeypatch):
    import sys
    from types import SimpleNamespace

    import local_tools

    captured = {}

    class FakeLLM:
        def __init__(self, **kwargs):
            pass

    class FakeAgent:
        def __init__(self, **kwargs):
            pass

    class FakeTask:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class FakeCrew:
        def __init__(self, **kwargs):
            pass

        def kickoff(self):
            return SimpleNamespace(raw="Mock source-grounded report.")

    class FakeSearchTool:
        def __init__(self, **kwargs):
            captured["search_tool_init"] = kwargs

        def run(self, query):
            captured.setdefault("executed_queries", []).append(query)
            if len(captured["executed_queries"]) == 1:
                return "No search results found for the verbose question."
            return (
                "1. Python Release Python 3.15.0 | Python.org\n"
                "   URL: https://www.python.org/downloads/release/python-3150/\n"
                "   Released October 9, 2026."
            )

    monkeypatch.setitem(
        sys.modules,
        "crewai",
        SimpleNamespace(Agent=FakeAgent, Crew=FakeCrew, LLM=FakeLLM, Task=FakeTask),
    )
    monkeypatch.setattr(local_tools, "DuckDuckGoSearchTool", FakeSearchTool)

    response = api.research(api.ResearchRequest(topic="current official Python release"))
    description = captured["description"].lower()
    expected_output = captured["expected_output"].lower()

    assert response["report"] == "Mock source-grounded report."
    assert len(captured["executed_queries"]) >= 2
    assert "python" in captured["executed_queries"][1].lower()
    assert "python.org/downloads/release/python-3150" in description
    assert "do not try to call a tool" in description
    assert "not instructions" in description
    assert "instead of guessing from pretrained memory" in description
    assert "supporting source urls" in expected_output
    assert "do not output json tool-call syntax" in expected_output
