"""Regression tests for the local tools. All external services are mocked."""

import sys
from types import SimpleNamespace

import pytest

from local_tools import YouTubeTranscriptTool, extract_video_id, fetch_transcript


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("hBEAF-pCkz4", "hBEAF-pCkz4"),
        ("https://www.youtube.com/watch?v=hBEAF-pCkz4", "hBEAF-pCkz4"),
        ("https://youtu.be/hBEAF-pCkz4?t=30", "hBEAF-pCkz4"),
        ("https://youtube.com/shorts/hBEAF-pCkz4", "hBEAF-pCkz4"),
        ("https://www.youtube.com/embed/hBEAF-pCkz4", "hBEAF-pCkz4"),
        ("https://youtube.com/live/hBEAF-pCkz4", "hBEAF-pCkz4"),
    ],
)
def test_extract_video_id_accepts_supported_formats(value, expected):
    assert extract_video_id(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "",
        "not a video",
        "https://example.com/watch?v=hBEAF-pCkz4",
        "https://youtube.com/watch?v=bad",
        "https://youtube.com/channel/UC123",
    ],
)
def test_extract_video_id_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        extract_video_id(value)


def test_fetch_transcript_falls_back_to_first_available_language(monkeypatch):
    class Snippet:
        def __init__(self, text):
            self.text = text

    class Fetched:
        def __iter__(self):
            return iter([Snippet("Hola"), Snippet("mundo")])

    class Track:
        def fetch(self):
            return Fetched()

    class TranscriptList:
        def __iter__(self):
            return iter([Track()])

    class MockYouTubeTranscriptApi:
        def fetch(self, video_id, languages):
            assert video_id == "hBEAF-pCkz4"
            assert languages == ["en"]
            raise RuntimeError("English captions unavailable")

        def list(self, video_id):
            assert video_id == "hBEAF-pCkz4"
            return TranscriptList()

    monkeypatch.setitem(
        sys.modules,
        "youtube_transcript_api",
        SimpleNamespace(YouTubeTranscriptApi=MockYouTubeTranscriptApi),
    )

    assert fetch_transcript("https://youtu.be/hBEAF-pCkz4") == "Hola mundo"


def test_fetch_transcript_returns_clear_error_for_invalid_url():
    result = fetch_transcript("https://example.com/watch?v=hBEAF-pCkz4")
    assert result.startswith("Could not fetch transcript:")
    assert "Expected a YouTube URL" in result


def test_youtube_summarizer_uses_configured_ollama_url(monkeypatch):
    import crewai
    import local_tools

    monkeypatch.setattr(local_tools, "fetch_transcript", lambda *args, **kwargs: "A short transcript.")
    captured = {}

    class FakeLLM:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def call(self, prompt):
            assert "A short transcript." in prompt
            return "Summary"

    monkeypatch.setattr(crewai, "LLM", FakeLLM)

    tool = YouTubeTranscriptTool(
        model="ollama/llama3.1",
        base_url="http://ollama.internal:11434",
    )
    assert tool._run("https://youtu.be/hBEAF-pCkz4") == "Summary"
    assert captured == {
        "model": "ollama/llama3.1",
        "base_url": "http://ollama.internal:11434",
    }
