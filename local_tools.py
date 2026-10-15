"""Keyless CrewAI tools for a fully local setup.

These tools replace the Gemini/Exa-based tools used in the original examples:

- ``DuckDuckGoSearchTool``: replaces Exa search / Gemini Google Search
  grounding with the free, no-API-key ``duckduckgo_search`` package.
- ``YouTubeTranscriptTool``: replaces Gemini video summarization by pulling
  the public transcript of a YouTube video (via ``youtube-transcript-api``)
  and summarizing it locally with an Ollama-hosted Llama 3.1 model.

No API keys are required for any of these tools.
"""

import re

from crewai.tools import BaseTool
from duckduckgo_search import DDGS


class DuckDuckGoSearchTool(BaseTool):
    """Web search using DuckDuckGo — no API key required."""

    name: str = "duckduckgo_search"
    description: str = (
        "Search the web for up-to-date information. "
        "Input should be a plain-text search query string."
    )

    max_results: int = 5

    def _run(self, query: str) -> str:
        try:
            results = list(DDGS().text(query, max_results=self.max_results))
        except Exception as exc:  # pragma: no cover - network errors
            return f"DuckDuckGo search failed: {exc}"

        if not results:
            return f"No search results found for '{query}'."

        lines = []
        for i, r in enumerate(results, start=1):
            title = r.get("title", "").strip()
            href = r.get("href", "").strip()
            body = r.get("body", "").strip()
            lines.append(f"{i}. {title}\n   URL: {href}\n   {body}")
        return "\n\n".join(lines)


def extract_video_id(url_or_id: str) -> str:
    """Extract a YouTube video id from a URL or return the id unchanged."""

    url_or_id = url_or_id.strip()
    patterns = [
        r"(?:v=|/shorts/|/embed/|/live/)([A-Za-z0-9_-]{11})",
        r"^([A-Za-z0-9_-]{11})$",
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
    return url_or_id


def fetch_transcript(video_url: str, language: str = "en") -> str:
    """Fetch the transcript text of a YouTube video (no API key needed)."""

    from youtube_transcript_api import YouTubeTranscriptApi

    video_id = extract_video_id(video_url)
    api = YouTubeTranscriptApi()
    try:
        fetched = api.fetch(video_id, languages=[language])
    except Exception:
        # Fall back to whatever transcripts are available.
        try:
            transcript_list = api.list(video_id)
            transcript = next(iter(transcript_list.find_transcripts([])))
            fetched = transcript.fetch()
        except Exception as exc:
            return f"Could not fetch transcript for '{video_id}': {exc}"

    return " ".join(snippet.text for snippet in fetched)


class YouTubeTranscriptTool(BaseTool):
    """Summarize a YouTube video locally: transcript + Ollama Llama 3.1."""

    name: str = "summarize_youtube_video"
    description: str = (
        "Summarize a YouTube video. Input should be a YouTube video URL "
        "(e.g. https://www.youtube.com/watch?v=...). The transcript is "
        "fetched from YouTube and summarized locally with Ollama/Llama 3.1."
    )

    model: str = "ollama/llama3.1"
    language: str = "en"

    def _run(self, video_url: str) -> str:
        transcript = fetch_transcript(video_url, language=self.language)
        if transcript.startswith("Could not fetch transcript"):
            return transcript

        # Keep the prompt within reasonable context limits.
        transcript_excerpt = transcript[:8000]

        from crewai import LLM

        llm = LLM(model=self.model)
        prompt = (
            "You are a video analyst. Summarize the following YouTube video "
            "transcript in Markdown. Include a short overview, key points, "
            "and action items if present.\n\n"
            f"Transcript:\n{transcript_excerpt}"
        )
        return llm.call(prompt)


if __name__ == "__main__":
    print(DuckDuckGoSearchTool().run("local AI agents"))
    print(
        YouTubeTranscriptTool().run(
            "https://www.youtube.com/watch?v=hBEAF-pCkz4"
        )
    )
