"""FastAPI server exposing the local CrewAI workflows.

Endpoints:
    GET  /                  -> service info
    GET  /health            -> health check (also verifies Ollama reachability)
    POST /research          -> DuckDuckGo-powered research crew (Ollama Llama 3.1)
    POST /summarize-youtube -> YouTube transcript + local Llama summary

Everything runs locally with no hosted LLM API keys. Start Ollama first:
    ollama pull llama3.1
    ollama serve
"""

import logging
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "ollama/llama3.1")

app = FastAPI(
    title="CrewAI Local Agents API",
    description=(
        "Local, keyless CrewAI workflows powered by Ollama + Llama 3.1, "
        "DuckDuckGo search, and the YouTube Transcript API."
    ),
    version="1.0.1",
)


class ResearchRequest(BaseModel):
    topic: str = Field(
        ...,
        min_length=1,
        max_length=300,
        description="Topic to research (1–300 characters)",
        examples=["local AI agents"],
    )
    todays_date: str = Field(
        default="",
        max_length=40,
        description="Optional current date string for recency context",
        examples=["October 16, 2026"],
    )


class YouTubeSummaryRequest(BaseModel):
    video_url: str = Field(
        ...,
        min_length=1,
        max_length=2048,
        description="YouTube video URL or 11-character video ID",
        examples=["https://www.youtube.com/watch?v=hBEAF-pCkz4"],
    )


@app.get("/")
def root():
    return {
        "service": "CrewAI Local Agents API",
        "model": OLLAMA_MODEL,
        "ollama_base_url": OLLAMA_BASE_URL,
        "endpoints": {
            "GET /health": "Health check",
            "POST /research": "Run a local web-research crew (DuckDuckGo + Llama 3.1)",
            "POST /summarize-youtube": "Summarize a YouTube video locally",
            "GET /docs": "Swagger UI",
        },
    }


@app.get("/health")
def health():
    """Check that the API is up and Ollama is reachable."""
    ollama_ok = False
    try:
        import httpx

        response = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        ollama_ok = response.status_code == 200
    except Exception:
        logger.warning("Ollama health check failed for configured endpoint %s", OLLAMA_BASE_URL)

    payload = {
        "status": "ok" if ollama_ok else "degraded",
        "api": "up",
        "ollama": "reachable" if ollama_ok else "unreachable — run ollama serve",
        "model": OLLAMA_MODEL,
    }
    # Return a failing HTTP status when the model provider is unavailable so
    # readiness probes do not treat a degraded dependency as fully healthy.
    return JSONResponse(status_code=200 if ollama_ok else 503, content=payload)


@app.post("/research")
def research(request: ResearchRequest):
    """Run a research crew using the configured Ollama model and DuckDuckGo."""
    try:
        from crewai import Agent, Crew, LLM, Task

        from local_tools import DuckDuckGoSearchTool

        llm = LLM(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
        search_tool = DuckDuckGoSearchTool()

        analyst = Agent(
            role="Research Specialist",
            goal=f"Conduct detailed research on {request.topic}",
            backstory="You are an expert at web research.",
            llm=llm,
            tools=[search_tool],
            verbose=True,
        )

        task = Task(
            description=(
                f"Research and analyze recent trends about {request.topic}"
                + (f" as of {request.todays_date}" if request.todays_date else "")
                + " using the web search tool."
            ),
            expected_output="A detailed Markdown report with key findings and sources.",
            agent=analyst,
        )

        crew = Crew(agents=[analyst], tasks=[task], memory=False, verbose=True)
        result = crew.kickoff()
        return {"topic": request.topic, "report": result.raw}
    except HTTPException:
        raise
    except Exception:
        # Keep internal paths, provider details, and exception text out of API responses.
        logger.exception("Research crew execution failed")
        raise HTTPException(
            status_code=502,
            detail="Research crew failed. Check server logs for diagnostic details.",
        ) from None


@app.post("/summarize-youtube")
def summarize_youtube(request: YouTubeSummaryRequest):
    """Summarize a YouTube video: transcript fetch + local Ollama model."""
    try:
        from local_tools import YouTubeTranscriptTool

        tool = YouTubeTranscriptTool(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
        summary = tool.run(request.video_url)
        if summary.startswith("Could not fetch transcript"):
            raise HTTPException(status_code=422, detail=summary)
        return {"video_url": request.video_url, "summary": summary}
    except HTTPException:
        raise
    except Exception:
        logger.exception("YouTube summary execution failed")
        raise HTTPException(
            status_code=502,
            detail="YouTube summarization failed. Check server logs for diagnostic details.",
        ) from None


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
