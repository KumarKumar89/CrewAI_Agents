"""FastAPI server exposing the local CrewAI workflows.

Endpoints:
    GET  /                  -> service info
    GET  /health            -> health check (also verifies Ollama reachability)
    POST /research          -> DuckDuckGo-powered research crew (Ollama Llama 3.1)
    POST /summarize-youtube -> YouTube transcript + local Llama summary

Everything runs locally with **no API keys**. Start Ollama first:

    ollama pull llama3.1
    ollama serve

Then run either:

    uvicorn app:app --host 0.0.0.0 --port 8000
    # or, Codespaces-friendly:
    python run_server.py
"""

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "ollama/llama3.1")

app = FastAPI(
    title="CrewAI Local Agents API",
    description=(
        "Local, keyless CrewAI workflows powered by Ollama + Llama 3.1, "
        "DuckDuckGo search, and the YouTube Transcript API."
    ),
    version="1.0.0",
)


class ResearchRequest(BaseModel):
    topic: str = Field(..., description="Topic to research", examples=["local AI agents"])
    todays_date: str = Field(
        default="",
        description="Optional current date string for recency context",
        examples=["October 16, 2026"],
    )


class YouTubeSummaryRequest(BaseModel):
    video_url: str = Field(
        ...,
        description="YouTube video URL to summarize",
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

        resp = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        ollama_ok = resp.status_code == 200
    except Exception:
        ollama_ok = False

    return {
        "status": "ok" if ollama_ok else "degraded",
        "api": "up",
        "ollama": "reachable" if ollama_ok else "unreachable — run `ollama serve`",
        "model": OLLAMA_MODEL,
    }


@app.post("/research")
def research(request: ResearchRequest):
    """Run a two-agent research crew using Ollama/Llama 3.1 + DuckDuckGo."""
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
            expected_output=(
                "A detailed Markdown report with key findings and sources."
            ),
            agent=analyst,
        )

        crew = Crew(agents=[analyst], tasks=[task], memory=False, verbose=True)
        result = crew.kickoff()
        return {"topic": request.topic, "report": result.raw}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Research crew failed: {exc}")


@app.post("/summarize-youtube")
def summarize_youtube(request: YouTubeSummaryRequest):
    """Summarize a YouTube video: transcript fetch + local Llama 3.1."""
    try:
        from local_tools import YouTubeTranscriptTool

        # OLLAMA_MODEL is already LiteLLM-namespaced (e.g. "ollama/llama3.1").
        tool = YouTubeTranscriptTool(model=OLLAMA_MODEL)
        summary = tool.run(request.video_url)
        if summary.startswith("Could not fetch transcript"):
            raise HTTPException(status_code=422, detail=summary)
        return {"video_url": request.video_url, "summary": summary}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"YouTube summary failed: {exc}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
