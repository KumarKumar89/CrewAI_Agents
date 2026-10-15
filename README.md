# Build Agents with CrewAI — Fully Local (No API Keys)

Welcome to the learning guide and tutorial for building intelligent, role-playing AI agents using **CrewAI** — running **entirely locally** with [Ollama](https://ollama.com) + **Llama 3.1**, a keyless **DuckDuckGo search tool**, and the **YouTube Transcript API**. No Gemini keys, no Exa keys, no OpenAI embeddings required.

## Table of Contents

* What is CrewAI?
* Why Local?
* Setup & LLM Configuration
* Defining Agents and Tasks
* Assembling & Running a Crew
* Agent Reasoning
* Tools (DuckDuckGo Search & YouTube Transcripts)
* FastAPI Server
* Notebooks
* Troubleshooting

---

## What is CrewAI?

CrewAI is a Python framework designed for orchestrating role-playing AI agents that collaborate seamlessly on complex tasks.

* **Role-Based Agents:** Each agent features a distinct role, goal, and backstory, making collaboration contextual and natural.
* **Task-Driven Pipeline:** Tasks clearly define what needs to be done, expected outputs, and responsible agents.
* **Flexible LLM Backend:** Plug in any model via LiteLLM — here we use **local models** served by Ollama.
* **Tool Ecosystem:** Equip agents with built-in tools or custom `BaseTool` subclasses (see `local_tools.py`).

---

## Why Local?

The original examples used Gemini models, Exa semantic search, and Gemini video understanding — all requiring paid/external API keys. This repository has been converted to a **keyless, local workflow**:

| Original (API-based) | Replacement (Local / keyless) |
| --- | --- |
| `gemini/gemini-2.5-flash-lite` etc. | `ollama/llama3.1` via Ollama |
| `EXASearchTool` (Exa API key) | `DuckDuckGoSearchTool` (no key) |
| Gemini Google Search grounding (custom LLM) | `DuckDuckGoSearchTool` (no key) |
| Gemini `video/*` summarization (`google-genai`) | `youtube-transcript-api` + local Llama 3.1 |
| Default CrewAI memory (OpenAI embeddings) | `memory=False` on every crew |

All embedded API keys have been removed from the notebooks.

---

## Setup & LLM Configuration

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Install Ollama (https://ollama.com/download), then pull the model
ollama pull llama3.1

# 3. Make sure the Ollama server is running (usually auto-started as a service)
ollama serve
```

Then configure CrewAI's `LLM` wrapper to point at your local model — **no API key needed**:

```python
from crewai import LLM

local_llm = LLM(
    model="ollama/llama3.1",
    base_url="http://localhost:11434",
)
```

> **Important:** Always pass `memory=False` when creating a `Crew`. Otherwise CrewAI falls back to its default embedding provider (OpenAI), which requires an external API key.

---

## Defining Agents and Tasks

```python
from crewai import Agent, Task
from local_tools import DuckDuckGoSearchTool

analyst = Agent(
    role="Research Specialist",
    goal="Conduct detailed research on {topic}",
    backstory="Expert at web research.",
    llm=local_llm,
    tools=[DuckDuckGoSearchTool()],
    reasoning=True,
    max_reasoning_attempts=3,
    verbose=True,
)

analysis_task = Task(
    description="Research and analyze recent trends and generate a detailed report on {topic}.",
    expected_output="A detailed executive report for C-suite readers.",
    agent=analyst,
    verbose=True,
)
```

---

## Assembling & Running a Crew

```python
from crewai import Crew

crew = Crew(
    agents=[analyst],
    tasks=[analysis_task],
    memory=False,   # avoids OpenAI embedding fallback — keeps everything local
    verbose=True,
)

result = crew.kickoff(inputs={
    "topic": "Influencer marketing",
    "todays_date": "July 31, 2025",
})

print(result.raw)
```

---

## Agent Reasoning

Enabling `reasoning=True` prompts the agent to reflect and plan before taking action.

* **Receive Task:** The agent intakes the objective.
* **Reflect & Plan:** Generates an internal plan outlining necessary information and steps (similar to built-in Chain-of-Thought prompting).
* **Execute & Return:** Executes the plan and returns the final result.

Note: reasoning works best with capable models; Llama 3.1 handles it well locally.

---

## Tools

`local_tools.py` provides two keyless CrewAI tools:

* **`DuckDuckGoSearchTool`** — web search via the free `duckduckgo-search` package. Replaces both Exa search and Gemini Google Search grounding.
* **`YouTubeTranscriptTool`** — fetches a video's transcript with `youtube-transcript-api` and summarizes it locally with Ollama/Llama 3.1. Replaces Gemini video summarization.

```python
from local_tools import DuckDuckGoSearchTool, YouTubeTranscriptTool

print(DuckDuckGoSearchTool().run("local AI agents"))
print(YouTubeTranscriptTool().run("https://www.youtube.com/watch?v=hBEAF-pCkz4"))
```

---

## FastAPI Server

`app.py` exposes the local workflows as HTTP endpoints, and `run_server.py` is a Codespaces-friendly runner.

### Endpoints

| Method | Path | Description |
| --- | --- | --- |
| GET | `/` | Service info |
| GET | `/health` | Health check (verifies Ollama reachability) |
| POST | `/research` | Run the DuckDuckGo + Llama 3.1 research crew |
| POST | `/summarize-youtube` | Summarize a YouTube video locally |
| GET | `/docs` | Swagger UI |

### Start in GitHub Codespaces

```bash
python -m pip install -r requirements.txt

ollama pull llama3.1
ollama serve

python run_server.py
```

The runner:

* Binds to `0.0.0.0`
* Uses port `8000` by default (or the `PORT` environment variable)
* Enables reload mode by default (set `RELOAD=false` to disable)
* Makes the API available through the Codespaces forwarded port

Open `/docs` for Swagger API documentation.

### Example requests

```bash
curl http://localhost:8000/health
```

```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"topic":"local AI agents"}'
```

```bash
curl -X POST http://localhost:8000/summarize-youtube \
  -H "Content-Type: application/json" \
  -d '{"video_url":"https://www.youtube.com/watch?v=hBEAF-pCkz4"}'
```

---

## Notebooks

| Notebook | Purpose |
| --- | --- |
| `crewai_reasoning (1).ipynb` | Agent reasoning with a local LLM + DuckDuckGo search crew |
| `gemini_search_grounding_crewai (1).ipynb` | Keyless web-search agent (Ollama + DuckDuckGo replaces Gemini grounding) |
| `summarize_video_gemini_crewai.ipynb` | YouTube transcript + local Llama 3.1 video summaries |

Run them after installing `requirements.txt` and starting Ollama.

---

## Troubleshooting

* **`Connection error` / model calls hang** — Ollama isn't running. Start it with `ollama serve`, and verify with `curl http://localhost:11434/api/tags`.
* **`model not found: llama3.1`** — run `ollama pull llama3.1` first.
* **OpenAI embedding/API-key errors** — make sure every `Crew(...)` uses `memory=False`.
* **DuckDuckGo rate limits** — reduce `max_results` or add a short delay between searches.
* **Transcript fetch fails** — the video may have no captions/subtitles; try another video or set `language=` on `YouTubeTranscriptTool`.
* **Slow responses** — Llama 3.1 (8B) needs ~5 GB RAM/VRAM; close other apps or use a smaller model such as `ollama/llama3.2:3b` (set `OLLAMA_MODEL`).
* **Codespaces port not visible** — confirm the server binds to `0.0.0.0` (the default in `run_server.py`) and check the *Ports* tab for forwarded port 8000.
