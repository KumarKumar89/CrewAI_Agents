"""Real-service smoke test run only in the dedicated live E2E GitHub Actions job.

Requires a running Ollama server with the configured model, network access for
DuckDuckGo and YouTube, and the FastAPI service started on localhost.
"""

import json
import os
import sys
import time
import traceback
from datetime import date
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RESULT_PATH = Path(os.getenv("E2E_RESULT_PATH", "live-e2e-results.json"))
API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "ollama/llama3.1").removeprefix("ollama/")
results = {"date": date.today().isoformat(), "model": MODEL_NAME, "checks": []}


def check(name, fn):
    item = {"name": name, "status": "running"}
    results["checks"].append(item)
    try:
        detail = fn()
        item.update(status="passed", detail=detail)
        print(f"PASS: {name} — {detail}", flush=True)
        return detail
    except Exception as exc:
        item.update(status="failed", error=str(exc), traceback=traceback.format_exc()[-5000:])
        print(f"FAIL: {name} — {exc}", file=sys.stderr, flush=True)
        raise
    finally:
        RESULT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")


def json_request(url, payload=None, timeout=300):
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"} if body is not None else {},
        method="GET" if body is None else "POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            return response.status, json.loads(raw) if raw else {}
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {raw[:1200]}") from exc
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(f"Request to {url} failed: {exc}") from exc


def ollama_inference():
    status, payload = json_request(
        f"{OLLAMA_BASE}/api/generate",
        {"model": MODEL_NAME, "prompt": "Reply exactly with OLLAMA_READY.", "stream": False,
         "options": {"num_predict": 24}},
        timeout=240,
    )
    text = str(payload.get("response", "")).strip()
    if status != 200 or "OLLAMA_READY" not in text:
        raise RuntimeError(f"Unexpected Ollama generation response: {text[:400]}")
    return f"model={MODEL_NAME}; response={text[:100]!r}"


def direct_search():
    from local_tools import DuckDuckGoSearchTool

    text = DuckDuckGoSearchTool(max_results=3).run("Python official latest release python.org")
    if "DuckDuckGo search failed:" in text or "No search results found" in text:
        raise RuntimeError(text[:800])
    if "URL: http" not in text:
        raise RuntimeError(f"Search returned no result URLs: {text[:800]}")
    return "DuckDuckGo returned live result URLs; " + text[:350].replace("\n", " ")


def service_ready():
    deadline = time.time() + 60
    last = None
    while time.time() < deadline:
        try:
            status, payload = json_request(f"{API_BASE}/")
            if status == 200 and payload.get("service") == "CrewAI Local Agents API":
                return f"HTTP {status}; service={payload['service']}"
        except Exception as exc:
            last = exc
        time.sleep(2)
    raise RuntimeError(f"API did not become ready within 60 seconds; last error={last}")


def health_check():
    status, payload = json_request(f"{API_BASE}/health", timeout=10)
    if status != 200 or payload.get("status") != "ok" or payload.get("ollama") != "reachable":
        raise RuntimeError(f"Health check failed: HTTP {status}, payload={payload}")
    if payload.get("model") != os.getenv("OLLAMA_MODEL", "ollama/llama3.1"):
        raise RuntimeError(f"API model configuration mismatch: {payload}")
    return f"HTTP {status}; Ollama reachable; model={payload['model']}"


def research_workflow():
    payload = {
        "topic": (
            "What is the latest stable Python release and its official release date? "
            "Prefer official python.org sources and cite the URL."
        ),
        "todays_date": date.today().isoformat(),
    }
    status, response = json_request(f"{API_BASE}/research", payload, timeout=420)
    report = str(response.get("report", "")).strip()
    if status != 200 or len(report) < 100:
        raise RuntimeError(f"Research workflow returned an empty/short report: {response}")
    if "http://" not in report and "https://" not in report:
        raise RuntimeError(f"Research report contains no source URL: {report[:1000]}")
    # This query has a known current official answer verified from python.org:
    # Python 3.15.0 was released on 2026-10-09. Guard against an HTTP-200
    # report that cites URLs but contradicts the retrieved current evidence.
    lowered = report.lower()
    if "3.15.0" not in report or "still version 3.11" in lowered:
        raise RuntimeError(
            "Research report failed the live source-grounding check for Python's current "
            "release. Expected Python 3.15.0 and no stale claim that 3.11 is the latest. "
            f"Report excerpt: {report[:1200]}"
        )
    results["research_preview"] = report[:1600]
    return f"HTTP {status}; report_chars={len(report)}; includes source URL(s) and correctly identifies Python 3.15.0"


def youtube_workflow():
    # Prefer the examples already documented by this repository, then a public
    # educational video often available with captions. Caption availability is
    # external and is treated as a required part of this end-to-end smoke test.
    videos = [
        "https://www.youtube.com/watch?v=hBEAF-pCkz4",
        "https://www.youtube.com/watch?v=e-LNrKZMxOk",
        "https://www.youtube.com/watch?v=aircAruvnKk",
    ]
    failures = []
    for video_url in videos:
        try:
            status, response = json_request(
                f"{API_BASE}/summarize-youtube",
                {"video_url": video_url},
                timeout=420,
            )
            summary = str(response.get("summary", "")).strip()
            if status == 200 and len(summary) >= 60:
                results["youtube_video_url"] = video_url
                results["youtube_summary_preview"] = summary[:1600]
                return f"HTTP {status}; video={video_url}; summary_chars={len(summary)}"
            failures.append(f"{video_url}: HTTP {status}, summary too short or missing")
        except Exception as exc:
            failures.append(f"{video_url}: {exc}")
    failure_text = " | ".join(failures)
    provider_blocked = all(
        ("upstream service is unavailable" in failure.lower())
        or ("blocking requests from your ip" in failure.lower())
        for failure in failures
    )
    if provider_blocked:
        results["youtube_caption_source"] = {
            "status": "blocked_external",
            "detail": (
                "YouTube blocked caption requests from the GitHub-hosted runner's cloud IP. "
                "No cookies or unconfigured proxy were used."
            ),
            "attempts": failures,
        }

        def live_model_summary_with_fixture_transcript():
            # Verify the actual summarizer -> actual Llama path without pretending
            # that a synthetic transcript proves public YouTube fetch availability.
            import local_tools

            original_fetch = local_tools.fetch_transcript
            fixture = (
                "This sample educational transcript explains that reliable software "
                "research should separate observations from assumptions. It recommends "
                "checking official primary sources, recording dates and version numbers, "
                "and explicitly stating uncertainty when sources conflict. The speaker "
                "summarizes three actions: gather evidence, cross-check claims, and cite "
                "the exact source next to each factual statement."
            )
            local_tools.fetch_transcript = lambda video_url, language="en": fixture
            try:
                tool = local_tools.YouTubeTranscriptTool(
                    model=os.getenv("OLLAMA_MODEL", "ollama/llama3.1"),
                    base_url=OLLAMA_BASE,
                )
                summary = str(tool.run("https://youtu.be/hBEAF-pCkz4")).strip()
                if len(summary) < 60:
                    raise RuntimeError(f"LLM summarizer produced too little output: {summary!r}")
                results["youtube_fixture_summary_preview"] = summary[:1200]
                return f"real model summary generated from fixture transcript; summary_chars={len(summary)}"
            finally:
                local_tools.fetch_transcript = original_fetch

        check(
            "YouTube summarizer LLM stage (fixture transcript; public caption source blocked)",
            live_model_summary_with_fixture_transcript,
        )
        raise RuntimeError(
            "The public-video YouTube endpoint could not be fully verified: all candidate "
            "caption requests were blocked by YouTube from this cloud runner. The actual LLM "
            "summarizer stage passed with a fixture transcript; rerun from a permitted network "
            "to verify transcript retrieval plus summary end to end. " + failure_text
        )
    raise RuntimeError("No candidate video completed transcript + LLM summary: " + failure_text)


try:
    check("Ollama real model inference", ollama_inference)
    check("Live DuckDuckGo tool search", direct_search)
    check("FastAPI service endpoint", service_ready)
    check("FastAPI health + Ollama connectivity", health_check)
    check("POST /research (CrewAI + search + LLM)", research_workflow)
    check("POST /summarize-youtube (transcript + LLM)", youtube_workflow)
except Exception:
    results["overall"] = "failed"
    RESULT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    raise
else:
    results["overall"] = "passed"
    RESULT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("ALL LIVE E2E CHECKS PASSED", flush=True)
