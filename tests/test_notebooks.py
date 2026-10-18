"""Sanity checks for shipped Jupyter notebooks (without running external services)."""

import ast
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    "crewai_reasoning (1).ipynb",
    "gemini_search_grounding_crewai (1).ipynb",
    "summarize_video_gemini_crewai.ipynb",
]


def code_sources(path):
    notebook = json.loads((REPO_ROOT / path).read_text(encoding="utf-8"))
    assert notebook.get("nbformat") == 4
    return [
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    ]


@pytest.mark.parametrize("path", NOTEBOOKS)
def test_notebook_code_cells_parse(path):
    """Catch Python syntax errors while skipping standalone IPython magic lines."""
    for cell_number, source in enumerate(code_sources(path)):
        python_lines = [
            line for line in source.splitlines()
            if not line.lstrip().startswith(("%", "!"))
        ]
        cleaned = "\n".join(python_lines)
        if cleaned.strip():
            try:
                ast.parse(cleaned)
            except SyntaxError as exc:
                pytest.fail(f"{path}, code cell {cell_number}: {exc}")


def test_reasoning_notebook_assigns_reporting_task_to_reporter():
    sources = code_sources("crewai_reasoning (1).ipynb")
    report_source = next(source for source in sources if "report_task = Task(" in source)
    report_task = report_source.split("report_task = Task(", 1)[1]
    assert "agent=reporter" in report_task
    assert "July 31, 2025" not in "\n".join(sources)


def test_video_notebook_uses_single_ollama_prefix_and_nonempty_crew_tasks():
    sources = code_sources("summarize_video_gemini_crewai.ipynb")
    all_code = "\n".join(sources)
    assert "ollama/ollama/" not in all_code
    assert "YouTubeTranscriptTool(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)" in all_code
    assert "LLM(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)" in all_code
    assert "Crew(agents=[video_analyst], tasks=[video_task]" in all_code
