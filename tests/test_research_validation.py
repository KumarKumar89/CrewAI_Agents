"""Regression tests for deterministic research grounding checks."""

from research_validation import validate_research_report


SOURCE = {
    "title": "Python Release Python 3.15.0 | Python.org",
    "url": "https://www.python.org/downloads/release/python-3150/",
    "snippet": "This is the stable release of Python 3.15.0. Release date: Oct. 9, 2026.",
}
TOPIC = "What is the latest stable Python release and its official release date?"


def test_accepts_latest_version_and_release_date_supported_by_source():
    report = (
        "Python 3.15.0 is the latest stable release, released October 9, 2026. "
        "Official source: https://www.python.org/downloads/release/python-3150/"
    )
    assert validate_research_report(report, TOPIC, [SOURCE]) == []


def test_rejects_stale_latest_version_even_when_a_url_is_present():
    report = (
        "Python 3.11 is still the latest stable release, released October 1, 2026. "
        "Source: https://www.python.org/downloads/release/python-3150/"
    )
    errors = validate_research_report(report, TOPIC, [SOURCE])
    assert any("older version" in error or "supported version 3.15.0" in error for error in errors)


def test_rejects_urls_not_returned_by_search():
    report = (
        "Python 3.15.0 is the latest stable release, released October 9, 2026. "
        "Source: https://example.invalid/latest-python/"
    )
    errors = validate_research_report(report, TOPIC, [SOURCE])
    assert any("not present in retrieved search results" in error for error in errors)


def test_rejects_wrong_release_date_even_when_version_is_correct():
    report = (
        "Python 3.15.0 is the latest stable release, released October 1, 2026. "
        "Source: https://www.python.org/downloads/release/python-3150/"
    )
    errors = validate_research_report(report, TOPIC, [SOURCE])
    assert any("does not state a release date supported by evidence" in error for error in errors)


def test_rejects_tool_call_text_as_a_final_answer():
    report = (
        'Here is the JSON function call with its proper arguments. '
        '{"name": "duckduckgo_search", "parameters": {"query": "latest Python"}} '
        "This should be a normal research answer, but it is not."
    )
    errors = validate_research_report(report, TOPIC, [SOURCE])
    assert any("tool-call syntax" in error for error in errors)
