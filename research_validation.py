"""Deterministic validation for source-grounded research reports.

This is a conservative guard, not a proof of truth: it rejects model output when
citations are not from retrieved evidence or when a current-release claim
contradicts the highest stable version present in the evidence.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any
from urllib.parse import urlsplit

URL_RE = re.compile(r"https?://[^\s<>\]\[)]+", re.IGNORECASE)
VERSION_RE = re.compile(
    r"(?<![A-Za-z0-9])v?(\d{1,3}(?:\.\d{1,3}){1,2})(?![A-Za-z0-9_.-])"
)
CURRENT_WORDS_RE = re.compile(
    r"\b(latest|newest|current|most recent|up[- ]to[- ]date|stable)\b",
    re.IGNORECASE,
)
CURRENT_VERSION_CLAIM_RE = re.compile(
    r"\b(?:latest|newest|current|most recent|up[- ]to[- ]date|stable)"
    r"\b[^.!?\n]{0,120}?\b(v?\d{1,3}(?:\.\d{1,3}){1,2})\b",
    re.IGNORECASE,
)
VERSION_IS_CURRENT_CLAIM_RE = re.compile(
    r"\b(v?\d{1,3}(?:\.\d{1,3}){1,2})\b[^.!?\n]{0,80}?"
    r"\b(?:is|remains|stays)\s+(?:still\s+)?(?:the\s+)?"
    r"(?:latest|newest|current|most recent|up[- ]to[- ]date|stable)\b",
    re.IGNORECASE,
)
MONTH_DATE_RE = re.compile(
    r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\.?\s+(\d{1,2}),?\s+(\d{4})\b",
    re.IGNORECASE,
)
ISO_DATE_RE = re.compile(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b")
MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def normalize_url(url: str) -> str:
    """Normalize URLs for evidence matching while keeping their path/query."""
    value = str(url).strip().rstrip(".,;:!?)]}")
    try:
        parsed = urlsplit(value)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            return ""
        host = parsed.hostname.lower().removeprefix("www.")
        port = f":{parsed.port}" if parsed.port else ""
        path = parsed.path.rstrip("/")
        query = f"?{parsed.query}" if parsed.query else ""
        return f"{parsed.scheme.lower()}://{host}{port}{path}{query}"
    except ValueError:
        return ""


def extract_urls(text: str) -> list[str]:
    return [url for match in URL_RE.findall(text) if (url := normalize_url(match))]


def _version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.lstrip("v").split("."))


def _version_strings(text: str) -> set[str]:
    versions = set()
    for match in VERSION_RE.finditer(text):
        value = match.group(1)
        parts = _version_tuple(value)
        # Exclude dotted calendar dates (for example, 2026.10.09).
        if parts and parts[0] >= 1900:
            continue
        versions.add(value)
    return versions


def _month_date(year: str, month: str, day: str) -> str | None:
    try:
        month_num = MONTHS[month[:3].lower()]
        return date(int(year), month_num, int(day)).isoformat()
    except (KeyError, TypeError, ValueError):
        return None


def extract_dates(text: str) -> set[str]:
    dates: set[str] = set()
    for match in MONTH_DATE_RE.finditer(text):
        value = _month_date(match.group(3), match.group(1), match.group(2))
        if value:
            dates.add(value)
    for match in ISO_DATE_RE.finditer(text):
        try:
            dates.add(date(int(match.group(1)), int(match.group(2)), int(match.group(3))).isoformat())
        except ValueError:
            continue
    return dates


def is_latest_release_question(topic: str) -> bool:
    """Identify questions where stale version claims are particularly risky."""
    text = topic.casefold()
    asks_current = bool(re.search(
        r"\b(latest|newest|current|most recent|stable|up[- ]to[- ]date)\b", text
    ))
    asks_release = bool(re.search(r"\b(version|release|released)\b", text))
    return asks_current and asks_release


def validate_research_report(
    report: str,
    topic: str,
    sources: list[dict[str, Any]],
) -> list[str]:
    """Return validation failures. An empty list means the configured checks passed."""
    errors: list[str] = []
    answer = str(report or "").strip()
    if len(answer) < 80:
        errors.append("The answer is empty or too short to substantiate its claims.")
        return errors

    lowered = answer.casefold()
    tool_call_markers = (
        "here is the json function call",
        '"name": "duckduckgo_search"',
        '"name":"duckduckgo_search"',
        "tool_calls",
        "function_call",
    )
    if any(marker in lowered for marker in tool_call_markers):
        errors.append("The model returned tool-call syntax instead of a user-facing report.")

    source_by_url: dict[str, dict[str, Any]] = {}
    for source in sources:
        normalized = normalize_url(str(source.get("url") or source.get("href") or ""))
        if normalized:
            source_by_url[normalized] = source

    reported_urls = extract_urls(answer)
    if not reported_urls:
        errors.append("The answer does not cite any URL.")
    ungrounded_urls = [url for url in reported_urls if url not in source_by_url]
    if ungrounded_urls:
        errors.append("The answer contains URL(s) that were not present in retrieved search results.")
    if reported_urls and not any(url in source_by_url for url in reported_urls):
        errors.append("No answer citation maps to a retrieved source.")

    if not is_latest_release_question(topic):
        return errors

    version_source_text: dict[str, list[tuple[str, str]]] = {}
    for normalized_url, source in source_by_url.items():
        title = str(source.get("title") or "")
        snippet = str(source.get("snippet") or source.get("body") or "")
        combined = f"{title}\n{snippet}"
        for version in _version_strings(combined):
            version_source_text.setdefault(version, []).append((normalized_url, combined))

    if not version_source_text:
        errors.append(
            "The question asks for a current/latest release, but retrieved evidence contains no parseable version."
        )
        return errors

    latest_version = max(version_source_text, key=_version_tuple)
    latest_sources = {url for url, _ in version_source_text[latest_version]}
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", answer) if part.strip()]
    current_claim_paragraphs = [
        paragraph for paragraph in paragraphs
        if CURRENT_WORDS_RE.search(paragraph) and VERSION_RE.search(paragraph)
    ]
    if not current_claim_paragraphs:
        errors.append(
            f"The answer does not clearly state which retrieved version is current/latest ({latest_version})."
        )
    else:
        for paragraph in current_claim_paragraphs:
            claims = []
            claims.extend(CURRENT_VERSION_CLAIM_RE.findall(paragraph))
            claims.extend(VERSION_IS_CURRENT_CLAIM_RE.findall(paragraph))
            claimed_versions = {claim.lstrip("v") for claim in claims}
            # Ignore release-history paragraphs that name the correct current version
            # and older releases together; direct current-version grammar is decisive.
            if claimed_versions and any(v != latest_version for v in claimed_versions):
                errors.append(
                    f"The answer labels an older version as current/latest; evidence supports {latest_version}."
                )
            paragraph_versions = _version_strings(paragraph)
            paragraph_urls = set(extract_urls(paragraph))
            if latest_version not in paragraph_versions:
                errors.append(
                    f"The current/latest claim does not mention the evidence-supported version {latest_version}."
                )
            if not paragraph_urls.intersection(latest_sources):
                errors.append(
                    f"The current/latest version claim is not cited to a retrieved source that supports {latest_version}."
                )

    asks_date = bool(re.search(r"\b(release date|date of release|when (?:was|is).*released|released on)\b", topic.casefold()))
    if asks_date:
        supported_dates: set[str] = set()
        for _, evidence in version_source_text[latest_version]:
            supported_dates.update(extract_dates(evidence))
        if not supported_dates:
            errors.append(
                f"Retrieved evidence identifies {latest_version} but does not include a release date."
            )
        else:
            answer_dates = extract_dates(answer)
            if not answer_dates.intersection(supported_dates):
                expected = ", ".join(sorted(supported_dates))
                errors.append(
                    f"The answer does not state a release date supported by evidence for {latest_version} "
                    f"(evidence date(s): {expected})."
                )
    return list(dict.fromkeys(errors))
