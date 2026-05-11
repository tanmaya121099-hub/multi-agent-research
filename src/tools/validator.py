"""Source and fact validation utilities."""

from __future__ import annotations

import re

import httpx
import structlog

logger = structlog.get_logger(__name__)

_HEADERS = {"User-Agent": "Mozilla/5.0 (research-agent/1.0)"}
_TIMEOUT = 5.0

# Known low-quality or aggregator-only domains
_LOW_QUALITY_DOMAINS = {
    "answers.com",
    "ehow.com",
    "about.com",
    "reference.com",
}


def is_url_reachable(url: str) -> bool:
    """HEAD request to check if a URL returns 2xx/3xx."""
    try:
        response = httpx.head(url, headers=_HEADERS, timeout=_TIMEOUT, follow_redirects=True)
        return response.status_code < 400
    except Exception:
        return False


def is_high_quality_source(url: str) -> bool:
    """Heuristic filter for obviously low-quality sources."""
    domain = re.sub(r"^https?://(?:www\.)?", "", url).split("/")[0]
    return domain not in _LOW_QUALITY_DOMAINS


def validate_sources(urls: list[str]) -> dict[str, bool]:
    """Return a dict of url → is_valid for each source."""
    return {
        url: is_url_reachable(url) and is_high_quality_source(url)
        for url in urls
    }
