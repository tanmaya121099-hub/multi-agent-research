from __future__ import annotations

import structlog
import httpx
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

logger = structlog.get_logger(__name__)

_HEADERS = {"User-Agent": "Mozilla/5.0 (research-agent/1.0)"}
_TIMEOUT = 10.0
_MAX_CHARS = 8000


class ArticleScraper:
    """Fetches a URL and extracts readable text content."""

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=4))
    def scrape(self, url: str) -> str:
        try:
            response = httpx.get(url, headers=_HEADERS, timeout=_TIMEOUT, follow_redirects=True)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            # Remove noise
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()

            # Prefer main content areas
            main = soup.find("main") or soup.find("article") or soup.body
            text = main.get_text(separator="\n", strip=True) if main else ""

            # Collapse blank lines
            lines = [line for line in text.splitlines() if line.strip()]
            content = "\n".join(lines)[:_MAX_CHARS]

            logger.debug("scraper.done", url=url[:80], chars=len(content))
            return content
        except Exception as exc:
            logger.warning("scraper.failed", url=url[:80], error=str(exc))
            return ""
