from __future__ import annotations

from typing import Any

import structlog
from tavily import TavilyClient
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings

logger = structlog.get_logger(__name__)


class WebSearchTool:
    def __init__(self) -> None:
        self._client = TavilyClient(api_key=settings.tavily_api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def search(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        try:
            response = self._client.search(
                query,
                max_results=max_results,
                include_raw_content=False,
            )
            results = response.get("results", [])
            logger.info("web_search.done", count=len(results), query=query[:80])
            return [
                {
                    "url": r.get("url", ""),
                    "title": r.get("title", ""),
                    "content": r.get("content", ""),
                    "score": r.get("score", 0.0),
                }
                for r in results
            ]
        except Exception as exc:
            logger.error("web_search.failed", error=str(exc))
            return []
