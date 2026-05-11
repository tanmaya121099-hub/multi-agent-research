from __future__ import annotations

import json
from typing import Any

import redis
import structlog

from src.config import settings
from src.orchestration.messages import Fact, ResearchPlan

logger = structlog.get_logger(__name__)


class ResearchMemory:
    """Short-term and working memory backed by Redis."""

    SHORT_TTL = settings.redis_short_ttl    # 1hr  — current task
    WORKING_TTL = settings.redis_working_ttl  # 24hr — session findings

    def __init__(self) -> None:
        self._r = redis.from_url(settings.redis_url, decode_responses=True)

    # ---------- Plan ----------

    def save_plan(self, session_id: str, plan: ResearchPlan | None) -> None:
        if not plan:
            return
        try:
            self._r.setex(
                f"plan:{session_id}",
                self.SHORT_TTL,
                plan.model_dump_json(),
            )
        except Exception as exc:
            logger.warning("memory.save_plan_error", error=str(exc))

    def get_plan(self, session_id: str) -> ResearchPlan | None:
        try:
            raw = self._r.get(f"plan:{session_id}")
            return ResearchPlan.model_validate_json(raw) if raw else None
        except Exception:
            return None

    # ---------- Findings ----------

    def save_findings(self, session_id: str, facts: list[Fact]) -> None:
        try:
            key = f"findings:{session_id}"
            existing_raw = self._r.get(key)
            existing: list[dict] = json.loads(existing_raw) if existing_raw else []
            existing.extend([f.model_dump() for f in facts])
            self._r.setex(key, self.WORKING_TTL, json.dumps(existing))
        except Exception as exc:
            logger.warning("memory.save_findings_error", error=str(exc))

    def get_findings(self, session_id: str) -> list[Fact]:
        try:
            raw = self._r.get(f"findings:{session_id}")
            if not raw:
                return []
            return [Fact(**item) for item in json.loads(raw)]
        except Exception:
            return []

    # ---------- Report ----------

    def save_report(self, session_id: str, report: str) -> None:
        try:
            self._r.setex(f"report:{session_id}", self.WORKING_TTL, report)
        except Exception as exc:
            logger.warning("memory.save_report_error", error=str(exc))

    def get_report(self, session_id: str) -> str | None:
        try:
            return self._r.get(f"report:{session_id}")
        except Exception:
            return None

    # ---------- Health ----------

    def health(self) -> bool:
        try:
            return self._r.ping()
        except Exception:
            return False
