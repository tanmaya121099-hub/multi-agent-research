"""Long-term memory: stores research sessions and summaries in Postgres."""

from __future__ import annotations

import json
from datetime import datetime

import structlog
from sqlalchemy import Column, DateTime, String, Text, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.config import settings

logger = structlog.get_logger(__name__)


class Base(DeclarativeBase):
    pass


class ResearchSession(Base):
    __tablename__ = "research_sessions"

    session_id = Column(String, primary_key=True)
    query = Column(Text, nullable=False)
    report = Column(Text)
    sources = Column(Text)          # JSON list
    critic_iterations = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)


class PostgresMemory:
    def __init__(self) -> None:
        self._engine = create_engine(
            settings.postgres_url,
            pool_size=settings.postgres_pool_size,
            pool_pre_ping=True,
        )
        self._Session = sessionmaker(bind=self._engine)
        self._init_db()

    def _init_db(self) -> None:
        try:
            Base.metadata.create_all(self._engine)
            logger.info("postgres.tables_ready")
        except Exception as exc:
            logger.error("postgres.init_failed", error=str(exc))

    def save_session(
        self,
        session_id: str,
        query: str,
        report: str,
        sources: list[str],
        critic_iterations: int,
    ) -> None:
        try:
            with self._Session() as session:
                row = ResearchSession(
                    session_id=session_id,
                    query=query,
                    report=report,
                    sources=json.dumps(sources),
                    critic_iterations=str(critic_iterations),
                )
                session.merge(row)
                session.commit()
        except Exception as exc:
            logger.error("postgres.save_session_error", error=str(exc))

    def get_session(self, session_id: str) -> dict | None:
        try:
            with self._Session() as session:
                row = session.get(ResearchSession, session_id)
                if not row:
                    return None
                return {
                    "session_id": row.session_id,
                    "query": row.query,
                    "report": row.report,
                    "sources": json.loads(row.sources or "[]"),
                    "critic_iterations": row.critic_iterations,
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                }
        except Exception as exc:
            logger.error("postgres.get_session_error", error=str(exc))
            return None

    def health(self) -> bool:
        try:
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False
