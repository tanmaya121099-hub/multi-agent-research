"""Semantic memory: pgvector-powered similarity search over stored facts."""

from __future__ import annotations

import structlog
from openai import OpenAI
from sqlalchemy import Column, String, Text, create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from src.config import settings

logger = structlog.get_logger(__name__)

_PGVECTOR_DDL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS fact_embeddings (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    fact_text TEXT NOT NULL,
    source_url TEXT,
    embedding vector({dim})
);
CREATE INDEX IF NOT EXISTS fact_embeddings_idx
    ON fact_embeddings USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
""".format(dim=settings.embedding_dimensions)


class SemanticMemory:
    """Store facts as embeddings, search by semantic similarity across sessions."""

    def __init__(self) -> None:
        self._engine = create_engine(settings.postgres_url, pool_pre_ping=True)
        self._openai = OpenAI(api_key=settings.openai_api_key)
        self._init_schema()

    def _init_schema(self) -> None:
        try:
            with self._engine.connect() as conn:
                for stmt in _PGVECTOR_DDL.strip().split(";"):
                    stmt = stmt.strip()
                    if stmt:
                        conn.execute(text(stmt))
                conn.commit()
            logger.info("semantic_memory.schema_ready")
        except Exception as exc:
            logger.warning("semantic_memory.schema_init_failed", error=str(exc))

    def _embed(self, text_: str) -> list[float]:
        response = self._openai.embeddings.create(
            input=text_,
            model=settings.embedding_model,
        )
        return response.data[0].embedding

    def store_fact(self, fact_id: str, session_id: str, text: str, source_url: str = "") -> None:
        try:
            vec = self._embed(text)
            vec_str = "[" + ",".join(str(x) for x in vec) + "]"
            with self._engine.connect() as conn:
                conn.execute(
                    text(
                        "INSERT INTO fact_embeddings (id, session_id, fact_text, source_url, embedding) "
                        "VALUES (:id, :sid, :text, :url, :vec::vector) "
                        "ON CONFLICT (id) DO NOTHING"
                    ),
                    {"id": fact_id, "sid": session_id, "text": text, "url": source_url, "vec": vec_str},
                )
                conn.commit()
        except Exception as exc:
            logger.warning("semantic_memory.store_error", error=str(exc))

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        try:
            vec = self._embed(query)
            vec_str = "[" + ",".join(str(x) for x in vec) + "]"
            with self._engine.connect() as conn:
                rows = conn.execute(
                    text(
                        "SELECT fact_text, source_url, session_id, "
                        "1 - (embedding <=> :vec::vector) AS similarity "
                        "FROM fact_embeddings "
                        "ORDER BY embedding <=> :vec::vector "
                        "LIMIT :k"
                    ),
                    {"vec": vec_str, "k": top_k},
                ).fetchall()
            return [
                {
                    "text": row[0],
                    "source_url": row[1],
                    "session_id": row[2],
                    "similarity": float(row[3]),
                }
                for row in rows
            ]
        except Exception as exc:
            logger.warning("semantic_memory.search_error", error=str(exc))
            return []
