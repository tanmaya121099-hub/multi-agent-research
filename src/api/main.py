from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Optional

import structlog
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel, Field

from src.memory.postgres_client import PostgresMemory
from src.memory.redis_client import ResearchMemory
from src.orchestration.graph import ResearchOrchestrator

logger = structlog.get_logger(__name__)

# ---------- Prometheus ----------
REQUEST_COUNT = Counter("research_requests_total", "Total research requests")
LATENCY = Histogram("research_latency_seconds", "Research latency", buckets=[5, 10, 20, 30, 60, 120])
CRITIC_RETRIES = Counter("research_critic_retries_total", "Critic retry count")

# ---------- App state ----------
_orchestrator: ResearchOrchestrator | None = None
_redis_memory: ResearchMemory | None = None
_pg_memory: PostgresMemory | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _orchestrator, _redis_memory, _pg_memory
    _orchestrator = ResearchOrchestrator()
    _redis_memory = ResearchMemory()
    _pg_memory = PostgresMemory()
    logger.info("app.started")
    yield
    logger.info("app.shutdown")


app = FastAPI(
    title="Multi-Agent Research API",
    description="Planner + Researcher + Critic + Writer agent pipeline",
    version="0.1.0",
    lifespan=lifespan,
)

app.mount("/metrics", make_asgi_app())


# ---------- Schemas ----------

class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=5, max_length=2000)
    max_subtasks: int = Field(default=4, ge=1, le=5)
    session_id: Optional[str] = None


class ResearchResponse(BaseModel):
    report: str
    sources: list[str]
    plan: list[str]
    critic_iterations: int
    session_id: str
    latency_ms: float


class SessionResponse(BaseModel):
    session_id: str
    query: str
    report: str
    sources: list[str]
    created_at: Optional[str]


class HealthResponse(BaseModel):
    status: str
    redis: str
    postgres: str


# ---------- Endpoints ----------

@app.post("/research", response_model=ResearchResponse)
async def research(req: ResearchRequest) -> ResearchResponse:
    assert _orchestrator is not None and _redis_memory is not None and _pg_memory is not None

    start = time.perf_counter()
    REQUEST_COUNT.inc()

    try:
        result = _orchestrator.run(req.query, session_id=req.session_id)
    except Exception as exc:
        logger.error("research.failed", error=str(exc))
        raise HTTPException(status_code=500, detail="Research pipeline failed") from exc

    latency_ms = (time.perf_counter() - start) * 1000
    LATENCY.observe(latency_ms / 1000)

    if result["critic_iterations"] > 0:
        CRITIC_RETRIES.inc(result["critic_iterations"])

    _pg_memory.save_session(
        session_id=result["session_id"],
        query=req.query,
        report=result["report"],
        sources=result["sources"],
        critic_iterations=result["critic_iterations"],
    )

    return ResearchResponse(**result, latency_ms=round(latency_ms, 2))


@app.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str) -> SessionResponse:
    assert _pg_memory is not None
    row = _pg_memory.get_session(session_id)
    if not row:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionResponse(**row)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    redis_ok = _redis_memory.health() if _redis_memory else False
    pg_ok = _pg_memory.health() if _pg_memory else False
    return HealthResponse(
        status="ok" if (redis_ok and pg_ok) else "degraded",
        redis="up" if redis_ok else "down",
        postgres="up" if pg_ok else "down",
    )
