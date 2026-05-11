"""LangGraph shared state schema for the multi-agent research workflow."""

from __future__ import annotations

from typing import Any

from typing_extensions import TypedDict

from src.orchestration.messages import (
    AgentMessage,
    CritiqueResult,
    Fact,
    ResearchPlan,
    SubTask,
)


class ResearchState(TypedDict, total=False):
    # Immutable inputs
    session_id: str
    original_query: str

    # Plan
    plan: ResearchPlan
    current_subtask_index: int

    # Research accumulation
    facts: list[Fact]
    sources: list[str]
    completed_subtasks: list[str]

    # Critic
    critique: CritiqueResult
    critic_iterations: int

    # Output
    report: str

    # Meta
    agent_history: list[AgentMessage]
    error: str | None
