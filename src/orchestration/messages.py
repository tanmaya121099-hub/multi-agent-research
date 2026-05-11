"""Typed message contracts for inter-agent communication."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class AgentRole(StrEnum):
    PLANNER = "planner"
    RESEARCHER = "researcher"
    CRITIC = "critic"
    WRITER = "writer"
    SYSTEM = "system"


class MessageAction(StrEnum):
    PLAN = "plan"
    RESEARCH = "research"
    CRITIQUE = "critique"
    WRITE = "write"
    DELEGATE = "delegate"
    COMPLETE = "complete"
    RETRY = "retry"


class AgentMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    sender: AgentRole
    receiver: AgentRole
    action: MessageAction
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    session_id: str = ""


class SubTask(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    description: str
    status: str = "pending"  # pending | in_progress | complete
    assigned_to: AgentRole = AgentRole.RESEARCHER


class ResearchPlan(BaseModel):
    query: str
    subtasks: list[SubTask]
    session_id: str


class Fact(BaseModel):
    text: str
    source_url: str = ""
    source_title: str = ""
    confidence: float = 1.0


class CritiqueResult(BaseModel):
    approved: bool
    issues: list[str] = Field(default_factory=list)
    missing_topics: list[str] = Field(default_factory=list)
    overall_score: float  # 0-1


class ResearchFindings(BaseModel):
    session_id: str
    facts: list[Fact] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    subtask_id: str = ""
