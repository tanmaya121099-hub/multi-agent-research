from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.agents.base import BaseAgent, with_retry
from src.config import settings
from src.orchestration.messages import AgentRole, ResearchPlan, SubTask

_SYSTEM = """You are a research planner. Given a complex research query, break it into
3–5 clear, sequential subtasks that a researcher can execute one at a time.

Each subtask should be:
- Self-contained and actionable
- Focused on a specific aspect of the topic
- Logically ordered (definitions first, comparisons later)

Return JSON matching the schema."""


class _PlanOutput(BaseModel):
    subtasks: list[SubTask] = Field(description="Ordered list of research subtasks", max_length=5)


class PlannerAgent(BaseAgent):
    role = AgentRole.PLANNER

    def __init__(self) -> None:
        super().__init__()
        self._llm = ChatOpenAI(
            model=settings.planner_model,
            api_key=settings.openai_api_key,
            temperature=0,
        ).with_structured_output(_PlanOutput)

    @with_retry
    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        query = state["original_query"]
        self._log_call("plan", query=query)

        messages = [
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=f"Research query: {query}"),
        ]
        output: _PlanOutput = self._llm.invoke(messages)

        plan = ResearchPlan(
            query=query,
            subtasks=output.subtasks,
            session_id=state["session_id"],
        )
        return {
            **state,
            "plan": plan,
            "current_subtask_index": 0,
            "facts": [],
            "sources": [],
            "completed_subtasks": [],
            "critic_iterations": 0,
        }
