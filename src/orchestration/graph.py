from __future__ import annotations

import uuid
from typing import Any

import structlog
from langgraph.graph import END, START, StateGraph

from src.agents.critic import CriticAgent
from src.agents.planner import PlannerAgent
from src.agents.researcher import ResearcherAgent
from src.agents.writer import WriterAgent
from src.config import settings
from src.memory.redis_client import ResearchMemory
from src.orchestration.state import ResearchState

logger = structlog.get_logger(__name__)


def _route_after_research(state: ResearchState) -> str:
    plan = state.get("plan")
    idx = state.get("current_subtask_index", 0)
    if plan and idx < len(plan.subtasks):
        return "researcher"  # more subtasks remaining
    return "critic"


def _route_after_critic(state: ResearchState) -> str:
    critique = state.get("critique")
    iterations = state.get("critic_iterations", 0)

    if critique and critique.approved:
        return "writer"

    if iterations >= settings.max_critic_retries:
        logger.warning("critic.max_retries_reached", session=state.get("session_id"))
        return "writer"  # degrade gracefully

    return "researcher"


def build_graph() -> Any:
    planner = PlannerAgent()
    researcher = ResearcherAgent()
    critic = CriticAgent()
    writer = WriterAgent()
    memory = ResearchMemory()

    def planner_node(state: ResearchState) -> ResearchState:
        result = planner.run(state)
        memory.save_plan(state["session_id"], result.get("plan"))
        return result

    def researcher_node(state: ResearchState) -> ResearchState:
        result = researcher.run(state)
        memory.save_findings(state["session_id"], result.get("facts", []))
        return result

    def critic_node(state: ResearchState) -> ResearchState:
        return critic.run(state)

    def writer_node(state: ResearchState) -> ResearchState:
        result = writer.run(state)
        memory.save_report(state["session_id"], result.get("report", ""))
        return result

    graph = StateGraph(ResearchState)
    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("critic", critic_node)
    graph.add_node("writer", writer_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "researcher")

    graph.add_conditional_edges(
        "researcher",
        _route_after_research,
        {"researcher": "researcher", "critic": "critic"},
    )

    graph.add_conditional_edges(
        "critic",
        _route_after_critic,
        {"writer": "writer", "researcher": "researcher"},
    )

    graph.add_edge("writer", END)

    return graph.compile()


class ResearchOrchestrator:
    def __init__(self) -> None:
        self._graph = build_graph()

    def run(self, query: str, session_id: str | None = None) -> dict[str, Any]:
        sid = session_id or str(uuid.uuid4())
        initial: ResearchState = {
            "session_id": sid,
            "original_query": query,
            "facts": [],
            "sources": [],
            "completed_subtasks": [],
            "critic_iterations": 0,
            "agent_history": [],
        }
        result = self._graph.invoke(initial)
        return {
            "report": result.get("report", ""),
            "sources": result.get("sources", []),
            "plan": [t.title for t in result.get("plan", {}).subtasks or []],
            "critic_iterations": result.get("critic_iterations", 0),
            "session_id": sid,
        }
