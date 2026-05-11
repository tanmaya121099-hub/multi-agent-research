from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.agents.base import BaseAgent, with_retry
from src.config import settings
from src.orchestration.messages import AgentRole, CritiqueResult, Fact

_SYSTEM = """You are a critical research evaluator. Review the collected facts and determine
whether they are sufficient to write a comprehensive, accurate report on the original query.

Evaluate:
1. Coverage — are all major aspects of the query addressed?
2. Depth — are facts specific enough (numbers, dates, named sources)?
3. Citations — does every significant claim have a verifiable source URL?
4. Contradictions — are there conflicting facts that need resolution?

Be strict. Return a structured critique."""


class _CritiqueOutput(BaseModel):
    approved: bool = Field(description="True if research is sufficient for a good report")
    issues: list[str] = Field(default_factory=list, description="Specific problems found")
    missing_topics: list[str] = Field(
        default_factory=list,
        description="Topics not covered that are needed to answer the query",
    )
    overall_score: float = Field(ge=0.0, le=1.0, description="Quality score 0-1")


class CriticAgent(BaseAgent):
    role = AgentRole.CRITIC

    def __init__(self) -> None:
        super().__init__()
        self._llm = ChatOpenAI(
            model=settings.critic_model,
            api_key=settings.openai_api_key,
            temperature=0,
        ).with_structured_output(_CritiqueOutput)

    @with_retry
    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        query = state["original_query"]
        facts: list[Fact] = state.get("facts", [])
        self._log_call("critique", fact_count=len(facts))

        facts_text = "\n".join(
            f"- {f.text} [source: {f.source_url or 'unknown'}]" for f in facts
        )

        messages = [
            SystemMessage(content=_SYSTEM),
            HumanMessage(
                content=f"Original query: {query}\n\nCollected facts:\n{facts_text}"
            ),
        ]
        output: _CritiqueOutput = self._llm.invoke(messages)

        critique = CritiqueResult(
            approved=output.approved,
            issues=output.issues,
            missing_topics=output.missing_topics,
            overall_score=output.overall_score,
        )

        self._log_call(
            "critique_result",
            approved=critique.approved,
            score=critique.overall_score,
            issues=len(critique.issues),
        )

        iterations = state.get("critic_iterations", 0) + 1
        return {**state, "critique": critique, "critic_iterations": iterations}
