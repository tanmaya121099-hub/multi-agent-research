from __future__ import annotations

from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import BaseAgent, with_retry
from src.config import settings
from src.orchestration.messages import AgentRole, Fact

_SYSTEM = """You are a senior technical writer. Using the provided research facts and sources,
write a comprehensive, well-structured markdown report.

Requirements:
- Start with an executive summary (2-3 sentences)
- Use clear H2 sections for each major topic
- Cite sources inline as [1], [2], etc.
- Include a ## References section at the end with full URLs
- Use tables for comparisons
- Use bullet points for lists of features or trade-offs
- Be objective — present multiple perspectives where they exist
- Do NOT add information not present in the provided facts

Write for a technical audience who wants depth, not just surface summaries."""


class WriterAgent(BaseAgent):
    role = AgentRole.WRITER

    def __init__(self) -> None:
        super().__init__()
        self._llm = ChatAnthropic(
            model=settings.writer_model,
            api_key=settings.anthropic_api_key,
        )

    @with_retry
    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        query = state["original_query"]
        facts: list[Fact] = state.get("facts", [])
        sources: list[str] = state.get("sources", [])

        self._log_call("write", fact_count=len(facts), source_count=len(sources))

        facts_text = "\n".join(
            f"{i+1}. {f.text} (source: {f.source_url or 'internal'})"
            for i, f in enumerate(facts)
        )
        sources_text = "\n".join(f"[{i+1}] {url}" for i, url in enumerate(sources))

        messages = [
            SystemMessage(content=_SYSTEM),
            HumanMessage(
                content=(
                    f"Research query: {query}\n\n"
                    f"Facts gathered:\n{facts_text}\n\n"
                    f"Available sources:\n{sources_text}"
                )
            ),
        ]
        response = self._llm.invoke(messages)
        return {**state, "report": response.content}
