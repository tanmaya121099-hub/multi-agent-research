from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.agents.base import BaseAgent, with_retry
from src.config import settings
from src.orchestration.messages import AgentRole, Fact, ResearchFindings
from src.tools.search import WebSearchTool
from src.tools.scraper import ArticleScraper

_SYSTEM = """You are a meticulous research agent. Given a subtask, you will:
1. Search for relevant information
2. Extract key facts with exact source citations
3. Report findings as structured JSON

Rules:
- Every fact MUST have a source URL
- If you cannot find a source, do not include the fact
- Prefer primary sources over aggregator sites
- Be specific — include numbers, dates, and named entities"""


class _FactExtractionOutput(BaseModel):
    facts: list[Fact] = Field(description="Extracted facts with citations")
    summary: str = Field(description="One-paragraph summary of findings")


class ResearcherAgent(BaseAgent):
    role = AgentRole.RESEARCHER

    def __init__(self) -> None:
        super().__init__()
        self._llm = ChatOpenAI(
            model=settings.researcher_model,
            api_key=settings.openai_api_key,
            temperature=0.1,
        ).with_structured_output(_FactExtractionOutput)
        self._search = WebSearchTool()
        self._scraper = ArticleScraper()

    @with_retry
    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        plan = state["plan"]
        idx = state.get("current_subtask_index", 0)
        subtask = plan.subtasks[idx]

        self._log_call("research", subtask=subtask.title)

        search_results = self._search.search(subtask.description, max_results=4)

        context_parts = []
        sources = list(state.get("sources", []))

        for result in search_results:
            url = result.get("url", "")
            content = result.get("content", "")
            if url and url not in sources:
                sources.append(url)
            if content:
                context_parts.append(f"[Source: {url}]\n{content[:2000]}")

        context = "\n\n---\n\n".join(context_parts)

        messages = [
            SystemMessage(content=_SYSTEM),
            HumanMessage(
                content=f"Subtask: {subtask.title}\n\nDescription: {subtask.description}\n\n"
                        f"Search results:\n{context}"
            ),
        ]
        output: _FactExtractionOutput = self._llm.invoke(messages)

        existing_facts = list(state.get("facts", []))
        existing_facts.extend(output.facts)

        completed = list(state.get("completed_subtasks", []))
        completed.append(subtask.title)

        return {
            **state,
            "facts": existing_facts,
            "sources": sources,
            "completed_subtasks": completed,
            "current_subtask_index": idx + 1,
        }
