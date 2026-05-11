"""Unit tests for individual agent logic."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.orchestration.messages import AgentRole, CritiqueResult, Fact, SubTask


class TestPlannerAgent:
    def test_plan_has_subtasks(self):
        with patch("src.agents.planner.ChatOpenAI") as MockLLM:
            from src.agents.planner import PlannerAgent
            from src.orchestration.messages import SubTask

            mock_output = MagicMock()
            mock_output.subtasks = [
                SubTask(title="Define RAG", description="What is RAG?"),
                SubTask(title="List architectures", description="Common RAG variants"),
            ]
            MockLLM.return_value.with_structured_output.return_value.invoke.return_value = mock_output

            agent = PlannerAgent()
            result = agent.run({"original_query": "Explain RAG", "session_id": "test-123"})

            assert "plan" in result
            assert len(result["plan"].subtasks) == 2
            assert result["current_subtask_index"] == 0
            assert result["facts"] == []

    def test_plan_initialises_counters(self):
        with patch("src.agents.planner.ChatOpenAI") as MockLLM:
            from src.agents.planner import PlannerAgent

            mock_output = MagicMock()
            mock_output.subtasks = [SubTask(title="T1", description="D1")]
            MockLLM.return_value.with_structured_output.return_value.invoke.return_value = mock_output

            agent = PlannerAgent()
            result = agent.run({"original_query": "Test", "session_id": "s1"})

            assert result["critic_iterations"] == 0
            assert result["sources"] == []


class TestCriticAgent:
    def test_critic_approves_good_research(self):
        with patch("src.agents.critic.ChatOpenAI") as MockLLM:
            from src.agents.critic import CriticAgent

            mock_output = MagicMock()
            mock_output.approved = True
            mock_output.issues = []
            mock_output.missing_topics = []
            mock_output.overall_score = 0.9
            MockLLM.return_value.with_structured_output.return_value.invoke.return_value = mock_output

            agent = CriticAgent()
            facts = [Fact(text="RAG combines retrieval and generation.", source_url="https://example.com")]
            result = agent.run({
                "original_query": "What is RAG?",
                "facts": facts,
                "critic_iterations": 0,
            })

            assert result["critique"].approved is True
            assert result["critic_iterations"] == 1

    def test_critic_rejects_missing_sources(self):
        with patch("src.agents.critic.ChatOpenAI") as MockLLM:
            from src.agents.critic import CriticAgent

            mock_output = MagicMock()
            mock_output.approved = False
            mock_output.issues = ["Facts lack source citations"]
            mock_output.missing_topics = ["cost comparison"]
            mock_output.overall_score = 0.4
            MockLLM.return_value.with_structured_output.return_value.invoke.return_value = mock_output

            agent = CriticAgent()
            result = agent.run({
                "original_query": "Compare RAG vs fine-tuning",
                "facts": [Fact(text="RAG uses retrieval.")],
                "critic_iterations": 0,
            })

            assert result["critique"].approved is False
            assert len(result["critique"].issues) > 0


class TestRoutingFunctions:
    def test_route_after_research_more_tasks(self):
        from src.orchestration.messages import ResearchPlan, SubTask
        from src.orchestration.graph import _route_after_research

        plan = ResearchPlan(
            query="test",
            session_id="s1",
            subtasks=[
                SubTask(title="T1", description="D1"),
                SubTask(title="T2", description="D2"),
            ],
        )
        state = {"plan": plan, "current_subtask_index": 0}
        assert _route_after_research(state) == "researcher"

    def test_route_after_research_done(self):
        from src.orchestration.messages import ResearchPlan, SubTask
        from src.orchestration.graph import _route_after_research

        plan = ResearchPlan(
            query="test",
            session_id="s1",
            subtasks=[SubTask(title="T1", description="D1")],
        )
        state = {"plan": plan, "current_subtask_index": 1}
        assert _route_after_research(state) == "critic"

    def test_route_after_critic_approved(self):
        from src.orchestration.graph import _route_after_critic

        state = {"critique": CritiqueResult(approved=True, overall_score=0.9), "critic_iterations": 1}
        assert _route_after_critic(state) == "writer"

    def test_route_after_critic_retry(self):
        from src.orchestration.graph import _route_after_critic

        state = {"critique": CritiqueResult(approved=False, overall_score=0.4), "critic_iterations": 1}
        assert _route_after_critic(state) == "researcher"

    def test_route_after_critic_max_retries_forces_write(self):
        from src.config import settings
        from src.orchestration.graph import _route_after_critic

        state = {
            "critique": CritiqueResult(approved=False, overall_score=0.3),
            "critic_iterations": settings.max_critic_retries,
        }
        assert _route_after_critic(state) == "writer"
