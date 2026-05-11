"""Integration tests for the multi-agent orchestration graph."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.orchestration.messages import CritiqueResult, Fact, ResearchPlan, SubTask


@pytest.fixture
def minimal_plan():
    return ResearchPlan(
        query="What is LangGraph?",
        session_id="test-session",
        subtasks=[
            SubTask(title="Define LangGraph", description="What is LangGraph and what does it do?"),
        ],
    )


@pytest.fixture
def sample_facts():
    return [
        Fact(
            text="LangGraph is a library for building stateful multi-actor applications with LLMs.",
            source_url="https://langchain.com/langgraph",
        )
    ]


class TestResearchMemory:
    def test_save_and_retrieve_plan(self, minimal_plan):
        with patch("src.memory.redis_client.redis.from_url") as mock_redis:
            mock_r = MagicMock()
            mock_redis.return_value = mock_r
            mock_r.get.return_value = minimal_plan.model_dump_json()
            mock_r.ping.return_value = True

            from src.memory.redis_client import ResearchMemory
            mem = ResearchMemory()
            mem.save_plan("test-session", minimal_plan)
            mock_r.setex.assert_called_once()

    def test_save_findings_appends(self, sample_facts):
        import json
        with patch("src.memory.redis_client.redis.from_url") as mock_redis:
            mock_r = MagicMock()
            mock_redis.return_value = mock_r
            mock_r.get.return_value = json.dumps([])

            from src.memory.redis_client import ResearchMemory
            mem = ResearchMemory()
            mem.save_findings("test-session", sample_facts)

            stored = json.loads(mock_r.setex.call_args[0][2])
            assert len(stored) == 1
            assert stored[0]["text"] == sample_facts[0].text

    def test_health_returns_bool(self):
        with patch("src.memory.redis_client.redis.from_url") as mock_redis:
            mock_r = MagicMock()
            mock_redis.return_value = mock_r
            mock_r.ping.return_value = True

            from src.memory.redis_client import ResearchMemory
            mem = ResearchMemory()
            assert mem.health() is True


class TestMessageTypes:
    def test_agent_message_has_id(self):
        from src.orchestration.messages import AgentMessage, AgentRole, MessageAction
        msg = AgentMessage(
            sender=AgentRole.PLANNER,
            receiver=AgentRole.RESEARCHER,
            action=MessageAction.DELEGATE,
            payload={"subtask": "Research LLM costs"},
        )
        assert msg.id is not None
        assert len(msg.id) > 0

    def test_critique_result_defaults(self):
        result = CritiqueResult(approved=True, overall_score=0.85)
        assert result.issues == []
        assert result.missing_topics == []

    def test_fact_serializes(self, sample_facts):
        fact = sample_facts[0]
        d = fact.model_dump()
        assert "text" in d
        assert "source_url" in d
