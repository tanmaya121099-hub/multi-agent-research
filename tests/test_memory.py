"""Unit tests for memory layer — Redis and semantic search."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.orchestration.messages import Fact, ResearchPlan, SubTask


@pytest.fixture
def sample_plan():
    return ResearchPlan(
        query="What are the trade-offs between RAG and fine-tuning?",
        session_id="test-001",
        subtasks=[
            SubTask(title="Define RAG", description="Explain what RAG is."),
            SubTask(title="Define fine-tuning", description="Explain what fine-tuning is."),
        ],
    )


@pytest.fixture
def sample_facts():
    return [
        Fact(text="RAG retrieves context at inference time.", source_url="https://example.com/rag"),
        Fact(text="Fine-tuning bakes knowledge into weights.", source_url="https://example.com/ft"),
    ]


class TestResearchMemoryPlan:
    def test_save_plan_calls_setex(self, sample_plan):
        with patch("src.memory.redis_client.redis.from_url") as mock_redis:
            mock_r = MagicMock()
            mock_redis.return_value = mock_r

            from src.memory.redis_client import ResearchMemory
            mem = ResearchMemory()
            mem.save_plan(sample_plan.session_id, sample_plan)
            mock_r.setex.assert_called_once()

    def test_get_plan_deserializes(self, sample_plan):
        with patch("src.memory.redis_client.redis.from_url") as mock_redis:
            mock_r = MagicMock()
            mock_redis.return_value = mock_r
            mock_r.get.return_value = sample_plan.model_dump_json()

            from src.memory.redis_client import ResearchMemory
            mem = ResearchMemory()
            result = mem.get_plan(sample_plan.session_id)
            assert result is not None
            assert result.query == sample_plan.query

    def test_get_plan_returns_none_on_miss(self):
        with patch("src.memory.redis_client.redis.from_url") as mock_redis:
            mock_r = MagicMock()
            mock_redis.return_value = mock_r
            mock_r.get.return_value = None

            from src.memory.redis_client import ResearchMemory
            mem = ResearchMemory()
            assert mem.get_plan("nonexistent-session") is None


class TestResearchMemoryFindings:
    def test_save_findings_stores_list(self, sample_facts):
        with patch("src.memory.redis_client.redis.from_url") as mock_redis:
            mock_r = MagicMock()
            mock_redis.return_value = mock_r
            mock_r.get.return_value = json.dumps([])

            from src.memory.redis_client import ResearchMemory
            mem = ResearchMemory()
            mem.save_findings("test-001", sample_facts)

            stored = json.loads(mock_r.setex.call_args[0][2])
            assert len(stored) == len(sample_facts)

    def test_save_findings_appends_to_existing(self, sample_facts):
        existing = [{"text": "existing fact", "source_url": "https://x.com"}]
        with patch("src.memory.redis_client.redis.from_url") as mock_redis:
            mock_r = MagicMock()
            mock_redis.return_value = mock_r
            mock_r.get.return_value = json.dumps(existing)

            from src.memory.redis_client import ResearchMemory
            mem = ResearchMemory()
            mem.save_findings("test-001", sample_facts)

            stored = json.loads(mock_r.setex.call_args[0][2])
            assert len(stored) == len(existing) + len(sample_facts)

    def test_get_findings_returns_empty_on_miss(self):
        with patch("src.memory.redis_client.redis.from_url") as mock_redis:
            mock_r = MagicMock()
            mock_redis.return_value = mock_r
            mock_r.get.return_value = None

            from src.memory.redis_client import ResearchMemory
            mem = ResearchMemory()
            assert mem.get_findings("no-session") == []


class TestResearchMemoryHealth:
    def test_health_true_when_redis_up(self):
        with patch("src.memory.redis_client.redis.from_url") as mock_redis:
            mock_r = MagicMock()
            mock_redis.return_value = mock_r
            mock_r.ping.return_value = True

            from src.memory.redis_client import ResearchMemory
            assert ResearchMemory().health() is True

    def test_health_false_when_redis_down(self):
        with patch("src.memory.redis_client.redis.from_url") as mock_redis:
            mock_r = MagicMock()
            mock_redis.return_value = mock_r
            mock_r.ping.side_effect = Exception("connection refused")

            from src.memory.redis_client import ResearchMemory
            assert ResearchMemory().health() is False
