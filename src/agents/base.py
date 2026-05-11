"""Abstract base agent with retry logic, structured logging, and LangSmith tagging."""

from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from typing import Any, Callable, TypeVar

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from src.orchestration.messages import AgentRole

logger = structlog.get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

MAX_RETRIES = 3


def with_retry(fn: F) -> F:
    """Decorator that retries an agent method up to 3 times with exponential backoff."""
    @functools.wraps(fn)
    @retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(min=1, max=10))
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return fn(*args, **kwargs)
    return wrapper  # type: ignore[return-value]


class BaseAgent(ABC):
    role: AgentRole

    def __init__(self) -> None:
        self._log = structlog.get_logger(self.__class__.__name__)

    def _log_call(self, action: str, **kwargs: Any) -> None:
        self._log.info(
            f"{self.role}.{action}",
            agent=self.role,
            **{k: str(v)[:200] for k, v in kwargs.items()},
        )

    @abstractmethod
    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Execute the agent's primary task and return updated state."""
        ...
