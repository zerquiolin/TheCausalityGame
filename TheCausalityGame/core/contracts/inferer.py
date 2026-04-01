"""The Causality Game - Inferer Contract."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from TheCausalityGame.core.contracts.agent import AgentContext
from TheCausalityGame.core.contracts.dto.agent import BeliefSnapshot, RoundObservation
from TheCausalityGame.core.contracts.serializable import Serializable
from TheCausalityGame.core.infrastructure.logger import Logger
from TheCausalityGame.core.lib.errors.agent import (
    AgentContextNotSetError,
    AgentLoggerNotSetError,
)


class Inferer(Serializable):
    """Passive-learning component used by composable agents."""

    _context: AgentContext | None = None
    _logger: Logger | None = None
    _spec: str = "TheCausalityGame.core.contracts.specs.inferer:InfererSpec"

    def set_context(self, ctx: AgentContext) -> None:
        """Assign runtime context to the inferer."""
        if self._context is None:
            self._context = ctx

    @property
    def context(self) -> AgentContext:
        """Return inferer runtime context."""
        if self._context is None:
            raise AgentContextNotSetError()
        return self._context

    def set_logger(self, logger: Logger) -> None:
        """Assign a logger to the inferer."""
        if self._logger is None:
            self._logger = logger

    @property
    def logger(self) -> Logger:
        """Return the assigned inferer logger."""
        if self._logger is None:
            raise AgentLoggerNotSetError()
        return self._logger

    @abstractmethod
    def update(self, observation: RoundObservation) -> None:
        """Update the inferer state from a new round observation."""
        raise NotImplementedError

    @abstractmethod
    def answer(self) -> Any:  # noqa: ANN401
        """Return the inferer's current task estimate."""
        raise NotImplementedError

    @abstractmethod
    def snapshot(self) -> BeliefSnapshot:
        """Return a decider-facing snapshot of the current belief state."""
        raise NotImplementedError
