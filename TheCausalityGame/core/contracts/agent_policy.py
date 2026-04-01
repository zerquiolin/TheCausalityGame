"""The Causality Game - Unified agent policy contract."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from TheCausalityGame.core.contracts.agent import AgentContext
from TheCausalityGame.core.contracts.dto.agent import RoundObservation
from TheCausalityGame.core.contracts.dto.environment import AvailableActions, RoundInfo
from TheCausalityGame.core.contracts.serializable import Serializable
from TheCausalityGame.core.infrastructure.decisions import Decision
from TheCausalityGame.core.infrastructure.logger import Logger
from TheCausalityGame.core.lib.errors.agent import (
    AgentContextNotSetError,
    AgentLoggerNotSetError,
)


class AgentPolicy(Serializable):
    """Unified component that jointly decides and answers."""

    _context: AgentContext | None = None
    _logger: Logger | None = None
    _spec: str = "TheCausalityGame.core.contracts.specs.agent_policy:AgentPolicySpec"

    def set_context(self, ctx: AgentContext) -> None:
        """Assign runtime context to the policy."""
        if self._context is None:
            self._context = ctx

    @property
    def context(self) -> AgentContext:
        """Return policy runtime context."""
        if self._context is None:
            raise AgentContextNotSetError()
        return self._context

    def set_logger(self, logger: Logger) -> None:
        """Assign a logger to the policy."""
        if self._logger is None:
            self._logger = logger

    @property
    def logger(self) -> Logger:
        """Return the assigned policy logger."""
        if self._logger is None:
            raise AgentLoggerNotSetError()
        return self._logger

    @abstractmethod
    def update(self, observation: RoundObservation) -> None:
        """Update the policy state from a new round observation."""
        raise NotImplementedError

    @abstractmethod
    def decide(
        self,
        round_info: RoundInfo,
        available_actions: AvailableActions,
    ) -> Decision:
        """Choose the next decision."""
        raise NotImplementedError

    @abstractmethod
    def answer(self) -> Any:  # noqa: ANN401
        """Return the policy's current task estimate."""
        raise NotImplementedError
