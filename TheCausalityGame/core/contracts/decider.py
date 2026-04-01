"""The Causality Game - Decider Contract."""

from __future__ import annotations

from abc import abstractmethod

from TheCausalityGame.core.contracts.agent import AgentContext
from TheCausalityGame.core.contracts.dto.agent import BeliefSnapshot, RoundObservation
from TheCausalityGame.core.contracts.dto.environment import AvailableActions, RoundInfo
from TheCausalityGame.core.contracts.serializable import Serializable
from TheCausalityGame.core.infrastructure.decisions import Decision
from TheCausalityGame.core.infrastructure.logger import Logger
from TheCausalityGame.core.lib.errors.agent import (
    AgentContextNotSetError,
    AgentLoggerNotSetError,
)


class Decider(Serializable):
    """Active-learning component used by composable agents."""

    _context: AgentContext | None = None
    _logger: Logger | None = None
    _spec: str = "TheCausalityGame.core.contracts.specs.decider:DeciderSpec"

    def set_context(self, ctx: AgentContext) -> None:
        """Assign runtime context to the decider."""
        if self._context is None:
            self._context = ctx

    @property
    def context(self) -> AgentContext:
        """Return decider runtime context."""
        if self._context is None:
            raise AgentContextNotSetError()
        return self._context

    def set_logger(self, logger: Logger) -> None:
        """Assign a logger to the decider."""
        if self._logger is None:
            self._logger = logger

    @property
    def logger(self) -> Logger:
        """Return the assigned decider logger."""
        if self._logger is None:
            raise AgentLoggerNotSetError()
        return self._logger

    def required_capabilities(self) -> frozenset[str]:
        """Return inferer capabilities required by this decider."""
        return frozenset()

    def update(self, observation: RoundObservation) -> None:
        """Update the decider state from a new round observation."""
        del observation

    @abstractmethod
    def decide(
        self,
        round_info: RoundInfo,
        available_actions: AvailableActions,
        belief: BeliefSnapshot,
    ) -> Decision:
        """Choose the next decision using the current inferer snapshot."""
        raise NotImplementedError
