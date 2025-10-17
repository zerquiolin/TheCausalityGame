"""The Causality Game - Agent Contract."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping  # noqa: UP035

from TheCausalityGame.core.contracts.dto.environment import (
    AvailableActions,
    Feedback,
    RoundInfo,
    SamplesCollection,
)
from TheCausalityGame.core.contracts.serializable import Serializable
from TheCausalityGame.core.infrastructure.decisions import Decision
from TheCausalityGame.core.infrastructure.logger import Logger
from TheCausalityGame.core.lib.errors.agent import (
    AgentContextNotSetError,
    AgentLoggerNotSetError,
)


@dataclass(frozen=True, slots=True)
class AgentContext:
    """Execution context provided to an agent during runtime.

    Attributes
    ----------
    mission : Mapping[str, str]
        Information about the mission, such as name and description.
    behavior_metric : Mapping[str, str]
        Details of the metric evaluating agent behavior.
    result_metric : Mapping[str, str]
        Details of the metric evaluating agent performance.
    custom_metrics : list[Mapping[str, str]]
        Additional custom metrics associated with the mission.
    seed : int
        Global seed used for deterministic operations.
    """

    mission: Mapping[str, str]
    behavior_metric: Mapping[str, str]
    result_metric: Mapping[str, str]
    custom_metrics: list[Mapping[str, str]]
    seed: int


class Agent(Serializable):
    """
    Base class for all agents in The Causality Game.

    Agents must implement three core methods:
      - `act(round_info, available_actions)`: choose an experiment or answer.
      - `inform(samples_collection, feedback)`: process observed outcomes.
      - `answer()`: return the current best estimate.

    Agents are serializable and can be reconstructed from specifications.
    """

    id: str
    _context: AgentContext | None = None
    _logger: Logger | None = None

    # ---------------- Context Management ---------------- #

    def set_context(self, ctx: AgentContext) -> None:
        """Assign the runtime context to the agent (only once)."""
        if self._context is None:
            self._context = ctx

    @property
    def context(self) -> AgentContext:
        """Return the agent's runtime context."""
        if self._context is None:
            raise AgentContextNotSetError()
        return self._context

    # ---------------- Logger Management ---------------- #

    def set_logger(self, logger: Logger) -> None:
        """Assign a logger to the agent (only once)."""
        if self._logger is None:
            self._logger = logger

    @property
    def logger(self) -> Logger:
        """Return the assigned logger instance."""
        if self._logger is None:
            raise AgentLoggerNotSetError()
        return self._logger

    # ---------------- Required Public API ---------------- #

    @abstractmethod
    def act(
        self,
        round_info: RoundInfo,
        available_actions: AvailableActions,
    ) -> Decision:
        """
        Select and return a decision based on the current round state.

        Parameters
        ----------
        round_info : RoundInfo
            Information about the current round of interaction.
        available_actions : AvailableActions
            Valid actions the agent may take at this step.

        Returns
        -------
        Decision
            The agent's selected decision.
        """
        raise NotImplementedError

    @abstractmethod
    def inform(
        self,
        samples_collection: SamplesCollection,
        feedback: Feedback,
    ) -> None:
        """
        Receive and process feedback from the environment after an action.

        Parameters
        ----------
        samples_collection : SamplesCollection
            The observational data resulting from the action.
        feedback : Feedback
            Metadata about rewards, penalties, or mission-specific signals.
        """
        raise NotImplementedError

    @abstractmethod
    def answer(self) -> Any:  # noqa :ANN401
        """
        Return the agent's current best causal estimate.

        Returns
        -------
        Any
            The agent's prediction or structured output.
        """
        raise NotImplementedError
