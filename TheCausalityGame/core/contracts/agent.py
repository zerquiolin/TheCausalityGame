"""The Causality Game - Agent Contract."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping  # noqa: UP035

from TheCausalityGame.core.contracts.decisions import Decision
from TheCausalityGame.core.contracts.dto.environment import (
    AvailableActions,
    Feedback,
    RoundInfo,
    SamplesCollection,
)
from TheCausalityGame.core.contracts.serializable import Serializable


@dataclass(frozen=True, slots=True)
class AgentContext:
    """Execution context stored on an agent instance.

    Attributes
    ----------
        base_seed: Optional global seed used as root for deterministic derivations.
        game_scenario: Hints about mission and constraints (max rounds, budgets, etc.).
            Recommended keys:
              - mission_id: str
              - max_rounds: int
              - budgets: {time_s: float|None, samples: int|None, memory_mb: int|None}
              - constraints: Mapping[str, Any]

    """

    mission: Mapping[str, str]
    behavior_metric: Mapping[str, str]
    result_metric: Mapping[str, str]
    custom_metrics: list[Mapping[str, str]]
    seed: int


class Agent(Serializable):
    """Serializable agent base with an explicit, minimal surface.

    Agents MUST implement:
      - act(round_info, available_actions) -> Decision
      - inform(outcome) -> None
      - answer() -> Any
    """

    id: str
    _context: AgentContext | None = None

    # -------- context management --------

    @abstractmethod
    def set_context(self, ctx: AgentContext) -> None:
        """Inject the runtime context exactly once per agent instance."""
        if self._context is None:
            self._context = ctx

    @property
    def context(self) -> AgentContext:
        """Return the injected runtime context."""
        if self._context is None:
            raise RuntimeError("Agent context not set")
        return self._context

    # -------- required public API --------

    @abstractmethod
    def act(
        self, round_info: RoundInfo, available_actions: AvailableActions
    ) -> Decision:
        """Select and return a Decision for this round."""
        raise NotImplementedError

    @abstractmethod
    def inform(self, samples_collection: SamplesCollection, feedback: Feedback) -> None:
        """Receive the result of the previously executed action (samples/feedback)."""
        raise NotImplementedError

    @abstractmethod
    def answer(self) -> Any:
        """Return the agent's current best result/estimate (pure query)."""
        raise NotImplementedError
