"""The Causality Game - Agent Contract."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping  # noqa: UP035

from TheCausalityGame.core.contracts.decisions import Decision
from TheCausalityGame.core.contracts.dto import (
    ActionOutcome,
    AvailableActions,
    RoundInfo,
)
from TheCausalityGame.core.contracts.serializable import Serializable


@dataclass(frozen=True, slots=True)
class AgentContext:
    """Execution context stored on an agent instance.

    Attributes
    ----------
        config: Agent configuration (immutable view).
        manifest_id: Id of the current ProblemInstance.
        agent_id: Id of the running agent.
        base_seed: Optional global seed used as root for deterministic derivations.
        game_scenario: Hints about mission and constraints (max rounds, budgets, etc.).
            Recommended keys:
              - mission_id: str
              - max_rounds: int
              - budgets: {time_s: float|None, samples: int|None, memory_mb: int|None}
              - constraints: Mapping[str, Any]

    """

    config: Mapping[str, Any]
    manifest_id: str
    agent_id: str
    base_seed: int | None = None
    game_scenario: Mapping[str, Any] | None = None


class Agent(Serializable):
    """Serializable agent base with an explicit, minimal surface.

    Agents MUST implement:
      - act(round_info, available_actions) -> Decision
      - inform(outcome) -> None
      - answer() -> Any
    """

    _context: AgentContext | None = None

    # -------- context management --------

    def set_context(self, ctx: AgentContext) -> None:
        """Inject the runtime context exactly once per agent instance."""
        if self._context is None:
            self._context = ctx

    @property
    def context(self) -> AgentContext:
        """Return the injected runtime context."""
        if self._context is None:
            raise RuntimeError(
                "Agent context not set. Orchestrator must call `agent.set_context(...)` "
                "before running the agent."
            )
        return self._context

    # -------- required public API --------

    @abstractmethod
    def act(
        self, round_info: RoundInfo, available_actions: AvailableActions
    ) -> Decision:
        """Select and return a Decision for this round."""
        raise NotImplementedError

    @abstractmethod
    def inform(self, outcome: ActionOutcome) -> None:
        """Receive the result of the previously executed action (samples/feedback)."""
        raise NotImplementedError

    @abstractmethod
    def answer(self) -> Any:
        """Return the agent's current best result/estimate (pure query)."""
        raise NotImplementedError
