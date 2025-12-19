"""Agent variant implementing an upper-confidence bound action policy."""

from __future__ import annotations

from typing import override

import numpy as np

from TheCausalityGame.core.contracts.dto.environment import (
    AvailableActions,
    RoundInfo,
    SamplesCollection,
)
from TheCausalityGame.core.contracts.specs.agent import AgentSpec
from TheCausalityGame.core.infrastructure.decisions import Decision

from ._action_utils import ActionKey, collect_single_variable_candidates
from .exhaustive import ExhaustiveAgent


class IntelligentAgent(ExhaustiveAgent):
    """Agent that balances exploration and exploitation via an upper-confidence bound."""

    def __init__(
        self,
        id: str,
        num_obs: int = 1,
        num_inter: int = 1,
        exploration_weight: float = 1.5,
        *,
        max_rounds: int | None = None,
        target_result: float | None = None,
        target_score: float | None = None,
        patience: int | None = None,
        tolerance: float = 1e-6,
    ) -> None:
        super().__init__(
            id=id,
            num_obs=num_obs,
            num_inter=num_inter,
            max_rounds=max_rounds,
            target_result=target_result,
            target_score=target_score,
            patience=patience,
            tolerance=tolerance,
        )
        self._exploration_weight = exploration_weight
        self._action_values: dict[ActionKey, float] = {}
        self._action_counts: dict[ActionKey, int] = {}
        self._last_action: ActionKey | None = None
        self._rng = np.random.default_rng()
        self._last_reward: float | None = None

    def _ucb_score(self, key: ActionKey) -> float:
        count = self._action_counts.get(key, 0)
        mean = self._action_values.get(key, 0.0)
        if count == 0:
            return float("inf")
        total = sum(self._action_counts.values())
        exploration = self._exploration_weight * np.sqrt(np.log(total + 1) / count)
        return mean + exploration

    @override
    def act(
        self, round_info: RoundInfo, available_actions: AvailableActions
    ) -> Decision:
        if self._stopping_policy.should_stop_on_round(round_info):
            self._should_answer = True

        if self._should_answer or (
            self._stopping_policy.max_rounds is None and round_info.round >= self._counter
        ):
            return Decision.answer()

        decision = Decision.experiment()
        decision.add_experiment(treatment=None, n=self._num_obs)

        candidates = collect_single_variable_candidates(available_actions)
        if not candidates:
            self._should_answer = True
            return Decision.answer()

        for key in candidates:
            self._action_values.setdefault(key, 0.0)
            self._action_counts.setdefault(key, 0)

        chosen_key = max(candidates.keys(), key=self._ucb_score)
        treatment = candidates[chosen_key]
        decision.add_experiment(treatment, n=self._num_inter)
        self._last_action = chosen_key
        return decision

    @override
    def inform(self, samples_collection: SamplesCollection) -> None:
        super().inform(samples_collection)

        if self._last_action is None or self._last_action not in self._action_values:
            return

        reward = self._last_reward
        if reward is None:
            return

        count = self._action_counts[self._last_action] + 1
        current = self._action_values[self._last_action]
        updated = current + (reward - current) / count

        self._action_counts[self._last_action] = count
        self._action_values[self._last_action] = updated
        self._last_action = None
        self._last_reward = None

    @override
    def to_spec(self) -> AgentSpec:
        base_spec = super().to_spec()
        params = dict(base_spec.params or {})
        params["exploration_weight"] = self._exploration_weight
        return AgentSpec(id=base_spec.id, class_=base_spec.class_, params=params)

    @classmethod
    @override
    def from_spec(cls, spec: AgentSpec) -> "IntelligentAgent":
        params = spec.params or {}
        return cls(
            id=spec.id,
            num_obs=params.get("num_obs", 1),
            num_inter=params.get("num_inter", 1),
            exploration_weight=params.get("exploration_weight", 1.5),
            max_rounds=params.get("max_rounds"),
            target_score=params.get("target_score", params.get("target_result")),
            patience=params.get("patience"),
            tolerance=params.get("tolerance", 1e-6),
        )

    @staticmethod
    def _estimate_reward(samples_collection: SamplesCollection) -> float | None:
        if not samples_collection:
            return None
        return float(-samples_collection.total_n())

    @override
    def _progress_score(self, samples_collection: SamplesCollection) -> float | None:
        reward = self._estimate_reward(samples_collection)
        self._last_reward = reward
        return reward

    @staticmethod
    def _estimate_reward(samples_collection: SamplesCollection) -> float | None:
        if not samples_collection:
            return None
        return float(-samples_collection.total_n())
