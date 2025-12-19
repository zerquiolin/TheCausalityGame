"""Random agent variant built on top of the exhaustive agent."""

from __future__ import annotations

from typing import override

import numpy as np

from TheCausalityGame.core.contracts.dto.environment import (
    AvailableActions,
    RoundInfo,
)
from TheCausalityGame.core.infrastructure.decisions import Decision

from ._action_utils import collect_single_variable_candidates
from .exhaustive import ExhaustiveAgent


class RandomAgent(ExhaustiveAgent):
    """Agent that selects a single intervention uniformly at random each round."""

    def __init__(
        self,
        id: str,
        num_obs: int = 1,
        num_inter: int = 1,
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
        self._rng = np.random.default_rng()

    @override
    def act(
        self, round_info: RoundInfo, available_actions: AvailableActions
    ) -> Decision:
        if self._stopping_policy.should_stop_on_round(round_info):
            self._should_answer = True

        if self._should_answer or (
            self._stopping_policy.max_rounds is None
            and round_info.round >= self._counter
        ):
            return Decision.answer()

        decision = Decision.experiment()
        decision.add_experiment(treatment=None, n=self._num_obs)

        candidates = collect_single_variable_candidates(available_actions)
        if not candidates:
            self._should_answer = True
            return Decision.answer()

        keys = list(candidates.keys())
        index = self._rng.integers(0, len(keys))
        chosen_key = keys[index]
        treatment = candidates[chosen_key]
        decision.add_experiment(treatment, n=self._num_inter)
        return decision
