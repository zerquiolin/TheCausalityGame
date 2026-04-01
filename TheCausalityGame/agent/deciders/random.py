"""Random decider implementation."""

from __future__ import annotations

from typing import override

import numpy as np

from TheCausalityGame.core.contracts.decider import Decider
from TheCausalityGame.core.contracts.dto.agent import BeliefSnapshot
from TheCausalityGame.core.contracts.dto.environment import AvailableActions, RoundInfo
from TheCausalityGame.core.contracts.specs.decider import DeciderSpec
from TheCausalityGame.core.infrastructure.decisions import Decision
from TheCausalityGame.core.infrastructure.registry import get_class_path


class RandomDecider(Decider):
    """Select a single observation or intervention uniformly at random each round."""

    def __init__(self, num_obs: int = 1, num_inter: int = 3, threshold: float = 0.5) -> None:
        self._num_obs = num_obs
        self._num_inter = num_inter
        self._threshold = threshold
        self._rng = np.random.default_rng(1)

    @override
    def decide(
        self,
        round_info: RoundInfo,
        available_actions: AvailableActions,
        belief: BeliefSnapshot,
    ) -> Decision:
        del round_info, belief
        decision = Decision.experiment()
        if self._rng.uniform(0, 1) < self._threshold:
            decision.add_experiment(treatment=None, n=self._num_obs)
            return decision

        index = self._rng.integers(0, len(available_actions.experiments))
        experiment = available_actions.experiments[index]
        low, high = experiment.domain
        if isinstance(low, str):
            decision.add_experiment(
                {experiment.name: self._rng.choice(experiment.domain)},
                n=self._num_inter,
            )
        else:
            value = self._rng.uniform(float(low), float(high))
            decision.add_experiment({experiment.name: value}, n=self._num_inter)
        return decision

    @override
    def to_spec(self) -> DeciderSpec:
        return DeciderSpec(
            class_=get_class_path(self.__class__),
            params={
                "num_obs": self._num_obs,
                "num_inter": self._num_inter,
                "threshold": self._threshold,
            },
        )

    @classmethod
    @override
    def from_spec(cls, spec: DeciderSpec) -> RandomDecider:
        params = spec.params or {}
        return cls(
            num_obs=int(params.get("num_obs", 1)),
            num_inter=int(params.get("num_inter", 3)),
            threshold=float(params.get("threshold", 0.5)),
        )
