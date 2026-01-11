"""Random agent variant built on top of the exhaustive agent."""

from __future__ import annotations

from typing import override

import numpy as np

from TheCausalityGame.agent.common import CommonAgent
from TheCausalityGame.core.contracts.dto.environment import (
    AvailableActions,
    RoundInfo,
)
from TheCausalityGame.core.contracts.specs.agent import AgentSpec
from TheCausalityGame.core.infrastructure.decisions import Decision
from TheCausalityGame.core.infrastructure.registry import get_class_path


class RandomAgent(CommonAgent):
    """Agent that selects a single observation / intervention uniformly at random each round."""

    def __init__(
        self,
        id: str,
        num_obs: int = 1,
        num_inter: int = 1,
        threshold: float = 0.5,
    ) -> None:
        self.id = id
        self._num_obs = num_obs
        self._num_inter = num_inter
        self._threshold = threshold
        self._rng = np.random.default_rng()

    @override
    def act(self, round_info: RoundInfo, available_actions: AvailableActions) -> Decision:
        decision = Decision.experiment()
        if self._rng.uniform(0, 1) < self._threshold:
            decision.add_experiment(treatment=None, n=self._num_obs)
            return decision

        index = self._rng.integers(0, len(available_actions.experiments))
        experiment = available_actions.experiments[index]
        low, high = experiment.domain
        if type(low) is str:
            decision.add_experiment(
                {experiment.name: self._rng.choice(experiment.domain)}, n=self._num_inter
            )
        else:
            value = self._rng.uniform(float(low), float(high))
            decision.add_experiment({experiment.name: value}, n=self._num_inter)
        return decision

    @override
    def to_spec(self) -> AgentSpec:
        params = {
            "num_obs": self._num_obs,
            "num_inter": self._num_inter,
            "threshold": self._threshold,
        }
        return AgentSpec(
            id=self.id,
            class_=get_class_path(self.__class__),
            params=params,
        )

    @classmethod
    @override
    def from_spec(cls, spec: AgentSpec) -> RandomAgent:
        if not spec.params:
            return cls(id=spec.id)

        return cls(
            id=spec.id,
            num_obs=spec.params.get("num_obs", 1),
            num_inter=spec.params.get("num_inter", 1),
            threshold=spec.params.get("threshold", 0.5),
        )
