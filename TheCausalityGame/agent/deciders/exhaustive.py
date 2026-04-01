"""Exhaustive decider implementation."""

from __future__ import annotations

from typing import override

import numpy as np

from TheCausalityGame.core.contracts.decider import Decider
from TheCausalityGame.core.contracts.dto.agent import BeliefSnapshot
from TheCausalityGame.core.contracts.dto.environment import AvailableActions, RoundInfo
from TheCausalityGame.core.contracts.specs.decider import DeciderSpec
from TheCausalityGame.core.infrastructure.decisions import Decision
from TheCausalityGame.core.infrastructure.registry import get_class_path


class ExhaustiveDecider(Decider):
    """Request observational data plus a broad sweep of interventions each round."""

    def __init__(self, num_obs: int = 1, num_inter: int = 1) -> None:
        self._num_obs = num_obs
        self._num_inter = num_inter
        self.rng = np.random.default_rng(911)

    @override
    def decide(
        self,
        round_info: RoundInfo,
        available_actions: AvailableActions,
        belief: BeliefSnapshot,
    ) -> Decision:
        del round_info, belief
        decision = Decision.experiment()
        decision.add_experiment(treatment=None, n=self._num_obs)

        for var in available_actions.experiments:
            low, high = var.domain
            if isinstance(low, str):
                for val in var.domain:
                    decision.add_experiment({var.name: val}, n=self._num_inter)
            else:
                for _ in range(5):
                    decision.add_experiment(
                        {var.name: self.rng.uniform(float(low), float(high))},
                        n=self._num_inter,
                    )

        return decision

    @override
    def to_spec(self) -> DeciderSpec:
        return DeciderSpec(
            class_=get_class_path(self.__class__),
            params={
                "num_obs": self._num_obs,
                "num_inter": self._num_inter,
            },
        )

    @classmethod
    @override
    def from_spec(cls, spec: DeciderSpec) -> ExhaustiveDecider:
        params = spec.params or {}
        return cls(
            num_obs=int(params.get("num_obs", 1)),
            num_inter=int(params.get("num_inter", 1)),
        )
