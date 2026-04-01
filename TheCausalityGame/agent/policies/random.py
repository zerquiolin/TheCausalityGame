"""Simple random unified agent policy."""

from __future__ import annotations

from typing import Any, override

import numpy as np

from TheCausalityGame.core.contracts.agent_policy import AgentPolicy
from TheCausalityGame.core.contracts.dto.agent import RoundObservation
from TheCausalityGame.core.contracts.dto.environment import AvailableActions, RoundInfo
from TheCausalityGame.core.contracts.specs.agent_policy import AgentPolicySpec
from TheCausalityGame.core.infrastructure.decisions import Decision
from TheCausalityGame.core.infrastructure.registry import get_class_path


class RandomAgentPolicy(AgentPolicy):
    """Reference combined policy mainly intended to exercise the combined path."""

    def __init__(self, num_obs: int = 1, num_inter: int = 1, threshold: float = 0.5) -> None:
        self._num_obs = int(num_obs)
        self._num_inter = int(num_inter)
        self._threshold = float(threshold)
        self._rng = np.random.default_rng(17)
        self._rounds_seen = 0

    @override
    def update(self, observation: RoundObservation) -> None:
        self._rounds_seen = max(self._rounds_seen, observation.round_info.round)

    @override
    def decide(
        self,
        round_info: RoundInfo,
        available_actions: AvailableActions,
    ) -> Decision:
        del round_info
        decision = Decision.experiment()
        if self._rng.uniform(0.0, 1.0) < self._threshold or not available_actions.experiments:
            decision.add_experiment(treatment=None, n=self._num_obs)
            return decision

        var = available_actions.experiments[self._rng.integers(0, len(available_actions.experiments))]
        dom = list(var.domain)
        if len(dom) >= 2 and all(isinstance(x, (int, float, np.integer, np.floating)) for x in dom[:2]):
            low, high = float(dom[0]), float(dom[1])
            if high < low:
                low, high = high, low
            value: Any = float(self._rng.uniform(low, high))
        else:
            value = dom[self._rng.integers(0, len(dom))]
        decision.add_experiment(treatment={var.name: value}, n=self._num_inter)
        return decision

    @override
    def answer(self) -> Any:
        return {"rounds_seen": self._rounds_seen}

    @override
    def to_spec(self) -> AgentPolicySpec:
        return AgentPolicySpec(
            class_=get_class_path(self.__class__),
            params={
                "num_obs": self._num_obs,
                "num_inter": self._num_inter,
                "threshold": self._threshold,
            },
        )

    @classmethod
    @override
    def from_spec(cls, spec: AgentPolicySpec) -> RandomAgentPolicy:
        params = spec.params or {}
        return cls(
            num_obs=int(params.get("num_obs", 1)),
            num_inter=int(params.get("num_inter", 1)),
            threshold=float(params.get("threshold", 0.5)),
        )
