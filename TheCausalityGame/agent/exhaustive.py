"""The Causality Game - Exhaustive Agent with multiple strategies."""

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


class ExhaustiveAgent(CommonAgent):
    """
    Agent that performs exhaustive experimentation and learns conditional treatment effects.

    Parameters
    ----------
    id : str
        Unique identifier for the agent.
    num_obs : int, optional
        Number of observational samples to collect, by default 1.
    num_inter : int, optional
        Number of interventional samples per treatment value, by default 1.
    """

    def __init__(
        self,
        id: str,
        num_obs: int = 1,
        num_inter: int = 1,
    ) -> None:
        self.id = id
        self._num_obs = num_obs
        self._num_inter = num_inter
        self.rng = np.random.default_rng()

    @override
    def act(self, round_info: RoundInfo, available_actions: AvailableActions) -> Decision:
        decision = Decision.experiment()
        decision.add_experiment(treatment=None, n=self._num_obs)

        for var in available_actions.experiments:
            low, high = var.domain
            if isinstance(low, str):  # Categorical variable
                for val in var.domain:
                    decision.add_experiment({var.name: val}, n=self._num_inter)
            else:  # Numerical variable
                # decision.add_experiment({var.name: low}, n=self._num_inter)
                # decision.add_experiment({var.name: high}, n=self._num_inter)
                # for val in np.linspace(float(low), float(high), num=3):
                #     decision.add_experiment({var.name: val}, n=self._num_inter)
                for _ in range(5):
                    decision.add_experiment(
                        {var.name: self.rng.uniform(float(low), float(high))}, n=self._num_inter
                    )

        return decision

    @override
    def to_spec(self) -> AgentSpec:
        params = {
            "num_obs": self._num_obs,
            "num_inter": self._num_inter,
        }
        return AgentSpec(
            id=self.id,
            class_=get_class_path(self.__class__),
            params=params,
        )

    @classmethod
    @override
    def from_spec(cls, spec: AgentSpec) -> ExhaustiveAgent:
        if not spec.params:
            return cls(id=spec.id)

        return cls(
            id=spec.id,
            num_obs=spec.params.get("num_obs", 1),
            num_inter=spec.params.get("num_inter", 1),
        )
