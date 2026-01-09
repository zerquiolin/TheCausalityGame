"""The Causality Game - Exhaustive Agent with multiple strategies."""

from __future__ import annotations

from typing import Any, override

import numpy as np

from TheCausalityGame.agent.strategies.cate_strategy import CATEStrategy
from TheCausalityGame.core.contracts.agent import Agent, AgentContext
from TheCausalityGame.core.contracts.dto.environment import (
    AvailableActions,
    RoundInfo,
    SamplesCollection,
)
from TheCausalityGame.core.contracts.specs.agent import AgentSpec
from TheCausalityGame.core.infrastructure.decisions import Decision
from TheCausalityGame.core.infrastructure.registry import get_class_path
from TheCausalityGame.core.infrastructure.strategy import Strategy


class ExhaustiveAgent(Agent):
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
        rng = np.random.default_rng()
        self.id = id
        self._num_obs = num_obs
        self._num_inter = num_inter
        self._counter = rng.integers(7000, 12500)
        self._should_answer = False

    @override
    def set_context(self, ctx: AgentContext) -> None:
        self._context = ctx

        strategies: dict[str, Strategy] = {
            "Conditional Average Treatment Effect Mission": CATEStrategy()
        }

        self.strategy = strategies[self._context.mission["name"]]
        self.strategy.initialize()

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
                for val in np.linspace(float(low), float(high), num=5):
                    decision.add_experiment({var.name: val}, n=self._num_inter)

        return decision

    @override
    def inform(self, samples_collection: SamplesCollection) -> None:
        self.strategy.learn(samples_collection)

    @override
    def answer(self) -> Any:
        return self.strategy.answer()

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
