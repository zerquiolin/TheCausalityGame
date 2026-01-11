"""The Causality Game - Exhaustive Agent with multiple strategies."""

from __future__ import annotations

from typing import Any, override

from TheCausalityGame.agent.strategies.cate_strategy import CATEStrategy
from TheCausalityGame.agent.strategies.dag_strategy import DAGDiscoveryStrategy
from TheCausalityGame.agent.strategies.scm_strategy import SCMDiscoveryStrategy
from TheCausalityGame.core.contracts.agent import Agent, AgentContext
from TheCausalityGame.core.contracts.dto.environment import SamplesCollection
from TheCausalityGame.core.infrastructure.strategy import Strategy


class CommonAgent(Agent):
    """Agent that performs over the strategies available."""

    @override
    def set_context(self, ctx: AgentContext) -> None:
        self._context = ctx

        strategies: dict[str, Strategy] = {
            "Conditional Average Treatment Effect Mission": CATEStrategy(),
            "DAG Discovery Mission": DAGDiscoveryStrategy(),
            "SCM Estimation Mission": SCMDiscoveryStrategy(),
        }

        self.strategy = strategies[self._context.mission["name"]]
        self.strategy.initialize()

    @override
    def inform(self, samples_collection: SamplesCollection) -> None:
        self.strategy.learn(samples_collection)

    @override
    def answer(self) -> Any:
        return self.strategy.answer()
