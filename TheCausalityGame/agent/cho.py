from __future__ import annotations

from typing import override

import numpy as np

from TheCausalityGame.agent.common import CommonAgent
from TheCausalityGame.agent.helpers.ridge_bootstrap_edge_belief import RidgeBootstrapEdgeBelief
from TheCausalityGame.core.contracts.dto.environment import AvailableActions, RoundInfo
from TheCausalityGame.core.contracts.specs.agent import AgentSpec
from TheCausalityGame.core.infrastructure.decisions import Decision
from TheCausalityGame.core.infrastructure.registry import get_class_path


class Cho2016ActiveGBNAgent(CommonAgent):
    """
    Cho-style active learning proxy:
      - pick the variable (WHERE) with highest incident uncertainty
      - intervene at a canonical baseline value (no HOW optimization)
    """

    def __init__(
        self,
        id: str,
        num_obs: int = 0,
        num_inter: int = 10,
        n_bootstrap: int = 32,
        ridge_lambda: float = 1e-2,
        coef_threshold: float = 1e-2,
        seed: int | None = None,
    ) -> None:
        self.id = id
        self._num_obs = int(num_obs)
        self._num_inter = int(num_inter)
        self._belief = RidgeBootstrapEdgeBelief(
            n_bootstrap=n_bootstrap,
            ridge_lambda=ridge_lambda,
            coef_threshold=coef_threshold,
            seed=seed,
        )
        self.rng = np.random.default_rng(seed)

    @override
    def inform(self, samples_collection) -> None:
        self.strategy.learn(samples_collection)
        self._belief.fit(list(samples_collection))

    @override
    def act(self, round_info: RoundInfo, available_actions: AvailableActions) -> Decision:
        decision = Decision.experiment()

        # modest observational probe (optional)
        if self._num_obs > 0:
            decision.add_experiment(treatment=None, n=self._num_obs)

        if self._num_inter <= 0 or not available_actions.experiments:
            return decision

        # Choose WHERE: max incident uncertainty
        best_var = None
        best_u = -1.0
        for var in available_actions.experiments:
            u = self._belief.incident_uncertainty(var.name)
            if u > best_u:
                best_u = u
                best_var = var

        if best_var is None:
            return decision

        # Choose HOW: sample interior value for numeric domains, else categorical
        dom = list(best_var.domain)

        # For numeric domains represented as [low, high], sample an interior value to avoid endpoint bias.
        if len(dom) >= 2 and all(
            isinstance(x, (int, float, np.integer, np.floating)) for x in dom[:2]
        ):
            low, high = float(dom[0]), float(dom[1])
            if high < low:
                low, high = high, low
            val = float(self.rng.uniform(low, high))
        else:
            # categorical / enumerated
            val = dom[self.rng.integers(0, len(dom))]

        decision.add_experiment(treatment={best_var.name: val}, n=self._num_inter)
        return decision

    @override
    def to_spec(self) -> AgentSpec:
        params = {
            "num_obs": self._num_obs,
            "num_inter": self._num_inter,
            "n_bootstrap": self._belief.n_bootstrap,
            "ridge_lambda": self._belief.ridge_lambda,
            "coef_threshold": self._belief.coef_threshold,
        }
        return AgentSpec(id=self.id, class_=get_class_path(self.__class__), params=params)

    @classmethod
    @override
    def from_spec(cls, spec: AgentSpec) -> "Cho2016ActiveGBNAgent":
        p = spec.params or {}
        return cls(
            id=spec.id,
            num_obs=p.get("num_obs", 1),
            num_inter=p.get("num_inter", 1),
            n_bootstrap=p.get("n_bootstrap", 32),
            ridge_lambda=p.get("ridge_lambda", 1e-2),
            coef_threshold=p.get("coef_threshold", 1e-2),
        )
