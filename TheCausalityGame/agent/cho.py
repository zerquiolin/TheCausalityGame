from __future__ import annotations

from typing import Any, override

import numpy as np

from TheCausalityGame.agent.common import CommonAgent
from TheCausalityGame.agent.helpers.ridge_bootstrap_edge_belief import RidgeBootstrapEdgeBelief
from TheCausalityGame.core.contracts.dto.environment import (
    AvailableActions,
    RoundInfo,
    SamplesCollection,
)
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
        num_obs: int = 5,
        num_inter: int = 1,
        n_bootstrap: int = 32,
        ridge_lambda: float = 1e-2,
        coef_threshold: float = 1e-2,
        k_intervene: int = 1,
        seed: int | None = 911,
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
        self.k_intervene = int(max(1, k_intervene))

    @override
    def inform(self, samples_collection: SamplesCollection) -> None:
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

        # Choose WHERE: greedily pick up to k_intervene variables by incident uncertainty
        candidates = sorted(
            list(available_actions.experiments),
            key=lambda v: (-float(self._belief.incident_uncertainty(v.name)), str(v.name)),
        )
        chosen = candidates[: min(self.k_intervene, len(candidates))]

        if not chosen:
            return decision

        # Choose HOW: sample interior value for numeric domains, else categorical
        treatment: dict[str, Any] = {}
        for v in chosen:
            dom = list(v.domain)
            if not dom:
                continue

            if len(dom) >= 2 and all(
                isinstance(x, (int, float, np.integer, np.floating)) for x in dom[:2]
            ):
                low, high = float(dom[0]), float(dom[1])
                if high < low:
                    low, high = high, low
                val = float(self.rng.uniform(low, high))
            else:
                val = dom[self.rng.integers(0, len(dom))]

            treatment[v.name] = val

        if not treatment:
            return decision

        decision.add_experiment(treatment=treatment, n=self._num_inter)
        return decision

    @override
    def to_spec(self) -> AgentSpec:
        params = {
            "num_obs": self._num_obs,
            "num_inter": self._num_inter,
            "n_bootstrap": self._belief.n_bootstrap,
            "ridge_lambda": self._belief.ridge_lambda,
            "coef_threshold": self._belief.coef_threshold,
            "k_intervene": self.k_intervene,
        }
        return AgentSpec(id=self.id, class_=get_class_path(self.__class__), params=params)

    @classmethod
    @override
    def from_spec(cls, spec: AgentSpec) -> "Cho2016ActiveGBNAgent":
        if not spec.params:
            return cls(id=spec.id)

        return cls(
            id=spec.id,
            num_obs=spec.params.num_obs or None,  # type: ignore
            num_inter=spec.params.num_inter or None,  # type: ignore
            n_bootstrap=spec.params.n_bootstrap or None,  # type: ignore
            ridge_lambda=spec.params.ridge_lambda or None,  # type: ignore
            coef_threshold=spec.params.coef_threshold or None,  # type: ignore
            k_intervene=spec.params.k_intervene or None,  # type: ignore
        )
