"""Cho-style active intervention decider."""

from __future__ import annotations

from typing import override

import numpy as np

from TheCausalityGame.agent.helpers.ridge_bootstrap_edge_belief import RidgeBootstrapEdgeBelief
from TheCausalityGame.core.contracts.decider import Decider
from TheCausalityGame.core.contracts.dto.agent import BeliefSnapshot, RoundObservation
from TheCausalityGame.core.contracts.dto.environment import AvailableActions, RoundInfo
from TheCausalityGame.core.contracts.specs.decider import DeciderSpec
from TheCausalityGame.core.infrastructure.decisions import Decision
from TheCausalityGame.core.infrastructure.registry import get_class_path

DOMAIN_BOUNDS_COUNT = 2


class Cho2016ActiveGBNDecider(Decider):
    """Cho-style active intervention policy."""

    def __init__(  # noqa: PLR0913
        self,
        num_obs: int = 5,
        num_inter: int = 1,
        n_bootstrap: int = 32,
        ridge_lambda: float = 1e-2,
        coef_threshold: float = 1e-2,
        k_intervene: int = 1,
        seed: int | None = 911,
    ) -> None:
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
    def update(self, observation: RoundObservation) -> None:
        self._belief.fit(list(observation.samples))

    @override
    def decide(
        self,
        round_info: RoundInfo,
        available_actions: AvailableActions,
        belief: BeliefSnapshot,
    ) -> Decision:
        del round_info, belief
        decision = Decision.experiment()

        if self._num_obs > 0:
            decision.add_experiment(treatment=None, n=self._num_obs)

        if self._num_inter <= 0 or not available_actions.experiments:
            return decision

        candidates = sorted(
            available_actions.experiments,
            key=lambda v: (-float(self._belief.incident_uncertainty(v.name)), str(v.name)),
        )
        chosen = candidates[: min(self.k_intervene, len(candidates))]

        if not chosen:
            return decision

        treatment: dict[str, object] = {}
        for v in chosen:
            dom = list(v.domain)
            if not dom:
                continue

            if len(dom) >= DOMAIN_BOUNDS_COUNT and all(
                isinstance(x, (int, float, np.integer, np.floating))
                for x in dom[:DOMAIN_BOUNDS_COUNT]
            ):
                low, high = float(dom[0]), float(dom[1])
                if high < low:
                    low, high = high, low
                val = float(self.rng.uniform(low, high))
            else:
                val = dom[self.rng.integers(0, len(dom))]

            treatment[v.name] = val

        if treatment:
            decision.add_experiment(treatment=treatment, n=self._num_inter)
        return decision

    @override
    def to_spec(self) -> DeciderSpec:
        return DeciderSpec(
            class_=get_class_path(self.__class__),
            params={
                "num_obs": self._num_obs,
                "num_inter": self._num_inter,
                "n_bootstrap": self._belief.n_bootstrap,
                "ridge_lambda": self._belief.ridge_lambda,
                "coef_threshold": self._belief.coef_threshold,
                "k_intervene": self.k_intervene,
            },
        )

    @classmethod
    @override
    def from_spec(cls, spec: DeciderSpec) -> Cho2016ActiveGBNDecider:
        params = spec.params or {}
        return cls(
            num_obs=int(params.get("num_obs", 5)),
            num_inter=int(params.get("num_inter", 1)),
            n_bootstrap=int(params.get("n_bootstrap", 32)),
            ridge_lambda=float(params.get("ridge_lambda", 1e-2)),
            coef_threshold=float(params.get("coef_threshold", 1e-2)),
            k_intervene=int(params.get("k_intervene", 1)),
        )
