"""He-Geng style minimax intervention decider."""

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


def _sample_value(rng: np.random.Generator, domain: list[object]) -> object:
    dom = list(domain)
    if not dom:
        return 0

    if len(dom) >= DOMAIN_BOUNDS_COUNT and all(
        isinstance(x, (int, float, np.integer, np.floating)) for x in dom[:DOMAIN_BOUNDS_COUNT]
    ):
        low, high = float(dom[0]), float(dom[1])  # type: ignore
        if high < low:
            low, high = high, low
        if np.isclose(low, high):
            return float(low)
        return float(rng.uniform(low, high))

    return dom[rng.integers(0, len(dom))]


class HeGeng2008MinimaxDecider(Decider):
    """He-Geng style minimax intervention design."""

    def __init__(  # noqa: PLR0913
        self,
        num_obs: int = 2,
        num_inter: int = 3,
        n_bootstrap: int = 32,
        ridge_lambda: float = 1e-2,
        coef_threshold: float = 1e-2,
        n_graphs: int = 32,
        k_intervene: int = 1,
        seed: int | None = 911,
    ) -> None:
        self._num_obs = int(num_obs)
        self._num_inter = int(num_inter)
        self.n_graphs = int(max(4, n_graphs))
        self.k_intervene = int(max(1, k_intervene))
        self.seed = seed if seed is not None else 911
        self._belief = RidgeBootstrapEdgeBelief(
            n_bootstrap=n_bootstrap,
            ridge_lambda=ridge_lambda,
            coef_threshold=coef_threshold,
            seed=self.seed,
        )
        self.rng = np.random.default_rng(self.seed)

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
        del belief
        decision = Decision.experiment()

        if self._num_obs > 0:
            decision.add_experiment(treatment=None, n=self._num_obs)

        if self._num_inter <= 0 or not available_actions.experiments:
            return decision

        summ = self._belief.summary()
        if summ is None:
            vars_sorted = sorted(available_actions.experiments, key=lambda v: str(v.name))
            var = vars_sorted[self.rng.integers(0, len(vars_sorted))]
            val = _sample_value(self.rng, list(var.domain))
            decision.add_experiment(treatment={var.name: val}, n=self._num_inter)
            return decision

        models = self._belief.sample_linear_dag_ensemble(
            n_graphs=self.n_graphs,
            seed=self.seed + round_info.round,
        )
        col_index = {c: i for i, c in enumerate(summ.columns)}

        cands = sorted(
            [v for v in available_actions.experiments if v.name in col_index],
            key=lambda v: str(v.name),
        )
        if not cands:
            vars_sorted = sorted(available_actions.experiments, key=lambda v: str(v.name))
            var = vars_sorted[self.rng.integers(0, len(vars_sorted))]
            treatment = {var.name: _sample_value(self.rng, list(var.domain))}
            decision.add_experiment(treatment=treatment, n=self._num_inter)
            return decision

        edges = []
        degs = []
        for _, adjacency, _ in models:
            adjacency_mask = (np.abs(adjacency) > 0).astype(int)
            edges.append(float(adjacency_mask.sum()))  # type: ignore
            degs.append(  # type: ignore
                adjacency_mask.sum(axis=0).astype(float) + adjacency_mask.sum(axis=1).astype(float)
            )

        chosen: list[object] = []
        chosen_idx: set[int] = set()

        k = min(self.k_intervene, len(cands))
        for _ in range(k):
            best_var = None
            best_obj = float("inf")

            for v in cands:
                idx = col_index[v.name]
                if idx in chosen_idx:
                    continue

                worst_remaining = -1.0
                for g in range(len(models)):
                    removed = sum(degs[g][col_index[x.name]] for x in chosen) + degs[g][idx]  # type: ignore
                    remaining = edges[g] - removed  # type: ignore
                    worst_remaining = max(worst_remaining, remaining)  # type: ignore

                if worst_remaining < best_obj:
                    best_obj = worst_remaining
                    best_var = v

            if best_var is None:
                break
            chosen.append(best_var)
            chosen_idx.add(col_index[best_var.name])

        treatment = {v.name: _sample_value(self.rng, list(v.domain)) for v in chosen}  # type: ignore
        decision.add_experiment(treatment=treatment, n=self._num_inter)  # type: ignore
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
                "n_graphs": self.n_graphs,
                "k_intervene": self.k_intervene,
            },
        )

    @classmethod
    @override
    def from_spec(cls, spec: DeciderSpec) -> HeGeng2008MinimaxDecider:
        params = spec.params or {}
        return cls(
            num_obs=int(params.get("num_obs", 2)),
            num_inter=int(params.get("num_inter", 3)),
            n_bootstrap=int(params.get("n_bootstrap", 32)),
            ridge_lambda=float(params.get("ridge_lambda", 1e-2)),
            coef_threshold=float(params.get("coef_threshold", 1e-2)),
            n_graphs=int(params.get("n_graphs", 32)),
            k_intervene=int(params.get("k_intervene", 1)),
        )
