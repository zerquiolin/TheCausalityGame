from __future__ import annotations

from typing import Any, override

import numpy as np

from TheCausalityGame.agent.common import CommonAgent
from TheCausalityGame.agent.helpers.ridge_bootstrap_edge_belief import RidgeBootstrapEdgeBelief
from TheCausalityGame.core.contracts.dto.environment import AvailableActions, RoundInfo
from TheCausalityGame.core.contracts.specs.agent import AgentSpec
from TheCausalityGame.core.infrastructure.decisions import Decision
from TheCausalityGame.core.infrastructure.registry import get_class_path


def _sample_value(rng, domain):
    dom = list(domain)
    if not dom:
        return 0

    # Numeric bounds convention: [low, high]
    if len(dom) >= 2 and all(isinstance(x, (int, float, np.integer, np.floating)) for x in dom[:2]):
        low, high = float(dom[0]), float(dom[1])
        if high < low:
            low, high = high, low
        if np.isclose(low, high):
            return float(low)
        return float(rng.uniform(low, high))

    # categorical/enumerated
    return dom[rng.integers(0, len(dom))]


class HeGeng2008MinimaxAgent(CommonAgent):
    """
    He & Geng (2008) style sequential design with a minimax criterion.

    Paper idea (high level):
      - maintain an equivalence class / uncertainty over causal structures
      - pick the next intervention to minimize the worst-case remaining uncertainty

    In this implementation:
      - we approximate the equivalence class with an ensemble of plausible DAGs sampled from
        RidgeBootstrapEdgeBelief.sample_linear_dag_ensemble(...)
      - minimax score for target X: minimize max_g (|E_g| - deg_g(X)), i.e. reduce worst-case
        remaining edges by intervening on nodes that have high degree even in the worst case.
    """

    def __init__(
        self,
        id: str,
        num_obs: int = 2,
        num_inter: int = 3,
        n_bootstrap: int = 32,
        ridge_lambda: float = 1e-2,
        coef_threshold: float = 1e-2,
        n_graphs: int = 32,
        k_intervene: int = 1,  # number of variables to intervene on at once
        seed: int | None = 911,
    ) -> None:
        self.id = id
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
    def inform(self, samples_collection) -> None:
        # keep your internal strategy pipeline intact if you use it elsewhere
        self.strategy.learn(samples_collection)
        self._belief.fit(list(samples_collection))

    def _pick_value(self, domain: list[Any]) -> Any:
        dom = list(domain)
        if not dom:
            return 0

        # Numeric bounds: [low, high] -> sample interior to avoid endpoint bias
        if len(dom) >= 2 and all(
            isinstance(x, (int, float, np.integer, np.floating)) for x in dom[:2]
        ):
            low, high = float(dom[0]), float(dom[1])
            if high < low:
                low, high = high, low
            if np.isclose(low, high):
                return float(low)
            return float(self.rng.uniform(low, high))

        # Categorical/enumerated
        return dom[self.rng.integers(0, len(dom))]

    @override
    def act(self, round_info: RoundInfo, available_actions: AvailableActions) -> Decision:
        decision = Decision.experiment()

        if self._num_obs > 0:
            decision.add_experiment(treatment=None, n=self._num_obs)

        if self._num_inter <= 0 or not available_actions.experiments:
            return decision

        # If no belief yet, fall back to random intervention
        summ = self._belief.summary()
        if summ is None:
            vars_sorted = sorted(list(available_actions.experiments), key=lambda v: str(v.name))
            var = vars_sorted[self.rng.integers(0, len(vars_sorted))]
            val = self._pick_value(list(var.domain))
            decision.add_experiment(treatment={var.name: val}, n=self._num_inter)
            return decision

        models = self._belief.sample_linear_dag_ensemble(
            n_graphs=self.n_graphs, seed=self.seed + round_info.round
        )
        col_index = {c: i for i, c in enumerate(summ.columns)}

        # allowed variables with indices
        cands = sorted(
            [v for v in available_actions.experiments if v.name in col_index],
            key=lambda v: str(v.name),
        )
        if not cands:
            # fallback random
            vars_sorted = sorted(list(available_actions.experiments), key=lambda v: str(v.name))
            var = vars_sorted[self.rng.integers(0, len(vars_sorted))]
            treatment = {var.name: _sample_value(self.rng, var.domain)}
            decision.add_experiment(treatment=treatment, n=self._num_inter)
            return decision

        # Precompute per-graph degrees for speed
        # degs[g][i] = degree of node i in graph g; edges[g] = total edges
        edges = []
        degs = []
        for _, B, _ in models:
            A = (np.abs(B) > 0).astype(int)
            edges.append(float(A.sum()))
            degs.append(A.sum(axis=0).astype(float) + A.sum(axis=1).astype(float))  # in+out

        chosen: list[Any] = []
        chosen_idx: set[int] = set()

        k = min(self.k_intervene, len(cands))
        for _ in range(k):
            best_var = None
            best_obj = float("inf")

            for v in cands:
                idx = col_index[v.name]
                if idx in chosen_idx:
                    continue

                # compute minimax objective if we add this variable
                worst_remaining = -1.0
                for g in range(len(models)):
                    removed = sum(degs[g][col_index[x.name]] for x in chosen) + degs[g][idx]
                    remaining = edges[g] - removed
                    worst_remaining = max(worst_remaining, remaining)

                if worst_remaining < best_obj:
                    best_obj = worst_remaining
                    best_var = v

            if best_var is None:
                break
            chosen.append(best_var)
            chosen_idx.add(col_index[best_var.name])

        treatment = {v.name: _sample_value(self.rng, v.domain) for v in chosen}
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
            "n_graphs": self.n_graphs,
            "k_intervene": self.k_intervene,
        }
        return AgentSpec(id=self.id, class_=get_class_path(self.__class__), params=params)

    @classmethod
    @override
    def from_spec(cls, spec: AgentSpec) -> HeGeng2008MinimaxAgent:
        if not spec.params:
            return cls(id=spec.id)

        return cls(
            id=spec.id,
            num_obs=spec.params.num_obs or None,  # type: ignore
            num_inter=spec.params.num_inter or None,  # type: ignore
            n_bootstrap=spec.params.n_bootstrap or None,  # type: ignore
            ridge_lambda=spec.params.ridge_lambda or None,  # type: ignore
            coef_threshold=spec.params.coef_threshold or None,  # type: ignore
            n_graphs=spec.params.n_graphs or None,  # type: ignore
            k_intervene=spec.params.k_intervene or None,  # type: ignore
        )
