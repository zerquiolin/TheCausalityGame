from __future__ import annotations

from typing import Any, override

import numpy as np
import pandas as pd

from TheCausalityGame.agent.common import CommonAgent
from TheCausalityGame.agent.helpers.ridge_bootstrap_edge_belief import RidgeBootstrapEdgeBelief
from TheCausalityGame.core.contracts.dto.environment import AvailableActions, RoundInfo
from TheCausalityGame.core.contracts.specs.agent import AgentSpec
from TheCausalityGame.core.infrastructure.decisions import Decision
from TheCausalityGame.core.infrastructure.registry import get_class_path


class Tigas2022CBEDAgent(CommonAgent):
    """
    CBED-style lookahead (proxy):
      - score (var,value) by expected entropy reduction using fantasized do-data
      - choose argmax

    This is compute heavier but actually 'BOED-ish' and distinct from Cho/CAASL.
    """

    def __init__(
        self,
        id: str,
        num_obs: int = 0,
        num_inter: int = 10,
        n_bootstrap: int = 32,
        ridge_lambda: float = 1e-2,
        coef_threshold: float = 1e-2,
        fantasies: int = 5,  # MC samples per candidate
        eval_bootstrap: int = 12,  # fewer bootstraps during candidate eval
        topk_vars: int = 4,  # only evaluate top-k uncertain vars for speed
        num_value_candidates: int = 7,  # candidate values for numeric ranges
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
        self.fantasies = int(fantasies)
        self.eval_bootstrap = int(eval_bootstrap)
        self.topk_vars = int(topk_vars)
        self.num_value_candidates = int(max(2, num_value_candidates))

        self.rng = np.random.default_rng(seed)

    @override
    def inform(self, samples_collection) -> None:
        self.strategy.learn(samples_collection)
        self._belief.fit(list(samples_collection))

    def _candidate_values(self, var) -> list[Any]:
        dom = list(var.domain)

        # If numeric domain is given as [low, high] bounds, generate interior candidates.
        if len(dom) >= 2 and all(
            isinstance(x, (int, float, np.integer, np.floating)) for x in dom[:2]
        ):
            low, high = float(dom[0]), float(dom[1])
            if high < low:
                low, high = high, low

            # Stratified grid including interior points.
            grid = np.linspace(low, high, num=self.num_value_candidates, dtype=float)

            # If the range is degenerate, just return the single value.
            if np.isclose(low, high):
                return [float(low)]

            return [float(v) for v in grid]

        # categorical / enumerated list: test all
        return dom

    def _expected_entropy_after(self, var: str, val: Any, n_add: int) -> float:
        s = self._belief.summary()
        if s is None:
            return float("inf")

        entropies = []
        for _ in range(self.fantasies):
            fant = self._belief.fantasize_do(var, val, n=n_add, rng=self.rng)
            if fant is None:
                continue
            df_hyp = pd.concat([s.df_numeric, fant], axis=0, ignore_index=True)
            ent = self._belief.entropy_of_df(df_hyp, n_bootstrap=self.eval_bootstrap)
            entropies.append(ent)

        if not entropies:
            return float("inf")
        return float(np.mean(entropies))

    @override
    def act(self, round_info: RoundInfo, available_actions: AvailableActions) -> Decision:
        decision = Decision.experiment()

        if self._num_obs > 0:
            decision.add_experiment(treatment=None, n=self._num_obs)

        if self._num_inter <= 0 or not available_actions.experiments:
            return decision

        summ = self._belief.summary()
        if summ is None:
            # no belief yet -> explore big-signal actions
            # pick random var and extreme value
            var = available_actions.experiments[
                self.rng.integers(0, len(available_actions.experiments))
            ]
            dom = list(var.domain)
            if len(dom) >= 2 and all(
                isinstance(x, (int, float, np.integer, np.floating)) for x in dom[:2]
            ):
                low, high = float(dom[0]), float(dom[1])
                if high < low:
                    low, high = high, low
                val = float(self.rng.uniform(low, high))
            else:
                vals = self._candidate_values(var)
                val = vals[self.rng.integers(0, len(vals))]

            decision.add_experiment(treatment={var.name: val}, n=self._num_inter)
            return decision

        current_entropy = summ.total_entropy

        # Only evaluate top-k variables by outgoing uncertainty to keep it tractable
        vars_scored = sorted(
            available_actions.experiments,
            key=lambda v: self._belief.outgoing_uncertainty(v.name),
            reverse=True,
        )
        vars_scored = vars_scored[: max(1, self.topk_vars)]

        best = None
        best_ig = -1e18

        for var in vars_scored:
            for val in self._candidate_values(var):
                exp_ent = self._expected_entropy_after(var.name, val, n_add=self._num_inter)
                ig = current_entropy - exp_ent
                if ig > best_ig:
                    best_ig = ig
                    best = (var.name, val)

        if best is None:
            # fallback
            var = vars_scored[0]
            val = self._belief.canonical_value(list(var.domain))
            best = (var.name, val)

        decision.add_experiment(treatment={best[0]: best[1]}, n=self._num_inter)
        return decision

    @override
    def to_spec(self) -> AgentSpec:
        params = {
            "num_obs": self._num_obs,
            "num_inter": self._num_inter,
            "n_bootstrap": self._belief.n_bootstrap,
            "ridge_lambda": self._belief.ridge_lambda,
            "coef_threshold": self._belief.coef_threshold,
            "fantasies": self.fantasies,
            "eval_bootstrap": self.eval_bootstrap,
            "topk_vars": self.topk_vars,
        }
        return AgentSpec(id=self.id, class_=get_class_path(self.__class__), params=params)

    @classmethod
    @override
    def from_spec(cls, spec: AgentSpec) -> "Tigas2022CBEDAgent":
        p = spec.params or {}
        return cls(
            id=spec.id,
            num_obs=p.get("num_obs", 1),
            num_inter=p.get("num_inter", 1),
            n_bootstrap=p.get("n_bootstrap", 32),
            ridge_lambda=p.get("ridge_lambda", 1e-2),
            coef_threshold=p.get("coef_threshold", 1e-2),
            fantasies=p.get("fantasies", 5),
            eval_bootstrap=p.get("eval_bootstrap", 12),
            topk_vars=p.get("topk_vars", 4),
        )
