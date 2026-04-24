"""Tigas-style CBED lookahead decider."""

from __future__ import annotations

from typing import override

import numpy as np
import pandas as pd

from TheCausalityGame.agent.helpers.ridge_bootstrap_edge_belief import RidgeBootstrapEdgeBelief
from TheCausalityGame.core.contracts.decider import Decider
from TheCausalityGame.core.contracts.dto.agent import BeliefSnapshot, RoundObservation
from TheCausalityGame.core.contracts.dto.environment import AvailableActions, RoundInfo
from TheCausalityGame.core.contracts.specs.decider import DeciderSpec
from TheCausalityGame.core.infrastructure.decisions import Decision
from TheCausalityGame.core.infrastructure.registry import get_class_path

DOMAIN_BOUNDS_COUNT = 2


class Tigas2022CBEDDecider(Decider):
    """CBED-style lookahead decider."""

    def __init__(  # noqa: PLR0913
        self,
        num_obs: int = 2,
        num_inter: int = 3,
        n_bootstrap: int = 32,
        ridge_lambda: float = 1e-2,
        coef_threshold: float = 1e-2,
        fantasies: int = 5,
        eval_bootstrap: int = 12,
        topk_vars: int = 4,
        num_value_candidates: int = 7,
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
        self.fantasies = int(fantasies)
        self.eval_bootstrap = int(eval_bootstrap)
        self.topk_vars = int(topk_vars)
        self.num_value_candidates = int(max(2, num_value_candidates))
        self.rng = np.random.default_rng(seed)

    @override
    def update(self, observation: RoundObservation) -> None:
        self._belief.fit(list(observation.samples))

    def _candidate_values(self, var: object) -> list[object]:
        dom = list(var.domain)  # type: ignore

        if (
            len(dom) >= DOMAIN_BOUNDS_COUNT  # type: ignore
            and all(
                isinstance(x, (int, float, np.integer, np.floating))
                for x in dom[:DOMAIN_BOUNDS_COUNT]  # type: ignore
            )
        ):
            low, high = float(dom[0]), float(dom[1])  # type: ignore
            if high < low:
                low, high = high, low
            grid = np.linspace(low, high, num=self.num_value_candidates, dtype=float)
            if np.isclose(low, high):
                return [float(low)]
            return [float(v) for v in grid]

        return dom  # type: ignore

    def _expected_entropy_after(self, var: str, val: object, n_add: int) -> float:
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
            entropies.append(ent)  # type: ignore

        if not entropies:
            return float("inf")
        return float(np.mean(entropies))  # type: ignore

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

        summ = self._belief.summary()
        if summ is None:
            var = available_actions.experiments[
                self.rng.integers(0, len(available_actions.experiments))
            ]
            dom = list(var.domain)
            if len(dom) >= DOMAIN_BOUNDS_COUNT and all(
                isinstance(x, (int, float, np.integer, np.floating))
                for x in dom[:DOMAIN_BOUNDS_COUNT]
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
            var = vars_scored[0]
            val = self._belief.canonical_value(list(var.domain))
            best = (var.name, val)

        decision.add_experiment(treatment={best[0]: best[1]}, n=self._num_inter)
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
                "fantasies": self.fantasies,
                "eval_bootstrap": self.eval_bootstrap,
                "topk_vars": self.topk_vars,
                "num_value_candidates": self.num_value_candidates,
            },
        )

    @classmethod
    @override
    def from_spec(cls, spec: DeciderSpec) -> Tigas2022CBEDDecider:
        params = spec.params or {}
        return cls(
            num_obs=int(params.get("num_obs", 2)),
            num_inter=int(params.get("num_inter", 3)),
            n_bootstrap=int(params.get("n_bootstrap", 32)),
            ridge_lambda=float(params.get("ridge_lambda", 1e-2)),
            coef_threshold=float(params.get("coef_threshold", 1e-2)),
            fantasies=int(params.get("fantasies", 5)),
            eval_bootstrap=int(params.get("eval_bootstrap", 12)),
            topk_vars=int(params.get("topk_vars", 4)),
            num_value_candidates=int(params.get("num_value_candidates", 7)),
        )
