from __future__ import annotations

from typing import Any, override

import numpy as np

from TheCausalityGame.agent.deciders.abci import (
    _context_query_spec,
    _domain_lookup,
    _random_value,
    _treatment_round_value,
)
from TheCausalityGame.agent.helpers.active_query_belief import ActiveQueryBelief
from TheCausalityGame.core.contracts.decider import Decider
from TheCausalityGame.core.contracts.dto.agent import BeliefSnapshot, RoundObservation
from TheCausalityGame.core.contracts.dto.environment import AvailableActions, RoundInfo
from TheCausalityGame.core.contracts.specs.decider import DeciderSpec
from TheCausalityGame.core.infrastructure.decisions import Decision
from TheCausalityGame.core.infrastructure.registry import get_class_path


class OptimalEffectDesignDecider(Decider):
    """Query-specific intervention-set heuristic for treatment-effect identification."""

    def __init__(
        self,
        num_obs: int = 1,
        num_inter: int = 3,
        n_bootstrap: int = 32,
        ridge_lambda: float = 1e-2,
        coef_threshold: float = 1e-2,
        n_graphs: int = 24,
        fantasy_samples: int = 48,
        num_value_candidates: int = 5,
        k_intervene: int = 1,
        treatment_bonus: float = 2.0,
        path_weight: float = 1.5,
        coverage_weight: float = 0.5,
        disagreement_weight: float = 1.0,
        seed: int | None = 911,
    ) -> None:
        self._num_obs = int(num_obs)
        self._num_inter = int(num_inter)
        self.n_graphs = int(max(4, n_graphs))
        self.fantasy_samples = int(max(8, fantasy_samples))
        self.num_value_candidates = int(max(2, num_value_candidates))
        self.k_intervene = int(max(1, k_intervene))
        self.treatment_bonus = float(max(0.0, treatment_bonus))
        self.path_weight = float(max(0.0, path_weight))
        self.coverage_weight = float(max(0.0, coverage_weight))
        self.disagreement_weight = float(max(0.0, disagreement_weight))
        self.seed = int(seed if seed is not None else 911)
        self.rng = np.random.default_rng(self.seed)
        self._belief = ActiveQueryBelief(
            n_bootstrap=n_bootstrap,
            ridge_lambda=ridge_lambda,
            coef_threshold=coef_threshold,
            seed=self.seed,
        )

    @override
    def update(self, observation: RoundObservation) -> None:
        self._belief.update(list(observation.samples))

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

        query = _context_query_spec(self)
        domain_lookup = _domain_lookup(available_actions)
        treatment_variable = next(
            (var for var in available_actions.experiments if var.name == query.treatment),
            None,
        )

        if query.treatment is None or query.outcome is None:
            variable = available_actions.experiments[
                self.rng.integers(0, len(available_actions.experiments))
            ]
            value = _random_value(self._belief, self.rng, variable, query, round_info)
            decision.add_experiment(treatment={variable.name: value}, n=self._num_inter)
            return decision

        if not self._belief.has_signal():
            if treatment_variable is None:
                treatment_variable = available_actions.experiments[
                    self.rng.integers(0, len(available_actions.experiments))
                ]
            treatment_value = _treatment_round_value(query, round_info)
            if treatment_value is None:
                treatment_value = _random_value(
                    self._belief,
                    self.rng,
                    treatment_variable,
                    query,
                    round_info,
                )
            decision.add_experiment(
                treatment={treatment_variable.name: treatment_value},
                n=self._num_inter,
            )
            return decision

        models = self._belief.sample_models(
            n_graphs=self.n_graphs,
            seed=self.seed + round_info.round,
        )
        if not models:
            if treatment_variable is None:
                treatment_variable = available_actions.experiments[
                    self.rng.integers(0, len(available_actions.experiments))
                ]
            value = _random_value(self._belief, self.rng, treatment_variable, query, round_info)
            decision.add_experiment(treatment={treatment_variable.name: value}, n=self._num_inter)
            return decision

        ranked: list[tuple[float, Any, Any]] = []
        for var_idx, variable in enumerate(available_actions.experiments):
            path_relevance = self._belief.query_path_relevance(
                models,
                treatment=query.treatment,
                outcome=query.outcome,
                candidate=variable.name,
            )
            edge_coverage = self._belief.query_edge_coverage(
                treatment=query.treatment,
                outcome=query.outcome,
                candidate=variable.name,
            )
            direct_bonus = self.treatment_bonus if variable.name == query.treatment else 0.0

            best_value = None
            best_disagreement = -float("inf")
            for val_idx, value in enumerate(
                self._belief.candidate_values(
                    variable,
                    num_value_candidates=self.num_value_candidates,
                )
            ):
                if variable.name == query.treatment:
                    treatment_value = _treatment_round_value(query, round_info)
                    if treatment_value is not None:
                        value = treatment_value

                local_rng = np.random.default_rng(
                    self.seed + 3037 * round_info.round + 211 * var_idx + val_idx
                )
                disagreement = self._belief.predictive_disagreement(
                    models,
                    intervention={variable.name: value},
                    focus_nodes=[query.outcome],
                    n=self.fantasy_samples,
                    rng=local_rng,
                    domain_lookup=domain_lookup,
                )
                if disagreement > best_disagreement:
                    best_disagreement = disagreement
                    best_value = value

            if best_value is None:
                best_value = _random_value(self._belief, self.rng, variable, query, round_info)
                best_disagreement = 0.0

            score = (
                direct_bonus
                + self.path_weight * path_relevance
                + self.coverage_weight * edge_coverage
                + self.disagreement_weight * best_disagreement
            )
            ranked.append((score, variable, best_value))

        ranked.sort(key=lambda item: item[0], reverse=True)
        chosen = ranked[: min(self.k_intervene, len(ranked))]
        treatment = {variable.name: value for _, variable, value in chosen}
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
                "n_graphs": self.n_graphs,
                "fantasy_samples": self.fantasy_samples,
                "num_value_candidates": self.num_value_candidates,
                "k_intervene": self.k_intervene,
                "treatment_bonus": self.treatment_bonus,
                "path_weight": self.path_weight,
                "coverage_weight": self.coverage_weight,
                "disagreement_weight": self.disagreement_weight,
                "seed": self.seed,
            },
        )

    @classmethod
    @override
    def from_spec(cls, spec: DeciderSpec) -> OptimalEffectDesignDecider:
        params = spec.params or {}
        return cls(
            num_obs=int(params.get("num_obs", 1)),
            num_inter=int(params.get("num_inter", 3)),
            n_bootstrap=int(params.get("n_bootstrap", 32)),
            ridge_lambda=float(params.get("ridge_lambda", 1e-2)),
            coef_threshold=float(params.get("coef_threshold", 1e-2)),
            n_graphs=int(params.get("n_graphs", 24)),
            fantasy_samples=int(params.get("fantasy_samples", 48)),
            num_value_candidates=int(params.get("num_value_candidates", 5)),
            k_intervene=int(params.get("k_intervene", 1)),
            treatment_bonus=float(params.get("treatment_bonus", 2.0)),
            path_weight=float(params.get("path_weight", 1.5)),
            coverage_weight=float(params.get("coverage_weight", 0.5)),
            disagreement_weight=float(params.get("disagreement_weight", 1.0)),
            seed=int(params.get("seed", 911)),
        )
