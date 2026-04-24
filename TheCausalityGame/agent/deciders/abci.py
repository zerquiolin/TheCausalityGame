from __future__ import annotations

from typing import Any, override

import numpy as np

from TheCausalityGame.agent.helpers.active_query_belief import ActiveQueryBelief, QuerySpec
from TheCausalityGame.core.contracts.decider import Decider
from TheCausalityGame.core.contracts.dto.agent import BeliefSnapshot, RoundObservation
from TheCausalityGame.core.contracts.dto.environment import AvailableActions, RoundInfo
from TheCausalityGame.core.contracts.specs.decider import DeciderSpec
from TheCausalityGame.core.infrastructure.decisions import Decision
from TheCausalityGame.core.infrastructure.registry import get_class_path


def _context_query_spec(decider: Decider) -> QuerySpec:
    merged: dict[str, Any] = {}

    mission_meta = decider.context.mission.get("metadata", {})
    result_meta = decider.context.result_metric.get("metadata", {})
    if isinstance(mission_meta, dict):
        merged.update(mission_meta)
    if isinstance(result_meta, dict):
        merged.update(result_meta)

    return QuerySpec(
        family=merged.get("query_family"),
        treatment=merged.get("treatment"),
        outcome=merged.get("outcome"),
        covariates=tuple(merged.get("covariates", ())),
        treatment_values=tuple(merged.get("treatment_values", ())),
    )


def _domain_lookup(available_actions: AvailableActions) -> dict[str, list[Any]]:
    return {var.name: list(var.domain) for var in available_actions.experiments}


def _treatment_round_value(query: QuerySpec, round_info: RoundInfo) -> Any | None:
    if len(query.treatment_values) < 2:
        return None
    values = list(query.treatment_values)
    index = (round_info.round + 1) % len(values)
    return values[index]


def _random_value(
    belief: ActiveQueryBelief,
    rng: np.random.Generator,
    variable: Any,
    query: QuerySpec,
    round_info: RoundInfo,
) -> Any:
    if query.treatment is not None and variable.name == query.treatment:
        chosen = _treatment_round_value(query, round_info)
        if chosen is not None:
            return chosen

    values = belief.candidate_values(variable)
    return values[rng.integers(0, len(values))]


class ABCIDecider(Decider):
    """Query-aware intervention design inspired by Active Bayesian Causal Inference."""

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
        query_focus_weight: float = 0.75,
        seed: int | None = 911,
    ) -> None:
        self._num_obs = int(num_obs)
        self._num_inter = int(num_inter)
        self.n_graphs = int(max(4, n_graphs))
        self.fantasy_samples = int(max(8, fantasy_samples))
        self.num_value_candidates = int(max(2, num_value_candidates))
        self.query_focus_weight = float(max(0.0, query_focus_weight))
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

        if not self._belief.has_signal():
            variable = next(
                (v for v in available_actions.experiments if v.name == query.treatment),
                None,
            )
            if variable is None:
                variable = available_actions.experiments[
                    self.rng.integers(0, len(available_actions.experiments))
                ]
            value = _random_value(self._belief, self.rng, variable, query, round_info)
            decision.add_experiment(treatment={variable.name: value}, n=self._num_inter)
            return decision

        models = self._belief.sample_models(
            n_graphs=self.n_graphs,
            seed=self.seed + round_info.round,
        )
        if not models:
            variable = available_actions.experiments[
                self.rng.integers(0, len(available_actions.experiments))
            ]
            value = _random_value(self._belief, self.rng, variable, query, round_info)
            decision.add_experiment(treatment={variable.name: value}, n=self._num_inter)
            return decision

        focus_nodes = [query.outcome] if query.outcome else None
        best_variable = None
        best_value = None
        best_score = -float("inf")

        for var_idx, variable in enumerate(available_actions.experiments):
            relevance = 1.0
            if query.treatment is not None and query.outcome is not None:
                relevance += self.query_focus_weight * self._belief.query_path_relevance(
                    models,
                    treatment=query.treatment,
                    outcome=query.outcome,
                    candidate=variable.name,
                )
                if variable.name == query.treatment:
                    relevance += 0.5 * self.query_focus_weight

            for val_idx, value in enumerate(
                self._belief.candidate_values(
                    variable,
                    num_value_candidates=self.num_value_candidates,
                )
            ):
                local_rng = np.random.default_rng(
                    self.seed + 1009 * round_info.round + 97 * var_idx + val_idx
                )
                disagreement = self._belief.predictive_disagreement(
                    models,
                    intervention={variable.name: value},
                    focus_nodes=focus_nodes,
                    n=self.fantasy_samples,
                    rng=local_rng,
                    domain_lookup=domain_lookup,
                )
                score = disagreement * relevance
                if score > best_score:
                    best_score = score
                    best_variable = variable
                    best_value = value

        if best_variable is None or best_value is None:
            best_variable = available_actions.experiments[
                self.rng.integers(0, len(available_actions.experiments))
            ]
            best_value = _random_value(self._belief, self.rng, best_variable, query, round_info)

        decision.add_experiment(treatment={best_variable.name: best_value}, n=self._num_inter)
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
                "query_focus_weight": self.query_focus_weight,
                "seed": self.seed,
            },
        )

    @classmethod
    @override
    def from_spec(cls, spec: DeciderSpec) -> ABCIDecider:
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
            query_focus_weight=float(params.get("query_focus_weight", 0.75)),
            seed=int(params.get("seed", 911)),
        )
