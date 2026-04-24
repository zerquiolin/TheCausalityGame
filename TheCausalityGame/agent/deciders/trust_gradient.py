"""Gradient-style active intervention decider."""

from __future__ import annotations

from typing import override

import numpy as np

from TheCausalityGame.agent.deciders.abci import (
    _context_query_spec,
    _domain_lookup,
    _random_value,
)
from TheCausalityGame.agent.helpers.active_query_belief import ActiveQueryBelief
from TheCausalityGame.core.contracts.decider import Decider
from TheCausalityGame.core.contracts.dto.agent import BeliefSnapshot, RoundObservation
from TheCausalityGame.core.contracts.dto.environment import AvailableActions, RoundInfo
from TheCausalityGame.core.contracts.specs.decider import DeciderSpec
from TheCausalityGame.core.infrastructure.decisions import Decision
from TheCausalityGame.core.infrastructure.registry import get_class_path


class TrustYourGradientDecider(Decider):
    """Gradient-magnitude intervention targeting inspired by GIT."""

    def __init__(  # noqa: PLR0913
        self,
        num_obs: int = 1,
        num_inter: int = 3,
        n_bootstrap: int = 32,
        ridge_lambda: float = 1e-2,
        coef_threshold: float = 1e-2,
        n_graphs: int = 20,
        fantasy_samples: int = 48,
        num_value_candidates: int = 5,
        uncertainty_weight: float = 0.25,
        seed: int | None = 911,
    ) -> None:
        self._num_obs = int(num_obs)
        self._num_inter = int(num_inter)
        self.n_graphs = int(max(4, n_graphs))
        self.fantasy_samples = int(max(8, fantasy_samples))
        self.num_value_candidates = int(max(2, num_value_candidates))
        self.uncertainty_weight = float(max(0.0, uncertainty_weight))
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
            uncertainty_bonus = 1.0 + self.uncertainty_weight * self._belief.outgoing_uncertainty(
                variable.name
            )
            for val_idx, value in enumerate(
                self._belief.candidate_values(
                    variable,
                    num_value_candidates=self.num_value_candidates,
                )
            ):
                local_rng = np.random.default_rng(
                    self.seed + 2027 * round_info.round + 173 * var_idx + val_idx
                )
                gradient_norm = self._belief.gradient_score(
                    models,
                    intervention={variable.name: value},
                    focus_nodes=focus_nodes,
                    n=self.fantasy_samples,
                    rng=local_rng,
                    domain_lookup=domain_lookup,
                )
                score = gradient_norm * uncertainty_bonus
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
                "uncertainty_weight": self.uncertainty_weight,
                "seed": self.seed,
            },
        )

    @classmethod
    @override
    def from_spec(cls, spec: DeciderSpec) -> TrustYourGradientDecider:
        params = spec.params or {}
        return cls(
            num_obs=int(params.get("num_obs", 1)),
            num_inter=int(params.get("num_inter", 3)),
            n_bootstrap=int(params.get("n_bootstrap", 32)),
            ridge_lambda=float(params.get("ridge_lambda", 1e-2)),
            coef_threshold=float(params.get("coef_threshold", 1e-2)),
            n_graphs=int(params.get("n_graphs", 20)),
            fantasy_samples=int(params.get("fantasy_samples", 48)),
            num_value_candidates=int(params.get("num_value_candidates", 5)),
            uncertainty_weight=float(params.get("uncertainty_weight", 0.25)),
            seed=int(params.get("seed", 911)),
        )
