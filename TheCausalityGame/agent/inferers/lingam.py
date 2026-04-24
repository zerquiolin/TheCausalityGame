"""Direct LiNGAM-style inferer for DAG discovery."""

from __future__ import annotations

from typing import override

import networkx as nx
import numpy as np

from TheCausalityGame.agent.inferers._dag_common import (
    empty_graph,
    fit_linear_coefficients,
    graph_from_adjacency,
    numeric_frame,
    observational_frame,
    residualize,
    standardize,
)
from TheCausalityGame.core.contracts.dto.agent import BeliefSnapshot, RoundObservation
from TheCausalityGame.core.contracts.dto.environment import Samples
from TheCausalityGame.core.contracts.inferer import Inferer
from TheCausalityGame.core.contracts.specs.inferer import InfererSpec
from TheCausalityGame.core.infrastructure.registry import get_class_path


class LiNGAMInferer(Inferer):
    """Approximate DirectLiNGAM using residual-independence ordering."""

    def __init__(
        self,
        *,
        ridge: float = 1e-3,
        coef_threshold: float = 0.1,
    ) -> None:
        self.ridge = ridge
        self.coef_threshold = coef_threshold
        self._samples: list[Samples] = []
        self._dag: nx.DiGraph | None = None

    @override
    def update(self, observation: RoundObservation) -> None:
        self._samples.extend(observation.samples)
        self._dag = None

    @override
    def answer(self) -> nx.DiGraph:
        if self._dag is not None:
            return self._dag

        obs_df = observational_frame(self._samples)
        if obs_df.empty:
            self._dag = nx.DiGraph()
            return self._dag

        numeric_df = numeric_frame(obs_df)
        if numeric_df.empty:
            self._dag = empty_graph(list(obs_df.columns))
            return self._dag

        columns = list(numeric_df.columns)
        X = standardize(numeric_df).to_numpy(dtype=float)  # noqa: N806
        order = self._causal_order(X)
        adjacency = self._fit_adjacency(X, order)
        self._dag = graph_from_adjacency(
            adjacency,
            columns,
            threshold=self.coef_threshold,
        )
        return self._dag

    def _causal_order(self, X: np.ndarray) -> list[int]:  # noqa: N803
        residuals = X.copy()
        remaining = list(range(X.shape[1]))
        order: list[int] = []

        while remaining:
            best_idx = remaining[0]
            best_score = float("inf")
            for candidate in remaining:
                score = self._exogeneity_score(residuals, remaining, candidate)
                if score < best_score:
                    best_score = score
                    best_idx = candidate

            order.append(best_idx)
            remaining.remove(best_idx)
            source = residuals[:, best_idx]
            for target in remaining:
                residuals[:, target] = residualize(
                    residuals[:, target],
                    source,
                    ridge=self.ridge,
                )

        return order

    def _exogeneity_score(
        self,
        data: np.ndarray,
        remaining: list[int],
        candidate: int,
    ) -> float:
        source = data[:, candidate]
        total = 0.0
        for target in remaining:
            if target == candidate:
                continue
            resid = residualize(data[:, target], source, ridge=self.ridge)
            total += self._independence_score(source, resid)
        return total

    @staticmethod
    def _independence_score(source: np.ndarray, resid: np.ndarray) -> float:
        """Approximate DirectLiNGAM residual dependence score."""
        transformed_source = np.tanh(source)
        transformed_resid = np.tanh(resid)
        corr_1 = abs(float(np.corrcoef(transformed_resid, source)[0, 1]))
        corr_2 = abs(float(np.corrcoef(resid, transformed_source)[0, 1]))
        if np.isnan(corr_1):
            corr_1 = 0.0
        if np.isnan(corr_2):
            corr_2 = 0.0
        return corr_1 + corr_2

    def _fit_adjacency(self, X: np.ndarray, order: list[int]) -> np.ndarray:  # noqa: N803
        d = X.shape[1]
        adjacency = np.zeros((d, d), dtype=float)
        for position, child in enumerate(order):
            parents = order[:position]
            if not parents:
                continue
            predictors = X[:, parents]
            target = X[:, child]
            coef = fit_linear_coefficients(predictors, target, ridge=self.ridge)
            for parent, weight in zip(parents, coef, strict=True):
                adjacency[parent, child] = float(weight)
        return adjacency

    @override
    def snapshot(self) -> BeliefSnapshot:
        obs_rows = len(observational_frame(self._samples))
        return BeliefSnapshot(
            estimate=self.answer(),
            summary={"observational_rows": obs_rows},
            capabilities=("dag_discovery", "lingam"),
        )

    @override
    def to_spec(self) -> InfererSpec:
        return InfererSpec(
            class_=get_class_path(self.__class__),
            params={
                "ridge": self.ridge,
                "coef_threshold": self.coef_threshold,
            },
        )

    @classmethod
    @override
    def from_spec(cls, spec: InfererSpec) -> LiNGAMInferer:
        params = spec.params or {}
        return cls(
            ridge=float(params.get("ridge", 1e-3)),
            coef_threshold=float(params.get("coef_threshold", 0.1)),
        )
