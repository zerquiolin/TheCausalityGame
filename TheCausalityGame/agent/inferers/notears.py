"""Linear NOTEARS inferer for DAG discovery."""

from __future__ import annotations

from typing import override

import networkx as nx
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

from TheCausalityGame.agent.inferers._dag_common import (
    empty_graph,
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


class NOTEARSInferer(Inferer):
    """Learn a linear DAG with the NOTEARS acyclicity constraint."""

    def __init__(
        self,
        *,
        lambda_l2: float = 0.01,
        w_threshold: float = 0.1,
        max_iter: int = 20,
        rho_max: float = 1e16,
        h_tol: float = 1e-8,
    ) -> None:
        self.lambda_l2 = lambda_l2
        self.w_threshold = w_threshold
        self.max_iter = max_iter
        self.rho_max = rho_max
        self.h_tol = h_tol
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
        adjacency = self._fit_notears(X)
        self._dag = self._orient_skeleton(adjacency, columns, X)
        return self._dag

    def _fit_notears(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        n, d = X.shape
        w = np.zeros((d, d), dtype=float)
        rho = 1.0
        alpha = 0.0

        def loss_and_grad(flat_w: np.ndarray) -> tuple[float, np.ndarray]:
            W = flat_w.reshape(d, d)  # noqa: N806
            np.fill_diagonal(W, 0.0)
            residual = X - X @ W
            loss = 0.5 / n * float(np.square(residual).sum())
            grad = -X.T @ residual / n
            return loss, grad

        def h_and_grad(W: np.ndarray) -> tuple[float, np.ndarray]:  # noqa: N803
            squared = W * W
            exp_squared = expm(squared)
            h_val = float(np.trace(exp_squared) - d)
            h_grad = 2.0 * exp_squared.T * W
            return h_val, h_grad

        for _ in range(self.max_iter):
            def objective(flat_w: np.ndarray) -> float:
                W = flat_w.reshape(d, d)  # noqa: N806
                loss, _ = loss_and_grad(flat_w)
                h_val, _ = h_and_grad(W)
                reg = 0.5 * self.lambda_l2 * float(np.square(W).sum())
                penalty = 0.5 * rho * h_val * h_val + alpha * h_val
                return loss + reg + penalty

            def gradient(flat_w: np.ndarray) -> np.ndarray:
                W = flat_w.reshape(d, d)  # noqa: N806
                _, grad_loss = loss_and_grad(flat_w)
                h_val, h_grad = h_and_grad(W)
                grad = grad_loss + self.lambda_l2 * W + (rho * h_val + alpha) * h_grad
                np.fill_diagonal(grad, 0.0)
                return grad.reshape(-1)

            result = minimize(
                objective,
                w.reshape(-1),
                jac=gradient,
                method="L-BFGS-B",
            )
            w = result.x.reshape(d, d)
            np.fill_diagonal(w, 0.0)
            h_val, _ = h_and_grad(w)
            if h_val <= self.h_tol or rho >= self.rho_max:
                break
            rho *= 10.0
            alpha += rho * h_val

        np.fill_diagonal(w, 0.0)
        return w

    def _orient_skeleton(
        self,
        adjacency: np.ndarray,
        columns: list[str],
        X: np.ndarray,  # noqa: N803
    ) -> nx.DiGraph:
        """Orient NOTEARS-discovered adjacencies using a residual-independence order."""
        d = len(columns)
        graph = empty_graph(columns)
        order = self._causal_order(X)
        rank = {node: index for index, node in enumerate(order)}
        for i in range(d):
            for j in range(i + 1, d):
                strength = max(abs(float(adjacency[i, j])), abs(float(adjacency[j, i])))
                if strength <= self.w_threshold:
                    continue
                if rank[i] <= rank[j]:
                    graph.add_edge(columns[i], columns[j])
                else:
                    graph.add_edge(columns[j], columns[i])
        return graph

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
                    ridge=self.lambda_l2,
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
            resid = residualize(data[:, target], source, ridge=self.lambda_l2)
            total += self._independence_score(source, resid)
        return total

    @staticmethod
    def _independence_score(source: np.ndarray, resid: np.ndarray) -> float:
        transformed_source = np.tanh(source)
        transformed_resid = np.tanh(resid)
        corr_1 = abs(float(np.corrcoef(transformed_resid, source)[0, 1]))
        corr_2 = abs(float(np.corrcoef(resid, transformed_source)[0, 1]))
        if np.isnan(corr_1):
            corr_1 = 0.0
        if np.isnan(corr_2):
            corr_2 = 0.0
        return corr_1 + corr_2

    @override
    def snapshot(self) -> BeliefSnapshot:
        obs_rows = len(observational_frame(self._samples))
        return BeliefSnapshot(
            estimate=self.answer(),
            summary={"observational_rows": obs_rows},
            capabilities=("dag_discovery", "notears"),
        )

    @override
    def to_spec(self) -> InfererSpec:
        return InfererSpec(
            class_=get_class_path(self.__class__),
            params={
                "lambda_l2": self.lambda_l2,
                "w_threshold": self.w_threshold,
                "max_iter": self.max_iter,
                "rho_max": self.rho_max,
                "h_tol": self.h_tol,
            },
        )

    @classmethod
    @override
    def from_spec(cls, spec: InfererSpec) -> NOTEARSInferer:
        params = spec.params or {}
        return cls(
            lambda_l2=float(params.get("lambda_l2", 0.01)),
            w_threshold=float(params.get("w_threshold", 0.1)),
            max_iter=int(params.get("max_iter", 20)),
            rho_max=float(params.get("rho_max", 1e16)),
            h_tol=float(params.get("h_tol", 1e-8)),
        )
