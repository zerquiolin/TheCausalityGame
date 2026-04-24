from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from TheCausalityGame.agent.helpers.ridge_bootstrap_edge_belief import (
    BeliefSummary,
    RidgeBootstrapEdgeBelief,
)


@dataclass(frozen=True, slots=True)
class QuerySpec:
    family: str | None = None
    treatment: str | None = None
    outcome: str | None = None
    covariates: tuple[str, ...] = ()
    treatment_values: tuple[Any, ...] = ()


class ActiveQueryBelief:
    """Small posterior proxy used by active causal deciders."""

    def __init__(
        self,
        *,
        n_bootstrap: int = 32,
        ridge_lambda: float = 1e-2,
        coef_threshold: float = 1e-2,
        seed: int | None = 911,
    ) -> None:
        self._belief = RidgeBootstrapEdgeBelief(
            n_bootstrap=n_bootstrap,
            ridge_lambda=ridge_lambda,
            coef_threshold=coef_threshold,
            seed=seed,
        )
        self._history: list[SimpleNamespace] = []
        self._seed = int(seed if seed is not None else 911)

    def update(self, samples: list[Any]) -> None:
        for sample in samples:
            data = getattr(sample, "data", None)
            if data is None or len(data) == 0:
                continue
            self._history.append(
                SimpleNamespace(
                    data=data.copy(),
                    interventions=getattr(sample, "interventions", None),
                )
            )

        self._belief.fit(self._history)

    def summary(self) -> BeliefSummary | None:
        return self._belief.summary()

    def has_signal(self) -> bool:
        return self.summary() is not None

    @property
    def n_bootstrap(self) -> int:
        return int(self._belief.n_bootstrap)

    @property
    def ridge_lambda(self) -> float:
        return float(self._belief.ridge_lambda)

    @property
    def coef_threshold(self) -> float:
        return float(self._belief.coef_threshold)

    def incident_uncertainty(self, var: str) -> float:
        return float(self._belief.incident_uncertainty(var))

    def outgoing_uncertainty(self, var: str) -> float:
        return float(self._belief.outgoing_uncertainty(var))

    def sample_models(
        self,
        *,
        n_graphs: int,
        seed: int | None = None,
    ) -> list[tuple[list[int], np.ndarray, np.ndarray]]:
        return self._belief.sample_linear_dag_ensemble(
            n_graphs=n_graphs,
            seed=self._seed if seed is None else seed,
        )

    def column_index(self) -> dict[str, int]:
        summary = self.summary()
        if summary is None:
            return {}
        return {name: idx for idx, name in enumerate(summary.columns)}

    def candidate_values(self, variable: Any, *, num_value_candidates: int = 5) -> list[Any]:
        dom = list(getattr(variable, "domain", []))
        if not dom:
            return [0.0]

        if len(dom) >= 2 and all(
            isinstance(x, (int, float, np.integer, np.floating)) for x in dom[:2]
        ):
            try:
                low = float(dom[0])
                high = float(dom[1])
                if high < low:
                    low, high = high, low
                if np.isclose(low, high):
                    return [float(low)]
                return np.linspace(low, high, num=max(2, num_value_candidates), dtype=float).tolist()
            except (TypeError, ValueError):
                pass

        return dom

    def canonical_value(self, variable: Any) -> Any:
        domain = list(getattr(variable, "domain", []))
        return self._belief.canonical_value(domain)

    def value_signal(self, var: str, value: Any) -> float:
        return float(self._belief.value_signal(var, value))

    def encode_value(self, var: str, value: Any, domain: list[Any] | None = None) -> float:
        del var
        try:
            return float(value)
        except (TypeError, ValueError):
            pass

        if domain and value in domain:
            return float(domain.index(value))

        return 0.0

    def simulate_intervention(
        self,
        model: tuple[list[int], np.ndarray, np.ndarray],
        *,
        intervention: dict[str, Any],
        n: int,
        rng: np.random.Generator,
        domain_lookup: dict[str, list[Any]] | None = None,
    ) -> pd.DataFrame | None:
        summary = self.summary()
        if summary is None:
            return None

        order, B, sigma2 = model
        cols = summary.columns
        means = np.asarray(summary.means, dtype=float)
        X = np.zeros((n, len(cols)), dtype=float)
        index = self.column_index()
        encoded_intervention = {
            index[name]: self.encode_value(name, value, None if domain_lookup is None else domain_lookup.get(name))
            for name, value in intervention.items()
            if name in index
        }

        for node in order:
            if node in encoded_intervention:
                X[:, node] = encoded_intervention[node]
                continue

            parents = np.flatnonzero(np.abs(B[:, node]) > 0.0)
            centered = np.zeros(n, dtype=float)
            if parents.size > 0:
                centered = (X[:, parents] - means[parents]) @ B[parents, node]
            noise = rng.normal(0.0, float(np.sqrt(max(sigma2[node], 1e-8))), size=n)
            X[:, node] = means[node] + centered + noise

        return pd.DataFrame(X, columns=cols)

    def predictive_disagreement(
        self,
        models: list[tuple[list[int], np.ndarray, np.ndarray]],
        *,
        intervention: dict[str, Any],
        focus_nodes: list[str] | None,
        n: int,
        rng: np.random.Generator,
        domain_lookup: dict[str, list[Any]] | None = None,
    ) -> float:
        summary = self.summary()
        if summary is None or not models:
            return 0.0

        focus = focus_nodes or summary.columns
        predictions: list[np.ndarray] = []

        for model in models:
            simulated = self.simulate_intervention(
                model,
                intervention=intervention,
                n=n,
                rng=rng,
                domain_lookup=domain_lookup,
            )
            if simulated is None:
                continue
            usable = [node for node in focus if node in simulated.columns]
            if not usable:
                continue
            predictions.append(simulated[usable].mean(axis=0).to_numpy(dtype=float))

        if len(predictions) < 2:
            return 0.0

        arr = np.stack(predictions, axis=0)
        return float(np.var(arr, axis=0).mean())

    def gradient_score(
        self,
        models: list[tuple[list[int], np.ndarray, np.ndarray]],
        *,
        intervention: dict[str, Any],
        focus_nodes: list[str] | None,
        n: int,
        rng: np.random.Generator,
        domain_lookup: dict[str, list[Any]] | None = None,
    ) -> float:
        summary = self.summary()
        if summary is None or not models:
            return 0.0

        B_mean = np.mean(np.stack([B for _, B, _ in models], axis=0), axis=0)
        means = np.asarray(summary.means, dtype=float)
        focus_idx = [
            self.column_index()[name]
            for name in (focus_nodes or summary.columns)
            if name in self.column_index()
        ]
        if not focus_idx:
            focus_idx = list(range(len(summary.columns)))

        scores: list[float] = []
        for model in models:
            simulated = self.simulate_intervention(
                model,
                intervention=intervention,
                n=n,
                rng=rng,
                domain_lookup=domain_lookup,
            )
            if simulated is None:
                continue

            X = simulated.to_numpy(dtype=float)
            centered = X - means
            total = 0.0
            for idx in focus_idx:
                pred = centered @ B_mean[:, idx]
                residual = centered[:, idx] - pred
                grad = centered.T @ residual / max(1, n)
                grad[idx] = 0.0
                total += float(np.linalg.norm(grad, ord=2))
            scores.append(total)

        if not scores:
            return 0.0
        return float(np.mean(scores))

    def query_effect_variance(
        self,
        models: list[tuple[list[int], np.ndarray, np.ndarray]],
        *,
        query: QuerySpec,
        n: int,
        rng: np.random.Generator,
        domain_lookup: dict[str, list[Any]] | None = None,
    ) -> float:
        summary = self.summary()
        if summary is None or not models or query.treatment is None or query.outcome is None:
            return 0.0
        if query.treatment not in summary.columns or query.outcome not in summary.columns:
            return 0.0

        treatment_values = list(query.treatment_values)
        if len(treatment_values) < 2 and domain_lookup is not None:
            treatment_values = list(domain_lookup.get(query.treatment, []))
        if len(treatment_values) < 2:
            return 0.0

        control_value = treatment_values[0]
        treated_value = treatment_values[-1]
        effects: list[float] = []

        for model in models:
            control = self.simulate_intervention(
                model,
                intervention={query.treatment: control_value},
                n=n,
                rng=rng,
                domain_lookup=domain_lookup,
            )
            treated = self.simulate_intervention(
                model,
                intervention={query.treatment: treated_value},
                n=n,
                rng=rng,
                domain_lookup=domain_lookup,
            )
            if control is None or treated is None:
                continue
            if query.outcome not in control.columns or query.outcome not in treated.columns:
                continue
            effects.append(float(treated[query.outcome].mean() - control[query.outcome].mean()))

        if len(effects) < 2:
            return 0.0
        return float(np.var(np.asarray(effects, dtype=float)))

    def query_path_relevance(
        self,
        models: list[tuple[list[int], np.ndarray, np.ndarray]],
        *,
        treatment: str,
        outcome: str,
        candidate: str,
    ) -> float:
        summary = self.summary()
        if summary is None or not models:
            return 0.0
        index = self.column_index()
        if treatment not in index or outcome not in index or candidate not in index:
            return 0.0

        treatment_idx = index[treatment]
        outcome_idx = index[outcome]
        candidate_idx = index[candidate]

        hit = 0.0
        for _, B, _ in models:
            if candidate_idx in {treatment_idx, outcome_idx}:
                hit += 1.0
                continue

            adjacency = {
                int(parent): [int(child) for child in np.flatnonzero(np.abs(B[parent]) > 0.0)]
                for parent in range(B.shape[0])
            }
            queue: deque[tuple[int, tuple[int, ...]]] = deque([(treatment_idx, (treatment_idx,))])
            found = False

            while queue and not found:
                node, path = queue.popleft()
                if node == outcome_idx:
                    if candidate_idx in path:
                        found = True
                    continue

                for child in adjacency.get(node, []):
                    if child in path:
                        continue
                    queue.append((child, (*path, child)))

            if found:
                hit += 1.0

        return float(hit / len(models))

    def query_edge_coverage(self, *, treatment: str, outcome: str, candidate: str) -> float:
        summary = self.summary()
        if summary is None:
            return 0.0
        index = self.column_index()
        if treatment not in index or outcome not in index or candidate not in index:
            return 0.0

        c = index[candidate]
        t = index[treatment]
        y = index[outcome]
        edge_entropy = summary.edge_entropy
        return float(
            edge_entropy[c, t]
            + edge_entropy[t, c]
            + edge_entropy[c, y]
            + edge_entropy[y, c]
        )
