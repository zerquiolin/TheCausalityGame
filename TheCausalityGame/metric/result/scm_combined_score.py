"""The Causality Game SCM Combined Score Result Metric."""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from typing import Any, override

import networkx as nx
import numpy as np
import pandas as pd

from TheCausalityGame.agent.inferers.scm import EstimatedANMSCM
from TheCausalityGame.core.contracts.mission import ResultMetric
from TheCausalityGame.core.contracts.scm import SCM
from TheCausalityGame.core.contracts.specs.metric import MetricSpec
from TheCausalityGame.core.infrastructure.registry import get_class_path
from TheCausalityGame.core.lib.errors.metric import (
    NotInitializedError,
    UnsupportedMetricTypeError,
)


class SCMCombinedScoreResultMetric(ResultMetric):
    """
    Computes a score based on Structural Hamming Distance (SHD) and SCM accuracy.

    This metric compares the true Directed Acyclic Graph (DAG) structure of the SCM
    to the one estimated by the agent and the generation of data from the estimated SCM.

    Attributes
    ----------
    name : str
        Human-readable name for the metric.
    description : str
        Description of the metric's purpose.
    kinds : list of str
        Types of expected answers this metric is compatible with.
    """

    name = "SCM Combined Score"
    description = (
        "Computes a score based on Structural Hamming Distance (SHD) and SCM accuracy."
        " This metric compares the true Directed Acyclic Graph (DAG) structure of the SCM"
        " to the one estimated by the agent and the generation of data from the estimated SCM."
    )
    kinds = ["SCM"]  # noqa: RUF012

    @override
    def mount(self, scm: SCM) -> None:
        self.seed = 911
        self.scm = scm
        self.rng = np.random.default_rng(self.seed)
        self._true_edges = set(tuple(edge) for edge in self.scm.dag.edges)  # type: ignore

        # Cache true samples and intervention specs to avoid recomputation in evaluate.
        num_samples = int(getattr(self, "num_samples", 256))
        num_trials = int(getattr(self, "num_trials", 5))
        self._cached_num_samples = num_samples
        self._cached_num_trials = num_trials
        self._trial_specs: list[dict[str, Any]] = []

        for _ in range(num_trials):
            intervention_var = self.rng.choice(self.scm.controllable_vars)
            low, high = self.scm.nodes[intervention_var].domain
            value = self.rng.uniform(float(low), float(high))

            true_samples = self.scm.generate_samples(
                num_samples=num_samples,
                interventions={intervention_var: value},
                cancel_noise=True,
                random_state=np.random.RandomState(self.seed),
            )

            self._trial_specs.append(
                {
                    "intervention_var": intervention_var,
                    "value": value,
                    "true_samples": true_samples,
                }
            )
        self.is_mounted = True

    def skeleton_edge_set(self, G: nx.DiGraph, nodes: Iterable[Hashable]) -> set[frozenset[str]]:  # type: ignore # noqa: N803
        """
        Compute the undirected edge set (“skeleton”) of a directed graph G by a given node set.

        Args
        ----
            G: A NetworkX directed graph.
            nodes: Iterable of node labels to include.

        Returns
        -------
            A set of frozensets, each representing an undirected edge
            {u, v}.

        """
        node_set = set(nodes)
        # Only consider edges where both endpoints are in node_set
        edges: set[frozenset[str]] = (  # type: ignore
            frozenset((u, v))  # type: ignore
            for u, v in G.edges  # type: ignore
            if u in node_set and v in node_set
        )
        return set(edges)

    @override
    def evaluate(self, kind: str, result: Any) -> float:
        """
        Evaluate agent's estimated causal graph against the true graph using *directed* SHD.

        Scoring convention (default):
        - missing edge: +1
        - extra edge: +1
        - reversed edge: +1   (can be changed by setting self.reversal_cost = 2)

        Notes
        -----
        - This metric compares directed edges, not just the skeleton.
        - Nodes are unioned so that missing/extra edges are counted consistently.
        """
        if not self.is_mounted:
            raise NotInitializedError(self.name)

        if kind != "SCM":
            raise UnsupportedMetricTypeError(kind)

        if not isinstance(result, EstimatedANMSCM):
            raise TypeError(f"Expected EstimatedANMSCM concrete class result, got {type(result)}")

        # Union of nodes
        all_nodes = set(self.scm.dag.nodes) | set(result.dag.nodes)  # type: ignore

        def directed_edge_set(G: nx.DiGraph) -> set[tuple[str, str]]:  # type: ignore # noqa: N803
            edges: set[tuple[str, str]] = set()
            for u, v in G.edges:  # type: ignore
                if u in all_nodes and v in all_nodes:
                    edges.add((u, v))  # type: ignore
            return edges

        true_edges = {
            edge for edge in self._true_edges if edge[0] in all_nodes and edge[1] in all_nodes
        }
        pred_edges = directed_edge_set(result.dag)  # type: ignore

        reversal_cost = getattr(self, "reversal_cost", 1)

        missing = 0
        reversed_ = 0

        remaining_pred = set(pred_edges)

        for u, v in true_edges:
            if (u, v) in remaining_pred:
                remaining_pred.remove((u, v))
            elif (v, u) in remaining_pred:
                remaining_pred.remove((v, u))
                reversed_ += 1
            else:
                missing += 1

        extra = len(remaining_pred)

        shd_score = float(missing + extra + reversal_cost * reversed_)

        # print(f"SHD Score: {shd_score} (missing: {missing}, extra: {extra}, reversed: {reversed_})")

        # Evaluate the accuracy of the estimated SCM in generating data.
        # IMPORTANT: comparing sample i to sample i across two independently-sampled SCMs
        # produces a very noisy score that does not converge to 0 even for a perfect model.
        # Instead we compare *distributional* summaries under interventions.

        eps = 1e-12
        num_samples = int(getattr(self, "_cached_num_samples", getattr(self, "num_samples", 256)))

        def _is_numeric(arr: np.ndarray) -> bool:
            try:
                a = arr.astype(float)
                return np.isfinite(a).mean() > 0.95
            except Exception:
                return False

        def _moment_score(true_arr: np.ndarray, est_arr: np.ndarray) -> float:
            """Scale-normalized mean/variance error for numeric variables."""
            t = np.asarray(true_arr, dtype=float)
            e = np.asarray(est_arr, dtype=float)
            t = t[np.isfinite(t)]
            e = e[np.isfinite(e)]
            if t.size < 5 or e.size < 5:
                return 1.0

            mu_t = float(t.mean())
            mu_e = float(e.mean())
            var_t = float(t.var())
            var_e = float(e.var())

            # normalized mean-squared error (scale-invariant)
            mean_term = ((mu_t - mu_e) ** 2) / (var_t + eps)
            # normalized variance error
            var_term = ((var_t - var_e) ** 2) / ((var_t**2) + eps)
            return float(mean_term + var_term)

        def _categorical_l1(true_arr: np.ndarray, est_arr: np.ndarray) -> float:
            """L1 distance between empirical category probabilities."""
            t = np.asarray(true_arr, dtype=object)
            e = np.asarray(est_arr, dtype=object)
            # Drop missing
            t = t[pd.notna(t)] if "pd" in globals() else t[t != None]  # noqa: E711
            e = e[pd.notna(e)] if "pd" in globals() else e[e != None]  # noqa: E711
            if t.size == 0 or e.size == 0:
                return 1.0
            cats = set(t.tolist()) | set(e.tolist())
            if not cats:
                return 1.0
            t_counts = {c: 0 for c in cats}
            e_counts = {c: 0 for c in cats}
            for v in t.tolist():
                t_counts[v] += 1
            for v in e.tolist():
                e_counts[v] += 1
            t_total = float(len(t))
            e_total = float(len(e))
            l1 = 0.0
            for c in cats:
                l1 += abs((t_counts[c] / t_total) - (e_counts[c] / e_total))
            return float(l1)

        scm_score = 0.0
        trial_count = 0

        for trial in self._trial_specs:
            intervention_var = trial["intervention_var"]
            value = trial["value"]
            true_samples = trial["true_samples"]

            # Estimated SCM: prefer deterministic generation if available to reduce variance
            try:
                estimated_samples = result.generate_samples(
                    num_samples=num_samples,
                    interventions={intervention_var: value},
                    deterministic=True,
                )
            except TypeError:
                estimated_samples = result.generate_samples(
                    num_samples=num_samples,
                    interventions={intervention_var: value},
                )

            # Score all variables except the intervened one
            per_trial = 0.0
            per_count = 0

            for var in self.scm.vars:
                if var == intervention_var:
                    continue

                if var not in true_samples.columns:
                    continue

                true_values = true_samples[var].values

                if var not in estimated_samples.columns:
                    # Missing variable output is a hard failure
                    per_trial += 5.0
                    per_count += 1
                    continue

                est_values = estimated_samples[var].values

                if _is_numeric(true_values) and _is_numeric(est_values):
                    per_trial += _moment_score(true_values, est_values)
                else:
                    per_trial += _categorical_l1(true_values, est_values)

                per_count += 1

            if per_count > 0:
                scm_score += per_trial / per_count
                trial_count += 1

        if trial_count > 0:
            scm_score /= trial_count
        else:
            scm_score = float("inf")

        # Combine SHD score and SCM distributional score.
        # Default: mostly SCM fit, with optional small SHD weight.
        alpha = 0.75
        final_score = alpha * scm_score + (1 - alpha) * shd_score
        return float(final_score)
        # return float(scm_score)

    @override
    def to_spec(self) -> MetricSpec:
        return MetricSpec(
            class_=get_class_path(self.__class__),
        )

    @classmethod
    @override
    def from_spec(cls, spec: MetricSpec) -> SCMCombinedScoreResultMetric:
        return cls()
