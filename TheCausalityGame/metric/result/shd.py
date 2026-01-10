"""The Causality Game - Structural Hamming Distance Result Metric."""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from typing import Any, override

import networkx as nx

from TheCausalityGame.core.contracts.mission import ResultMetric
from TheCausalityGame.core.contracts.scm import SCM
from TheCausalityGame.core.contracts.specs.metric import MetricSpec
from TheCausalityGame.core.infrastructure.registry import get_class_path
from TheCausalityGame.core.lib.errors.metric import (
    NotInitializedError,
    UnsupportedMetricTypeError,
)


class SHDResultMetric(ResultMetric):
    """
    Computes the Structural Hamming Distance (SHD).

    This metric compares the true Directed Acyclic Graph (DAG) structure of the SCM
    to the one estimated by the agent.

    Attributes
    ----------
    name : str
        Human-readable name for the metric.
    description : str
        Description of the metric's purpose.
    kinds : list of str
        Types of expected answers this metric is compatible with.
    """

    name = "SHD"
    description = (
        "Computes the Structural Hamming Distance (SHD) "
        "between the true and estimated Directed Acyclic Graph (DAG) structures."
    )
    kinds = ["DAG"]  # noqa: RUF012

    @override
    def mount(self, scm: SCM) -> None:
        self.scm = scm
        print(self.scm.dag.edges)
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

        if kind != "DAG":
            raise UnsupportedMetricTypeError(kind)

        if not isinstance(result, nx.DiGraph):
            raise TypeError(f"Expected nx.DiGraph result, got {type(result)}")

        # Union of nodes
        all_nodes = set(self.scm.dag.nodes) | set(result.nodes)  # type: ignore

        def directed_edge_set(G: nx.DiGraph) -> set[tuple[str, str]]:  # type: ignore # noqa: N803
            edges: set[tuple[str, str]] = set()
            for u, v in G.edges:  # type: ignore
                if u in all_nodes and v in all_nodes:
                    edges.add((u, v))  # type: ignore
            return edges

        true_edges = directed_edge_set(self.scm.dag)  # type: ignore
        pred_edges = directed_edge_set(result)

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

        return float(missing + extra + reversal_cost * reversed_)

    @override
    def to_spec(self) -> MetricSpec:
        return MetricSpec(
            class_=get_class_path(self.__class__),
        )

    @classmethod
    @override
    def from_spec(cls, spec: MetricSpec) -> SHDResultMetric:
        return cls()
