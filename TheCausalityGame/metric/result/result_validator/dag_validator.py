"""The Causality Game - DAG Result Validator."""

from __future__ import annotations

from typing import Any, override

import networkx as nx

from TheCausalityGame.core.contracts.dto.causal_graph import PartiallyDirectedGraph
from TheCausalityGame.core.contracts.result_validator import ResultValidator
from TheCausalityGame.core.contracts.specs.result_validator import ResultValidatorSpec
from TheCausalityGame.core.infrastructure.registry import get_class_path
from TheCausalityGame.core.lib.errors.result_validator import InvalidResultTypeError


class DAGResultValidator(ResultValidator):
    """
    Validator for treatment effect estimation functions.

    This result validator ensures the agent returns a valid Directed Acyclic Graph (DAG)
    represented by a NetworkX DiGraph object.

    Attributes
    ----------
    _kind : str
        The kind of result this validator supports.
    """

    _kind = "DAG"

    def __init__(self, allowed_nodes: list[str] | None = None) -> None:
        self.allowed_nodes = None if allowed_nodes is None else set(allowed_nodes)

    @override
    def validate(self, result: Any) -> Any:
        if not isinstance(result, nx.DiGraph | PartiallyDirectedGraph):
            raise InvalidResultTypeError(
                type=str(type(result)),  # type: ignore
                expected="nx.DiGraph or PartiallyDirectedGraph",
            )

        nodes = list(result.nodes)
        if any(not isinstance(node, str) or not node for node in nodes):
            raise ValueError("Graph node labels must be nonempty strings.")
        if len(nodes) != len(set(nodes)):
            raise ValueError("Graph node labels must be unique.")
        if self.allowed_nodes is not None and not set(nodes).issubset(
            self.allowed_nodes
        ):
            raise ValueError("Graph contains unknown node labels.")
        if isinstance(result, nx.DiGraph):
            if result.is_multigraph() or not nx.is_directed_acyclic_graph(result):
                raise ValueError(
                    "A DAG result must be simple and acyclic, without self-loops."
                )
        else:
            graph = nx.DiGraph()
            graph.add_nodes_from(nodes)
            graph.add_edges_from(result.directed_edges)
            edges = list(result.directed_edges) + list(result.undirected_edges)
            if any(u == v or u not in nodes or v not in nodes for u, v in edges):
                raise ValueError("Partial edges must connect distinct declared nodes.")
            if not nx.is_directed_acyclic_graph(graph):
                raise ValueError("Partial graph contains a directed cycle.")
            directed = {frozenset(edge) for edge in result.directed_edges}
            if directed.intersection(
                frozenset(edge) for edge in result.undirected_edges
            ):
                raise ValueError(
                    "A partial edge cannot be both directed and unresolved."
                )
            unresolved = nx.Graph()
            unresolved.add_nodes_from(nodes)
            unresolved.add_edges_from(result.undirected_edges)
            components = {
                node: i
                for i, component in enumerate(nx.connected_components(unresolved))
                for node in component
            }
            component_graph = nx.DiGraph()
            for u, v in result.directed_edges:
                if components[u] == components[v]:
                    raise ValueError("Partial graph contains a semidirected cycle.")
                component_graph.add_edge(components[u], components[v])
            if not nx.is_directed_acyclic_graph(component_graph):
                raise ValueError("Partial graph contains a semidirected cycle.")
            for component in nx.connected_components(unresolved):
                if not nx.is_chordal(unresolved.subgraph(component)):
                    raise ValueError("CPDAG undirected components must be chordal.")
        return result

    @override
    def to_spec(self) -> ResultValidatorSpec:
        return ResultValidatorSpec(
            class_=get_class_path(self.__class__),
            params={
                "allowed_nodes": (
                    sorted(self.allowed_nodes)
                    if self.allowed_nodes is not None
                    else None
                )
            },
        )

    @classmethod
    @override
    def from_spec(cls, spec: ResultValidatorSpec) -> DAGResultValidator:
        return cls(**(spec.params or {}))
