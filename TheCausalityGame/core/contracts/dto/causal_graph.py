"""Typed causal-graph estimates shared by inferers and metrics."""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from typing import TypeAlias

import networkx as nx
from pydantic import field_validator

from TheCausalityGame.core.contracts.dto.common import CommonDTO

Edge = tuple[str, str]


class PartiallyDirectedGraph(CommonDTO):
    """A CPDAG with separate directed and unresolved edge collections."""

    nodes: tuple[str, ...] = ()
    directed_edges: tuple[Edge, ...] = ()
    undirected_edges: tuple[Edge, ...] = ()

    @field_validator("undirected_edges")
    @classmethod
    def normalize_undirected_edges(cls, edges: tuple[Edge, ...]) -> tuple[Edge, ...]:
        """Canonicalize undirected endpoints and reject self edges."""
        normalized: set[Edge] = set()
        for source, target in edges:
            if source == target:
                raise ValueError("A CPDAG cannot contain a self edge.")
            normalized.add((source, target) if source < target else (target, source))
        return tuple(sorted(normalized))

    def to_networkx(self) -> nx.DiGraph:
        """Return a display graph, encoding unresolved edges in both directions."""
        graph = nx.DiGraph()
        graph.add_nodes_from(self.nodes)
        graph.add_edges_from(self.directed_edges)
        for source, target in self.undirected_edges:
            graph.add_edge(source, target, unresolved=True)
            graph.add_edge(target, source, unresolved=True)
        return graph


CausalGraphResult: TypeAlias = nx.DiGraph | PartiallyDirectedGraph


def graph_nodes(graph: CausalGraphResult) -> set[Hashable]:
    """Return nodes from either supported graph representation."""
    if isinstance(graph, PartiallyDirectedGraph):
        return set(graph.nodes)
    return set(graph.nodes)


def graph_directed_edges(graph: CausalGraphResult) -> set[tuple[Hashable, Hashable]]:
    """Return only compelled/directed edges."""
    if isinstance(graph, PartiallyDirectedGraph):
        return set(graph.directed_edges)
    return set(graph.edges)


def graph_undirected_edges(graph: CausalGraphResult) -> set[frozenset[Hashable]]:
    """Return unresolved CPDAG adjacencies."""
    if not isinstance(graph, PartiallyDirectedGraph):
        return set()
    return {frozenset(edge) for edge in graph.undirected_edges}


def graph_adjacencies(
    graph: CausalGraphResult,
    nodes: Iterable[Hashable] | None = None,
) -> set[frozenset[Hashable]]:
    """Return the skeleton of either graph representation."""
    allowed = graph_nodes(graph) if nodes is None else set(nodes)
    directed = {
        frozenset((source, target))
        for source, target in graph_directed_edges(graph)
        if source in allowed and target in allowed
    }
    unresolved = {
        edge for edge in graph_undirected_edges(graph) if edge.issubset(allowed)
    }
    return directed | unresolved
