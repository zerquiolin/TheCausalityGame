"""The Causality Game - Directed Acyclic Graph (DAG) Contract."""

from abc import abstractmethod
from typing import Any

import networkx as nx

from TheCausalityGame.core.contracts.serializable import Serializable
from TheCausalityGame.core.lib.errors.dag import DAGCycleError


class DAG(Serializable):
    """
    Abstract base class for Directed Acyclic Graph (DAG) structures using NetworkX.

    Provides utility methods for DAG inspection, including access to nodes and edges,
    identification of root, leaf, and intermediate nodes, and retrieval of parent nodes.

    Subclasses must implement the `plot` method for visualization.

    Parameters
    ----------
    graph : nx.DiGraph
        A directed acyclic graph.

    Raises
    ------
    ValueError
        If the provided graph contains cycles.
    """

    def __init__(self, graph: nx.DiGraph) -> None:
        self.graph: nx.DiGraph = graph
        if not nx.is_directed_acyclic_graph(self.graph):
            raise DAGCycleError()

    @property
    def nodes(self) -> list[str]:
        """
        List of all nodes in the DAG.

        Returns
        -------
        list[str]
            A list of node identifiers.
        """
        return list(self.graph.nodes())  # type: ignore

    @property
    def edges(self) -> list[tuple[str, str]]:
        """
        List of all edges in the DAG.

        Returns
        -------
        list[tuple[str, str]]
            A list of (source, target) tuples representing edges.
        """
        return list(self.graph.edges())  # type: ignore

    def get_parents(self, node: str) -> list[str]:
        """
        Retrieve all parent (predecessor) nodes of a given node.

        Parameters
        ----------
        node : str
            The target node.

        Returns
        -------
        list[str]
            A list of parent node identifiers.
        """
        return list(self.graph.predecessors(node))  # type: ignore

    def get_node_types(self) -> tuple[list[str], list[str], list[str]]:
        """
        Categorize nodes.

        - roots (no incoming edges)
        - leaves (no outgoing edges)
        - intermediates (having both incoming and outgoing edges).

        Returns
        -------
        tuple[list[str], list[str], list[str]]
            A tuple of (roots, leaves, intermediates).
        """
        roots: list[str] = []
        leaves: list[str] = []
        intermediates: list[str] = []
        for node in self.nodes:
            if self.graph.in_degree(node) == 0:  # type: ignore
                roots.append(node)
            elif self.graph.out_degree(node) == 0:  # type: ignore
                leaves.append(node)
            else:
                intermediates.append(node)
        return roots, leaves, intermediates

    def get_structured_nodes(self) -> dict[Any, list[Any]]:
        """
        Create a mapping of each node to its list of parent nodes.

        Returns
        -------
        dict[Any, list[Any]]
            Dictionary mapping each node to its parent nodes.
        """
        return {node: self.get_parents(node) for node in self.nodes}

    @abstractmethod
    def plot(self, spacing_factor: float = 2.0) -> None:
        """
        Visualize the DAG structure.

        Parameters
        ----------
        spacing_factor : float, optional
            Factor to control spacing between nodes in the plot, by default 2.0

        Raises
        ------
        NotImplementedError
            If the method is not implemented by a subclass.
        """
        raise NotImplementedError("Subclasses must implement this method.")
