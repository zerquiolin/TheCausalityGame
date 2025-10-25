"""The Causality Game - Core DAG implementation using NetworkX."""

from typing import Any, override

import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from networkx.readwrite import json_graph

from TheCausalityGame.core.contracts.dag import DAG
from TheCausalityGame.core.contracts.specs.dag import DAGSpec
from TheCausalityGame.core.infrastructure.registry import get_class_path
from TheCausalityGame.core.lib.errors.dag import DAGCycleError


class CoreDAG(DAG):
    """
    Core DAG implementation using NetworkX.

    This class represents a Directed Acyclic Graph (DAG) and provides support for
    serialization, deserialization, and plotting using NetworkX.

    Parameters
    ----------
    graph : nx.DiGraph
        The directed acyclic graph to manage.

    Raises
    ------
    ValueError
        If the provided graph is not a valid DAG.
    """

    def __init__(self, graph: nx.DiGraph) -> None:
        if not nx.is_directed_acyclic_graph(graph):
            raise DAGCycleError()
        super().__init__(graph)

    @override
    def to_spec(self) -> DAGSpec:
        dag = json_graph.node_link_data(self.graph, edges="edges")  # type: ignore

        return DAGSpec(
            class_=get_class_path(self.__class__),
            nodes=[x["id"] for x in dag["nodes"]],  # type: ignore
            edges=[(e["source"], e["target"]) for e in dag["edges"]],  # type: ignore
        )

    @classmethod
    @override
    def from_spec(cls, spec: DAGSpec) -> "CoreDAG":
        dag_description_for_nx: dict[str, Any] = {
            "directed": True,
            "multigraph": False,
            "graph": {},
            "nodes": [{"id": x} for x in spec.nodes],
            "edges": [{"source": e[0], "target": e[1]} for e in spec.edges],
        }

        dag_graph = json_graph.node_link_graph(dag_description_for_nx, edges="edges")  # type: ignore
        return cls(dag_graph)  # type: ignore

    @override
    def generate_figure(
        self, title: str = "", spacing_factor: float = 2.0
    ) -> matplotlib.figure.Figure:
        roots, leaves, _ = self.get_node_types()

        # Assign node colors and sizes based on node role
        node_colors: list[str] = []
        node_sizes: list[int] = []
        for node in self.graph.nodes():  # type: ignore
            if node in roots:
                node_colors.append("#81ff00")  # green for root
                node_sizes.append(1000)
            elif node in leaves:
                node_colors.append("#ff1c00")  # red for leaf
                node_sizes.append(1000)
            else:
                node_colors.append("#ffffff")  # white for intermediate
                node_sizes.append(800)

        # Spring layout positioning
        pos = nx.spring_layout(self.graph, k=1.2, scale=spacing_factor * 10, seed=42)  # type: ignore

        fig, ax = plt.subplots(figsize=(10, 8))  # type: ignore

        # Draw nodes
        nx.draw_networkx_nodes(  # type: ignore
            self.graph,
            pos,
            node_color=node_colors,  # type: ignore
            edgecolors="black",
            node_size=node_sizes,  # type: ignore
            ax=ax,
        )

        # Draw node labels
        nx.draw_networkx_labels(  # type: ignore
            self.graph,
            pos,
            labels={node: node for node in self.graph.nodes()},  # type: ignore
            font_size=12,
            font_weight="bold",
            ax=ax,
        )

        # Draw directed edges with arrowheads
        for u, v in self.graph.edges():  # type: ignore
            start, end = np.array(pos[u]), np.array(pos[v])  # type: ignore
            direction = end - start  # type: ignore
            norm_dir = direction / np.linalg.norm(direction)  # type: ignore
            node_radius = 0.03 * np.linalg.norm(list(ax.get_xlim()))

            # Offset arrows to avoid overlap with node markers
            start = start + norm_dir * node_radius
            end = end - norm_dir * node_radius

            ax.annotate(  # type: ignore
                "",
                xy=end,  # type: ignore
                xytext=start,  # type: ignore
                arrowprops={
                    "arrowstyle": "-|>",
                    "lw": 1.5,
                    "color": "gray",
                    "shrinkA": 10,
                    "shrinkB": 10,
                    "clip_on": False,
                    "connectionstyle": "arc3,rad=0.1",
                },
            )

        plt.title(title or "DAG Structure", fontsize=16)  # type: ignore
        plt.axis("off")  # type: ignore

        return fig

    @override
    def plot(
        self, title: str = "Directed Acyclic Graph (DAG)", spacing_factor: float = 2.0
    ) -> None:
        fig = self.generate_figure(title, spacing_factor)
        # Ensure this figure is the active one in pyplot, then show via pyplot.
        plt.figure(fig.number)  # type: ignore
        plt.show()  # type: ignore
