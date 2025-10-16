# Math
# Plotting
import matplotlib.pyplot as plt

# Graph
import networkx as nx
import numpy as np
from networkx.readwrite import json_graph

# Internal imports
from TheCausalityGame.core.contracts.dag import DAG
from TheCausalityGame.core.contracts.specs.dag import DAGSpec
from TheCausalityGame.core.infrastructure.registry import get_class_path


class CoreDAG(DAG):
    """
    DAG is a concrete implementation of the BaseDAG abstract class.
    It uses a NetworkX DiGraph to represent a Directed Acyclic Graph and provides
    methods to serialize, deserialize, and visualize the graph.
    """

    def __init__(self, graph: nx.DiGraph):
        """
        Initializes the DAG instance.

        Args:
            graph (nx.DiGraph): A NetworkX directed acyclic graph.

        Raises:
            ValueError: If the provided graph is not a valid DAG.
        """
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError(
                "The provided graph is not a directed acyclic graph (DAG)."
            )
        super().__init__(graph)

    def to_spec(self) -> DAGSpec:
        dag = json_graph.node_link_data(self.graph, edges="edges")

        # simplify node and edge encoding for shorter format
        return DAGSpec(
            class_=get_class_path(self.__class__),
            nodes=[x["id"] for x in dag["nodes"]],
            edges=[[e["source"], e["target"]] for e in dag["edges"]],
        )

    @classmethod
    def from_spec(cls, spec: DAGSpec) -> "CoreDAG":
        dag_description_for_nx = {
            "directed": True,
            "multigraph": False,
            "graph": {},
        }

        # recover description as required by nx from simplified format
        dag_description_for_nx.update(
            {
                "nodes": [{"id": x} for x in spec.nodes],
                "edges": [{"source": e[0], "target": e[1]} for e in spec.edges],
            }
        )
        dag_graph = json_graph.node_link_graph(dag_description_for_nx, edges="edges")
        return cls(dag_graph)

    def plot(self, title="", spacing_factor: float = 2.0) -> None:
        roots, leaves, intermediates = self.get_node_types()

        node_colors, node_sizes = [], []
        for node in self.graph.nodes():
            if node in roots:
                node_colors.append("#81ff00")
                node_sizes.append(1000)
            elif node in leaves:
                node_colors.append("#ff1c00")
                node_sizes.append(1000)
            else:
                node_colors.append("#ffffff")
                node_sizes.append(800)

        pos = nx.spring_layout(self.graph, k=1.2, scale=spacing_factor * 10, seed=42)
        fig, ax = plt.subplots(figsize=(10, 8))

        nx.draw_networkx_nodes(
            self.graph,
            pos,
            node_color=node_colors,
            edgecolors="black",
            node_size=node_sizes,
            ax=ax,
        )

        nx.draw_networkx_labels(
            self.graph,
            pos,
            labels={node: node for node in self.graph.nodes()},
            font_size=12,
            font_weight="bold",
            ax=ax,
        )

        for u, v in self.graph.edges():
            start, end = np.array(pos[u]), np.array(pos[v])
            direction = end - start
            norm_dir = direction / np.linalg.norm(direction)
            node_radius = 0.03 * np.linalg.norm(list(ax.get_xlim()))
            start = start + norm_dir * node_radius
            end = end - norm_dir * node_radius

            ax.annotate(
                "",
                xy=end,
                xytext=start,
                arrowprops=dict(
                    arrowstyle="-|>",
                    lw=1.5,
                    color="gray",
                    shrinkA=10,
                    shrinkB=10,
                    clip_on=False,
                    connectionstyle="arc3,rad=0.1",
                ),
            )

        plt.title(title or "DAG Structure", fontsize=16)
        plt.axis("off")
        plt.show()
