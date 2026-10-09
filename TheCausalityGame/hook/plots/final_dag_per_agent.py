from __future__ import annotations

from pathlib import Path
from typing import override

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from TheCausalityGame.core.contracts.dto.causal_graph import PartiallyDirectedGraph
from TheCausalityGame.core.contracts.dto.transcript import Transcript, TranscriptEntry
from TheCausalityGame.core.contracts.hook import Hook
from TheCausalityGame.core.lib.enum.hook import HookEvent
from TheCausalityGame.scm.dag.core import CoreDAG


def _cpdag_figure(graph: PartiallyDirectedGraph) -> plt.Figure:
    """Render directed and unresolved CPDAG edges without imposing acyclicity on display arcs."""
    layout_graph = nx.Graph()
    layout_graph.add_nodes_from(graph.nodes)
    layout_graph.add_edges_from(graph.directed_edges)
    layout_graph.add_edges_from(graph.undirected_edges)
    positions = nx.spring_layout(layout_graph, seed=42)
    figure, axis = plt.subplots(figsize=(10, 8))
    nx.draw_networkx_nodes(
        layout_graph, positions, node_color="white", edgecolors="black", ax=axis
    )
    nx.draw_networkx_labels(layout_graph, positions, ax=axis)
    nx.draw_networkx_edges(
        layout_graph,
        positions,
        edgelist=list(graph.directed_edges),
        arrows=True,
        arrowstyle="-|>",
        edge_color="gray",
        ax=axis,
    )
    nx.draw_networkx_edges(
        layout_graph,
        positions,
        edgelist=list(graph.undirected_edges),
        arrows=False,
        style="dashed",
        edge_color="gray",
        ax=axis,
    )
    axis.set_title("CPDAG Structure")
    axis.axis("off")
    return figure


def to_imshow_array(obj) -> np.ndarray:
    # Case 1: a Matplotlib Figure -> render to RGBA pixels
    if isinstance(obj, plt.Figure):
        canvas = FigureCanvas(obj)
        canvas.draw()
        arr = np.asarray(canvas.buffer_rgba())
        plt.close(obj)  # optional: avoid leaking figures
        return arr

    # Case 2: PIL image / numpy array / list-of-lists, etc.
    arr = np.asarray(obj)

    # If numpy ends up with dtype=object, force a numeric conversion
    if arr.dtype == object:
        arr = np.array(obj, dtype=np.uint8)

    return arr


class FinalDAGPerAgent(Hook):
    id: str = "Agents DAGs"
    step: HookEvent = HookEvent.BENCHMARK_END

    @override
    def run(
        self, hooks_dir: Path, context: dict[str, Transcript] | TranscriptEntry | None
    ) -> None:
        if context is None or not isinstance(context, dict):
            return

        fig, axes = plt.subplots(1, 7, figsize=(18, 4))
        fig.suptitle("Agent Final DAGs", fontsize=14)

        for i, (id, transcript) in enumerate(context.items()):
            if transcript.invalidated or not transcript.entries:
                continue

            last_entry = transcript.entries[-1]
            if last_entry.result is None:
                continue

            result = last_entry.result
            if isinstance(result, PartiallyDirectedGraph):
                figure_obj = _cpdag_figure(result)
            else:
                figure_obj = CoreDAG(result).generate_figure()
            img = to_imshow_array(figure_obj)

            ax = axes[i]
            ax.set_title(f"Agent: {id}")
            ax.imshow(img)
            ax.axis("off")

        plt.subplots_adjust(top=0.83)
        # Save the figure to a file
        plot_path = hooks_dir / "agents_dags.png"
        fig.savefig(plot_path)  # type: ignore
