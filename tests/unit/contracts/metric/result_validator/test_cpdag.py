"""Tests for CPDAG validation and scoring semantics."""

from types import SimpleNamespace

import networkx as nx
from matplotlib import pyplot as plt

from TheCausalityGame.core.contracts.dto.causal_graph import PartiallyDirectedGraph
from TheCausalityGame.hook.plots.final_dag_per_agent import _cpdag_figure
from TheCausalityGame.metric.result.result_validator.dag_validator import (
    DAGResultValidator,
)
from TheCausalityGame.metric.result.shd import SHDResultMetric


def test_dag_validator_accepts_cpdag() -> None:
    graph = PartiallyDirectedGraph(nodes=("X", "Y"), undirected_edges=(("Y", "X"),))

    assert DAGResultValidator().validate(graph) == graph
    assert graph.undirected_edges == (("X", "Y"),)


def test_cpdag_serialization_round_trip() -> None:
    graph = PartiallyDirectedGraph(
        nodes=("X", "Y", "Z"),
        directed_edges=(("Y", "Z"),),
        undirected_edges=(("Y", "X"),),
    )

    assert PartiallyDirectedGraph.model_validate(graph.model_dump()) == graph


def test_cpdag_plot_accepts_unresolved_edges() -> None:
    graph = PartiallyDirectedGraph(
        nodes=("X", "Y", "Z"),
        directed_edges=(("Y", "Z"),),
        undirected_edges=(("X", "Y"),),
    )

    figure = _cpdag_figure(graph)

    assert figure.axes
    plt.close(figure)


def test_matching_unresolved_edge_has_no_orientation_penalty() -> None:
    truth = nx.DiGraph([("X", "Y")])
    result = PartiallyDirectedGraph(nodes=("X", "Y"), undirected_edges=(("X", "Y"),))
    metric = SHDResultMetric()
    metric.mount(SimpleNamespace(dag=truth))

    assert metric.evaluate("DAG", result) == 0.0


def test_contradicted_cpdag_arrowhead_costs_one() -> None:
    truth = nx.DiGraph([("X", "Y")])
    result = PartiallyDirectedGraph(nodes=("X", "Y"), directed_edges=(("Y", "X"),))
    metric = SHDResultMetric()
    metric.mount(SimpleNamespace(dag=truth))

    assert metric.evaluate("DAG", result) == 1.0
