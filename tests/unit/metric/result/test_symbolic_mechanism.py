"""Tests for symbolic mechanism result metrics."""

from __future__ import annotations

import networkx as nx
import numpy as np
import sympy as sp

from TheCausalityGame.agent.inferers.symbolic_scm import (
    EstimatedSymbolicSCM,
    SymbolicMechanism,
)
from TheCausalityGame.metric.result.symbolic_mechanism import (
    SymbolicMechanismFunctionalErrorMetric,
)
from TheCausalityGame.scm.core import CoreSCM
from TheCausalityGame.scm.dag.core import CoreDAG
from TheCausalityGame.scm.nodes.sympy import EquationBasedNumericalSCMNode
from TheCausalityGame.scm.noise.uniform import UniformNoiseDistribution


def _product_scm() -> CoreSCM:
    """Create a small SCM with mechanism F = a*m."""
    graph = nx.DiGraph()
    graph.add_edges_from([("a", "F"), ("m", "F")])
    dag = CoreDAG(graph=graph)
    a = EquationBasedNumericalSCMNode(
        name="a",
        evaluation=None,
        domain=[1.0, 3.0],
        noise_distribution=UniformNoiseDistribution(),
    )
    m = EquationBasedNumericalSCMNode(
        name="m",
        evaluation=None,
        domain=[2.0, 4.0],
        noise_distribution=UniformNoiseDistribution(),
    )
    f = EquationBasedNumericalSCMNode(
        name="F",
        evaluation=sp.sympify("a*m"),
        domain=[2.0, 12.0],
        parents=["a", "m"],
        noise_distribution=UniformNoiseDistribution(),
    )
    return CoreSCM(dag=dag, nodes=[a, m, f], random_state=np.random.RandomState(911))


def test_symbolic_mechanism_metric_scores_exact_expression_near_zero() -> None:
    """An exact symbolic expression scores near zero functional error."""
    metric = SymbolicMechanismFunctionalErrorMetric(num_samples=128)
    metric.mount(_product_scm())
    graph = nx.DiGraph()
    graph.add_edges_from([("a", "F"), ("m", "F")])
    result = EstimatedSymbolicSCM(
        graph,
        {"F": SymbolicMechanism("F", ["a", "m"], sp.sympify("a*m"))},
    )

    assert metric.evaluate("SymbolicSCM", result) < 1e-10


def test_symbolic_mechanism_metric_penalizes_missing_expression() -> None:
    """A missing target expression receives the configured penalty."""
    metric = SymbolicMechanismFunctionalErrorMetric(num_samples=128, missing_penalty=7.0)
    metric.mount(_product_scm())
    result = EstimatedSymbolicSCM(nx.DiGraph(), {})

    assert metric.evaluate("SymbolicSCM", result) == 7.0


def test_symbolic_mechanism_metric_scores_equivalent_expression_near_zero() -> None:
    """An algebraically equivalent expression scores near zero functional error."""
    metric = SymbolicMechanismFunctionalErrorMetric(num_samples=128)
    metric.mount(_product_scm())
    graph = nx.DiGraph()
    graph.add_edges_from([("a", "F"), ("m", "F")])
    result = EstimatedSymbolicSCM(
        graph,
        {"F": SymbolicMechanism("F", ["a", "m"], sp.sympify("m*a + 0"))},
    )

    assert metric.evaluate("SymbolicSCM", result) < 1e-10
