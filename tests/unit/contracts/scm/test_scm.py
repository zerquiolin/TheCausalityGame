"""The Causality Game - Structural Causal Model (SCM) Tests."""

from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import sympy as sp
from core.lib.enum.nodes import NodeAccessibility

from TheCausalityGame.core.contracts.scm import SCM
from TheCausalityGame.core.contracts.specs.scm import SCMSpec
from TheCausalityGame.core.infrastructure.registry import (
    build_from_spec,
    load_subclasses_from_path,
)
from TheCausalityGame.core.lib.utils.tests import (
    assert_dicts_equal,
)
from TheCausalityGame.scm.dag.core import CoreDAG
from TheCausalityGame.scm.nodes.sympy import EquationBasedNumericalSCMNode
from TheCausalityGame.scm.noise.uniform import UniformNoiseDistribution

# Search Classes
base_path = "TheCausalityGame/scm"
classes = load_subclasses_from_path(SCM, base_path)


# Tests
@pytest.fixture(params=classes, scope="module")
def scm_instance(request: Any) -> None:  # noqa: ANN401
    """Test class construction."""
    cls = request.param
    # Arguments for a simple SCM
    graph = nx.DiGraph()
    graph.add_edges_from([("a", "F"), ("m", "F")])  # type: ignore
    dag = CoreDAG(graph=graph)

    # Create Nodes
    a = EquationBasedNumericalSCMNode(
        name="a",
        evaluation=None,
        domain=[-1e11, 1e11],
        noise_distribution=UniformNoiseDistribution(),
    )
    m = EquationBasedNumericalSCMNode(
        name="m",
        evaluation=None,
        domain=[0, 1e11],
        noise_distribution=UniformNoiseDistribution(),
    )
    F = EquationBasedNumericalSCMNode(  # noqa: N806
        name="F",
        evaluation=sp.sympify("a*m"),  # type: ignore
        domain=[3, 15],
        parents=["a", "m"],
        noise_distribution=UniformNoiseDistribution(),
        accessibility=NodeAccessibility.MEASURABLE,
    )

    return cls(
        dag=dag,
        nodes=[a, m, F],
        random_state=np.random.RandomState(911),
    )


def test_scm_properties(scm_instance: Any) -> None:  # noqa: ANN401
    """Test SCM properties."""
    # Node/edge access
    assert len(scm_instance.nodes) == 3  # noqa: PLR2004
    assert all(node in {"a", "m", "F"} for node in scm_instance.nodes)

    # Node accessibility
    assert scm_instance.nodes["F"].accessibility == NodeAccessibility.MEASURABLE
    assert scm_instance.nodes["m"].accessibility == NodeAccessibility.CONTROLLABLE
    assert scm_instance.nodes["a"].accessibility == NodeAccessibility.CONTROLLABLE


def test_scm_serialization_roundtrip(scm_instance: Any) -> None:  # noqa: ANN401
    """Test serialization roundtrip."""
    spec = scm_instance.to_spec()
    assert isinstance(spec, SCMSpec)
    dag2 = build_from_spec(spec)
    assert_dicts_equal(scm_instance.to_dict(), dag2.to_dict())


def test_data_generation(scm_instance: Any) -> None:  # noqa: ANN401
    """Test data generation."""
    df = pd.DataFrame(index=range(5))
    # Generate data
    result = scm_instance.generate_samples(
        df, num_samples=5, random_state=np.random.RandomState(911)
    )
    # Assertion
    assert len(result) == len(df)
    # Generate again to check for consistency
    result_second = scm_instance.generate_samples(
        df, num_samples=5, random_state=np.random.RandomState(911)
    )

    # Compare
    assert result.equals(result_second)
