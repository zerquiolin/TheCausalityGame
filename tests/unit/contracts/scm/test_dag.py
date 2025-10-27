"""The Causality Game - Directed Acyclic Graph (DAG) Tests."""

from typing import Any

import networkx as nx
import pytest

from TheCausalityGame.core.contracts.dag import DAG
from TheCausalityGame.core.contracts.specs.dag import DAGSpec
from TheCausalityGame.core.infrastructure.registry import (
    build_from_spec,
    load_subclasses_from_path,
)
from TheCausalityGame.core.lib.utils.tests import (
    assert_dicts_equal,
)

# Search Classes
base_path = "TheCausalityGame/scm/dag"
classes = load_subclasses_from_path(DAG, base_path)


# Tests
@pytest.fixture(params=classes, scope="module")
def dag_instance(request: Any) -> None:  # noqa: ANN401
    """Test class construction."""
    cls = request.param
    # Create simple graph
    g = nx.DiGraph()
    g.add_edges_from([("X", "Y"), ("Z", "Y")])  # type: ignore
    return cls(g)


def test_dag_properties(dag_instance: Any) -> None:  # noqa: ANN401
    """Test DAG properties."""
    # Node/edge access
    assert set(dag_instance.nodes) == {"X", "Y", "Z"}
    assert ("X", "Y") in set(dag_instance.edges)

    # Node types (roots/leaves)
    roots, leaves, middle = dag_instance.get_node_types()
    assert set(roots) == {"X", "Z"}
    assert set(leaves) == {"Y"}
    assert set(middle) == set()


def test_dag_serialization_roundtrip(dag_instance: Any) -> None:  # noqa: ANN401
    """Test serialization roundtrip."""
    spec = dag_instance.to_spec()
    assert isinstance(spec, DAGSpec)
    dag2 = build_from_spec(spec)
    assert_dicts_equal(dag_instance.to_dict(), dag2.to_dict())


@pytest.mark.parametrize("cls", classes)
def test_coredag_rejects_cycle(cls: Any) -> None:  # noqa: ANN401
    """Thest that CoreDAG rejects cycles."""
    g = nx.DiGraph()
    g.add_edges_from([("A", "B"), ("B", "A")])  # type: ignore
    with pytest.raises(Exception):  # noqa: B017
        _ = cls(graph=g)
