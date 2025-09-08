from __future__ import annotations

import networkx as nx
import pytest

# If you renamed this class/module, adjust here:
CoreDAG = pytest.importorskip("TheCausalityGame.scm.dag.core").CoreDAG
DAGSpec = pytest.importorskip("TheCausalityGame.core.contracts.specs.dag").DAGSpec


def test_coredag_create_and_properties():
    g = nx.DiGraph()
    g.add_edges_from([("X", "Y"), ("Z", "Y")])
    dag = CoreDAG(graph=g)

    # Node/edge access
    assert set(dag.nodes) == {"X", "Y", "Z"}
    assert ("X", "Y") in set(dag.edges)

    # Node types (roots/leaves)
    roots, leaves, middle = dag.get_node_types()
    assert set(roots) == {"X", "Z"}
    assert set(leaves) == {"Y"}
    assert set(middle) == set()


def test_coredag_serialization_roundtrip():
    g = nx.DiGraph()
    g.add_edges_from([("A", "B"), ("B", "C")])
    dag = CoreDAG(graph=g)

    spec = dag.to_spec()
    assert isinstance(spec, DAGSpec)
    dag2 = CoreDAG.from_spec(spec)

    assert set(dag2.nodes) == set(dag.nodes)
    assert set(dag2.edges) == set(dag.edges)


def test_coredag_rejects_cycle():
    g = nx.DiGraph()
    g.add_edges_from([("A", "B"), ("B", "A")])
    with pytest.raises(Exception):
        _ = CoreDAG(graph=g)
