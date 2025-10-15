"""The Causality Game - Test Serialized DAGs."""

# Registry
# networkx
import networkx as nx

from TheCausalityGame.core.infraestructure.registry import build_from_spec

# DAG
from TheCausalityGame.scm.dag.core import CoreDAG

# Create DAG
graph = nx.DiGraph()
graph.add_edges_from([("Z", "X"), ("X", "Y"), ("Z", "Y")])
dag = CoreDAG(graph=graph)
dag.plot()


# Serialize noise distribution to JSON
dag_json = dag.to_json()

# Deserialize noise distribution from JSON
dag_deserialized = build_from_spec(spec=dag_json)


# Check if both generated noises are the same
assert dag_json == dag_deserialized.to_json()

print("Original and deserialized DAG's are identical.")
