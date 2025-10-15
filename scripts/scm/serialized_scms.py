"""The Causality Game - Test Serialized scms."""

# Registry
import json

import networkx as nx
import numpy as np
import sympy as sp

from TheCausalityGame.core.infraestructure.registry import build_from_spec

# Noises
from TheCausalityGame.scm.core import CoreSCM
from TheCausalityGame.scm.dag.core import CoreDAG
from TheCausalityGame.scm.nodes.sympy import EquationBasedNumericalSCMNode
from TheCausalityGame.scm.noise.uniform import UniformNoiseDistribution

# Create DAG
graph = nx.DiGraph()
graph.add_edges_from([("Z", "X"), ("X", "Y"), ("Z", "Y")])
dag = CoreDAG(graph=graph)

# Create Nodes
Z = EquationBasedNumericalSCMNode(
    name="Z",
    evaluation=None,
    domain=[1, 5],
    noise_distribution=UniformNoiseDistribution(),
    accessibility="controllable",
    parents=None,
    parent_mappings=None,
)
X = EquationBasedNumericalSCMNode(
    name="X",
    evaluation=sp.sympify("2*Z"),
    domain=[2, 10],
    noise_distribution=UniformNoiseDistribution(),
    accessibility="observable",
    parents=["Z"],
    parent_mappings=None,
)
Y = EquationBasedNumericalSCMNode(
    name="Y",
    evaluation=sp.sympify("X+2*Z"),
    domain=[3, 15],
    noise_distribution=UniformNoiseDistribution(),
    accessibility="observable",
    parents=["Z", "X"],
    parent_mappings=None,
)

# Create scm
scm = CoreSCM(
    dag=dag,
    nodes=[Z, X, Y],
    random_state=np.random.RandomState(911),
)

# Serialize SCM to JSON
scm_json = scm.to_json()

# Deserialize SCM from JSON
scm_deserialized = build_from_spec(spec=scm_json)

# Generate values
fist = scm.generate_samples(num_samples=10)
second = scm_deserialized.generate_samples(num_samples=10)

# Check if both generated samples are the same
assert all(fist == second)


def deep_dict_equal(d1, d2, path=""):
    if isinstance(d1, dict) and isinstance(d2, dict):
        if d1.keys() != d2.keys():
            print(f"Key mismatch at {path}: {d1.keys()} vs {d2.keys()}")
            return False
        for key in d1:
            new_path = f"{path}.{key}" if path else key
            if not deep_dict_equal(d1[key], d2[key], new_path):
                return False
        return True

    elif isinstance(d1, list) and isinstance(d2, list):
        if len(d1) != len(d2):
            print(f"List length mismatch at {path}: {len(d1)} vs {len(d2)}")
            return False
        for index, (item1, item2) in enumerate(zip(d1, d2)):
            new_path = f"{path}[{index}]"
            if not deep_dict_equal(item1, item2, new_path):
                return False
        return True

    else:
        if d1 != d2:
            print(f"Value mismatch at {path}: {d1} vs {d2}")
            return False
        return True


# Check if both SCM's are the same
deep_dict_equal(scm.to_dict(), scm_deserialized.to_dict())

print("Original and deserialized SCM's are identical.")

# Save scm to json file
path = "scm_test.json"
cjson = scm.to_dict()

new_nodes = []
for node in scm.nodes.values():
    node_dict = node.to_dict()
    del node_dict["random_state"]
    new_nodes.append(node_dict)
cjson["vars"] = new_nodes
del cjson["random_state"]

with open(path, "w") as f:
    f.write(json.dumps(cjson, indent=4))
