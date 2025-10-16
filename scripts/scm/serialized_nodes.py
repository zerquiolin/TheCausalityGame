"""The Causality Game - Test Serialized Nodes."""

# TODO: Complete the test with categorical node once implemented, also for the other types of nodes.

import numpy as np
import pandas as pd
import sympy as sp

# Registry
from TheCausalityGame.core.infrastructure.registry import build_from_spec

# Nodes
from TheCausalityGame.scm.nodes.sympy import (
    EquationBasedCategoricalSCMNode,
    EquationBasedNumericalSCMNode,
)
from TheCausalityGame.scm.noise.uniform import UniformNoiseDistribution

# Create noise distribution
numerical = EquationBasedNumericalSCMNode(
    name="F",
    evaluation=sp.sympify("a*m"),
    domain=[-1e11, 1e11],
    noise_distribution=UniformNoiseDistribution(),
    accessibility="controllable",
    parents=["a", "m"],
    parent_mappings=None,
)

# categorical = EquationBasedCategoricalSCMNode(
#     name="categorical_noise",
#     equation="Categorical({0: 0.5, 1: 0.5})",
# )

# Serialize nodes to JSON
numerical_json = numerical.to_json()
# categorical_json = categorical.to_json()
# Deserialize nodes from JSON
numerical_deserialized = build_from_spec(spec=numerical_json)
# categorical_deserialized = build_from_spec(spec=categorical_json)

# Parent Values
parent_values = pd.DataFrame({"a": [2, 3, 4, 5, 6, 7], "m": [3, 4, 5, 6, 7, 8]})

# Generate noise values
numerical_first = numerical.generate_values(
    parent_values,
    random_state=np.random.RandomState(911),
)
numerical_second = numerical.generate_values(
    parent_values,
    random_state=np.random.RandomState(911),
)

# categorical_first = numerical.generate_values()
# categorical_second = numerical.generate_values()

# Check if both nodes are identical
assert numerical_json == numerical_deserialized.to_json(), "Numerical nodes differ!"
# assert (
#     categorical_json == categorical_deserialized.to_json()
# ), "Categorical nodes differ!"

# Check if both generated values are the same
assert all(numerical_first == numerical_second), "Numerical noise distributions differ!"
# assert (
#     categorical_first == categorical_second
# ), "Categorical noise distributions differ!"


print("Original and deserialized nodes are identical.")
