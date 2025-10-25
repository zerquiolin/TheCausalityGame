"""The Causality Game - Test Serialized Nodes."""

import numpy as np
import pandas as pd
import sympy as sp

# Registry
from TheCausalityGame.core.infrastructure.registry import build_from_spec
from TheCausalityGame.core.lib.enum.nodes import NodeAccessibility
from TheCausalityGame.scm.nodes.bn import BayesianNetworkSCMNode

# Nodes
from TheCausalityGame.scm.nodes.sympy import (
    EquationBasedCategoricalSCMNode,
    EquationBasedNumericalSCMNode,
)
from TheCausalityGame.scm.noise.uniform import UniformNoiseDistribution

# Create Numerical Node
numerical = EquationBasedNumericalSCMNode(
    name="F",
    evaluation=sp.sympify("a*m"),
    domain=[-1e11, 1e11],
    noise_distribution=UniformNoiseDistribution(),
    accessibility=NodeAccessibility.CONTROLLABLE,
    parents=["a", "m"],
    parent_mappings=None,
)

# Create Categorical Node
categorical = EquationBasedCategoricalSCMNode(
    name="categorical",
    evaluation=None,
    domain=["a", "b", "c"],
    noise_distribution=UniformNoiseDistribution(),
)

# Create Bayesian Network Node
bayesian = BayesianNetworkSCMNode(
    name="Binary",
    accessibility=NodeAccessibility.CONTROLLABLE,
    parents=[],
    domain=["1", "0"],
    probability_distribution=[0.5, 0.5],
)


# Serialize nodes to JSON
numerical_json = numerical.to_json()
bayesian_json = bayesian.to_json()
categorical_json = categorical.to_json()
# Deserialize nodes from JSON
numerical_deserialized = build_from_spec(spec=numerical_json)
bayesian_deserialized = build_from_spec(spec=bayesian_json)
categorical_deserialized = build_from_spec(spec=categorical_json)

# Parent Values
parent_values = pd.DataFrame({"a": [2, 3, 4, 5, 6, 7], "m": [3, 4, 5, 6, 7, 8]})

# Generate numerical values
numerical_first = numerical.generate_values(
    parent_values,
    random_state=np.random.RandomState(911),
)
numerical_second = numerical.generate_values(
    parent_values,
    random_state=np.random.RandomState(911),
)

# Generate categorical values
categorical_first = categorical.generate_values(
    pd.DataFrame(index=range(10)),
    random_state=np.random.RandomState(911),
)
categorical_first = categorical.generate_values(
    pd.DataFrame(index=range(10)),
    random_state=np.random.RandomState(911),
)

# Generate bayesian network values
bayesian_first = bayesian.generate_values(
    pd.DataFrame(index=range(10)),
    random_state=np.random.RandomState(911),
)
bayesian_second = bayesian.generate_values(
    pd.DataFrame(index=range(10)),
    random_state=np.random.RandomState(911),
)

# Check if both nodes are identical
assert numerical_json == numerical_deserialized.to_json(), "Numerical nodes differ!"
assert (
    categorical_json == categorical_deserialized.to_json()
), "Categorical nodes differ!"
assert bayesian_json == bayesian_deserialized.to_json(), "Bayesian nodes differ!"

# Check if both generated values are the same
assert all(numerical_first == numerical_second), "Numerical noise distributions differ!"
# assert (
#     categorical_first == categorical_second
# ), "Categorical noise distributions differ!"
assert all(bayesian_first == bayesian_second), "Bayesian noise distributions differ!"


print("Original and deserialized nodes are identical.")
