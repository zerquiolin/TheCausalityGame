"""The Causality Game - Structural Causal Model (SCM) Node Tests."""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from TheCausalityGame.core.contracts.scm_node import SCMNode
from TheCausalityGame.core.infrastructure.registry import (
    build_from_spec,
    load_subclasses_from_path,
)
from TheCausalityGame.core.lib.enum.nodes import NodeAccessibility
from TheCausalityGame.core.lib.utils.tests import (
    assert_dicts_equal,
    get_required_init_args,
)
from TheCausalityGame.scm.noise.uniform import UniformNoiseDistribution

# Search Classes
base_path = "TheCausalityGame/scm/nodes"
classes = load_subclasses_from_path(SCMNode, base_path)


# Tests
@pytest.fixture(params=classes, scope="module")
def scm_node_instance(request: Any) -> None:  # noqa: ANN401
    """Test class construction."""
    cls = request.param
    # Common arguments
    args = {
        "name": cls.__name__,
        "evaluation": None,
        "domain": [0, 1],
        "noise_distribution": UniformNoiseDistribution(),
        "accessibility": NodeAccessibility.CONTROLLABLE,
        "parents": [],
        "probability_distribution": [0.5, 0.5],
    }
    # Filter parameters
    args = {k: v for k, v in args.items() if k in get_required_init_args(cls)}
    return cls(**args)


def test_scm_node_properties(scm_node_instance: Any) -> None:  # noqa: ANN401
    """Test SCM Node properties."""
    # Check name
    assert scm_node_instance.name == scm_node_instance.__class__.__name__
    # Check domain
    assert scm_node_instance.domain == [0, 1]
    # Check accessibility
    assert scm_node_instance.accessibility == NodeAccessibility.CONTROLLABLE
    # Check noise distribution
    if scm_node_instance.noise_distribution:
        assert isinstance(
            scm_node_instance.noise_distribution, UniformNoiseDistribution
        )


def test_scm_node_serialization_roundtrip(
    scm_node_instance: Any,  # noqa: ANN401
) -> None:
    """Test serialization roundtrip."""
    # Build json
    cls_json = scm_node_instance.to_json()
    # Deserialize
    cls_deserialized = build_from_spec(cls_json)
    # Compare
    assert_dicts_equal(scm_node_instance.to_dict(), cls_deserialized.to_dict())


def test_data_generation(scm_node_instance: Any) -> None:  # noqa: ANN401
    """Test data generation."""
    df = pd.DataFrame(index=range(5))
    # Generate data
    result = scm_node_instance.generate_values(
        df, random_state=np.random.RandomState(911)
    )
    # Assertion
    assert len(result) == len(df)
    # Generate again to check for consistency
    result_second = scm_node_instance.generate_values(
        df, random_state=np.random.RandomState(911)
    )
    # Compare
    assert np.array_equal(result, result_second)
