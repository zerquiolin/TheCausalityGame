"""The Causality Game - Structural Causal Model (SCM) Noise Distribution Tests."""

from typing import Any

import pytest

from TheCausalityGame.core.contracts.noise import NoiseDistribution
from TheCausalityGame.core.contracts.specs.noise import NoiseDistributionSpec
from TheCausalityGame.core.infrastructure.registry import (
    build_from_spec,
    load_subclasses_from_path,
)
from TheCausalityGame.core.lib.utils.tests import (
    assert_dicts_equal,
)

# Search Classes
base_path = "TheCausalityGame/scm/noise"
classes = load_subclasses_from_path(NoiseDistribution, base_path)


# Tests
@pytest.fixture(params=classes, scope="module")
def noise_distribution_instance(request: Any) -> None:  # noqa: ANN401
    """Test class construction."""
    cls = request.param
    # We assume a no-arg constructor for noise distributions
    return cls()


def test_noise_distribution_serialization_roundtrip(
    noise_distribution_instance: Any,  # noqa: ANN401
) -> None:
    """Test serialization roundtrip."""
    spec = noise_distribution_instance.to_spec()
    assert isinstance(spec, NoiseDistributionSpec)
    noise_distribution_2 = build_from_spec(spec)
    assert_dicts_equal(
        noise_distribution_instance.to_dict(), noise_distribution_2.to_dict()
    )


def test_noise_distribution_generate_consistency(
    noise_distribution_instance: Any,  # noqa: ANN401
) -> None:
    """Test that two instances generate the same noise after serialization roundtrip."""
    spec = noise_distribution_instance.to_spec()
    noise_distribution_2 = build_from_spec(spec)

    first = noise_distribution_instance.generate(size=10, random_state=911)
    second = noise_distribution_2.generate(size=10, random_state=911)

    assert (first == second).all()
