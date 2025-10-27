"""The Causality Game - Infrastructure Registry Tests."""

import pytest

from TheCausalityGame.agent.exhaustive import ExhaustiveAgent
from TheCausalityGame.core.infrastructure.registry import (
    build_from_spec,
    get_class_path,
    load_class,
)
from TheCausalityGame.core.lib.errors.registry import (
    MissingAttributeError,
    NotAllowedByPolicyError,
)
from TheCausalityGame.scm.noise.uniform import UniformNoiseDistribution


def test_load_class_disallowed_package_rejected() -> None:
    """Test that loading a class from a disallowed package raises an error."""
    with pytest.raises(NotAllowedByPolicyError):
        load_class("notallowed.module:ClassName")


def test_build_from_spec_requires_class() -> None:
    """Test that building from a spec without a class raises an error."""
    with pytest.raises(MissingAttributeError):
        build_from_spec({})


def test_get_class_path() -> None:
    """Test getting the class path of a class."""
    assert (
        get_class_path(ExhaustiveAgent)
        == "TheCausalityGame.agent.exhaustive:ExhaustiveAgent"
    )
    assert (
        get_class_path(UniformNoiseDistribution)
        == "TheCausalityGame.scm.noise.uniform:UniformNoiseDistribution"
    )


def test_build_from_spec() -> None:
    """Test building an object from its specification."""
    nd = UniformNoiseDistribution()
    spec = nd.to_spec()
    noise = build_from_spec(spec)
    assert isinstance(noise, UniformNoiseDistribution)
    assert noise.low == nd.low
    assert noise.high == nd.high
