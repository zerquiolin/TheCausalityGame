from __future__ import annotations

import pytest

from TheCausalityGame.agent.dummy_agent import DummyAgent
from TheCausalityGame.core.infrastructure.registry import (
    build_from_spec,
    get_class_path,
    load_class,
)
from TheCausalityGame.core.lib.classes.errors import LoadError
from TheCausalityGame.scm.noise.uniform import UniformNoiseDistribution


def test_load_class_disallowed_package_rejected() -> None:
    with pytest.raises(LoadError):
        load_class("notallowed.module:ClassName")


def test_build_from_spec_requires_class() -> None:
    with pytest.raises(LoadError):
        build_from_spec({})


def test_get_class_path() -> None:
    assert get_class_path(DummyAgent) == "TheCausalityGame.agent.dummy_agent:DummyAgent"
    assert (
        get_class_path(UniformNoiseDistribution)
        == "TheCausalityGame.scm.noise.uniform:UniformNoiseDistribution"
    )


def test_build_from_spec() -> None:
    nd = UniformNoiseDistribution()
    spec = nd.to_json()
    noise = build_from_spec(spec)
    assert isinstance(noise, UniformNoiseDistribution)
    assert noise.mu == nd.mu
    assert noise.sigma == nd.sigma
