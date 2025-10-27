"""The Causality Game - Agent Tests."""

from typing import Any

import pytest

from TheCausalityGame.core.contracts.agent import Agent
from TheCausalityGame.core.contracts.specs.agent import AgentSpec
from TheCausalityGame.core.infrastructure.registry import (
    build_from_spec,
    load_subclasses_from_path,
)
from TheCausalityGame.core.lib.utils.tests import (
    assert_dicts_equal,
)

# Search Classes
base_path = "TheCausalityGame/agent"
classes = load_subclasses_from_path(Agent, base_path)


# Tests
@pytest.fixture(params=classes, scope="module")
def agent_instance(request: Any) -> None:  # noqa: ANN401
    """Test class construction."""
    cls = request.param
    # Assume default constructor with id only
    return cls(id=cls.__name__)


def test_agent_serialization_roundtrip(agent_instance: Any) -> None:  # noqa: ANN401
    """Test serialization roundtrip."""
    spec = agent_instance.to_spec()
    assert isinstance(spec, AgentSpec)
    agent2 = build_from_spec(spec)
    assert_dicts_equal(agent_instance.to_dict(), agent2.to_dict())
