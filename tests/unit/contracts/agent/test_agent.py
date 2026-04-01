"""The Causality Game - Agent Tests."""

from typing import Any

import pytest

from TheCausalityGame.agent.combined import CombinedAgent
from TheCausalityGame.agent.composable import ComposableAgent
from TheCausalityGame.agent.deciders.exhaustive import ExhaustiveDecider
from TheCausalityGame.agent.deciders.random import RandomDecider
from TheCausalityGame.agent.inferers.cate import CATEInferer
from TheCausalityGame.agent.inferers.dag import DAGDiscoveryInferer
from TheCausalityGame.agent.policies.random import RandomAgentPolicy
from TheCausalityGame.core.contracts.specs.agent import AgentSpec
from TheCausalityGame.core.infrastructure.registry import build_from_spec
from TheCausalityGame.core.lib.utils.tests import assert_dicts_equal


@pytest.fixture(
    params=[
        ComposableAgent(id="cate_random", inferer=CATEInferer(), decider=RandomDecider()),
        ComposableAgent(id="dag_exhaustive", inferer=DAGDiscoveryInferer(), decider=ExhaustiveDecider()),
        CombinedAgent(id="combined_random", policy=RandomAgentPolicy()),
    ],
    scope="module",
)
def agent_instance(request: Any) -> Any:  # noqa: ANN401
    """Construct representative agent wrappers."""
    return request.param


def test_agent_serialization_roundtrip(agent_instance: Any) -> None:  # noqa: ANN401
    """Test serialization roundtrip."""
    spec = agent_instance.to_spec()
    assert isinstance(spec, AgentSpec)
    agent2 = build_from_spec(spec)
    assert_dicts_equal(agent_instance.to_dict(), agent2.to_dict())
