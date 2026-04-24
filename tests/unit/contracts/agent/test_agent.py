"""The Causality Game - Agent Tests."""

from typing import Any

import pytest

from TheCausalityGame.agent.combined import CombinedAgent
from TheCausalityGame.agent.composable import ComposableAgent
from TheCausalityGame.agent.deciders.abci import ABCIDecider
from TheCausalityGame.agent.deciders.exhaustive import ExhaustiveDecider
from TheCausalityGame.agent.deciders.optimal_effect_design import OptimalEffectDesignDecider
from TheCausalityGame.agent.deciders.random import RandomDecider
from TheCausalityGame.agent.deciders.trust_gradient import TrustYourGradientDecider
from TheCausalityGame.agent.inferers.dag import DAGDiscoveryInferer
from TheCausalityGame.agent.inferers.outcome_regression import OutcomeRegressionInferer
from TheCausalityGame.agent.inferers.scm import SCMDiscoveryInferer
from TheCausalityGame.agent.policies.random import RandomAgentPolicy
from TheCausalityGame.core.contracts.specs.agent import AgentSpec
from TheCausalityGame.core.infrastructure.registry import build_from_spec
from TheCausalityGame.core.lib.utils.tests import assert_dicts_equal


@pytest.fixture(
    params=[
        ComposableAgent(
            id="cate_random",
            inferer=OutcomeRegressionInferer(),
            decider=RandomDecider(),
        ),
        ComposableAgent(
            id="cate_abci",
            inferer=OutcomeRegressionInferer(),
            decider=ABCIDecider(),
        ),
        ComposableAgent(
            id="cate_optimal_effect",
            inferer=OutcomeRegressionInferer(),
            decider=OptimalEffectDesignDecider(),
        ),
        ComposableAgent(
            id="dag_exhaustive",
            inferer=DAGDiscoveryInferer(),
            decider=ExhaustiveDecider(),
        ),
        ComposableAgent(
            id="scm_trust_gradient",
            inferer=SCMDiscoveryInferer(),
            decider=TrustYourGradientDecider(),
        ),
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
