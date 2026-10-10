"""Backend and action boundaries must reject invalid scientific representations."""

import numpy as np
import pytest

from TheCausalityGame.agent.deciders.exhaustive import ExhaustiveDecider
from TheCausalityGame.agent.deciders.random import RandomDecider
from TheCausalityGame.agent.inferers._data import adjacency_to_digraph
from TheCausalityGame.core.contracts.dto.environment import (
    AvailableActions,
    ExperimentVariable,
    RoundInfo,
)


@pytest.mark.parametrize(
    "matrix",
    [
        np.zeros((3, 3)),
        np.zeros((1, 2)),
        np.array([[0, np.nan], [0, 0]]),
        np.array([[0, np.inf], [0, 0]]),
    ],
)
def test_backend_adjacency_must_be_finite_and_match_variable_count(matrix):
    with pytest.raises(ValueError):
        adjacency_to_digraph(matrix, ["X", "Y"])


def test_random_decider_supports_three_category_domain():
    actions = AvailableActions(
        experiments=[ExperimentVariable(name="X", domain=["red", "green", "blue"])]
    )
    decision = RandomDecider(threshold=0).decide(RoundInfo(round=1), actions, None)
    assert decision.experiments[0].treatment["X"] in {"red", "green", "blue"}


def test_random_decider_can_observe_when_no_controls_exist():
    decision = RandomDecider(threshold=0).decide(
        RoundInfo(round=1), AvailableActions(experiments=[]), None
    )
    assert decision.experiments[0].treatment is None


def test_exhaustive_decider_enumerates_string_categories():
    actions = AvailableActions(
        experiments=[ExperimentVariable(name="X", domain=["small", "medium", "large"])]
    )
    decision = ExhaustiveDecider().decide(RoundInfo(round=1), actions, None)
    assert {item.treatment["X"] for item in decision.experiments if item.treatment} == {
        "small",
        "medium",
        "large",
    }
