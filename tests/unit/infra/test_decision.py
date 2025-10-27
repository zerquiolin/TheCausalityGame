"""The Causality Game - Tests for the Decision infrastructure helpers."""

import pytest

from TheCausalityGame.core.infrastructure.decisions import Decision, Experiment
from TheCausalityGame.core.lib.enum.environment import ActionKind
from TheCausalityGame.core.lib.errors.environment import DecisionMismatchError


def test_experiment_factory_initializes_empty_decision() -> None:
    """Test that the experiment factory method creates a Decision."""
    decision = Decision.experiment()

    assert decision.kind is ActionKind.EXPERIMENT
    assert decision.is_experiment
    assert not decision.is_answer
    assert decision.experiments == []


def test_add_experiment_appends_experiment_instances() -> None:
    """Test that add_experiment adds an Experiment to the Decision."""
    decision = Decision.experiment()

    returned = decision.add_experiment(treatment={"X": 1}, n=10)

    assert returned is decision
    assert len(decision.experiments) == 1
    experiment = decision.experiments[0]
    assert isinstance(experiment, Experiment)
    assert experiment.treatment == {"X": 1}
    assert experiment.n == 10  # noqa: PLR2004


def test_extend_accepts_mixed_input_types() -> None:
    """Test that extend accepts both Experiment instances and tuples."""
    decision = Decision.experiment()
    observational = Experiment(treatment=None, n=15)

    decision.extend(
        [
            ({"X": 1, "Z": 0}, 20),
            observational,
        ]
    )

    assert len(decision.experiments) == 2  # noqa: PLR2004
    first, second = decision.experiments
    assert first.treatment == {"X": 1, "Z": 0}
    assert first.n == 20  # noqa: PLR2004
    assert second is observational
    assert second.is_observational


def test_to_dict_serializes_kind_and_experiments() -> None:
    """Test that to_dict serializes the Decision correctly."""
    decision = Decision.experiment()
    decision.add_experiment(treatment={"X": 1}, n=10)
    decision.extend([(None, 5)])

    payload = decision.to_dict()

    assert payload["kind"] == ActionKind.EXPERIMENT.value
    assert payload["experiments"] == [
        {"treatment": {"X": 1}, "n": 10},
        {"treatment": None, "n": 5},
    ]


def test_answer_decision_rejects_experiments() -> None:
    """Test that an answer Decision cannot accept experiments."""
    answer = Decision.answer()

    assert answer.kind is ActionKind.ANSWER
    assert answer.is_answer
    assert not answer.is_experiment

    with pytest.raises(DecisionMismatchError):
        answer.add_experiment({"X": 1}, 10)
