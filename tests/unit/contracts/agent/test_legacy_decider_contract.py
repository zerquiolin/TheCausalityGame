"""Domain, degeneracy, reproducibility and budget-boundary tests for deciders."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pytest

from TheCausalityGame.agent.deciders.annadani import Annadani2024CAASLOnlineDecider
from TheCausalityGame.agent.deciders.cho import Cho2016ActiveGBNDecider
from TheCausalityGame.agent.deciders.exhaustive import ExhaustiveDecider
from TheCausalityGame.agent.deciders.geng import HeGeng2008MinimaxDecider
from TheCausalityGame.agent.deciders.gies import GIESDecider
from TheCausalityGame.agent.deciders.random import RandomDecider
from TheCausalityGame.agent.deciders.tigas import Tigas2022CBEDDecider
from TheCausalityGame.core.contracts.decider import Decider
from TheCausalityGame.core.contracts.dto.agent import RoundObservation
from TheCausalityGame.core.contracts.dto.environment import (
    AvailableActions,
    BudgetSnapshot,
    ExperimentVariable,
    RoundInfo,
    Samples,
    SamplesCollection,
)
from TheCausalityGame.core.contracts.specs.budget import BudgetSpec
from TheCausalityGame.core.infrastructure.decisions import Decision
from TheCausalityGame.core.lib.errors.budget import SamplesBudgetExceededError
from TheCausalityGame.core.managers.budget import BudgetManager

DeciderFactory = Callable[[], Decider]


def _decider_factories() -> list[tuple[str, DeciderFactory]]:
    return [
        ("random", lambda: RandomDecider(num_obs=1, num_inter=2, threshold=0)),
        ("exhaustive", lambda: ExhaustiveDecider(num_obs=1, num_inter=1)),
        ("annadani", lambda: Annadani2024CAASLOnlineDecider(num_obs=1, num_inter=2, seed=31)),
        ("cho", lambda: Cho2016ActiveGBNDecider(num_obs=1, num_inter=2, seed=31)),
        ("geng", lambda: HeGeng2008MinimaxDecider(num_obs=1, num_inter=2, seed=31)),
        ("gies", lambda: GIESDecider(num_obs=1, num_inter=2, seed=31)),
        ("tigas", lambda: Tigas2022CBEDDecider(num_obs=1, num_inter=2, seed=31)),
    ]


def _actions() -> AvailableActions:
    return AvailableActions(
        experiments=[
            ExperimentVariable(name="Z", domain=["control", "treated", "follow-up"]),
            ExperimentVariable(name="X", domain=[-2.0, 4.0]),
        ]
    )


def _constant_observation() -> RoundObservation:
    data = pd.DataFrame({"Z": ["only"] * 24, "X": [1.5] * 24, "Y": [2.0] * 24})
    sample = Samples(
        kind="observational", n=len(data), data=data, interventions=None
    )
    return RoundObservation(
        round_info=RoundInfo(round=1),
        decision=Decision.experiment().add_experiment(None, n=len(data)),
        samples=SamplesCollection([sample]),
    )


def _varied_observation() -> RoundObservation:
    """Provide nondegenerate confounded-looking data to activate model paths."""
    rows = 64
    treatment = ["control", "treated"] * (rows // 2)
    covariate = [float(index % 8) for index in range(rows)]
    outcome = [
        1.5 * covariate[index] + (2.0 if treatment[index] == "treated" else 0.0)
        for index in range(rows)
    ]
    data = pd.DataFrame({"Z": treatment, "X": covariate, "Y": outcome})
    sample = Samples(kind="observational", n=rows, data=data, interventions=None)
    return RoundObservation(
        round_info=RoundInfo(round=1),
        decision=Decision.experiment().add_experiment(None, n=rows),
        samples=SamplesCollection([sample]),
    )
def _assert_domain_valid(decision: Decision) -> None:
    domains = {variable.name: variable.domain for variable in _actions().experiments}
    assert decision.experiments
    for experiment in decision.experiments:
        assert experiment.n > 0
        if experiment.treatment is None:
            continue
        for name, value in experiment.treatment.items():
            domain = domains[name]
            if all(isinstance(label, str) for label in domain):
                assert isinstance(value, str) and value in domain
            else:
                low, high = domain
                assert isinstance(value, (int, float)) and low <= value <= high


@pytest.mark.parametrize(
    "name,factory",
    _decider_factories(),
    ids=lambda item: item if isinstance(item, str) else None,
)
def test_legacy_deciders_emit_domain_valid_reproducible_actions(
    name: str, factory: DeciderFactory
) -> None:
    """Same-seed cold starts are repeatable and proposals stay in declared domains."""
    del name
    actions = _actions()
    round_info = RoundInfo(round=1, budget_snapshot=BudgetSnapshot(samples_left=100))
    first = factory().decide(round_info, actions, None)  # type: ignore[arg-type]
    second = factory().decide(round_info, actions, None)  # type: ignore[arg-type]

    _assert_domain_valid(first)
    assert first.to_dict() == second.to_dict()


@pytest.mark.parametrize(
    "_name,factory",
    _decider_factories(),
    ids=lambda item: item if isinstance(item, str) else None,
)
def test_legacy_deciders_handle_constant_observations(
    _name: str, factory: DeciderFactory
) -> None:
    """Degenerate all-constant samples do not produce invalid actions or crash."""
    decider = factory()
    decider.update(_constant_observation())
    decision = decider.decide(
        RoundInfo(round=2, budget_snapshot=BudgetSnapshot(samples_left=100)),
        _actions(),
        None,  # type: ignore[arg-type]
    )

    _assert_domain_valid(decision)


@pytest.mark.parametrize(
    "_name,factory",
    _decider_factories(),
    ids=lambda item: item if isinstance(item, str) else None,
)
def test_environment_budget_preflight_rejects_oversized_legacy_design_atomically(
    _name: str, factory: DeciderFactory
) -> None:
    """A decider's multi-experiment proposal cannot bypass the sample budget."""
    decision = factory().decide(
        RoundInfo(round=1, budget_snapshot=BudgetSnapshot(samples_left=1)),
        _actions(),
        None,  # type: ignore[arg-type]
    )
    budget = BudgetManager(BudgetSpec(rounds=2, samples=1))

    with pytest.raises(SamplesBudgetExceededError):
        budget.preflight(decision.experiments)
    assert budget.samples_used == 0


@pytest.mark.parametrize(
    "_name,factory",
    _decider_factories(),
    ids=lambda item: item if isinstance(item, str) else None,
)
def test_legacy_model_informed_paths_handle_varied_observations(
    _name: str, factory: DeciderFactory
) -> None:
    """Nonconstant evidence activates learned-design paths without invalid actions."""
    decider = factory()
    decider.update(_varied_observation())
    decision = decider.decide(
        RoundInfo(round=2, budget_snapshot=BudgetSnapshot(samples_left=100)),
        _actions(),
        None,  # type: ignore[arg-type]
    )
    _assert_domain_valid(decision)
