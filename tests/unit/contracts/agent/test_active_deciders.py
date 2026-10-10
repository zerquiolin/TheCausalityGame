"""Tests for query-aware active deciders."""

from __future__ import annotations

import pandas as pd
import pytest

from TheCausalityGame.agent.deciders.abci import ABCIDecider
from TheCausalityGame.agent.deciders.optimal_effect_design import OptimalEffectDesignDecider
from TheCausalityGame.agent.deciders.trust_gradient import TrustYourGradientDecider
from TheCausalityGame.core.contracts.agent import AgentContext
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

OBSERVATION_BATCH_SIZE = 2


def _agent_context() -> AgentContext:
    return AgentContext(
        mission={
            "id": "conditional_average_treatment_effect",
            "name": "Conditional Average Treatment Effect Mission",
            "description": "Estimate a treatment effect.",
            "metadata": {
                "query_family": "treatment_effect",
                "estimand_kind": "cate",
                "treatment": "Z",
                "outcome": "Y",
                "covariates": ["X"],
                "treatment_values": ["0", "1"],
            },
        },
        behavior_metric={"name": "behavior", "description": "behavior", "metadata": {}},
        result_metric={
            "name": "PEHE",
            "description": "PEHE",
            "metadata": {
                "query_family": "treatment_effect",
                "estimand_kind": "cate",
                "treatment": "Z",
                "outcome": "Y",
                "covariates": ["X"],
                "treatment_values": ["0", "1"],
            },
        },
        custom_metrics=[],
    )


def _available_actions() -> AvailableActions:
    return AvailableActions(
        experiments=[
            ExperimentVariable(name="Z", domain=["0", "1"]),
            ExperimentVariable(name="W", domain=[0.0, 1.0]),
        ],
        answer="submit",
    )


def _round_observation() -> RoundObservation:
    df = pd.DataFrame(
        {
            "Z": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "X": [0.1, 0.8, 0.2, 1.0, 0.3, 0.9, 0.2, 1.1, 0.1, 1.2, 0.2, 1.0],
            "Y": [0.2, 1.0, 0.4, 1.4, 0.5, 1.6, 0.3, 1.7, 0.2, 1.8, 0.4, 1.5],
            "W": [0.5, 0.4, 0.55, 0.45, 0.5, 0.35, 0.6, 0.5, 0.65, 0.45, 0.55, 0.4],
        }
    )
    sample = Samples(
        kind="observational",
        n=len(df),
        data=df,
        interventions=None,
    )
    return RoundObservation(
        round_info=RoundInfo(round=1),
        decision=Decision.experiment().add_experiment(treatment=None, n=len(df)),
        samples=SamplesCollection([sample]),
    )


def test_optimal_effect_design_prefers_treatment_without_history() -> None:
    """The query-specific design policy starts by probing the treatment."""
    decider = OptimalEffectDesignDecider()
    decider.set_context(_agent_context())

    decision = decider.decide(
        round_info=RoundInfo(round=1),
        available_actions=_available_actions(),
        belief=None,  # type: ignore[arg-type]
    )

    assert decision.is_experiment
    assert decision.experiments[-1].treatment is not None
    assert "Z" in decision.experiments[-1].treatment


def test_abci_decider_returns_valid_intervention_after_update() -> None:
    """ABCI returns a controllable intervention after receiving samples."""
    decider = ABCIDecider(num_obs=0)
    decider.set_context(_agent_context())
    decider.update(_round_observation())

    decision = decider.decide(
        round_info=RoundInfo(round=2),
        available_actions=_available_actions(),
        belief=None,  # type: ignore[arg-type]
    )

    assert decision.is_experiment
    assert decision.experiments
    treatment = decision.experiments[-1].treatment or {}
    assert set(treatment).issubset({"Z", "W"})


def test_trust_gradient_returns_valid_intervention_after_update() -> None:
    """The gradient policy returns a controllable intervention after learning."""
    decider = TrustYourGradientDecider(num_obs=0)
    decider.set_context(_agent_context())
    decider.update(_round_observation())

    decision = decider.decide(
        round_info=RoundInfo(round=2),
        available_actions=_available_actions(),
        belief=None,  # type: ignore[arg-type]
    )

    assert decision.is_experiment
    assert decision.experiments
    treatment = decision.experiments[-1].treatment or {}
    assert set(treatment).issubset({"Z", "W"})


def test_active_deciders_respect_string_category_and_numeric_interval_domains() -> None:
    """Query-aware agents must return values from each variable's declared domain."""
    actions = AvailableActions(
        experiments=[
            ExperimentVariable(name="Z", domain=["0", "1", "follow-up"]),
            ExperimentVariable(name="W", domain=[-2.0, 3.0]),
        ]
    )
    active_deciders = [
        OptimalEffectDesignDecider(num_obs=0, num_inter=2, seed=13),
        ABCIDecider(num_obs=0, num_inter=2, seed=13),
        TrustYourGradientDecider(num_obs=0, num_inter=2, seed=13),
    ]

    for decider in active_deciders:
        decider.set_context(_agent_context())
        decision = decider.decide(
            RoundInfo(round=2), actions, belief=None  # type: ignore[arg-type]
        )
        assert decision.experiments, type(decider).__name__
        for experiment in decision.experiments:
            assert experiment.n > 0, type(decider).__name__
            if experiment.treatment is None:
                continue
            assert len(experiment.treatment) > 0, type(decider).__name__
            for variable, value in experiment.treatment.items():
                domain = next(item.domain for item in actions.experiments if item.name == variable)
                if all(isinstance(bound, str) for bound in domain):
                    assert isinstance(value, str), (type(decider).__name__, variable, value)
                    assert value in domain, (type(decider).__name__, variable, value)
                else:
                    low, high = domain
                    assert isinstance(value, (int, float)), (
                        type(decider).__name__,
                        variable,
                        value,
                    )
                    assert low <= value <= high, (type(decider).__name__, variable, value)


def test_query_aware_deciders_observe_when_controls_are_unavailable() -> None:
    """Without controllable variables, these agents can only request observations."""
    actions = AvailableActions(experiments=[])
    for decider in (
        OptimalEffectDesignDecider(num_obs=2, num_inter=3),
        ABCIDecider(num_obs=2, num_inter=3),
        TrustYourGradientDecider(num_obs=2, num_inter=3),
    ):
        decision = decider.decide(
            RoundInfo(round=1), actions, belief=None  # type: ignore[arg-type]
        )
        assert [experiment.treatment for experiment in decision.experiments] == [None]
        assert decision.experiments[0].n == OBSERVATION_BATCH_SIZE


def _query_aware_factories():  # noqa: ANN202
    return (
        lambda: OptimalEffectDesignDecider(num_obs=0, num_inter=2, seed=29),
        lambda: ABCIDecider(num_obs=0, num_inter=2, seed=29),
        lambda: TrustYourGradientDecider(num_obs=0, num_inter=2, seed=29),
    )


def _assert_query_aware_domain(decider: Decider, decision: Decision) -> None:
    available = _available_actions()
    domains = {variable.name: variable.domain for variable in available.experiments}
    assert decision.experiments
    for experiment in decision.experiments:
        assert experiment.n > 0
        for name, value in (experiment.treatment or {}).items():
            domain = domains[name]
            if all(isinstance(item, str) for item in domain):
                assert isinstance(value, str) and value in domain, type(decider).__name__
            else:
                assert domain[0] <= value <= domain[1], type(decider).__name__


def test_query_aware_deciders_repeat_seeded_actions_and_handle_constant_data() -> None:
    """Degenerate evidence must not destabilize actions or their declared domains."""
    actions = _available_actions()
    for factory in _query_aware_factories():
        first, replay = factory(), factory()
        first.set_context(_agent_context())
        replay.set_context(_agent_context())
        round_info = RoundInfo(round=2, budget_snapshot=BudgetSnapshot(samples_left=100))
        first_action = first.decide(round_info, actions, None)  # type: ignore[arg-type]
        replay_action = replay.decide(round_info, actions, None)  # type: ignore[arg-type]
        _assert_query_aware_domain(first, first_action)
        assert first_action.to_dict() == replay_action.to_dict()

        constant = pd.DataFrame({"Z": [0] * 24, "X": [1.5] * 24, "Y": [2.0] * 24})
        observation = RoundObservation(
            round_info=RoundInfo(round=1),
            decision=Decision.experiment().add_experiment(None, n=len(constant)),
            samples=SamplesCollection(
                [Samples(kind="observational", n=len(constant), data=constant)]
            ),
        )
        first.update(observation)
        after_constant = first.decide(round_info, actions, None)  # type: ignore[arg-type]
        _assert_query_aware_domain(first, after_constant)


def test_query_aware_proposals_cannot_bypass_environment_sample_preflight() -> None:
    """The environment's sample preflight rejects an oversized query atomically."""
    for factory in _query_aware_factories():
        decider = factory()
        decider.set_context(_agent_context())
        decision = decider.decide(
            RoundInfo(round=1, budget_snapshot=BudgetSnapshot(samples_left=1)),
            _available_actions(),
            None,  # type: ignore[arg-type]
        )
        budget = BudgetManager(BudgetSpec(rounds=2, samples=1))
        with pytest.raises(SamplesBudgetExceededError):
            budget.preflight(decision.experiments)
        assert budget.samples_used == 0
