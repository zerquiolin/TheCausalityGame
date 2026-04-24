"""Tests for query-aware active deciders."""

from __future__ import annotations

import pandas as pd

from TheCausalityGame.agent.deciders.abci import ABCIDecider
from TheCausalityGame.agent.deciders.optimal_effect_design import OptimalEffectDesignDecider
from TheCausalityGame.agent.deciders.trust_gradient import TrustYourGradientDecider
from TheCausalityGame.core.contracts.agent import AgentContext
from TheCausalityGame.core.contracts.dto.agent import RoundObservation
from TheCausalityGame.core.contracts.dto.environment import (
    AvailableActions,
    ExperimentVariable,
    RoundInfo,
    Samples,
    SamplesCollection,
)
from TheCausalityGame.core.infrastructure.decisions import Decision


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
