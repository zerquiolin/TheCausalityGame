"""Check price inheritance, linear weights and nested stopping arithmetic."""

import pytest

from TheCausalityGame.core.contracts.dto.transcript import Transcript, TranscriptEntry
from TheCausalityGame.core.contracts.specs.acquisition import (
    AcquisitionPricing,
    PriceRule,
    StopCondition,
)
from TheCausalityGame.core.contracts.specs.budget import BudgetSpec
from TheCausalityGame.core.infrastructure.registry import build_from_spec
from TheCausalityGame.metric.behavior.acquisition import (
    LinearBehaviorMetric,
    RoundCountBehaviorMetric,
)


def test_prices_inherit_and_specific_override():
    pricing = AcquisitionPricing(
        per_sample=1,
        rules=[
            PriceRule(treatment={"X": None}, per_sample=2),
            PriceRule(treatment={"X": 1}, per_sample=3),
        ],
    )
    assert pricing.quote(None, 4) == 4
    assert pricing.quote({"X": 0, "Y": 2}, 4) == 8
    assert pricing.quote({"X": 1, "Y": 2}, 4) == 12


def test_price_conflict_is_explicit():
    pricing = AcquisitionPricing(
        rules=[
            PriceRule(treatment={"X": None}, per_sample=2),
            PriceRule(treatment={"Y": None}, per_sample=3),
        ]
    )
    with pytest.raises(ValueError, match="disagree"):
        pricing.quote({"X": 0, "Y": 0}, 3)


def test_composite_stopping_boundary():
    stop = StopCondition(
        mode="all",
        conditions=[
            StopCondition(resource="samples", threshold=5),
            StopCondition(
                mode="any",
                conditions=[
                    StopCondition(resource="cost", threshold=10),
                    StopCondition(resource="rounds", threshold=3),
                ],
            ),
        ],
    )
    assert not stop.reached({"samples": 4, "cost": 20, "rounds": 3})
    assert stop.reached({"samples": 5, "cost": 10, "rounds": 1})
    assert stop.reached({"samples": 5, "cost": 0, "rounds": 3})
    assert StopCondition.model_validate(stop.model_dump()) == stop


def test_linear_weights_need_not_sum_to_one():
    metric = LinearBehaviorMetric(
        [RoundCountBehaviorMetric(), RoundCountBehaviorMetric()], [2, -0.5]
    )
    transcript = Transcript(
        agent_id="A",
        mission_id="M",
        manifest_id="P",
        budget=BudgetSpec(),
        entries=[TranscriptEntry(round=1), TranscriptEntry(round=2)],
    )
    assert metric.evaluate(transcript) == 3
    assert build_from_spec(metric.to_spec()).evaluate(transcript) == 3
