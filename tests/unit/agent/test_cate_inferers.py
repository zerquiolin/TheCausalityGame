"""Tests for CATE inferer implementations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from TheCausalityGame.agent.inferers.causal_tree import HonestCausalTreeInferer
from TheCausalityGame.agent.inferers.outcome_regression import OutcomeRegressionInferer
from TheCausalityGame.agent.inferers.transformed_outcome import TransformedOutcomeInferer
from TheCausalityGame.core.contracts.dto.agent import RoundObservation
from TheCausalityGame.core.contracts.dto.environment import RoundInfo, Samples, SamplesCollection
from TheCausalityGame.core.infrastructure.decisions import Decision
from TheCausalityGame.core.infrastructure.registry import build_from_spec
from TheCausalityGame.metric.result.result_validator.cate_function_validator import (
    ConditionalAverageTreatmentEffectFunctionValidator,
)

TWO_ROWS = 2
REGION_THRESHOLD = 0.5
LOW_EFFECT_UPPER_BOUND = 2.0
HIGH_EFFECT_LOWER_BOUND = 3.0


def _observation(
    data: pd.DataFrame,
    interventions: dict[str, Any] | None = None,
    kind: str = "observational",
) -> RoundObservation:
    """Build a single-round observation for inferer tests."""
    sample = Samples(
        kind=kind,
        n=len(data),
        data=data,
        interventions=interventions,
    )
    return RoundObservation(
        round_info=RoundInfo(round=1),
        decision=Decision.experiment().add_experiment(treatment=interventions, n=len(data)),
        samples=SamplesCollection([sample]),
    )


def _query(values: np.ndarray) -> tuple[list[str], str, str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Build a treatment-effect query over the synthetic covariate values."""
    control = pd.DataFrame({"T": np.zeros(len(values)), "Z": values})
    treated = pd.DataFrame({"T": np.ones(len(values)), "Z": values})
    return ["Z"], "T", "Y", (control, treated)


def _linear_effect_data(n: int = 80) -> pd.DataFrame:
    """Create data with a linear heterogeneous treatment effect."""
    z = np.linspace(0.0, 1.0, n)
    t = np.tile([0.0, 1.0], n // 2)
    y = 1.0 + z + t * (2.0 + z)
    return pd.DataFrame({"T": t, "Z": z, "Y": y})


def _interventional_effect_data(value: float, n: int = 40) -> pd.DataFrame:
    """Create one controlled treatment arm for transformed-outcome tests."""
    z = np.linspace(0.0, 1.0, n)
    y = 1.0 + z + value * (2.0 + z)
    return pd.DataFrame({"T": np.full(n, value), "Z": z, "Y": y})


@pytest.mark.parametrize(
    "inferer",
    [
        OutcomeRegressionInferer(),
        TransformedOutcomeInferer(),
        HonestCausalTreeInferer(min_leaf_size=4, min_treatment_count=2),
    ],
)
def test_cate_inferer_empty_answer_shape_and_validation(inferer: Any) -> None:  # noqa: ANN401
    """Inferers return validator-compatible zero effects before seeing data."""
    answer = inferer.answer()
    validated = ConditionalAverageTreatmentEffectFunctionValidator().validate(answer)
    X, treatment, outcome, covariate_values = _query(np.array([0.0, 1.0]))  # noqa: N806
    result = validated(X, treatment, outcome, covariate_values)

    assert list(result.columns) == ["treatment_effect"]
    assert len(result) == TWO_ROWS
    assert np.allclose(result["treatment_effect"], 0.0)


@pytest.mark.parametrize(
    "inferer",
    [
        OutcomeRegressionInferer(),
        TransformedOutcomeInferer(),
        HonestCausalTreeInferer(min_leaf_size=4, min_treatment_count=2),
    ],
)
def test_cate_inferer_spec_roundtrip(inferer: Any) -> None:  # noqa: ANN401
    """Inferer specs rebuild to equivalent inferers."""
    rebuilt = build_from_spec(inferer.to_spec())

    assert rebuilt.to_dict() == inferer.to_dict()


def test_outcome_regression_recovers_linear_heterogeneous_effect() -> None:
    """Outcome regression recovers a simple linear CATE surface."""
    inferer = OutcomeRegressionInferer(alpha=1e-8)
    inferer.update(_observation(_linear_effect_data()))

    result = inferer.answer()(*_query(np.array([0.0, 0.5, 1.0])))

    assert np.allclose(result["treatment_effect"], [2.0, 2.5, 3.0], atol=0.08)


def test_transformed_outcome_prefers_interventional_data() -> None:
    """Transformed outcome uses controlled batches when both arms are present."""
    inferer = TransformedOutcomeInferer(alpha=1e-8, min_group_size=2)
    inferer.update(
        _observation(
            _interventional_effect_data(0.0),
            interventions={"T": 0.0},
            kind="interventional",
        )
    )
    inferer.update(
        _observation(
            _interventional_effect_data(1.0),
            interventions={"T": 1.0},
            kind="interventional",
        )
    )

    result = inferer.answer()(*_query(np.array([0.0, 0.5, 1.0])))

    assert np.allclose(result["treatment_effect"], [2.0, 2.5, 3.0], atol=0.15)


def test_transformed_outcome_observational_propensity_fallback() -> None:
    """Transformed outcome falls back to observational propensity estimates."""
    inferer = TransformedOutcomeInferer(alpha=1e-8, propensity_alpha=1e-8, min_group_size=2)
    inferer.update(_observation(_linear_effect_data()))

    result = inferer.answer()(*_query(np.array([0.0, 0.5, 1.0])))

    assert np.allclose(result["treatment_effect"], [2.0, 2.5, 3.0], atol=0.3)


def test_honest_causal_tree_learns_two_effect_regions() -> None:
    """Honest causal trees recover separate low and high effect regions."""
    z = np.tile(np.linspace(0.0, 1.0, 40), 2)
    t = np.repeat([0.0, 1.0], 40)
    effect = np.where(z <= REGION_THRESHOLD, 1.0, 4.0)
    y = z + t * effect
    data = pd.DataFrame({"T": t, "Z": z, "Y": y})
    inferer = HonestCausalTreeInferer(
        max_depth=2,
        min_leaf_size=8,
        min_treatment_count=3,
        random_state=911,
    )
    inferer.update(_observation(data))

    result = inferer.answer()(*_query(np.array([0.2, 0.8])))

    assert result["treatment_effect"].iloc[0] < LOW_EFFECT_UPPER_BOUND
    assert result["treatment_effect"].iloc[1] > HIGH_EFFECT_LOWER_BOUND
