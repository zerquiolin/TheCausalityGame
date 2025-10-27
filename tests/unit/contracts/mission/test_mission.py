"""The Causality Game - Mission Tests."""

from typing import Any

import pytest

from TheCausalityGame.core.contracts.mission import Mission
from TheCausalityGame.core.contracts.specs.mission import MissionSpec
from TheCausalityGame.core.infrastructure.registry import (
    build_from_spec,
    load_subclasses_from_path,
)
from TheCausalityGame.core.lib.utils.tests import (
    assert_dicts_equal,
)
from TheCausalityGame.metric.behavior.rounds import RoundsBehaviorMetric
from TheCausalityGame.metric.result.pehe import PEHEResultMetric
from TheCausalityGame.metric.result.result_validator.cate_function_validator import (
    ConditionalAverageTreatmentEffectFunctionValidator,
)

# Search Classes
base_path = "TheCausalityGame/mission"
classes = load_subclasses_from_path(Mission, base_path)


# Tests
@pytest.fixture(params=classes, scope="module")
def mission_instance(request: Any) -> None:  # noqa: ANN401
    """Test class construction."""
    # Metrics
    behavior = RoundsBehaviorMetric()
    result = PEHEResultMetric()

    # Result Validator
    validator = ConditionalAverageTreatmentEffectFunctionValidator()

    cls = request.param
    return cls(
        behavior_metric=behavior,
        result_metric=result,
        result_validator=validator,
    )


def test_mission_serialization_roundtrip(
    mission_instance: Any,  # noqa: ANN401
) -> None:
    """Test serialization roundtrip."""
    spec = mission_instance.to_spec()
    assert isinstance(spec, MissionSpec)
    mission2 = build_from_spec(spec)
    assert_dicts_equal(mission_instance.to_dict(), mission2.to_dict())
