"""The Causality Game - Result Validator Tests."""

from typing import Any

import pytest

from TheCausalityGame.core.contracts.result_validator import ResultValidator
from TheCausalityGame.core.contracts.specs.result_validator import ResultValidatorSpec
from TheCausalityGame.core.infrastructure.registry import (
    build_from_spec,
    load_subclasses_from_path,
)
from TheCausalityGame.core.lib.utils.tests import (
    assert_dicts_equal,
)

# Search Classes
base_path = "TheCausalityGame/metric/result/result_validator"
classes = load_subclasses_from_path(ResultValidator, base_path)


# Tests
@pytest.fixture(params=classes, scope="module")
def result_validator_instance(request: Any) -> None:  # noqa: ANN401
    """Test class construction."""
    cls = request.param
    # We assume a no-arg constructor (or default args)
    return cls()


def test_result_validator_serialization_roundtrip(
    result_validator_instance: Any,  # noqa: ANN401
) -> None:
    """Test serialization roundtrip."""
    spec = result_validator_instance.to_spec()
    assert isinstance(spec, ResultValidatorSpec)
    result_validator2 = build_from_spec(spec)
    assert_dicts_equal(result_validator_instance.to_dict(), result_validator2.to_dict())
