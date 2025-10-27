"""The Causality Game - Concrete CATE function Result Validator."""

import pytest

from TheCausalityGame.core.infrastructure.registry import build_from_spec
from TheCausalityGame.core.lib.errors.result_validator import (
    InvalidNumberOfArgumentsError,
    InvalidResultArgumentsError,
    InvalidResultTypeError,
)
from TheCausalityGame.metric.result.result_validator.cate_function_validator import (
    ConditionalAverageTreatmentEffectFunctionValidator,
)


def _valid_fn(X, treatment, outcome, covariate_values) -> int:  # type: ignore  # noqa: ANN001, ARG001, N803
    """Test function: valid signature for CATE function."""
    return 1


def _wrong_arity_fn(X, treatment, outcome) -> int:  # type: ignore  # noqa: ANN001, ARG001, N803
    """Test function: wrong arity for CATE function."""
    return 0


def _wrong_names_fn(features, treatment, outcome, covariate_values) -> int:  # type: ignore  # noqa: ANN001, ARG001
    """Test function: wrong argument names for CATE function."""
    return 0


def test_validate_requires_callable() -> None:
    """Test that validate method requires a callable input."""
    validator = ConditionalAverageTreatmentEffectFunctionValidator()

    with pytest.raises(InvalidResultTypeError):
        validator.validate(123)


def test_validate_enforces_argument_count() -> None:
    """Test that validate method enforces correct number of arguments."""
    validator = ConditionalAverageTreatmentEffectFunctionValidator()

    with pytest.raises(InvalidNumberOfArgumentsError):
        validator.validate(_wrong_arity_fn)


def test_validate_enforces_argument_names() -> None:
    """Test that validate method enforces correct argument names."""
    validator = ConditionalAverageTreatmentEffectFunctionValidator()

    with pytest.raises(InvalidResultArgumentsError):
        validator.validate(_wrong_names_fn)


def test_validator_serialization_roundtrip_preserves_behavior() -> None:
    """Test that serialization and deserialization of the validator preserves its behavior."""
    validator = ConditionalAverageTreatmentEffectFunctionValidator()

    spec_dict = validator.to_dict()
    assert spec_dict["class_"].endswith(
        "ConditionalAverageTreatmentEffectFunctionValidator"
    )
    assert spec_dict["spec_"].endswith("ResultValidatorSpec")

    serialized = validator.to_json()
    rebuilt = build_from_spec(serialized)

    assert (
        serialized == rebuilt.to_json()
    ), "Serialized JSON does not match after rebuild."
    assert (
        validator.validate(_valid_fn) is _valid_fn
    ), "Original validator did not validate the function correctly."
    assert (
        rebuilt.validate(_valid_fn) is _valid_fn
    ), "Rebuilt validator did not validate the function correctly."
