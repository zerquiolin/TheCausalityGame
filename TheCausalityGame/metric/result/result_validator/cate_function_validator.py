"""The Causality Game - CATE Function Result Validator."""

from typing import Any, override

from TheCausalityGame.core.contracts.result_validator import ResultValidator
from TheCausalityGame.core.contracts.specs.result_validator import ResultValidatorSpec
from TheCausalityGame.core.infrastructure.registry import get_class_path
from TheCausalityGame.core.lib.errors.result_validator import (
    InvalidNumberOfArgumentsError,
    InvalidResultArgumentsError,
    InvalidResultTypeError,
)


class ConditionalAverageTreatmentEffectFunctionValidator(ResultValidator):
    """
    Validator for treatment effect estimation functions.

    This result validator ensures the agent returns a callable function that
    adheres to a specific signature for estimating Conditional Average Treatment Effects (CATE).

    Expected signature:
        def func(X: list[str], treatment: str, outcome: str, covariate_values: tuple[DataFrame, DataFrame]) -> DataFrame

    Attributes
    ----------
    _kind : str
        The kind of result this validator supports.
    """  # noqa: E501

    _kind = "Treatment Effect Function"

    @override
    def validate(self, result: Any) -> Any:
        # Ensure the result is a callable (function or similar)
        if not callable(result):
            raise InvalidResultTypeError(
                type=type(result).__name__, expected="function"
            )

        # Ensure the function has exactly 4 arguments
        if result.__code__.co_argcount != 4:  # noqa: PLR2004
            raise InvalidNumberOfArgumentsError(
                arguments=result.__code__.co_argcount, expected=4
            )

        # Check the names of the arguments
        expected_args = ("X", "treatment", "outcome", "covariate_values")
        actual_args = result.__code__.co_varnames[:4]

        if actual_args != expected_args:
            raise InvalidResultArgumentsError(
                arguments=list(actual_args),
                expected=list(expected_args),
            )

        return result

    @override
    def to_spec(self) -> ResultValidatorSpec:
        return ResultValidatorSpec(
            class_=get_class_path(self.__class__),
        )

    @classmethod
    @override
    def from_spec(
        cls, spec: ResultValidatorSpec
    ) -> "ConditionalAverageTreatmentEffectFunctionValidator":
        return cls()
