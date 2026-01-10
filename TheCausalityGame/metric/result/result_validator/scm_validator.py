"""The Causality Game - SCM Result Validator."""

from __future__ import annotations

from typing import Any, override

from TheCausalityGame.agent.strategies.scm_strategy import EstimatedANMSCM
from TheCausalityGame.core.contracts.result_validator import ResultValidator
from TheCausalityGame.core.contracts.specs.result_validator import ResultValidatorSpec
from TheCausalityGame.core.infrastructure.registry import get_class_path
from TheCausalityGame.core.lib.errors.result_validator import InvalidResultTypeError


class SCMResultValidator(ResultValidator):
    """
    Validator for SCM estimation functions.

    This result validator ensures the agent returns a valid Structural Causal Model (SCM)
    represented by a 'EstimatedANMSCM' object.

    Attributes
    ----------
    _kind : str
        The kind of result this validator supports.
    """

    _kind = "SCM"

    @override
    def validate(self, result: Any) -> Any:
        if not isinstance(result, EstimatedANMSCM):
            raise InvalidResultTypeError(
                type=str(type(result)),  # type: ignore
                expected=str(EstimatedANMSCM),
            )

        return result

    @override
    def to_spec(self) -> ResultValidatorSpec:
        return ResultValidatorSpec(
            class_=get_class_path(self.__class__),
        )

    @classmethod
    @override
    def from_spec(cls, spec: ResultValidatorSpec) -> SCMResultValidator:
        return cls()
