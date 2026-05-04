"""Symbolic SCM result validator."""

from __future__ import annotations

from typing import Any, override

import sympy as sp

from TheCausalityGame.agent.inferers.symbolic_scm import EstimatedSymbolicSCM
from TheCausalityGame.core.contracts.result_validator import ResultValidator
from TheCausalityGame.core.contracts.specs.result_validator import ResultValidatorSpec
from TheCausalityGame.core.infrastructure.registry import get_class_path
from TheCausalityGame.core.lib.errors.result_validator import InvalidResultTypeError


class SymbolicSCMResultValidator(ResultValidator):
    """Validator for symbolic SCM discovery outputs."""

    _kind = "SymbolicSCM"

    @override
    def validate(self, result: Any) -> EstimatedSymbolicSCM:
        if not isinstance(result, EstimatedSymbolicSCM):
            raise InvalidResultTypeError(
                type=str(type(result)),
                expected=str(EstimatedSymbolicSCM),
            )

        for mechanism in result.mechanisms.values():
            try:
                sp.sympify(mechanism.expression)
            except Exception as error:  # noqa: BLE001
                raise InvalidResultTypeError(
                    type=str(type(mechanism.expression)),
                    expected="SymPy-compatible expression",
                ) from error

        return result

    @override
    def to_spec(self) -> ResultValidatorSpec:
        return ResultValidatorSpec(
            class_=get_class_path(self.__class__),
        )

    @classmethod
    @override
    def from_spec(cls, spec: ResultValidatorSpec) -> SymbolicSCMResultValidator:
        return cls()
