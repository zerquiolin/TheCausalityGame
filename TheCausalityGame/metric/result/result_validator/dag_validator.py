"""The Causality Game - DAG Result Validator."""

from __future__ import annotations

from typing import Any, override

import networkx as nx

from TheCausalityGame.core.contracts.result_validator import ResultValidator
from TheCausalityGame.core.contracts.specs.result_validator import ResultValidatorSpec
from TheCausalityGame.core.infrastructure.registry import get_class_path
from TheCausalityGame.core.lib.errors.result_validator import InvalidResultTypeError


class DAGResultValidator(ResultValidator):
    """
    Validator for treatment effect estimation functions.

    This result validator ensures the agent returns a valid Directed Acyclic Graph (DAG)
    represented by a NetworkX DiGraph object.

    Attributes
    ----------
    _kind : str
        The kind of result this validator supports.
    """

    _kind = "DAG"

    @override
    def validate(self, result: Any) -> Any:
        if not isinstance(result, nx.DiGraph):
            raise InvalidResultTypeError(
                type=str(type(result)),  # type: ignore
                expected=str(nx.DiGraph),
            )

        return result  # type: ignore

    @override
    def to_spec(self) -> ResultValidatorSpec:
        return ResultValidatorSpec(
            class_=get_class_path(self.__class__),
        )

    @classmethod
    @override
    def from_spec(cls, spec: ResultValidatorSpec) -> DAGResultValidator:
        return cls()
