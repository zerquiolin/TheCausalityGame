import numpy as np

from TheCausalityGame.core.contracts.result_validator import ResultValidator
from TheCausalityGame.core.contracts.specs.result_validator import ResultValidatorSpec
from TheCausalityGame.core.infrastructure.registry import get_class_path


class ListResultValidator(ResultValidator):

    def validate(self, result: any) -> bool:
        return isinstance(result, list)

    def normalize(self, result: list):
        return [float(x) ** 2 for x in result]

    def to_spec(self) -> ResultValidatorSpec:
        return ResultValidatorSpec(
            class_=get_class_path(self.__class__),
            params={},
        )

    @classmethod
    def from_spec(cls, spec: ResultValidatorSpec) -> "ListResultValidator":
        return ListResultValidator(**spec.params)
