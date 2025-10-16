from TheCausalityGame.core.contracts.result_validator import ResultValidator
from TheCausalityGame.core.contracts.specs.result_validator import ResultValidatorSpec
from TheCausalityGame.core.infrastructure.registry import get_class_path


class TreatmentEffectFunctionValidator(ResultValidator):
    _kind = "Treatment Effect Function"

    def validate(self, result: any) -> any:
        # Check of result is a function
        if not callable(result):
            raise ValueError("Result is not a function")
        # Check the arguments of the function
        if result.__code__.co_argcount != 4:
            raise ValueError("Result function must have exactly three arguments")
        # Check the argument names of the function
        if result.__code__.co_varnames[:4] != (
            "X",
            "treatment",
            "outcome",
            "covariate_values",
        ):
            raise ValueError("Arguments don't match expected names")

        return result

    def to_spec(self) -> ResultValidatorSpec:
        return ResultValidatorSpec(
            class_=get_class_path(self.__class__),
            params={},
        )

    @classmethod
    def from_spec(cls, spec: ResultValidatorSpec) -> "TreatmentEffectFunctionValidator":
        return TreatmentEffectFunctionValidator(**spec.params)
