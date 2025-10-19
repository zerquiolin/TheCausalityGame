"""The Causality Game - Environment Errors."""


class InvalidActionError(Exception):
    """Invalid action submitted to a mission/environment."""

    def __init__(self) -> None:
        super().__init__("The action provided is invalid for the current environment.")


class BudgetExceededError(Exception):
    """Sample budget exceeded."""

    def __init__(self, message: str = "The budget has been exceeded.") -> None:
        super().__init__(message)


class DecisionMismatchError(Exception):
    """The decision does not match the environment's expectations."""

    def __init__(
        self,
        message: str = "The decision kind does not match the environment's expectations.",
    ) -> None:
        super().__init__(message)


class UnknownVariableError(Exception):
    """The variable is not recognized in the current SCM."""

    def __init__(
        self,
        variable_name: str,
    ) -> None:
        message = (
            f"The variable '{variable_name}' is not recognized in the current SCM."
        )
        super().__init__(message)


class NonControllableVariableError(Exception):
    """The variable is not controllable in the current SCM."""

    def __init__(
        self,
        variable_name: str,
    ) -> None:
        message = (
            f"The variable '{variable_name}' is not controllable in the current SCM."
        )
        super().__init__(message)


class ExperimentOutOfDomainError(Exception):
    """The experiment value is out of the variable's domain."""

    def __init__(
        self,
        variable_name: str,
        value: float,
        domain: list[float | str],
    ) -> None:
        message = f"The value '{value}' for variable '{variable_name}' is out of the domain {domain}."  # noqa: E501
        super().__init__(message)


class UnsupportedMetricTypeError(Exception):
    """The metric type is not supported in the current environment."""

    def __init__(
        self,
        metric_type: str,
    ) -> None:
        message = f"The metric type '{metric_type}' is not supported in the current environment."
        super().__init__(message)
