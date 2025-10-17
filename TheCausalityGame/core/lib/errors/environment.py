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
