"""The Causality Game - Budget Manager Errors."""


class TimeBudgetExceededError(RuntimeError):
    """Raised when the time budget is exceeded."""

    def __init__(self, used: float, allowed: float) -> None:
        super().__init__(
            f"Time budget exceeded: {used:.2f}s used / {allowed:.2f}s allowed."
        )


class RoundsBudgetExceededError(RuntimeError):
    """Raised when the rounds budget is exceeded."""

    def __init__(self, used: int, allowed: int) -> None:
        super().__init__(f"Rounds budget exceeded: {used} used / {allowed} allowed.")


class SamplesBudgetExceededError(RuntimeError):
    """Raised when the samples budget is exceeded."""

    def __init__(self, used: int, allowed: int) -> None:
        super().__init__(f"Samples budget exceeded: {used} used / {allowed} allowed.")


class MemoryBudgetExceededError(RuntimeError):
    """Raised when the memory budget is exceeded."""

    def __init__(self, used_mb: float, allowed_mb: float) -> None:
        super().__init__(
            f"Memory budget exceeded: {used_mb:.2f}MB used / {allowed_mb:.2f}MB allowed."
        )
