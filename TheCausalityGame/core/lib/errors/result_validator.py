"""The Causality Game - Result Validator Errors."""


class InvalidResultTypeError(Exception):
    """Exception raised when the result type is invalid."""

    def __init__(self, type: str, expected: str) -> None:
        super().__init__(f"Invalid result type: {type}. Expected: {expected}.")


class InvalidNumberOfArgumentsError(Exception):
    """Exception raised when the result function has invalid number of arguments."""

    def __init__(self, arguments: int, expected: int) -> None:
        super().__init__(
            f"Invalid number of arguments: {arguments}. Expected: {expected}."
        )


class InvalidResultArgumentsError(Exception):
    """Exception raised when the result function has invalid arguments."""

    def __init__(self, arguments: list[str], expected: list[str]) -> None:
        super().__init__(
            f"Invalid result function arguments: {arguments}. Expected: {expected}."
        )
