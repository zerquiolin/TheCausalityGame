"""The Causality Game - Mission Errors."""


class NotMountedError(Exception):
    """Exception raised when a mission is not mounted but an operation requires it."""

    def __init__(self) -> None:
        super().__init__("Mission is not mounted.")
