"""The Causality Game - Error Definitions."""


class ConfigurationError(Exception):
    """Configuration-related error."""

    def __init__(self, message: str = "Invalid configuration.") -> None:
        """Initialize the error."""
        super().__init__(message)


class LoadError(Exception):
    """Dynamic import or plugin loading error."""

    def __init__(self, message: str = "Failed to load component.") -> None:
        """Initialize the error."""
        super().__init__(message)
