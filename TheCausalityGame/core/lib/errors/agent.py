"""The Causality Game - Agent Errors."""


class AgentContextNotSetError(RuntimeError):
    """Raised when accessing agent context before initialization."""

    def __init__(self) -> None:
        super().__init__("Agent context not set.")


class AgentLoggerNotSetError(RuntimeError):
    """Raised when accessing agent logger before initialization."""

    def __init__(self) -> None:
        super().__init__("Agent logger not set.")
