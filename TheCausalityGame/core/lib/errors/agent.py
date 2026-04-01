"""The Causality Game - Agent Errors."""


class AgentContextNotSetError(RuntimeError):
    """Raised when accessing agent context before initialization."""

    def __init__(self) -> None:
        super().__init__("Agent context not set.")


class AgentLoggerNotSetError(RuntimeError):
    """Raised when accessing agent logger before initialization."""

    def __init__(self) -> None:
        super().__init__("Agent logger not set.")


class AgentPendingObservationMissingError(RuntimeError):
    """Raised when an agent receives samples before acting."""

    def __init__(self) -> None:
        super().__init__("Agent is missing pending round metadata required to build an observation.")


class IncompatibleAgentCompositionError(RuntimeError):
    """Raised when a decider requires inferer capabilities that are not available."""

    def __init__(self, missing_capabilities: set[str]) -> None:
        missing = ", ".join(sorted(missing_capabilities)) or "<unknown>"
        super().__init__(f"Incompatible inferer/decider composition. Missing capabilities: {missing}.")
