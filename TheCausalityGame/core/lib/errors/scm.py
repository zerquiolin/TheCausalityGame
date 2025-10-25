"""The Causality Game - Structural Causal Model (SCM) Errors."""


class NoNumericLeafNodeError(Exception):
    """Raised when no numeric leaf node is available in the SCM for metrics requiring one."""

    def __init__(self) -> None:
        """Initialize the error."""
        super().__init__("No numeric leaf node available in the SCM.")


class NoControllableNodeError(Exception):
    """Raised when no controllable node is available in the SCM for metrics requiring one."""

    def __init__(self) -> None:
        """Initialize the error."""
        super().__init__("No controllable node available in the SCM.")
