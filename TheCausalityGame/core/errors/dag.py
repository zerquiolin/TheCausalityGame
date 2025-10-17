"""The Causality Game - Directed Acyclic Graph (DAG) Errors."""


class DAGCycleError(RuntimeError):
    """Raised when a cycle is detected in a Directed Acyclic Graph (DAG)."""

    def __init__(self) -> None:
        super().__init__("The provided graph is not a Directed Acyclic Graph (DAG).")
