from enum import Enum


class RunMode(str, Enum):
    """How to schedule multiple agents."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
