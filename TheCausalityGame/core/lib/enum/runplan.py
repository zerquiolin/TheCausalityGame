from enum import Enum


class RunPlanExecution(str, Enum):
    """How to execute multiple agents."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"


class RunPlanParallelBackEnd(str, Enum):
    """Parallel backend for parallel execution."""

    THREAD = "thread"
    PROCESS = "process"
