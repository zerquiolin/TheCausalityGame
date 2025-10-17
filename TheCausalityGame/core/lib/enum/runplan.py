"""The Causality Game - RunPlan Enums."""

from enum import Enum


class RunPlanExecution(str, Enum):
    """Defines the strategy for executing multiple agents.

    - SEQUENTIAL: Agents are run one after the other.
    - PARALLEL: Agents are run concurrently using threads or processes.
    """

    SEQUENTIAL = "sequential"
    """Run agents one at a time."""

    PARALLEL = "parallel"
    """Run agents concurrently."""


class RunPlanParallelBackEnd(str, Enum):
    """Backend type for parallel execution mode.

    - THREAD: Use multithreading (recommended for I/O-bound workloads).
    - PROCESS: Use multiprocessing (recommended for CPU-bound workloads).
    """

    THREAD = "thread"
    """Thread-based parallelism (I/O-bound)."""

    PROCESS = "process"
    """Process-based parallelism (CPU-bound)."""
