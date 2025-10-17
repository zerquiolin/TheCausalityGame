"""The Causality Game - Enum for Agent Scheduling Mode."""

from enum import Enum


class RunMode(str, Enum):
    """Defines how multiple agents are scheduled for execution."""

    SEQUENTIAL = "sequential"
    """Run agents one at a time, in sequence."""

    PARALLEL = "parallel"
    """Run agents concurrently using threads or processes."""
