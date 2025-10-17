"""The Causality Game - Plot Enums."""

from enum import Enum


class PlotKind(str, Enum):
    """Types of plot generation hooks during game execution."""

    GAME_END = "run_end"
    """Triggered at the end of a full game run."""

    ROUND_END = "round_end"
    """Triggered at the end of each round."""

    BENCHMARK_END = "benchmark_end"
    """Triggered after all benchmark evaluations have completed."""
