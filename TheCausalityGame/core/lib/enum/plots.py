from __future__ import annotations

from enum import Enum


class PlotKind(str, Enum):
    GAME_END = "run_end"
    ROUND_END = "round_end"
    BENCHMARK_END = "benchmark_end"
