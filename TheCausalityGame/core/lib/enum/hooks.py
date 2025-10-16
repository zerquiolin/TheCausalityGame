from __future__ import annotations

from enum import Enum


class HookEvent(str, Enum):
    GAME_START = "run_start"
    GAME_END = "run_end"

    TRANSCRIPTION_START = "transcription_start"
    BUDGET_SNAPSHOT = "budget_snapshot"

    ROUND_START = "round_start"
    ROUND_END = "round_end"

    BEFORE_ACT = "before_act"
    AFTER_ACT = "after_act"

    BEFORE_EVAL = "before_eval"
    AFTER_EVAL = "after_eval"

    BEFORE_INFORM = "before_inform"
    AFTER_INFORM = "after_inform"
