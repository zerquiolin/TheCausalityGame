from __future__ import annotations

from enum import Enum


class HookEvent(str, Enum):
    RUN_START = "run_start"
    RUN_END = "run_end"
    ROUND_START = "round_start"
    ROUND_END = "round_end"
    BEFORE_ACT = "before_act"
    AFTER_ACT = "after_act"
    BEFORE_EVAL = "before_eval"
    AFTER_EVAL = "after_eval"
    BEFORE_INFORM = "before_inform"
    AFTER_INFORM = "after_inform"
    NEW_SNAPSHOT = "new_snapshot"
