from __future__ import annotations

from enum import Enum


class HookEvent(str, Enum):
    RUN_START = "run_start"
    RUN_END = "run_end"
    ROUND_START = "round_start"
    ROUND_END = "round_end"
    BEFORE_ACT = "before_act"
    AFTER_ACT = "after_act"
    SUBMIT_FINAL = "submit_final"


class StepKind(str, Enum):
    STATUS = "status"
    DATASET_BATCH = "dataset_batch"
    FEEDBACK = "feedback"
    ACTION_EXPERIMENT = "experiment"
    ACTION_SUBMIT_FINAL = "submit_final"
    ACTION_UNKNOWN = "unknown"
