"""The Causality Game - Hook Enums."""

from enum import Enum


class HookEvent(str, Enum):
    """Canonical lifecycle events available for hook subscriptions."""

    GAME_START = "run_start"
    """Triggered at the beginning of the game run."""

    GAME_END = "run_end"
    """Triggered at the end of the game run."""

    TRANSCRIPTION_START = "transcription_start"
    """Triggered when a new transcript is created."""

    BUDGET_SNAPSHOT = "budget_snapshot"
    """Triggered when a budget snapshot is taken."""

    ROUND_START = "round_start"
    """Triggered at the start of a round."""

    ROUND_END = "round_end"
    """Triggered at the end of a round."""

    BEFORE_ACT = "before_act"
    """Triggered before the agent performs an action."""

    AFTER_ACT = "after_act"
    """Triggered after the agent performs an action."""

    BEFORE_EVAL = "before_eval"
    """Triggered before the agent's answer is evaluated."""

    AFTER_EVAL = "after_eval"
    """Triggered after the agent's answer is evaluated."""

    BEFORE_INFORM = "before_inform"
    """Triggered before the agent is informed of feedback."""

    AFTER_INFORM = "after_inform"
    """Triggered after the agent is informed of feedback."""
