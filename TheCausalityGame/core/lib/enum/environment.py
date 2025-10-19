"""The Causality Game - Environment Enums."""

from enum import Enum


class ActionKind(str, Enum):
    """Type of actions an agent can take during the game."""

    EXPERIMENT = "experiment"
    """Perform an experiment."""

    ANSWER = "answer"
    """Provide a final answer to the mission."""


class SamplesKind(str, Enum):
    """Type of samples generated in the environment."""

    OBSERVATIONAL = "observational"
    """Samples generated without interventions."""

    INTERVENTIONAL = "interventional"
    """Samples generated with interventions."""
