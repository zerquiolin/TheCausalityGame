"""The Causality Game - Node Enums."""

from enum import Enum


class NodeAccessibility(str, Enum):
    """Accessibility levels for SCM nodes.

    Indicates how a variable can be observed or manipulated during gameplay.
    """

    LATENT = "latent"
    """The variable is hidden and cannot be observed or controlled."""

    MEASURABLE = "measurable"
    """The variable can be measured but not manipulated."""

    CONTROLLABLE = "controllable"
    """The variable can be directly manipulated by the agent."""
