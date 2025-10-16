from enum import Enum


class NodeAccessibility(str, Enum):
    LATENT = "latent"
    MEASURABLE = "measurable"
    CONTROLLABLE = "controllable"
