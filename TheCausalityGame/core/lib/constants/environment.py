from __future__ import annotations

from enum import Enum


class PossibleActions(str, Enum):
    EXPERIMENT = "experiment"
    SUBMIT = "submit"
