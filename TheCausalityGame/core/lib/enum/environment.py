from enum import Enum


class ActionKind(str, Enum):
    EXPERIMENT = "experiment"
    ANSWER = "answer"
