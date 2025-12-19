"""Agent variants available in The Causality Game package."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "ExhaustiveAgent",
    "RandomAgent",
    "ReinforcementLearningAgent",
    "IntelligentAgent",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_map = {
        "ExhaustiveAgent": "TheCausalityGame.agent.exhaustive",
        "RandomAgent": "TheCausalityGame.agent.random_agent",
        "ReinforcementLearningAgent": "TheCausalityGame.agent.reinforcement_learning_agent",
        "IntelligentAgent": "TheCausalityGame.agent.intelligent_agent",
    }

    module = import_module(module_map[name])
    return getattr(module, name)
