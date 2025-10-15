from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "DAG",
    "SCM",
    "Agent",
    "AgentContext",
    "BehaviorMetric",
    "Hook",
    "HookEvent",
    "Metric",
    "Mission",
    "NoiseDistribution",
    "ProblemInstance",
    "ResultMetric",
    "ResultValidator",
    "SCMNode",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "Agent": ("TheCausalityGame.core.contracts.agent", "Agent"),
    "AgentContext": ("TheCausalityGame.core.contracts.agent", "AgentContext"),
    "BehaviorMetric": ("TheCausalityGame.core.contracts.metric", "BehaviorMetric"),
    "DAG": ("TheCausalityGame.core.contracts.dag", "DAG"),
    "Hook": ("TheCausalityGame.core.contracts.hooks", "Hook"),
    "HookEvent": ("TheCausalityGame.core.contracts.hooks", "HookEvent"),
    "Metric": ("TheCausalityGame.core.contracts.metric", "Metric"),
    "Mission": ("TheCausalityGame.core.contracts.mission", "Mission"),
    "NoiseDistribution": (
        "TheCausalityGame.core.contracts.noise",
        "NoiseDistribution",
    ),
    "ProblemInstance": (
        "TheCausalityGame.core.contracts.problem_instance",
        "ProblemInstance",
    ),
    "ResultMetric": ("TheCausalityGame.core.contracts.metric", "ResultMetric"),
    "ResultValidator": (
        "TheCausalityGame.core.contracts.result_validator",
        "ResultValidator",
    ),
    "SCM": ("TheCausalityGame.core.contracts.scm", "SCM"),
    "SCMNode": ("TheCausalityGame.core.contracts.scm_node", "SCMNode"),
}


def __getattr__(name: str) -> Any:
    try:
        module_path, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_path)
    attr = getattr(module, attr_name)
    globals()[name] = attr  # cache for future lookups
    return attr


def __dir__() -> list[str]:
    return sorted(__all__)
