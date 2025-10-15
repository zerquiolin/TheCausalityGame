"""Lazy accessors for commonly used spec models.

Keeping imports lazy helps avoid circular import issues during package
initialization and reduces startup overhead for modules that only need a few
specs.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "AgentSpec",
    "BudgetSpec",
    "DAGSpec",
    "HookEvent",
    "HookSpec",
    "MetricSpec",
    "MissionSpec",
    "NoiseDistributionSpec",
    "PlotSpec",
    "ProblemInstanceSpec",
    "ResultValidatorSpec",
    "RunPlanSpec",
    "RuntimeSettingsSpec",
    "SCMNodeSpec",
    "SCMSpec",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "AgentSpec": ("TheCausalityGame.core.specs.agent", "AgentSpec"),
    "BudgetSpec": ("TheCausalityGame.core.specs.budget", "BudgetSpec"),
    "DAGSpec": ("TheCausalityGame.core.specs.dag", "DAGSpec"),
    "HookEvent": ("TheCausalityGame.core.specs.hook", "HookEvent"),
    "HookSpec": ("TheCausalityGame.core.specs.hook", "HookSpec"),
    "MetricSpec": ("TheCausalityGame.core.specs.metric", "MetricSpec"),
    "MissionSpec": ("TheCausalityGame.core.specs.mission", "MissionSpec"),
    "NoiseDistributionSpec": (
        "TheCausalityGame.core.specs.noise",
        "NoiseDistributionSpec",
    ),
    "PlotSpec": ("TheCausalityGame.core.specs.plot", "PlotSpec"),
    "ProblemInstanceSpec": (
        "TheCausalityGame.core.specs.problem_instance",
        "ProblemInstanceSpec",
    ),
    "ResultValidatorSpec": (
        "TheCausalityGame.core.specs.result_validator",
        "ResultValidatorSpec",
    ),
    "RunPlanSpec": ("TheCausalityGame.core.specs.run", "RunPlanSpec"),
    "RuntimeSettingsSpec": (
        "TheCausalityGame.core.specs.settings",
        "RuntimeSettingsSpec",
    ),
    "SCMNodeSpec": ("TheCausalityGame.core.specs.scm_node", "SCMNodeSpec"),
    "SCMSpec": ("TheCausalityGame.core.specs.scm", "SCMSpec"),
}


def __getattr__(name: str) -> Any:
    try:
        module_path, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_path)
    attr = getattr(module, attr_name)
    globals()[name] = attr
    return attr


def __dir__() -> list[str]:
    return sorted(__all__)
