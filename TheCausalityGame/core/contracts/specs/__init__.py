"""Lazy accessors for commonly used spec models.

Keeping imports lazy helps avoid circular import issues during package
initialization and reduces startup overhead for modules that only need a few
specs.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

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
    "AgentSpec": ("TheCausalityGame.core.contracts.specs.agent", "AgentSpec"),
    "BudgetSpec": ("TheCausalityGame.core.contracts.specs.budget", "BudgetSpec"),
    "DAGSpec": ("TheCausalityGame.core.contracts.specs.dag", "DAGSpec"),
    "HookEvent": ("TheCausalityGame.core.contracts.specs.hook", "HookEvent"),
    "HookSpec": ("TheCausalityGame.core.contracts.specs.hook", "HookSpec"),
    "MetricSpec": ("TheCausalityGame.core.contracts.specs.metric", "MetricSpec"),
    "MissionSpec": ("TheCausalityGame.core.contracts.specs.mission", "MissionSpec"),
    "NoiseDistributionSpec": (
        "TheCausalityGame.core.contracts.specs.noise",
        "NoiseDistributionSpec",
    ),
    "PlotSpec": ("TheCausalityGame.core.contracts.specs.plot", "PlotSpec"),
    "ProblemInstanceSpec": (
        "TheCausalityGame.core.contracts.specs.problem_instance",
        "ProblemInstanceSpec",
    ),
    "ResultValidatorSpec": (
        "TheCausalityGame.core.contracts.specs.result_validator",
        "ResultValidatorSpec",
    ),
    "RunPlanSpec": ("TheCausalityGame.core.contracts.specs.run", "RunPlanSpec"),
    "RuntimeSettingsSpec": (
        "TheCausalityGame.core.contracts.specs.settings",
        "RuntimeSettingsSpec",
    ),
    "SCMNodeSpec": ("TheCausalityGame.core.contracts.specs.scm_node", "SCMNodeSpec"),
    "SCMSpec": ("TheCausalityGame.core.contracts.specs.scm", "SCMSpec"),
}

if TYPE_CHECKING:
    from TheCausalityGame.core.contracts.specs.agent import AgentSpec as _AgentSpec
    from TheCausalityGame.core.contracts.specs.budget import BudgetSpec as _BudgetSpec
    from TheCausalityGame.core.contracts.specs.dag import DAGSpec as _DAGSpec
    from TheCausalityGame.core.contracts.specs.hook import (
        HookEvent as _HookEvent,
        HookSpec as _HookSpec,
    )
    from TheCausalityGame.core.contracts.specs.metric import (
        MetricSpec as _MetricSpec,
    )
    from TheCausalityGame.core.contracts.specs.mission import (
        MissionSpec as _MissionSpec,
    )
    from TheCausalityGame.core.contracts.specs.noise import (
        NoiseDistributionSpec as _NoiseDistributionSpec,
    )
    from TheCausalityGame.core.contracts.specs.plot import PlotSpec as _PlotSpec
    from TheCausalityGame.core.contracts.specs.problem_instance import (
        ProblemInstanceSpec as _ProblemInstanceSpec,
    )
    from TheCausalityGame.core.contracts.specs.result_validator import (
        ResultValidatorSpec as _ResultValidatorSpec,
    )
    from TheCausalityGame.core.contracts.specs.run import RunPlanSpec as _RunPlanSpec
    from TheCausalityGame.core.contracts.specs.scm import SCMSpec as _SCMSpec
    from TheCausalityGame.core.contracts.specs.scm_node import (
        SCMNodeSpec as _SCMNodeSpec,
    )
    from TheCausalityGame.core.contracts.specs.settings import (
        RuntimeSettingsSpec as _RuntimeSettingsSpec,
    )

    # Re-exported names for static analyzers.
    AgentSpec = _AgentSpec
    BudgetSpec = _BudgetSpec
    DAGSpec = _DAGSpec
    HookEvent = _HookEvent
    HookSpec = _HookSpec
    MetricSpec = _MetricSpec
    MissionSpec = _MissionSpec
    NoiseDistributionSpec = _NoiseDistributionSpec
    PlotSpec = _PlotSpec
    ProblemInstanceSpec = _ProblemInstanceSpec
    ResultValidatorSpec = _ResultValidatorSpec
    RunPlanSpec = _RunPlanSpec
    RuntimeSettingsSpec = _RuntimeSettingsSpec
    SCMNodeSpec = _SCMNodeSpec
    SCMSpec = _SCMSpec


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
