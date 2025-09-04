from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from TheCausalityGame.core.contracts.agent import BaseAgent
from TheCausalityGame.core.engine.evaluator import (
    MetricsEvaluator,
)
from TheCausalityGame.core.engine.evaluator import (
    MetricsSpec as MetricsBundle,
)
from TheCausalityGame.core.runtime.game_instance import (
    AgentSpecModel,
    ComponentSpec,
    GameInstance,
    MetricsSpecModel,
)
from TheCausalityGame.core.runtime.orchestrator import Orchestrator, RunMode


# TODO: This method already exists in the registry module.
def _import_class(path: str) -> type:
    """Import 'pkg.mod:Class' or 'pkg.mod.Class'."""
    if ":" in path:
        mod, cls = path.split(":", 1)
    elif "." in path:
        mod, cls = path.rsplit(".", 1)
    else:
        raise ValueError(f"Invalid class path: {path}")
    return getattr(importlib.import_module(mod), cls)


def _instantiate(spec: ComponentSpec) -> Any:
    """Instantiate a component from a ComponentSpec."""
    return _import_class(spec.cls)(**spec.params)


# TODO: This is also a requirement in the metric classes, should implement an abstract class like 'from_spec" or similar.
def _make_metrics(m: MetricsSpecModel | None) -> MetricsEvaluator | None:
    if not m:
        return None
    behavior = _instantiate(m.behavior)
    result = _instantiate(m.result)
    custom = tuple(_instantiate(x) for x in m.custom)
    return MetricsEvaluator(
        spec=MetricsBundle(behavior=behavior, result=result, custom=custom)
    )


# TODO: This method is not necessary, that is what the extend of the serialization class methosd are used for.
def _agent_factory_from_spec(
    s: AgentSpecModel,
) -> tuple[str, Callable[[dict[str, Any]], BaseAgent]]:
    cls = _import_class(s.component.cls)
    params = dict(s.component.params)

    def factory(_cfg: dict[str, Any]) -> BaseAgent:
        return cls(**params)  # type: ignore[return-value]

    return s.id, factory


@dataclass(slots=True)
class Game:
    """A thin façade over Orchestrator for running a GameInstance."""

    instance: GameInstance
    run_dir: Path

    def _make_orchestrator(self) -> Orchestrator:
        mi = self.instance

        def build_scm(_manifest: dict[str, Any]) -> Any:
            return _instantiate(mi.scm)

        def build_mission(_manifest: dict[str, Any]) -> Any:
            return _instantiate(mi.mission)

        def build_metrics(_manifest: dict[str, Any]) -> MetricsEvaluator | None:
            return _make_metrics(mi.metrics)

        return Orchestrator(
            run_dir=self.run_dir,
            build_scm=build_scm,
            build_mission=build_mission,
            build_metrics=build_metrics,
            max_parallel_workers=mi.run_plan.max_parallel_workers,
        )

    def run(self) -> dict[str, dict[str, Any]]:
        """Run all agents per the instance's RunPlan and return summaries per agent."""
        orch = self._make_orchestrator()
        agents = tuple(_agent_factory_from_spec(a) for a in self.instance.agents)
        run_mode = (
            RunMode(self.instance.run_plan.mode)
            if self.instance.run_plan.mode in RunMode._value2member_map_
            else RunMode.SEQUENTIAL
        )

        return orch.run_many(
            run_mode=run_mode,
            manifest={
                "id": self.instance.id,
                "base_seed": self.instance.base_seed,
            },  # light context
            agents=agents,
            rounds=self.instance.run_plan.rounds,
            time_limit_s=self.instance.run_plan.budgets.time_s,
            sample_limit=self.instance.run_plan.budgets.samples,
            memory_mb_limit=self.instance.run_plan.budgets.memory_mb,
            trusted=True,
            base_seed=self.instance.base_seed,
        )
