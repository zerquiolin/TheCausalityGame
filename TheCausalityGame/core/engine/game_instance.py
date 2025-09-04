from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True, slots=True)
class GameInstance:
    """Typed, validated manifest for a single benchmark session."""

    # TODO: This should either be contracts or specs.
    id: str
    base_seed: int
    scm: ComponentSpec
    mission: ComponentSpec
    metrics: Optional[MetricsSpecModel]
    agents: Tuple[AgentSpecModel, ...]
    run_plan: RunPlan

    @staticmethod
    def from_manifest(m: Dict[str, Any]) -> "GameInstance":
        """Build a GameInstance from a plain dict (manifest JSON)."""

        def as_comp(d: Dict[str, Any]) -> ComponentSpec:
            return ComponentSpec(cls=d["class"], params=d.get("params", {}) or {})

        def as_agent(d: Dict[str, Any]) -> AgentSpecModel:
            return AgentSpecModel(id=d["id"], component=as_comp(d))

        metrics = None
        if m.get("metrics"):
            metrics = MetricsSpecModel(
                behavior=as_comp(m["metrics"]["behavior"]),
                result=as_comp(m["metrics"]["result"]),
                custom=tuple(as_comp(x) for x in m["metrics"].get("custom", [])),
            )

        rp_raw = m.get("run_plan", {}) or {}
        budgets = BudgetPlan(
            time_s=rp_raw.get("budgets", {}).get("time_s"),
            samples=rp_raw.get("budgets", {}).get("samples"),
            memory_mb=rp_raw.get("budgets", {}).get("memory_mb"),
        )
        run_plan = RunPlan(
            mode=(rp_raw.get("mode") or "sequential").lower(),
            rounds=int(rp_raw.get("rounds", 1)),
            max_parallel_workers=int(rp_raw.get("max_parallel_workers", 0)),
            budgets=budgets,
        )

        return GameInstance(
            id=str(m.get("id", "manifest")),
            base_seed=int(m.get("base_seed", 0)),
            scm=as_comp(m["scm"]),
            mission=as_comp(m["mission"]),
            metrics=metrics,
            agents=tuple(as_agent(a) for a in (m.get("agents") or [])),
            run_plan=run_plan,
        )
