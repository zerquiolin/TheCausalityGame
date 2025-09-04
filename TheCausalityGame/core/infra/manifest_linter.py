from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Optional


CLASS_PAT = re.compile(r"^[A-Za-z_][\w\.]*(:|\.)[A-Za-z_]\w*$")


class LintLevel(str, Enum):
    INFO = "info"
    WARN = "warn"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class LintIssue:
    level: LintLevel
    code: str
    message: str
    path: str  # JSON path for the field, e.g. "metrics.result.class"


@dataclass(frozen=True, slots=True)
class LintPolicy:
    """Policy toggles for linting.

    Attributes:
        allowed_module_prefixes: if provided, component classes must originate from these module/package prefixes.
        require_metrics: require a metrics section.
        max_rounds: hard upper bound for rounds (0 disables).
        max_parallel_workers: upper bound for workers (0 disables).
        allow_parallel: whether 'parallel' mode is allowed at all.
        restricted_mode: if True, additional checks apply (e.g., forbid custom metrics if desired).
        forbid_custom_metrics: optionally forbid custom metrics list in restricted mode.
    """

    allowed_module_prefixes: tuple[str, ...] = ()
    require_metrics: bool = False
    max_rounds: int = 0
    max_parallel_workers: int = 0
    allow_parallel: bool = True
    restricted_mode: bool = False
    forbid_custom_metrics: bool = False


def _validate_component(
    spec: dict[str, Any], path: str, issues: list[LintIssue], policy: LintPolicy
) -> None:
    cls = spec.get("class")
    if not cls or not isinstance(cls, str):
        issues.append(
            LintIssue(
                LintLevel.ERROR,
                "missing-class",
                "Component missing 'class' string",
                path + ".class",
            )
        )
        return
    if not CLASS_PAT.match(cls):
        issues.append(
            LintIssue(
                LintLevel.ERROR,
                "class-format",
                f"Invalid class path '{cls}'",
                path + ".class",
            )
        )
    if policy.allowed_module_prefixes:
        mod = cls.split(":")[0] if ":" in cls else cls.rsplit(".", 1)[0]
        if not any(mod.startswith(pref) for pref in policy.allowed_module_prefixes):
            issues.append(
                LintIssue(
                    LintLevel.WARN,
                    "module-not-allowlisted",
                    f"Module '{mod}' not in allowlist",
                    path + ".class",
                )
            )
    params = spec.get("params", {})
    if params is not None and not isinstance(params, dict):
        issues.append(
            LintIssue(
                LintLevel.ERROR,
                "params-type",
                "'params' must be an object",
                path + ".params",
            )
        )


def lint_manifest(
    manifest: dict[str, Any], policy: Optional[LintPolicy] = None
) -> tuple[bool, list[LintIssue]]:
    """Return (ok, issues) for a manifest dict given a policy."""
    pol = policy or LintPolicy()
    issues: list[LintIssue] = []

    # Required sections
    if "scm" not in manifest:
        issues.append(
            LintIssue(LintLevel.ERROR, "missing-scm", "Manifest must have 'scm'", "scm")
        )
    if "mission" not in manifest:
        issues.append(
            LintIssue(
                LintLevel.ERROR,
                "missing-mission",
                "Manifest must have 'mission'",
                "mission",
            )
        )
    if (
        "agents" not in manifest
        or not isinstance(manifest.get("agents"), list)
        or not manifest.get("agents")
    ):
        issues.append(
            LintIssue(
                LintLevel.ERROR,
                "missing-agents",
                "Manifest must have non-empty 'agents' array",
                "agents",
            )
        )
    if pol.require_metrics and "metrics" not in manifest:
        issues.append(
            LintIssue(
                LintLevel.ERROR,
                "missing-metrics",
                "Metrics required by policy",
                "metrics",
            )
        )

    # Run plan
    rp = manifest.get("run_plan", {}) or {}
    rounds = rp.get("rounds")
    if not isinstance(rounds, int) or rounds <= 0:
        issues.append(
            LintIssue(
                LintLevel.ERROR,
                "rounds-invalid",
                "'run_plan.rounds' must be a positive integer",
                "run_plan.rounds",
            )
        )
    elif pol.max_rounds and rounds > pol.max_rounds:
        issues.append(
            LintIssue(
                LintLevel.WARN,
                "rounds-high",
                f"rounds {rounds} exceeds policy max {pol.max_rounds}",
                "run_plan.rounds",
            )
        )

    mode = (rp.get("mode") or "sequential").lower()
    if mode not in ("sequential", "parallel"):
        issues.append(
            LintIssue(
                LintLevel.ERROR,
                "mode-invalid",
                "mode must be 'sequential' or 'parallel'",
                "run_plan.mode",
            )
        )
    if mode == "parallel" and not pol.allow_parallel:
        issues.append(
            LintIssue(
                LintLevel.WARN,
                "parallel-disallowed",
                "parallel mode not allowed by policy",
                "run_plan.mode",
            )
        )

    workers = rp.get("max_parallel_workers", 0)
    if not isinstance(workers, int) or workers < 0:
        issues.append(
            LintIssue(
                LintLevel.ERROR,
                "workers-invalid",
                "max_parallel_workers must be a non-negative integer",
                "run_plan.max_parallel_workers",
            )
        )
    elif pol.max_parallel_workers and workers > pol.max_parallel_workers:
        issues.append(
            LintIssue(
                LintLevel.WARN,
                "workers-high",
                f"workers {workers} exceeds policy max {pol.max_parallel_workers}",
                "run_plan.max_parallel_workers",
            )
        )

    budgets = rp.get("budgets", {}) or {}
    for k in ("time_s", "samples", "memory_mb"):
        v = budgets.get(k)
        if v is not None and (not isinstance(v, (int, float)) or v < 0):
            issues.append(
                LintIssue(
                    LintLevel.ERROR,
                    "budget-invalid",
                    f"Budget '{k}' must be non-negative number",
                    f"run_plan.budgets.{k}",
                )
            )

    # Components
    if "scm" in manifest and isinstance(manifest["scm"], dict):
        _validate_component(manifest["scm"], "scm", issues, pol)
    if "mission" in manifest and isinstance(manifest["mission"], dict):
        _validate_component(manifest["mission"], "mission", issues, pol)
    metrics = manifest.get("metrics")
    if metrics:
        if not isinstance(metrics, dict):
            issues.append(
                LintIssue(
                    LintLevel.ERROR,
                    "metrics-type",
                    "'metrics' must be an object",
                    "metrics",
                )
            )
        else:
            if "behavior" in metrics:
                _validate_component(
                    metrics["behavior"], "metrics.behavior", issues, pol
                )
            if "result" in metrics:
                _validate_component(metrics["result"], "metrics.result", issues, pol)
            if "custom" in metrics:
                if (
                    pol.restricted_mode
                    and pol.forbid_custom_metrics
                    and metrics.get("custom")
                ):
                    issues.append(
                        LintIssue(
                            LintLevel.WARN,
                            "custom-metrics-forbidden",
                            "Custom metrics disabled in restricted mode",
                            "metrics.custom",
                        )
                    )
                if not isinstance(metrics.get("custom"), list):
                    issues.append(
                        LintIssue(
                            LintLevel.ERROR,
                            "metrics-custom-type",
                            "'metrics.custom' must be an array",
                            "metrics.custom",
                        )
                    )

    # Agents
    if isinstance(manifest.get("agents"), list):
        for i, a in enumerate(manifest["agents"]):
            if not isinstance(a, dict):
                issues.append(
                    LintIssue(
                        LintLevel.ERROR,
                        "agent-type",
                        "Each agent must be an object",
                        f"agents[{i}]",
                    )
                )
                continue
            if not a.get("id"):
                issues.append(
                    LintIssue(
                        LintLevel.ERROR,
                        "agent-id-missing",
                        "Agent missing 'id'",
                        f"agents[{i}].id",
                    )
                )
            comp = a.get("class")
            if not comp:
                issues.append(
                    LintIssue(
                        LintLevel.ERROR,
                        "agent-class-missing",
                        "Agent missing 'class'",
                        f"agents[{i}].class",
                    )
                )
            else:
                _validate_component(
                    {"class": comp, "params": a.get("params", {}) or {}},
                    f"agents[{i}]",
                    issues,
                    pol,
                )

    ok = not any(ix.level == LintLevel.ERROR for ix in issues)
    return ok, issues
