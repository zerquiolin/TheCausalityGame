from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable

from pydantic import BaseModel, ValidationError

from TheCausalityGame.core.contracts.specs.common import CommonSpec
from TheCausalityGame.core.contracts.specs.problem_instance import ProblemInstanceSpec
from TheCausalityGame.core.contracts.specs.run import RunPlanSpec

CLASS_PATH_RE = re.compile(r"^[A-Za-z_][\w\.]*(:|\.)[A-Za-z_]\w*$")


class LintLevel(str, Enum):
    INFO = "info"
    WARN = "warn"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class LintIssue:
    level: LintLevel
    code: str
    message: str
    path: str


@dataclass(frozen=True, slots=True)
class LintPolicy:
    """Policy toggles applied after schema validation."""

    allowed_module_prefixes: tuple[str, ...] = ()
    allowed_schema_versions: tuple[str, ...] | None = None
    min_agents: int = 1
    max_agents: int | None = None
    max_rounds: int | None = None
    max_workers: int | None = None
    allow_parallel: bool = True
    forbid_custom_metrics: bool = False


def lint_manifest(
    manifest: dict[str, Any], policy: LintPolicy | None = None
) -> tuple[bool, list[LintIssue]]:
    """Validate a serialized problem instance manifest."""

    pol = policy or LintPolicy()
    issues: list[LintIssue] = []

    try:
        spec = ProblemInstanceSpec.model_validate(manifest)
    except ValidationError as err:
        issues.extend(_issues_from_validation(err))
        return False, issues

    _validate_problem_instance(spec, pol, issues)
    ok = not any(issue.level == LintLevel.ERROR for issue in issues)
    return ok, issues


def _issues_from_validation(err: ValidationError) -> list[LintIssue]:
    collected: list[LintIssue] = []
    for error in err.errors():
        loc = _format_loc(error.get("loc", ()))
        code = f"validation.{error.get('type', 'unknown')}"
        msg = error.get("msg", "Invalid value")
        collected.append(LintIssue(LintLevel.ERROR, code, msg, loc))
    return collected


def _format_loc(loc: Iterable[Any]) -> str:
    parts: list[str] = ["manifest"]
    for token in loc:
        if isinstance(token, int):
            parts[-1] = f"{parts[-1]}[{token}]"
        else:
            name = str(token)
            if name == "__root__":
                continue
            parts.append(name)
    return ".".join(parts)


def _validate_problem_instance(
    spec: ProblemInstanceSpec, policy: LintPolicy, issues: list[LintIssue]
) -> None:
    if policy.allowed_schema_versions and spec.schema_version not in policy.allowed_schema_versions:
        issues.append(
            LintIssue(
                LintLevel.ERROR,
                "schema-version-unsupported",
                f"Schema version '{spec.schema_version}' is not allowed",
                "manifest.schema_version",
            )
        )

    agent_count = len(spec.agents)
    if agent_count < policy.min_agents:
        issues.append(
            LintIssue(
                LintLevel.ERROR,
                "agents-too-few",
                f"Manifest must declare at least {policy.min_agents} agent(s)",
                "manifest.agents",
            )
        )
    if policy.max_agents is not None and agent_count > policy.max_agents:
        issues.append(
            LintIssue(
                LintLevel.WARN,
                "agents-too-many",
                f"Manifest declares {agent_count} agents; policy maximum is {policy.max_agents}",
                "manifest.agents",
            )
        )

    _check_duplicate_agent_ids(spec, issues)
    _validate_run_plan(spec.run_plan, policy, issues)

    if policy.forbid_custom_metrics and spec.custom_metrics:
        issues.append(
            LintIssue(
                LintLevel.WARN,
                "custom-metrics-forbidden",
                "Custom metrics are disabled by policy",
                "manifest.custom_metrics",
            )
        )

    for path, component in _walk_common_specs(spec, "manifest"):
        _validate_common_component(path, component, policy, issues)


def _check_duplicate_agent_ids(
    spec: ProblemInstanceSpec, issues: list[LintIssue]
) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for agent in spec.agents:
        if agent.id in seen:
            duplicates.add(agent.id)
        else:
            seen.add(agent.id)
    if duplicates:
        dup_list = ", ".join(sorted(duplicates))
        issues.append(
            LintIssue(
                LintLevel.ERROR,
                "agent-id-duplicate",
                f"Agent identifiers must be unique; duplicates: {dup_list}",
                "manifest.agents",
            )
        )


def _validate_run_plan(
    run_plan: RunPlanSpec, policy: LintPolicy, issues: list[LintIssue]
) -> None:
    path = "manifest.run_plan"

    if run_plan.execution == "parallel" and not policy.allow_parallel:
        issues.append(
            LintIssue(
                LintLevel.WARN,
                "parallel-disallowed",
                "Parallel execution is disabled by policy",
                f"{path}.execution",
            )
        )

    if run_plan.max_workers is not None:
        if run_plan.max_workers <= 0:
            issues.append(
                LintIssue(
                    LintLevel.ERROR,
                    "workers-invalid",
                    "max_workers must be a positive integer",
                    f"{path}.max_workers",
                )
            )
        if (
            policy.max_workers is not None
            and run_plan.max_workers > policy.max_workers
        ):
            issues.append(
                LintIssue(
                    LintLevel.WARN,
                    "workers-high",
                    f"Requested workers ({run_plan.max_workers}) exceed policy maximum {policy.max_workers}",
                    f"{path}.max_workers",
                )
            )

    rounds = run_plan.budget.rounds
    rounds_path = f"{path}.budget.rounds"
    if rounds is None:
        issues.append(
            LintIssue(
                LintLevel.ERROR,
                "rounds-missing",
                "Budget must declare a positive 'rounds' limit",
                rounds_path,
            )
        )
    elif rounds <= 0:
        issues.append(
            LintIssue(
                LintLevel.ERROR,
                "rounds-invalid",
                "'rounds' must be a positive integer",
                rounds_path,
            )
        )
    elif policy.max_rounds is not None and rounds > policy.max_rounds:
        issues.append(
            LintIssue(
                LintLevel.WARN,
                "rounds-exceed-policy",
                f"'rounds' value {rounds} exceeds policy maximum {policy.max_rounds}",
                rounds_path,
            )
        )

    for field_name in ("time_s", "samples", "memory_mb"):
        value = getattr(run_plan.budget, field_name)
        if value is not None and value < 0:
            issues.append(
                LintIssue(
                    LintLevel.ERROR,
                    "budget-negative",
                    f"'{field_name}' must not be negative",
                    f"{path}.budget.{field_name}",
                )
            )


def _walk_common_specs(
    value: Any, path: str
) -> Iterable[tuple[str, CommonSpec]]:
    if isinstance(value, CommonSpec):
        yield path, value

    if isinstance(value, BaseModel):
        for field_name in value.model_fields:
            attr = getattr(value, field_name)
            if attr is None:
                continue
            child_path = f"{path}.{field_name}"
            yield from _walk_common_specs(attr, child_path)
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            if item is None:
                continue
            child_path = f"{path}[{idx}]"
            yield from _walk_common_specs(item, child_path)


def _validate_common_component(
    path: str,
    component: CommonSpec,
    policy: LintPolicy,
    issues: list[LintIssue],
) -> None:
    class_path = component.class_
    class_field = f"{path}.class_"
    if not CLASS_PATH_RE.match(class_path):
        issues.append(
            LintIssue(
                LintLevel.ERROR,
                "class-path-invalid",
                f"Invalid class path '{class_path}'",
                class_field,
            )
        )
    else:
        module = _module_from_class_path(class_path)
        if policy.allowed_module_prefixes and not any(
            module.startswith(prefix) for prefix in policy.allowed_module_prefixes
        ):
            issues.append(
                LintIssue(
                    LintLevel.WARN,
                    "module-not-allowlisted",
                    f"Module '{module}' is not allowed by policy",
                    class_field,
                )
            )

    spec_path = component.spec_
    if not isinstance(spec_path, str) or not CLASS_PATH_RE.match(spec_path):
        issues.append(
            LintIssue(
                LintLevel.ERROR,
                "spec-path-invalid",
                "Spec must provide a valid 'spec_' class path",
                f"{path}.spec_",
            )
        )

    if component.params is not None and not isinstance(component.params, dict):
        issues.append(
            LintIssue(
                LintLevel.ERROR,
                "params-type",
                "'params' must be a JSON object",
                f"{path}.params",
            )
        )


def _module_from_class_path(class_path: str) -> str:
    if ":" in class_path:
        return class_path.split(":", 1)[0]
    if "." in class_path:
        return class_path.rsplit(".", 1)[0]
    return class_path
