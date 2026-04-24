"""Tests for inferer-grouped problem-instance scripts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from TheCausalityGame.core.contracts.specs.problem_instance import ProblemInstanceSpec
from TheCausalityGame.core.infrastructure.registry import build_from_spec


@pytest.mark.parametrize(
    "script_path",
    sorted(Path("scripts/problem_instances_by_inferer").rglob("*.json")),
)
def test_inferer_grouped_problem_instance_loads(script_path: Path) -> None:
    """Each inferer-grouped problem instance is valid and buildable."""
    with script_path.open() as f:
        spec = ProblemInstanceSpec(**json.load(f))

    problem_instance = build_from_spec(spec)

    assert problem_instance.id == spec.id
