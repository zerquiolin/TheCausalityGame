from __future__ import annotations

import random
from typing import Any

from TheCausalityGame.core.contracts.dto import (
    AgentSpec,
    MetricSpec,
    MetricSpecs,
    ProblemInstance,
    RunPlan,
)
from TheCausalityGame.core.contracts.specs.settings import RuntimeSettings
from TheCausalityGame.core.runtime.runner import GameRunner

# ---------------- Dummy components ---------------- #


class DummySCM:
    """A minimal dummy SCM that just returns random numbers."""

    def generate_samples(
        self, *, n: int | None, interventions: dict[str, Any] | None, seed: int | None
    ) -> dict[str, Any]:
        rng = random.Random(seed)
        return {
            "samples": [rng.random() for _ in range(n or 5)],
            "interventions": interventions,
        }


class DummyMission:
    """A dummy mission that accepts any payload and returns it back."""

    def validate_submit(self, payload: dict[str, Any], trusted: bool) -> dict[str, Any]:
        return {"validated_payload": payload, "trusted": trusted}

    def truth_handle_for_metrics(self, scm: DummySCM) -> dict[str, Any]:
        return {"ground_truth": "dummy"}


class DummyAgent:
    """A simple agent that always submits the same answer."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}

    def run(self, env: Any) -> dict[str, Any]:
        return {"answer": "42", "config_used": self._config}


class DummyMetric:
    """A metric that checks if the agent answered '42'."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}

    def evaluate(
        self, deliverable: dict[str, Any], truth: dict[str, Any]
    ) -> dict[str, Any]:
        return {"score": 1.0 if deliverable.get("answer") == "42" else 0.0}


# ---------------- Manifest builder ---------------- #


def build_dummy_manifest() -> ProblemInstance:
    """Build a ProblemInstance manifest with dummy components."""
    scm_spec = {"class": "examples.dummy_runner.DummySCM", "config": {}}
    mission_spec = {"class": "examples.dummy_runner.DummyMission", "config": {}}

    agents = [
        AgentSpec(class_="examples.dummy_runner.DummyAgent", config={"id": "agentA"}),
        AgentSpec(class_="examples.dummy_runner.DummyAgent", config={"id": "agentB"}),
    ]

    metric_specs = MetricSpecs(
        behavior=MetricSpec(class_="examples.dummy_runner.DummyMetric", config={}),
        result=MetricSpec(class_="examples.dummy_runner.DummyMetric", config={}),
        custom_metric_specs=[],
    )

    run_plan = RunPlan(policy="round_robin")

    return ProblemInstance(
        id="dummy-problem",
        scm=scm_spec,
        mission=mission_spec,
        agents=agents,
        metric_specs=metric_specs,
        run_plan=run_plan,
    )


# ---------------- Entry point ---------------- #

if __name__ == "__main__":
    settings = RuntimeSettings.from_sources(mode="dev", debug=True, trusted=True)
    runner = GameRunner(settings=settings)

    manifest = build_dummy_manifest()
    results = runner.run_manifest(manifest=manifest, runs_dir="runs")

    print("\n=== Dummy Run Results ===")
    for agent, summary in results.items():
        print(agent, summary)
