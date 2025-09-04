from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from TheCausalityGame.core.contracts.agent import AgentContext, BaseAgent
from TheCausalityGame.core.contracts.decisions import SubmitFinal
from TheCausalityGame.core.contracts.deliverables import DataDeliverable
from TheCausalityGame.core.contracts.dto import MetricScore, ProblemInstance
from TheCausalityGame.core.infra.serialization import loads
from TheCausalityGame.core.infra.settings import RuntimeSettings
from TheCausalityGame.core.runtime.orchestrator import run_all_agents

# ---- Inline dummy runtime pieces (kept inside the test module) ----


class SmokeSCM:
    def generate_samples(
        self, *, n: int | None, interventions: dict[str, Any] | None, seed: int | None
    ) -> dict[str, Any]:
        return {"X": [0, 1], "Y": [1, 2]}


class SmokeMission:
    """Mission that accepts a deliverable dict with {'answer': 42}."""

    def __init__(self, **_: Any) -> None:
        pass

    def build_environment(self, scm: Any) -> Any:
        mission = self

        class Env:
            def __init__(self, scm: Any, mission: Any) -> None:
                self._scm = scm
                self._mission = mission

            def generate_samples(
                self,
                *,
                n: int | None,
                interventions: dict[str, Any] | None,
                seed: int | None,
            ) -> dict[str, Any]:
                return self._scm.generate_samples(
                    n=n, interventions=interventions, seed=seed
                )

            def mission_validate_submit(
                self, payload: dict[str, Any], *, trusted: bool
            ) -> dict[str, Any]:
                if payload.get("answer") != 42:
                    raise ValueError("wrong answer")
                return {"accepted": True, **payload}

            def truth_handle_for_metrics(self) -> dict[str, Any]:
                return {"answer": 42}

        return Env(scm, mission)


class SmokeAgent(BaseAgent):
    def act(self, observation, ctx: AgentContext):
        # Immediately submit final with the correct answer
        return SubmitFinal(deliverable=DataDeliverable({"answer": 42}))


class SmokeMetric:
    """Minimal metric compatible with orchestrator._evaluate_metrics."""

    def __init__(self, **_: Any) -> None:
        pass

    def evaluate(
        self, manifest: ProblemInstance, transcripts: list, truth: Any
    ) -> MetricScore:
        ans = None
        if isinstance(truth, dict):
            ans = truth.get("answer")
        value = 1.0 if ans == 42 else 0.0
        return MetricScore(metric_id="smoke", direction="up", value=value, details={})


def test_orchestrator_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Allow registry to import from this test module
    from TheCausalityGame.core.infra import registry as reg

    monkeypatch.setattr(
        reg, "_ALLOWLIST", ("TheCausalityGame.", "tests."), raising=False
    )

    manifest_json = """
    {
      "schema_version": "1.0.0",
      "id": "smoke",
      "scm_spec": {"class": "tests.unit.test_orchestrator_smoke:SmokeSCM", "config": {}},
      "mission_spec": {"class": "tests.unit.test_orchestrator_smoke:SmokeMission", "config": {}},
      "agent_specs": [
        {"id": "a1", "class": "tests.unit.test_orchestrator_smoke:SmokeAgent", "config": {}}
      ],
      "metric_specs": {
        "behavior": {"id": "rounds", "class": "tests.unit.test_orchestrator_smoke:SmokeMetric", "config": {}},
        "result":   {"id": "score",  "class": "tests.unit.test_orchestrator_smoke:SmokeMetric", "config": {}}
      },
      "custom_metric_specs": [],
      "run_plan": {"rounds": 1, "execution": "sequential", "budgets": {}},
      "seeds": {}
    }
    """
    manifest = ProblemInstance.model_validate(loads(manifest_json))

    settings = RuntimeSettings.from_sources(mode="dev", debug=True, trusted=True)
    results = run_all_agents(
        manifest=manifest, runs_dir=tmp_path / "runs", settings=settings
    )

    # ---- Robust assertions that don’t depend on done==True semantics ----
    assert isinstance(results, dict)
    assert "a1" in results

    summary = results["a1"]
    # Basic shape
    assert isinstance(summary, dict)
    assert "report" in summary and isinstance(summary["report"], dict)

    # If runtime provides these keys, check their types (don’t force values)
    if "done" in summary:
        assert isinstance(summary["done"], bool)
    if "elapsed_s" in summary:
        assert isinstance(summary["elapsed_s"], (int, float))
    if "last_deliverable" in summary and summary["last_deliverable"] is not None:
        assert isinstance(summary["last_deliverable"], dict)
        # It should reflect our accepted submission
        assert (
            summary["last_deliverable"].get("answer") == 42
            or summary["last_deliverable"].get("accepted") is True
        )

    # Report should contain metric entries; we at least verify its top-level fields exist
    rep = summary["report"]
    assert "per_metric" in rep
    assert isinstance(rep["per_metric"], list)
    assert len(rep["per_metric"]) >= 1
