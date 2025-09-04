from __future__ import annotations

from TheCausalityGame.core.contracts.dto import ProblemInstance


def test_problem_instance_validates_minimal() -> None:
    """Minimal valid manifest with new execution + metric specs shape."""
    manifest = {
        "schema_version": "1.0.0",
        "id": "smoke",
        "scm_spec": {"class": "TheCausalityGame.scm.example:Dummy", "config": {}},
        "mission_spec": {
            "class": "TheCausalityGame.missions.example:Dummy",
            "config": {},
        },
        "agent_specs": [
            {
                "id": "random",
                "class": "TheCausalityGame.agents.random:RandomAgent",
                "config": {},
            }
        ],
        "metric_specs": {
            "behavior": {
                "id": "rounds",
                "class": "TheCausalityGame.evaluators.behavior:RoundsUsed",
            },
            "result": {
                "id": "mse",
                "class": "TheCausalityGame.evaluators.regression:MSEMetric",
            },
        },
        "custom_metric_specs": [],
        "run_plan": {"rounds": 1, "budgets": {}},
        "seeds": {},
    }
    m = ProblemInstance.model_validate(manifest)
    assert m.id == "smoke"
    # Defaults for execution policy:
    assert m.run_plan.execution == "sequential"
    assert m.run_plan.parallel_backend == "thread"
    assert m.run_plan.max_workers is None
    # Metrics present:
    assert m.metric_specs.behavior.id == "rounds"
    assert m.metric_specs.result.id == "mse"
    assert m.custom_metric_specs == []


def test_problem_instance_rejects_extra_fields() -> None:
    """Extra top-level fields are rejected (extra='forbid')."""
    bad = {
        "schema_version": "1.0.0",
        "id": "bad",
        "scm_spec": {},
        "mission_spec": {},
        "agent_specs": [],
        "metric_specs": {
            "behavior": {"id": "b", "class": "X:Y"},
            "result": {"id": "r", "class": "X:Z"},
        },
        "run_plan": {"rounds": 1, "budgets": {}},
        "unexpected": 123,
    }
    try:
        ProblemInstance.model_validate(bad)
    except Exception as e:
        assert "unexpected" in str(e).lower()
    else:
        raise AssertionError("should fail validation")


def test_problem_instance_parallel_fields() -> None:
    """Parallel execution accepts backend + max_workers."""
    manifest = {
        "schema_version": "1.0.0",
        "id": "parallel",
        "scm_spec": {"class": "TheCausalityGame.scm.example:Dummy", "config": {}},
        "mission_spec": {
            "class": "TheCausalityGame.missions.example:Dummy",
            "config": {},
        },
        "agent_specs": [
            {
                "id": "a1",
                "class": "TheCausalityGame.agents.random:RandomAgent",
                "config": {},
            },
            {
                "id": "a2",
                "class": "TheCausalityGame.agents.random:RandomAgent",
                "config": {},
            },
        ],
        "metric_specs": {
            "behavior": {
                "id": "rounds",
                "class": "TheCausalityGame.evaluators.behavior:RoundsUsed",
            },
            "result": {
                "id": "mse",
                "class": "TheCausalityGame.evaluators.regression:MSEMetric",
            },
        },
        "custom_metric_specs": [],
        "run_plan": {
            "rounds": 5,
            "execution": "parallel",
            "parallel_backend": "process",
            "max_workers": 2,
            "budgets": {"time_s": 60, "samples": 1000, "memory_mb": 256},
        },
    }
    m = ProblemInstance.model_validate(manifest)
    assert m.run_plan.execution == "parallel"
    assert m.run_plan.parallel_backend == "process"
    assert m.run_plan.max_workers == 2
