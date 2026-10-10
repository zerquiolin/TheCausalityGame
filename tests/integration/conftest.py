"""Small real problem instances for runtime integration tests."""

from __future__ import annotations

import pytest

from TheCausalityGame.core.contracts.specs.problem_instance import ProblemInstanceSpec


@pytest.fixture
def tiny_problem() -> ProblemInstanceSpec:
    """Return a fresh seeded SCM with one hidden root and two observable nodes."""
    return ProblemInstanceSpec.model_validate(
        {
            "class_": "TheCausalityGame.core.contracts.problem_instance:ProblemInstance",
            "id": "runtime_test",
            "agents": [
                {
                    "id": name,
                    "class_": "TheCausalityGame.agent.composable:ComposableAgent",
                    "inferer": {
                        "class_": "TheCausalityGame.agent.inferers.empirical.pc:PCInferer"
                    },
                    "decider": {
                        "class_": "TheCausalityGame.agent.deciders.exhaustive:ExhaustiveDecider",
                        "params": {"num_obs": 10, "num_inter": 2},
                    },
                }
                for name in ("first", "second")
            ],
            "scm": {
                "class_": "TheCausalityGame.scm.core:CoreSCM",
                "vars": [
                    {
                        "class_": "TheCausalityGame.scm.nodes.sympy:EquationBasedNumericalSCMNode",
                        "name": name,
                        "domain": [-1.0, 1.0],
                        "accessibility": access,
                        **({"equation": "L + 2*X"} if name == "Y" else {}),
                    }
                    for name, access in (
                        ("X", "controllable"),
                        ("L", "latent"),
                        ("Y", "measurable"),
                    )
                ],
                "dag": {
                    "class_": "TheCausalityGame.scm.dag.core:CoreDAG",
                    "edges": [["X", "Y"], ["L", "Y"]],
                },
            },
            "mission": {
                "id": "tiny_graph",
                "class_": "TheCausalityGame.mission.dag_discovery:DAGDiscoveryMission",
                "behavior_metric": {
                    "class_": "TheCausalityGame.metric.behavior.rounds:RoundsBehaviorMetric"
                },
                "result_metric": {
                    "class_": "TheCausalityGame.metric.result.shd:SHDResultMetric"
                },
                "result_validator": {
                    "class_": (
                        "TheCausalityGame.metric.result.result_validator."
                        "dag_validator:DAGResultValidator"
                    )
                },
            },
            "run_plan": {
                "execution": "sequential",
                "max_workers": 2,
                "budget": {"rounds": 2, "samples": 40},
            },
            "runtime": {"mode": "dev", "debug_level": 40},
        }
    )
