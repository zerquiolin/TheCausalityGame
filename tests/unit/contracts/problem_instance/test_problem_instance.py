"""The Causality Game - Problem Instance Tests."""

from typing import Any

import networkx as nx
import numpy as np
import pytest
import sympy as sp

from TheCausalityGame.agent.exhaustive import ExhaustiveAgent
from TheCausalityGame.core.contracts.problem_instance import ProblemInstance
from TheCausalityGame.core.contracts.specs.budget import BudgetSpec
from TheCausalityGame.core.contracts.specs.problem_instance import ProblemInstanceSpec
from TheCausalityGame.core.contracts.specs.run import RunPlanSpec
from TheCausalityGame.core.contracts.specs.settings import RuntimeSettingsSpec
from TheCausalityGame.core.infrastructure.registry import build_from_spec
from TheCausalityGame.core.lib.enum.nodes import NodeAccessibility
from TheCausalityGame.core.lib.enum.runplan import (
    RunPlanExecution,
    RunPlanParallelBackEnd,
)
from TheCausalityGame.core.lib.enum.runtime import RuntimeDebugLevel, RuntimeMode
from TheCausalityGame.core.lib.utils.tests import assert_dicts_equal
from TheCausalityGame.metric.behavior.rounds import RoundsBehaviorMetric
from TheCausalityGame.metric.result.pehe import PEHEResultMetric
from TheCausalityGame.metric.result.result_validator.cate_function_validator import (
    ConditionalAverageTreatmentEffectFunctionValidator,
)
from TheCausalityGame.mission.conditional_average_treatment_effect import (
    ConditionalAverageTreatmentEffectMission,
)
from TheCausalityGame.scm.core import CoreSCM
from TheCausalityGame.scm.dag.core import CoreDAG
from TheCausalityGame.scm.nodes.sympy import EquationBasedNumericalSCMNode
from TheCausalityGame.scm.noise.uniform import UniformNoiseDistribution


# Tests
@pytest.fixture(params=[ProblemInstance], scope="module")
def problem_instance(request: Any) -> None:  # noqa: ANN401
    """Test class construction."""
    # Arguments for a simple SCM
    graph = nx.DiGraph()
    graph.add_edges_from([("a", "F"), ("m", "F")])  # type: ignore
    dag = CoreDAG(graph=graph)

    # Create Nodes
    a = EquationBasedNumericalSCMNode(
        name="a",
        evaluation=None,
        domain=[-1e11, 1e11],
        noise_distribution=UniformNoiseDistribution(),
    )
    m = EquationBasedNumericalSCMNode(
        name="m",
        evaluation=None,
        domain=[0, 1e11],
        noise_distribution=UniformNoiseDistribution(),
    )
    F = EquationBasedNumericalSCMNode(  # noqa: N806
        name="F",
        evaluation=sp.sympify("a*m"),  # type: ignore
        domain=[3, 15],
        parents=["a", "m"],
        noise_distribution=UniformNoiseDistribution(),
        accessibility=NodeAccessibility.MEASURABLE,
    )

    # Create scm
    scm = CoreSCM(
        dag=dag,
        nodes=[a, m, F],
        random_state=np.random.RandomState(911),
    )

    # Create Behavior and Result Metrics
    behavior = RoundsBehaviorMetric()
    result = PEHEResultMetric()

    # Create Result Validator
    validator = ConditionalAverageTreatmentEffectFunctionValidator()

    # Create Mission
    mission = ConditionalAverageTreatmentEffectMission(
        behavior_metric=behavior,
        result_metric=result,
        result_validator=validator,
    )

    # Create Agent
    agent = ExhaustiveAgent(id="911")

    # Create Budget
    budget = BudgetSpec(rounds=100, time_s=60.0, samples=100, memory_mb=512)

    # Create Run Plan
    run_plan = RunPlanSpec(
        execution=RunPlanExecution.PARALLEL,
        parallel_backend=RunPlanParallelBackEnd.THREAD,
        max_workers=None,
        budget=budget,
    )

    # Create Runtime Settings
    runtime_settings = RuntimeSettingsSpec(
        mode=RuntimeMode.DEV,
        debug_level=RuntimeDebugLevel.DEBUG,
    )

    # Create Problem Instance
    cls = request.param
    return cls(
        schema_version="0.1.0",
        id="test_problem_instance",
        scm=scm,
        mission=mission,
        agents=[agent],
        custom_metrics=[],
        run_plan=run_plan,
        seeds={"agent": 42, "scm": 911, "misc": 7},
        runtime=runtime_settings,
    )


def test_agent_serialization_roundtrip(problem_instance: Any) -> None:  # noqa: ANN401
    """Test serialization roundtrip."""
    spec = problem_instance.to_spec()
    assert isinstance(spec, ProblemInstanceSpec)
    problem_instance2 = build_from_spec(spec)
    assert_dicts_equal(problem_instance.to_dict(), problem_instance2.to_dict())
