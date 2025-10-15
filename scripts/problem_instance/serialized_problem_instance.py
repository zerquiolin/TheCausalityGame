"""The Causality Game - Test Serialized DAGs."""

import networkx as nx
import numpy as np
import sympy as sp

from TheCausalityGame.agent.exhaustive import ExhaustiveAgent
from TheCausalityGame.core.contracts.problem_instance import ProblemInstance
from TheCausalityGame.core.contracts.specs.budget import BudgetSpec
from TheCausalityGame.core.contracts.specs.run import RunPlanSpec
from TheCausalityGame.core.contracts.specs.settings import RuntimeSettingsSpec
from TheCausalityGame.core.infraestructure.registry import build_from_spec
from TheCausalityGame.mission.conditional_average_treatment_effect import (
    ConditionalAverageTreatmentEffectMission,
)
from TheCausalityGame.mission.metric.behavior.rounds import RoundsBehaviorMetric
from TheCausalityGame.mission.metric.result.pehe import PEHEResultMetric
from TheCausalityGame.mission.result_validator.tef_validator import (
    TreatmentEffectFunctionValidator,
)
from TheCausalityGame.scm.core import CoreSCM
from TheCausalityGame.scm.dag.core import CoreDAG
from TheCausalityGame.scm.nodes.sympy import EquationBasedNumericalSCMNode
from TheCausalityGame.scm.noise.uniform import UniformNoiseDistribution

# Create DAG
graph = nx.DiGraph()
graph.add_edges_from([("Z", "X"), ("X", "Y"), ("Z", "Y")])
dag = CoreDAG(graph=graph)

# Create Nodes
Z = EquationBasedNumericalSCMNode(
    name="Z",
    evaluation=None,
    domain=[1, 5],
    noise_distribution=UniformNoiseDistribution(),
    accessibility="controllable",
    parents=None,
    parent_mappings=None,
)
X = EquationBasedNumericalSCMNode(
    name="X",
    evaluation=sp.sympify("2*Z"),
    domain=[2, 10],
    noise_distribution=UniformNoiseDistribution(),
    accessibility="observable",
    parents=["Z"],
    parent_mappings=None,
)
Y = EquationBasedNumericalSCMNode(
    name="Y",
    evaluation=sp.sympify("X+2*Z"),
    domain=[3, 15],
    noise_distribution=UniformNoiseDistribution(),
    accessibility="observable",
    parents=["Z", "X"],
    parent_mappings=None,
)

# Create scm
scm = CoreSCM(
    dag=dag,
    nodes=[Z, X, Y],
    random_state=np.random.RandomState(911),
)

# Create Behavior and Result Metrics
behavior = RoundsBehaviorMetric()
result = PEHEResultMetric()

# Create Result Validator
validator = TreatmentEffectFunctionValidator()

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
    execution="parallel",
    parallel_backend="thread",
    max_workers=None,
    budget=budget,
)

# Create Runtime Settings
runtime_settings = RuntimeSettingsSpec(
    mode="dev",
    debug=True,
    trusted=True,
)

# Create Problem Instance
problem_instance = ProblemInstance(
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

# Serialize to json
problem_instance_json = problem_instance.to_json()

# Dump json to file
with open("test_problem_instance.json", "w") as f:
    f.write(problem_instance_json)

# Deserialize from json
problem_instance_deserialized = build_from_spec(problem_instance_json)


# Check serialization
def deep_dict_equal(d1, d2, path=""):
    if isinstance(d1, dict) and isinstance(d2, dict):
        if d1.keys() != d2.keys():
            print(f"Key mismatch at {path}: {d1.keys()} vs {d2.keys()}")
            return False
        for key in d1:
            new_path = f"{path}.{key}" if path else key
            if not deep_dict_equal(d1[key], d2[key], new_path):
                return False
        return True

    elif isinstance(d1, list) and isinstance(d2, list):
        if len(d1) != len(d2):
            print(f"List length mismatch at {path}: {len(d1)} vs {len(d2)}")
            return False
        for index, (item1, item2) in enumerate(zip(d1, d2)):
            new_path = f"{path}[{index}]"
            if not deep_dict_equal(item1, item2, new_path):
                return False
        return True

    else:
        if d1 != d2:
            print(f"Value mismatch at {path}: {d1} vs {d2}")
            return False
        return True


deep_dict_equal(problem_instance.to_dict(), problem_instance_deserialized.to_dict())

print("All checks passed.")
