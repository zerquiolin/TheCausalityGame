"""The Causality Game - Main Script to Run Problem Instances."""

import json

from TheCausalityGame.core.contracts.specs.problem_instance import ProblemInstanceSpec
from TheCausalityGame.core.runtime.runner import Runner

# Hill
problem_instance_path = "scripts/problem_instances/treatment_effect/hill.json"

# Hill Extended
# problem_instance_path = "scripts/problem_instances/treatment_effect/hill_extended.json"

# Graph Discovery
# problem_instance_path = "scripts/problem_instances/graph_discovery/rc_circuit.json"

# SCM Discovery
# problem_instance_path = "scripts/problem_instances/scm_discovery/rc_circuit.json"

# Read problem instance
with open(problem_instance_path) as f:
    problem_instance_spec = json.load(f)
    problem_instance_spec = ProblemInstanceSpec(**problem_instance_spec)

# Create orchestrator
orchestrator = Runner(problem_instance=problem_instance_spec)

# Run
orchestrator.run()
