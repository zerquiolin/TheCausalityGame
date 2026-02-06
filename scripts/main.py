"""The Causality Game - Main Script to Run Problem Instances."""

import json
import os

from TheCausalityGame.core.contracts.specs.problem_instance import ProblemInstanceSpec
from TheCausalityGame.core.runtime.runner import Runner

# Hill
# problem_instance_path = "scripts/problem_instances/treatment_effect/hill.json"
problem_instance_path, problem_instance_file = "scripts/problem_instances", "treatment_effect/hill"

# Hill Extended
# problem_instance_path = "scripts/problem_instances/treatment_effect/hill_extended.json"

# Graph Discovery
# problem_instance_path = "scripts/problem_instances/graph_discovery/rc_circuit.json"

# SCM Discovery
# problem_instance_path = "scripts/problem_instances/scm_discovery/rc_circuit.json"

seeds = (1, 42, 59, 34, 57, 91, 17, 83, 27, 99, 123, 473, 501, 701, 875, 101, 911)

# Read problem instance
for seed in seeds:
    with open(
        os.path.join(problem_instance_path, "rss", f"{problem_instance_file}_rs-{seed}.json")
    ) as f:
        problem_instance_spec = json.load(f)
        problem_instance_spec = ProblemInstanceSpec(**problem_instance_spec)

    # Create orchestrator
    orchestrator = Runner(problem_instance=problem_instance_spec)

    # Run
    orchestrator.run()
