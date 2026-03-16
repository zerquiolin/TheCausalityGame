"""The Causality Game - Main Script to Run Problem Instances."""

import json
import os

import numpy as np

from TheCausalityGame.core.contracts.specs.problem_instance import ProblemInstanceSpec
from TheCausalityGame.core.runtime.runner import Runner

problem_instance_path = []
problem_instance_file = []

# Hill
# problem_instance_path = "scripts/problem_instances/treatment_effect/hill.json"
# problem_instance_path, problem_instance_file = "scripts/problem_instances", "treatment_effect/hill"
# problem_instance_path.append("scripts/problem_instances")
# problem_instance_file.append("treatment_effect/hill")

# Hill Extended
# problem_instance_path = "scripts/problem_instances/treatment_effect/hill_extended.json"
# problem_instance_path, problem_instance_file = (
#     "scripts/problem_instances",
#     "treatment_effect/hill_extended",
# )
# problem_instance_path.append("scripts/problem_instances")
# problem_instance_file.append("treatment_effect/hill_extended")


# Graph Discovery
# problem_instance_path = "scripts/problem_instances/graph_discovery/rc_circuit.json"
# problem_instance_path.append("scripts/problem_instances")
# problem_instance_file.append("graph_discovery/rc_circuit")

# SCM Discovery
# problem_instance_path = "scripts/problem_instances/scm_discovery/rc_circuit.json"
problem_instance_path.append("scripts/problem_instances")
problem_instance_file.append("scm_discovery/rc_circuit")

seeds = list(np.random.default_rng(911).integers(0, 999, size=20))
seeds.append(911)  # Just Because

# Read problem instance

for path, file in zip(problem_instance_path, problem_instance_file):
    for seed in seeds[2:]:
        with open(os.path.join(path, "rss", f"{file}_rs-{seed}.json")) as f:
            problem_instance_spec = json.load(f)
            problem_instance_spec = ProblemInstanceSpec(**problem_instance_spec)

        # Create orchestrator
        orchestrator = Runner(problem_instance=problem_instance_spec)

        # Run
        orchestrator.run()
