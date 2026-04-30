"""Run inferer-grouped problem instances."""

import json
from pathlib import Path

from TheCausalityGame.core.contracts.specs.problem_instance import ProblemInstanceSpec
from TheCausalityGame.core.runtime.runner import Runner

problem_instance_roots = []

# Graph discovery inferer families
problem_instance_roots.append(Path("scripts/problem_instances_by_inferer/graph_discovery"))  # type: ignore

# Treatment-effect inferer families
problem_instance_roots.append(Path("scripts/problem_instances_by_inferer/treatment_effect"))  # type: ignore


for root in problem_instance_roots:  # type: ignore
    for problem_instance_path in sorted(root.rglob("*.json")):  # type: ignore
        with problem_instance_path.open() as f:  # type: ignore
            problem_instance_spec = ProblemInstanceSpec(**json.load(f))  # type: ignore

        orchestrator = Runner(problem_instance=problem_instance_spec)
        orchestrator.run()
