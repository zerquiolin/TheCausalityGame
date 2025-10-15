import json

from TheCausalityGame.core.engine.runner import Runner
from TheCausalityGame.core.infra.registry import build_from_spec

# Get problem instance
problem_instance_path = "test_problem_instance.json"
# REad problem instance
with open(problem_instance_path, "r") as f:
    problem_instance_spec = json.load(f)
# Build problem instance
problem_instance = build_from_spec(problem_instance_spec)

# Create orchestrator
orchestrator = Runner(problem_instance=problem_instance)

# Run
orchestrator.run()
