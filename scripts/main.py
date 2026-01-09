import json

from TheCausalityGame.core.contracts.specs.problem_instance import ProblemInstanceSpec
from TheCausalityGame.core.runtime.runner import Runner

# Get problem instance
problem_instance_path = "scripts/problem_instances/cate.json"
# REad problem instance
with open(problem_instance_path) as f:
    problem_instance_spec = json.load(f)
    problem_instance_spec = ProblemInstanceSpec(**problem_instance_spec)

# Create orchestrator
orchestrator = Runner(problem_instance=problem_instance_spec)

# Run
orchestrator.run()
