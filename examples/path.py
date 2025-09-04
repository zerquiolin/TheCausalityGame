"""_summary_."""

from TheCausalityGame.agents.dummy_agent import DummyAgent
from TheCausalityGame.core.infra.serialization import get_class_path

print(
    get_class_path(DummyAgent)
)  # Should print "TheCausalityGame.agents.dummy_agent:DummyAgent"
