# agents/dummy_agent.py
from TheCausalityGame.core.contracts.agent import BaseAgent
from TheCausalityGame.core.contracts.dto import Action, Observation
from TheCausalityGame.core.contracts.errors import AgentError


class DummyAgent(BaseAgent):
    """A minimal agent that just returns a fixed action."""

    def act(self, observation: Observation) -> Action:
        """Generate an action based on the provided observation."""
        try:
            return Action(
                result={"guess": 42}, metadata={"note": "Dummy agent always guesses 42"}
            )
        except Exception as e:
            raise AgentError(
                "DummyAgent failed to produce an action", details={"error": str(e)}
            ) from e
