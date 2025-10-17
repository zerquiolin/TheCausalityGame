"""The Causality Game - Agent Specification."""

from pydantic import Field

from TheCausalityGame.core.contracts.specs.common import CommonSpec


class AgentSpec(CommonSpec):
    """
    Specification for constructing an agent.

    Inherits from `CommonSpec` to support dynamic loading and configuration.

    Attributes
    ----------
    class_ : str
        Import path of the agent class (e.g., "my_module.agent:MyAgent").
    spec_ : str | None
        Optional reference to the agent spec class.
    params : dict[str, Any]
        Optional configuration parameters for the agent.
    id : str
        Unique identifier for the agent instance.
    """

    id: str = Field(..., description="Unique identifier for the agent.")
