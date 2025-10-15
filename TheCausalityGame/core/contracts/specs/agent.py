"""The Causality Game - Agent Spec."""

from pydantic import BaseModel, ConfigDict, Field

from TheCausalityGame.core.contracts.types.common import JsonDict
from TheCausalityGame.core.contracts.specs.common import CommonSpec


class AgentSpec(CommonSpec):
    """Specification for constructing an agent.

    Attributes
    ----------
        class_: Import path 'module:Class' (aliased from 'class' in JSON).
        params: Optional agent configuration payload.
    """

    id: str = Field(..., description="Unique identifier for the agent.")
