"""The Causality Game - Agent Specification."""

from __future__ import annotations

from pydantic import Field, model_validator

from TheCausalityGame.core.contracts.specs.agent_policy import AgentPolicySpec
from TheCausalityGame.core.contracts.specs.common import CommonSpec
from TheCausalityGame.core.contracts.specs.decider import DeciderSpec
from TheCausalityGame.core.contracts.specs.inferer import InfererSpec


class AgentSpec(CommonSpec):
    """
    Specification for constructing an agent.

    Inherits from `CommonSpec` to support dynamic loading and configuration.

    Attributes
    ----------
    class_ : str
        Import path of the agent class (e.g., "my_module.agent:MyAgent").
    params : dict[str, Any]
        Optional configuration parameters for the agent.
    id : str
        Unique identifier for the agent instance.
    """

    id: str = Field(..., description="Unique identifier for the agent.")
    active: bool | None = Field(
        default=True, description="Flag indicating whether the agent is active in the game."
    )
    inferer: InfererSpec | None = Field(
        default=None,
        description="Passive-learning component used by a composable agent.",
    )
    decider: DeciderSpec | None = Field(
        default=None,
        description="Active-learning component used by a composable agent.",
    )
    policy: AgentPolicySpec | None = Field(
        default=None,
        description="Unified policy component used by a combined agent.",
    )

    @model_validator(mode="after")
    def validate_component_shape(self) -> AgentSpec:
        """Ensure the spec encodes exactly one valid agent composition."""
        has_composed = self.inferer is not None or self.decider is not None
        if has_composed and not (self.inferer is not None and self.decider is not None):
            raise ValueError("Composable agents require both 'inferer' and 'decider'.")

        if has_composed and self.policy is not None:
            raise ValueError("Agent specs must define either inferer+decider or policy, not both.")

        if not has_composed and self.policy is None:
            raise ValueError("Agent specs must define either inferer+decider or policy.")

        return self
