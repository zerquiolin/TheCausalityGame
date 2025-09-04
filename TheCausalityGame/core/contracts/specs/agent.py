from contracts.types.common import JsonDict
from pydantic import BaseModel, ConfigDict, Field


class AgentSpec(BaseModel):
    """Specification for constructing an agent.

    Attributes
    ----------
        id: Unique agent identifier for the run.
        class_: Import path 'module:Class' (aliased from 'class' in JSON).
        config: Optional agent configuration payload.
        priority: Optional priority hint (lower → earlier in sequential UIs).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    class_: str = Field(alias="class")
    params: JsonDict = {}

    priority: int | None = Field(default=None, ge=0)
