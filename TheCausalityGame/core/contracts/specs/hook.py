"""The Causality Game - Hook Specification."""

from pydantic import Field

from TheCausalityGame.core.contracts.specs.common import CommonSpec
from TheCausalityGame.core.lib.enum.hook import HookEvent


class HookSpec(CommonSpec):
    """
    Specification for a runtime hook subscription.

    Inherits from `CommonSpec` to support dynamic loading and configuration.

    Attributes
    ----------
    class_ : str
        Fully qualified import path (aliased from 'class' in JSON).
    params : dict
        Optional noise distribution configuration payload.
    id : str
        Unique identifier for the hook.
    step : HookEvent
        The canonical event this hook should trigger on.
    priority : int
        Priority level (higher is earlier). Defaults to 1.
    """

    id: str | None = Field(default=None, description="Unique identifier for the hook.")
    step: HookEvent = Field(description="The hook event this subscription listens to.")
    priority: int = Field(
        default=1, ge=0, description="Execution priority. Higher values run earlier."
    )
