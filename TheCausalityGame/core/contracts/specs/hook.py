from typing import Literal

from TheCausalityGame.core.contracts.enum.hooks import HookEvent
from TheCausalityGame.core.contracts.specs.common import CommonSpec


class HookSpec(CommonSpec):
    """Specification for a runtime hook subscription.

    Attributes
    ----------
    events : list[HookEvent]
        List of events to subscribe to.
    priority : int, optional
        Priority of the hook. Higher priority hooks are executed first. Default is 1.
    on_error : Literal["ignore", "warn", "fail"], optional
        Behavior when an error occurs in the hook. Default is "warn".
    """

    events: list[HookEvent]
    priority: int = 1
    on_error: Literal["ignore", "warn", "fail"] = "warn"
