from TheCausalityGame.core.contracts.specs.common import CommonSpec
from TheCausalityGame.core.lib.enum.hooks import HookEvent


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

    id: str
    step: HookEvent
    priority: int = 1
