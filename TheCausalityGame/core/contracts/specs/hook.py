from typing import Literal

from contracts.types.common import JsonDict
from pydantic import BaseModel, ConfigDict


class HookSpec(BaseModel):
    """Specification for a runtime hook subscription.

    Attributes
    ----------
        id: Logical identifier for the hook subscription.
        class_path: Import path for a hook object implementing __call__(event,payload).
        entry_point: Alternative import path (kept for compatibility).
        events: Optional whitelist of event names this hook wants.
        priority: Ordering among hooks (lower runs earlier).
        requires: Other hook IDs that must run before this one.
        on_error: Policy when a hook raises ('ignore'|'warn'|'fail').
        config: Hook-specific configuration.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    class_path: str | None = None
    entry_point: str | None = None
    events: list[str] | None = None
    priority: int = 100
    requires: list[str] = []
    on_error: Literal["ignore", "warn", "fail"] = "warn"
    config: JsonDict = {}
