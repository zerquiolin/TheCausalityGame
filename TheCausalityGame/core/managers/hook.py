from collections import defaultdict
from typing import Any

from TheCausalityGame.core.contracts.hook import Hook
from TheCausalityGame.core.contracts.specs.hook import HookSpec
from TheCausalityGame.core.infrastructure.registry import build_from_spec
from TheCausalityGame.core.lib.enum.hook import HookEvent


class HookManager:
    def __init__(self, hooks: list[HookSpec]) -> None:
        # Store hooks
        self.hooks = hooks
        # Sort hooks by priority
        self.hooks.sort(key=lambda h: h.priority)
        # Group hooks by step
        self.hooks_by_step: dict[HookEvent, list[Hook]] = defaultdict(list)
        for hook in hooks:
            self.hooks_by_step[hook.step].append(build_from_spec(hook))

    def trigger(self, step: HookEvent, context: Any = None) -> None:
        for hook in self.hooks_by_step.get(step, []):
            hook.run(context)
