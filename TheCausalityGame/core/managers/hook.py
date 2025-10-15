from collections import defaultdict
from typing import Any

from TheCausalityGame.core.contracts.enum.hooks import HookEvent
from TheCausalityGame.core.contracts.hooks import Hook


class HookManager:
    def __init__(self, hooks: list[Hook]) -> None:
        self.hooks = hooks
        self.hooks_by_step: dict[HookEvent, list[Hook]] = defaultdict(list)
        for hook in hooks:
            self.hooks_by_step[hook.step].append(hook)

    def trigger(self, step: HookEvent, context: any) -> None:
        for hook in self.hooks_by_step.get(step, []):
            hook.run(context) if context else hook.run()
