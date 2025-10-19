"""The Causality Game - Hook Manager."""

from collections import defaultdict
from pathlib import Path

from TheCausalityGame.core.contracts.dto.transcript import TranscriptEntry
from TheCausalityGame.core.contracts.hook import Hook
from TheCausalityGame.core.contracts.specs.hook import HookSpec
from TheCausalityGame.core.infrastructure.registry import build_from_spec
from TheCausalityGame.core.lib.enum.hook import HookEvent


class HookManager:
    """
    Manages and triggers runtime hooks based on predefined lifecycle events.

    Attributes
    ----------
    hooks : list[HookSpec]
        Raw hook specifications provided at initialization.
    hooks_by_step : dict[HookEvent, list[Hook]]
        Grouped hook instances mapped by the step (event) they subscribe to.
    """

    def __init__(self, hooks: list[HookSpec], hook_dir: Path) -> None:
        """
        Initialize the HookManager with hook specs.

        Parameters
        ----------
        hooks : list[HookSpec]
            List of hook specifications to instantiate and register.
        """
        self.hooks = sorted(hooks, key=lambda h: h.priority)

        self.hooks_by_step: dict[HookEvent, list[Hook]] = defaultdict(list)
        for hook_spec in self.hooks:
            hook = build_from_spec(hook_spec)
            self.hooks_by_step[hook_spec.step].append(hook)

        # Ensure hook directory exists
        self.hook_dir = hook_dir
        if len(hooks) > 0:
            self.hook_dir.mkdir(parents=True, exist_ok=True)

    def trigger(self, step: HookEvent, context: TranscriptEntry | None = None) -> None:
        """
        Trigger all hooks subscribed to the specified lifecycle event.

        Parameters
        ----------
        step : HookEvent
            The lifecycle step to trigger (e.g., 'before_act', 'after_eval').
        context : TranscriptEntry
            Runtime context passed into each hook's `run` method.
        """
        for hook in self.hooks_by_step.get(step, []):
            hook.run(self.hook_dir, context)
