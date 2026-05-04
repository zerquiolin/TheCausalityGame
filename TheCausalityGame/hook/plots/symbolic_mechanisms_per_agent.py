"""Plot symbolic SCM mechanisms per agent."""

from __future__ import annotations

from pathlib import Path
from typing import override

import matplotlib.pyplot as plt

from TheCausalityGame.core.contracts.dto.transcript import Transcript, TranscriptEntry
from TheCausalityGame.core.contracts.hook import Hook
from TheCausalityGame.core.contracts.specs.hook import HookSpec
from TheCausalityGame.core.infrastructure.registry import get_class_path
from TheCausalityGame.core.lib.enum.hook import HookEvent


class SymbolicMechanismsPerAgentPlotHook(Hook):
    """Render discovered symbolic mechanisms for each agent."""

    id: str = "Agents Symbolic Mechanisms"
    step: HookEvent = HookEvent.BENCHMARK_END

    @override
    def run(self, hooks_dir: Path, context: dict[str, Transcript] | TranscriptEntry | None) -> None:
        if context is None or not isinstance(context, dict):
            return

        rows: list[tuple[str, list[str]]] = []
        for agent_id, transcript in context.items():
            if transcript.invalidated or not transcript.entries:
                continue

            result = transcript.entries[-1].result
            mechanisms = getattr(result, "mechanisms", None)
            if not isinstance(mechanisms, dict):
                continue

            expressions = [
                f"{node} = {mechanism.expression}"
                for node, mechanism in sorted(mechanisms.items())
                if hasattr(mechanism, "expression")
            ]
            rows.append((agent_id, expressions or ["No symbolic mechanisms discovered."]))

        if not rows:
            return

        height = max(3.0, 1.2 + sum(max(1, len(expressions)) for _, expressions in rows) * 0.45)
        fig, ax = plt.subplots(figsize=(12, height))
        ax.axis("off")
        ax.set_title("Discovered Symbolic SCM Mechanisms", fontsize=14, pad=16)

        y = 0.95
        line_height = 0.08 if len(rows) <= 3 else 0.055
        for agent_id, expressions in rows:
            ax.text(
                0.02,
                y,
                agent_id,
                transform=ax.transAxes,
                fontsize=11,
                fontweight="bold",
                va="top",
            )
            y -= line_height
            for expression in expressions:
                ax.text(
                    0.05,
                    y,
                    expression,
                    transform=ax.transAxes,
                    fontsize=10,
                    family="monospace",
                    va="top",
                )
                y -= line_height
            y -= line_height * 0.35

        fig.savefig(hooks_dir / "symbolic_mechanisms_per_agent.png", bbox_inches="tight")
        plt.close(fig)

    @override
    def to_spec(self) -> HookSpec:
        return HookSpec(
            class_=get_class_path(self.__class__),
            id=self.id,
            step=self.step,
        )

    @classmethod
    @override
    def from_spec(cls, spec: HookSpec) -> SymbolicMechanismsPerAgentPlotHook:
        return cls()
