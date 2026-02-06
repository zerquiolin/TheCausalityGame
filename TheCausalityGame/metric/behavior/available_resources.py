"""Behavior metric that reports remaining resources as a percentage (stateless)."""

from __future__ import annotations

from typing import override

from TheCausalityGame.core.contracts.dto.transcript import Transcript
from TheCausalityGame.core.contracts.metric import BehaviorMetric
from TheCausalityGame.core.contracts.scm import SCM
from TheCausalityGame.core.contracts.specs.metric import MetricSpec
from TheCausalityGame.core.infrastructure.registry import get_class_path


class AvailableResourcesBehaviorMetric(BehaviorMetric):
    """
    Weighted percentage of remaining resources (higher is better).

    This metric is designed for per-round curves: when called with the transcript-so-far,
    it returns the weighted remaining percentage at the current (last) entry.

    Totals are taken from `transcript.budget` (a BudgetSpec). Remaining values are taken
    from the current entry's `budget_snapshot`. Missing resources are ignored.

    If a total is missing in `transcript.budget` (None), we fall back to computing
    remaining/(used+remaining) for that resource when possible.
    """

    name: str = "Available Resources Metric"
    description: str = (
        "Behavior metric that reports remaining rounds, samples, time, and memory "
        "as a single weighted percentage (higher is better)."
    )

    def __init__(
        self,
        *,
        rounds_weight: float = 0.1,
        samples_weight: float = 0.45,
        time_weight: float = 0.35,
        memory_weight: float = 0.1,
    ) -> None:
        weights = [rounds_weight, samples_weight, time_weight, memory_weight]
        if sum(weights) != 1:
            raise ValueError("The sum of resource weights must be 1.")

        self._rounds_weight = float(rounds_weight)
        self._samples_weight = float(samples_weight)
        self._time_weight = float(time_weight)
        self._memory_weight = float(memory_weight)

    @override
    def mount(self, scm: SCM) -> None:
        self.acc_rounds = 0.0
        self.acc_samples = 0.0
        self.acc_time_s = 0.0
        self.acc_memory_mb = 0.0

    @override
    def evaluate(self, transcript: Transcript) -> float:
        if not transcript.entries:
            return 0.0

        entry = transcript.entries[-1]
        snap = entry.budget_snapshot

        if snap is None:
            return 0.0

        score: float = 0.0

        # Rounds
        score += self._rounds_weight * self._percentage_remaining(
            used=(
                transcript.budget.rounds - snap.rounds_left if snap.rounds_left is not None else 0
            ),
            total=transcript.budget.rounds or 0,
        )

        # Samples
        score += self._samples_weight * self._percentage_remaining(
            used=(
                transcript.budget.samples - snap.samples_left
                if snap.samples_left is not None and transcript.budget.samples is not None
                else 0
            ),
            total=transcript.budget.samples or 0,
        )

        # Time
        time_used_s = (
            (transcript.budget.time_s - snap.time_s_left)
            if snap.time_s_left is not None and transcript.budget.time_s is not None
            else 0.0
        )
        score += self._time_weight * self._percentage_remaining(
            used=time_used_s,
            total=transcript.budget.time_s or 0.0,
        )

        # Memory
        memory_used_mb = (
            (transcript.budget.memory_mb - snap.memory_mb_left)
            if snap.memory_mb_left is not None and transcript.budget.memory_mb is not None
            else 0.0
        )
        score += self._memory_weight * self._percentage_remaining(
            used=memory_used_mb,
            total=transcript.budget.memory_mb or 0.0,
        )

        return score

    @override
    def to_spec(self) -> MetricSpec:
        return MetricSpec(
            class_=get_class_path(self.__class__),
            params={
                "rounds_weight": self._rounds_weight,
                "samples_weight": self._samples_weight,
                "time_weight": self._time_weight,
                "memory_weight": self._memory_weight,
            },
        )

    @classmethod
    @override
    def from_spec(cls, spec: MetricSpec) -> AvailableResourcesBehaviorMetric:
        return cls(**(spec.params or {}))

    def _percentage_remaining(self, used: int | float, total: int | float) -> float:
        return (total - used) * 100.0 / total if total > 0.0 else 0.0
