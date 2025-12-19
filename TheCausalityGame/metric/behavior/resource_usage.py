"""Behavior metric that accounts for rounds, samples, and memory usage."""

from __future__ import annotations

from typing import override

from TheCausalityGame.core.contracts.dto.transcript import Transcript
from TheCausalityGame.core.contracts.metric import BehaviorMetric
from TheCausalityGame.core.contracts.scm import SCM
from TheCausalityGame.core.contracts.specs.metric import MetricSpec
from TheCausalityGame.core.infrastructure.registry import get_class_path
from TheCausalityGame.core.lib.utils.metrics import log_penalty


class ResourceUsageBehaviorMetric(BehaviorMetric):
    """
    Penalize overall resource consumption across rounds, samples, and memory.

    The metric aggregates three components—round count, samples requested, and
    memory consumed—using configurable weights. Each component is mapped through
    a logarithmic penalty so that higher resource usage produces a lower score,
    while still keeping the metric bounded away from zero.
    """

    name: str = "Resource Usage Metric"
    description: str = (
        "Behavior metric that penalizes how many rounds, samples, and memory "
        "an agent consumes before stopping."
    )

    def __init__(
        self,
        *,
        rounds_weight: float = 0.4,
        samples_weight: float = 0.4,
        memory_weight: float = 0.2,
        rounds_alpha: float = 0.1,
        samples_alpha: float = 0.01,
        memory_alpha: float = 0.05,
    ) -> None:
        weights = [rounds_weight, samples_weight, memory_weight]
        if all(weight <= 0 for weight in weights):
            raise ValueError("At least one resource weight must be positive.")

        self._rounds_weight = rounds_weight
        self._samples_weight = samples_weight
        self._memory_weight = memory_weight

        self._rounds_alpha = rounds_alpha
        self._samples_alpha = samples_alpha
        self._memory_alpha = memory_alpha
        self.samples_used = 0
        self.memory_used_bytes = 0
        self.memory_used_mb = 0

    @override
    def mount(self, scm: SCM) -> None:
        # This metric does not require SCM-specific context.
        return None

    @override
    def evaluate(self, transcript: Transcript) -> float:
        if not transcript.entries:
            return 0.0

        last_entry = transcript.entries[-1]
        snapshot = last_entry.budget_snapshot

        rounds_used = last_entry.round
        collection = last_entry.samples_collection
        if collection is not None:
            self.samples_used += collection.total_n()
            self.memory_used_bytes += collection.total_bytes()
        self.memory_used_mb = self.memory_used_bytes / (1024.0 * 1024.0)

        rounds_ratio = self._usage_ratio(
            used=rounds_used,
            remaining=None if snapshot is None else snapshot.rounds_left,
        )
        samples_ratio = self._usage_ratio(
            used=self.samples_used,
            remaining=None if snapshot is None else snapshot.samples_left,
        )
        memory_ratio = self._usage_ratio(
            used=self.memory_used_mb,
            remaining=None if snapshot is None else snapshot.memory_mb_left,
        )

        score = 0.0
        weight_total = 0.0

        if self._rounds_weight > 0:
            score += self._rounds_weight * log_penalty(
                rounds_ratio, alpha=self._rounds_alpha
            )
            weight_total += self._rounds_weight

        if self._samples_weight > 0:
            score += self._samples_weight * log_penalty(
                samples_ratio, alpha=self._samples_alpha
            )
            weight_total += self._samples_weight

        if self._memory_weight > 0:
            score += self._memory_weight * log_penalty(
                memory_ratio, alpha=self._memory_alpha
            )
            weight_total += self._memory_weight

        if weight_total == 0.0:
            return 0.0

        return score / weight_total

    @override
    def to_spec(self) -> MetricSpec:
        return MetricSpec(
            class_=get_class_path(self.__class__),
            params={
                "rounds_weight": self._rounds_weight,
                "samples_weight": self._samples_weight,
                "memory_weight": self._memory_weight,
                "rounds_alpha": self._rounds_alpha,
                "samples_alpha": self._samples_alpha,
                "memory_alpha": self._memory_alpha,
            },
        )

    @classmethod
    @override
    def from_spec(cls, spec: MetricSpec) -> "ResourceUsageBehaviorMetric":
        params = spec.params or {}
        return cls(**params)

    @staticmethod
    def _usage_ratio(used: float, remaining: float | None) -> float:
        """
        Convert resource usage to a percentage-style value for penalty scaling.

        If `remaining` is provided, the ratio is expressed as a percentage of the
        total budget consumed so far. Otherwise, the raw usage value is returned.
        """
        if remaining is None:
            return float(used)

        total = used + remaining
        if total <= 0.0:
            return 0.0

        return float(used) / total * 100.0
