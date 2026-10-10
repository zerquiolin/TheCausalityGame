"""Actual acquisition prices and arbitrary linear behavior-metric composition."""

from __future__ import annotations

import math

from TheCausalityGame.core.contracts.metric import BehaviorMetric
from TheCausalityGame.core.contracts.specs.metric import MetricSpec
from TheCausalityGame.core.infrastructure.registry import (
    build_from_spec,
    get_class_path,
)


class AcquisitionCostBehaviorMetric(BehaviorMetric):
    """Sum declared per-experiment and per-sample prices for acquired batches."""

    name = "Acquisition cost"
    description = (
        "Financial acquisition cost in manifest-defined units; lower is better."
    )

    def mount(self, scm) -> None:
        self.is_mounted = True

    def evaluate(self, transcript) -> float:
        return float(
            sum(
                transcript.budget.pricing.quote(batch.interventions, batch.n)
                for entry in transcript.entries
                for batch in (entry.samples_collection or [])
            )
        )

    def to_spec(self) -> MetricSpec:
        return MetricSpec(class_=get_class_path(type(self)))

    @classmethod
    def from_spec(cls, spec: MetricSpec) -> AcquisitionCostBehaviorMetric:
        return cls()


class RoundCountBehaviorMetric(AcquisitionCostBehaviorMetric):
    """Raw attempted interaction rounds, including submission/forced-stop rounds."""

    name = "Round count"
    description = "Number of attempted interaction rounds; lower is better."

    def evaluate(self, transcript) -> float:
        return float(len(transcript.entries))


class LinearBehaviorMetric(BehaviorMetric):
    """Weighted sum of behavior metrics; weights are finite and need not sum to one."""

    name = "Linear behavior score"
    description = "Unnormalized linear combination of declared behavior quantities."

    def __init__(
        self,
        metrics: list[BehaviorMetric] | None = None,
        weights: list[float] | None = None,
    ) -> None:
        metrics = [RoundCountBehaviorMetric()] if metrics is None else metrics
        weights = [1.0] if weights is None else weights
        if (
            not metrics
            or len(metrics) != len(weights)
            or not all(math.isfinite(w) for w in weights)
        ):
            raise ValueError(
                "Linear metrics need matching nonempty components and finite weights."
            )
        if not all(isinstance(metric, BehaviorMetric) for metric in metrics):
            raise TypeError("Linear behavior components must implement BehaviorMetric.")
        self.metrics, self.weights = metrics, weights

    def mount(self, scm) -> None:
        for metric in self.metrics:
            metric.mount(scm)
        self.is_mounted = True

    def evaluate(self, transcript) -> float:
        return float(
            sum(
                weight * metric.evaluate(transcript)
                for metric, weight in zip(self.metrics, self.weights, strict=True)
            )
        )

    def to_spec(self) -> MetricSpec:
        return MetricSpec(
            class_=get_class_path(type(self)),
            params={
                "metrics": [metric.to_spec().model_dump() for metric in self.metrics],
                "weights": self.weights,
            },
        )

    @classmethod
    def from_spec(cls, spec: MetricSpec) -> LinearBehaviorMetric:
        params = spec.params or {}
        return (
            cls(
                metrics=[build_from_spec(metric) for metric in params["metrics"]],
                weights=params["weights"],
            )
            if params
            else cls()
        )
