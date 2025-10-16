"""The Causality Game - Metric Spec."""

from pydantic import Field

from TheCausalityGame.core.contracts.specs.common import CommonSpec


class MetricSpec(CommonSpec):
    """Specification for constructing a metric."""


class MetricsSpec(CommonSpec):
    """Canonical metric pair for each mission."""

    behavior: MetricSpec
    result: MetricSpec
    custom: list[MetricSpec] | None = Field(
        default=None,
        description="Optional custom metrics.",
    )
