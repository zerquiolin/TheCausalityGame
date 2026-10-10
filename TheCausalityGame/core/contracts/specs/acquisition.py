"""Acquisition pricing and composable, agent-visible resource stopping rules."""

from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PriceRule(BaseModel):
    """Match a subset of intervention targets; None matches any value of a target."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    treatment: dict[str, int | float | str | None] = Field(default_factory=dict)
    observational: bool | None = None
    per_experiment: float = Field(default=0, ge=0)
    per_sample: float = Field(default=0, ge=0)


class AcquisitionPricing(BaseModel):
    """Use the most specific matching price; conflicting equally specific rules fail."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    per_experiment: float = Field(default=0, ge=0)
    per_sample: float = Field(default=0, ge=0)
    rules: list[PriceRule] = Field(default_factory=list)

    def quote(self, treatment: dict | None, n: int) -> float:
        """Price one experiment without consuming resources."""
        arms = treatment or {}
        matches = []
        for rule in self.rules:
            if rule.observational is not None and rule.observational != (not arms):
                continue
            if all(
                key in arms and (value is None or arms[key] == value)
                for key, value in rule.treatment.items()
            ):
                specificity = (
                    len(rule.treatment),
                    sum(v is not None for v in rule.treatment.values()),
                    int(rule.observational is not None),
                )
                matches.append((specificity, rule))
        if not matches:
            return float(
                Decimal(str(self.per_experiment)) + n * Decimal(str(self.per_sample))
            )
        most = max(s for s, _ in matches)
        prices = {(r.per_experiment, r.per_sample) for s, r in matches if s == most}
        if len(prices) != 1:
            raise ValueError("Equally specific acquisition price rules disagree.")
        fixed, sample = prices.pop()
        return float(Decimal(str(fixed)) + n * Decimal(str(sample)))


class StopCondition(BaseModel):
    """Stop when a used-resource threshold or all/any child conditions are met."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    mode: Literal["threshold", "all", "any"] = "threshold"
    resource: (
        Literal["rounds", "samples", "experiments", "cost", "time_s", "memory_mb"]
        | None
    ) = None
    threshold: float | None = Field(default=None, ge=0)
    conditions: list[StopCondition] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_form(self) -> StopCondition:
        """Reject ambiguous or empty stopping expressions."""
        if self.mode == "threshold":
            if self.resource is None or self.threshold is None or self.conditions:
                raise ValueError(
                    "A threshold needs resource/threshold and no children."
                )
        elif (
            not self.conditions
            or self.resource is not None
            or self.threshold is not None
        ):
            raise ValueError(
                "A composite stop needs nonempty children and no threshold."
            )
        return self

    def reached(self, used: dict[str, float]) -> bool:
        """Evaluate against counters only, without access to latent truth or scores."""
        if self.mode == "threshold":
            return used[self.resource] >= self.threshold
        outcomes = [condition.reached(used) for condition in self.conditions]
        return all(outcomes) if self.mode == "all" else any(outcomes)
