"""The Causlity Game - Dirac Noise Distribution."""

from __future__ import annotations

from typing import override

import numpy as np

from TheCausalityGame.core.contracts.noise import NoiseDistribution
from TheCausalityGame.core.contracts.specs.noise import NoiseDistributionSpec
from TheCausalityGame.core.infrastructure.registry import get_class_path


class DiracNoiseDistribution(NoiseDistribution):
    """
    Dirac (deterministic) noise distribution.

    This distribution always returns a constant value (i.e., zero variance),
    effectively simulating a deterministic system. Useful for debugging,
    ablations, or testing non-stochastic SCMs.

    Parameters
    ----------
    val : int or float
        Constant value to be returned by the distribution.
    """

    def __init__(self, val: int | float = 0.1) -> None:
        self.val = val

    @override
    def generate(self, size: int, random_state: int | None = 911) -> np.ndarray:
        return np.full(size, self.val)

    @override
    def to_spec(self) -> NoiseDistributionSpec:
        return NoiseDistributionSpec(
            class_=get_class_path(self.__class__),
            params={"val": self.val},
        )

    @classmethod
    @override
    def from_spec(cls, spec: NoiseDistributionSpec) -> DiracNoiseDistribution:
        return cls(**spec.params)
