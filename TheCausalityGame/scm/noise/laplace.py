"""Scalar Laplace noise for non-Gaussian structural mechanisms."""

from __future__ import annotations

import numpy as np

from TheCausalityGame.core.contracts.noise import NoiseDistribution
from TheCausalityGame.core.contracts.specs.noise import NoiseDistributionSpec
from TheCausalityGame.core.infrastructure.registry import get_class_path


class LaplaceNoiseDistribution(NoiseDistribution):
    """Generate independent Laplace noise with finite location and positive scale."""

    def __init__(self, loc: float = 0, scale: float = 1) -> None:
        if not np.isfinite([loc, scale]).all() or scale <= 0:
            raise ValueError("Laplace loc must be finite and scale positive.")
        self.loc, self.scale = loc, scale

    def generate(
        self, size: int | tuple, random_state: int | np.random.RandomState | None = 911
    ) -> np.ndarray:
        rs = (
            random_state
            if isinstance(random_state, np.random.RandomState)
            else np.random.RandomState(random_state)
        )
        return rs.laplace(self.loc, self.scale, size=size)

    def to_spec(self) -> NoiseDistributionSpec:
        return NoiseDistributionSpec(
            class_=get_class_path(type(self)),
            params={"loc": self.loc, "scale": self.scale},
        )

    @classmethod
    def from_spec(cls, spec: NoiseDistributionSpec) -> LaplaceNoiseDistribution:
        return cls(**(spec.params or {}))
