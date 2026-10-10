"""Simplex-valued Dirichlet noise for categorical score vectors."""

from __future__ import annotations

import numpy as np

from TheCausalityGame.core.contracts.noise import NoiseDistribution
from TheCausalityGame.core.contracts.specs.noise import NoiseDistributionSpec
from TheCausalityGame.core.infrastructure.registry import get_class_path


class DirichletNoiseDistribution(NoiseDistribution):
    """Draw a probability vector per row; this is not scalar numerical-node noise."""

    def __init__(self, alpha: list[float] | None = None) -> None:
        self.alpha = [1.0, 1.0] if alpha is None else list(alpha)
        if not self.alpha or not np.isfinite(self.alpha).all() or min(self.alpha) <= 0:
            raise ValueError(
                "Dirichlet concentration must be a nonempty positive finite vector."
            )

    def generate(
        self, size: int | tuple, random_state: int | np.random.RandomState | None = 911
    ) -> np.ndarray:
        rs = (
            random_state
            if isinstance(random_state, np.random.RandomState)
            else np.random.RandomState(random_state)
        )
        if isinstance(size, tuple):
            if len(size) != 2 or size[1] != len(self.alpha):
                raise ValueError("Dirichlet shape must be (rows, categories).")
            size = size[0]
        return rs.dirichlet(self.alpha, size=size)

    def to_spec(self) -> NoiseDistributionSpec:
        return NoiseDistributionSpec(
            class_=get_class_path(type(self)), params={"alpha": self.alpha}
        )

    @classmethod
    def from_spec(cls, spec: NoiseDistributionSpec) -> DirichletNoiseDistribution:
        return cls(**(spec.params or {}))
