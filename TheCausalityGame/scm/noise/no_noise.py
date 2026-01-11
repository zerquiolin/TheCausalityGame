"""The Causality Game - Uniform Noise Distribution."""

from __future__ import annotations

from typing import override

import numpy as np

from TheCausalityGame.core.contracts.noise import NoiseDistribution
from TheCausalityGame.core.contracts.specs.noise import NoiseDistributionSpec
from TheCausalityGame.core.infrastructure.registry import get_class_path


class NoNoiseDistribution(NoiseDistribution):
    """
    Uniform noise distribution implementation.

    This class generates no noise.
    Useful for testing purposes.
    """

    def __init__(self) -> None:
        pass

    @override
    def generate(self, size: int, random_state: int | None = 911) -> np.ndarray:
        """
        Zeros noise generation.

        Parameters
        ----------
        size : int
            Number of samples to generate.
        random_state : int, optional
            Seed for reproducibility. Default is 911.

        Returns
        -------
        float | np.ndarray
            Array of sampled noise values from zeros.
        """
        return np.zeros(size)

    @override
    def to_spec(self) -> NoiseDistributionSpec:
        return NoiseDistributionSpec(
            class_=get_class_path(self.__class__),
        )

    @classmethod
    @override
    def from_spec(cls, spec: NoiseDistributionSpec) -> NoNoiseDistribution:
        return cls()
