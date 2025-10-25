"""The Causality Game - No Noise Distribution."""

from __future__ import annotations

from typing import override

import numpy as np
from scipy.stats import norm  # type: ignore

from TheCausalityGame.core.contracts.noise import NoiseDistribution
from TheCausalityGame.core.contracts.specs.noise import NoiseDistributionSpec
from TheCausalityGame.core.infrastructure.registry import get_class_path


class GaussianNoiseDistribution(NoiseDistribution):
    """
    Uniform noise distribution implementation.

    This class generates random noise from a Gaussian distribution with mean and std.
    Useful for SCM stochasticity or synthetic data generation.

    Parameters
    ----------
    mean: float
        Mean of the Gaussian distribution.
    std: float
        Standard deviation of the Gaussian distribution.
    """

    def __init__(self, mean: float = 0, std: float = 1.0) -> None:
        self.mean = mean
        self.std = std

    @override
    def generate(self, size: int, random_state: int | None = 911) -> np.ndarray:
        """
        Generate gaussian (normal) noise samples.

        Parameters
        ----------
        size : int
            Number of samples to generate.
        random_state : int, optional
            Seed for reproducibility. Default is 911.

        Returns
        -------
        float | np.ndarray
            Array of sampled noise values from Norm[low, high).
        """
        return norm.rvs(  # type: ignore
            loc=self.mean,
            scale=self.std,
            size=size,
            random_state=random_state,
        )

    @override
    def to_spec(self) -> NoiseDistributionSpec:
        return NoiseDistributionSpec(
            class_=get_class_path(self.__class__),
            params={"mean": self.mean, "std": self.std},
        )

    @classmethod
    @override
    def from_spec(cls, spec: NoiseDistributionSpec) -> GaussianNoiseDistribution:
        return cls(**spec.params)
