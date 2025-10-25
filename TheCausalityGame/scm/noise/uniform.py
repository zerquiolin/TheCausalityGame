"""The Causality Game - Uniform Noise Distribution."""

from typing import override

import numpy as np
from scipy.stats import uniform  # type: ignore

from TheCausalityGame.core.contracts.noise import NoiseDistribution
from TheCausalityGame.core.contracts.specs.noise import NoiseDistributionSpec
from TheCausalityGame.core.infrastructure.registry import get_class_path


class UniformNoiseDistribution(NoiseDistribution):
    """
    Uniform noise distribution implementation.

    This class generates random noise from a uniform distribution within the
    interval [low, high). Useful for SCM stochasticity or synthetic data generation.

    Parameters
    ----------
    low : float, optional
        Lower bound of the distribution (inclusive). Default is -1.0.
    high : float, optional
        Upper bound of the distribution (exclusive). Default is 1.0.
    """

    def __init__(self, low: float = -1.0, high: float = 1.0) -> None:
        self.low = low
        self.high = high

    @override
    def generate(self, size: int, random_state: int | None = 911) -> np.ndarray:
        """
        Generate uniform noise samples.

        Parameters
        ----------
        size : int
            Number of samples to generate.
        random_state : int, optional
            Seed for reproducibility. Default is 911.

        Returns
        -------
        float | np.ndarray
            Array of sampled noise values from Uniform[low, high).
        """
        return uniform.rvs(  # type: ignore
            loc=self.low,
            scale=self.high - self.low,
            size=size,
            random_state=random_state,
        )

    @override
    def to_spec(self) -> NoiseDistributionSpec:
        return NoiseDistributionSpec(
            class_=get_class_path(self.__class__),
            params={"low": self.low, "high": self.high},
        )

    @classmethod
    @override
    def from_spec(cls, spec: NoiseDistributionSpec) -> "UniformNoiseDistribution":
        return cls(**spec.params)
