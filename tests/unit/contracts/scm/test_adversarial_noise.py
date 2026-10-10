"""Analytical distribution properties, independent of serialization symmetry."""

import numpy as np
import pytest

from TheCausalityGame.scm.noise.dirichlet import DirichletNoiseDistribution
from TheCausalityGame.scm.noise.gaussian import GaussianNoiseDistribution
from TheCausalityGame.scm.noise.laplace import LaplaceNoiseDistribution
from TheCausalityGame.scm.noise.uniform import UniformNoiseDistribution


@pytest.mark.parametrize(
    "noise,mean,variance",
    [
        (GaussianNoiseDistribution(mean=3, std=2), 3, 4),
        (UniformNoiseDistribution(low=2, high=8), 5, 3),
        (LaplaceNoiseDistribution(loc=3, scale=2), 3, 8),
    ],
)
def test_noise_matches_analytical_first_two_moments(noise, mean, variance):
    values = noise.generate(100_000, random_state=731)
    assert abs(values.mean() - mean) < 0.04
    assert abs(values.var() - variance) < 0.2


def test_dirichlet_owns_concentration_configuration():
    alpha = [2.0, 6.0]
    noise = DirichletNoiseDistribution(alpha)
    alpha[:] = [6.0, 2.0]
    values = noise.generate(20_000, random_state=731)
    np.testing.assert_allclose(values.sum(axis=1), 1.0, atol=1e-12)
    assert abs(values[:, 0].mean() - 0.25) < 0.01
