"""Check new noise families against their defining statistical contracts."""

import numpy as np
import pytest

from TheCausalityGame.core.infrastructure.registry import build_from_spec
from TheCausalityGame.scm.noise.dirichlet import DirichletNoiseDistribution
from TheCausalityGame.scm.noise.laplace import LaplaceNoiseDistribution


def test_laplace_nondefault_roundtrip_and_population_moments():
    noise = LaplaceNoiseDistribution(loc=3.0, scale=2.0)
    restored = build_from_spec(noise.to_spec())
    sample = restored.generate(100_000, random_state=911)
    assert abs(sample.mean() - 3.0) < 0.04
    assert abs(sample.var() - 8.0) < 0.2
    with pytest.raises(ValueError):
        LaplaceNoiseDistribution(scale=0)
    with pytest.raises(ValueError):
        LaplaceNoiseDistribution(loc=float("nan"))


def test_dirichlet_simplex_shape_chunking_and_nondefault_roundtrip():
    noise = build_from_spec(DirichletNoiseDistribution([0.5, 2.0, 4.0]).to_spec())
    stream = np.random.RandomState(911)
    chunked = np.concatenate(
        [noise.generate((7, 3), stream), noise.generate((4, 3), stream)]
    )
    whole = noise.generate((11, 3), 911)
    np.testing.assert_array_equal(chunked, whole)
    np.testing.assert_allclose(whole.sum(axis=1), 1.0)
    assert (whole > 0).all()
    with pytest.raises(ValueError):
        noise.generate((5, 2))
    with pytest.raises(ValueError):
        DirichletNoiseDistribution([1.0, 0.0])
