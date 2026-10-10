"""Protect author-defined RNG precedence and state continuation."""

import numpy as np
import pytest

from TheCausalityGame.core.lib.utils.random_state_serialization import (
    random_state_to_json,
    resolve_scm_random_state,
)


def test_serialized_rng_continues_consumed_state_despite_conflicting_seeds():
    original = np.random.RandomState(150)
    original.normal(size=17)
    restored = resolve_scm_random_state(
        random_state_to_json(original), {"scm": 911, "global": 42}
    )
    np.testing.assert_array_equal(restored.normal(size=12), original.normal(size=12))


@pytest.mark.parametrize(
    "seeds,expected",
    [({"scm": 150, "global": 42}, 150), ({"global": 42}, 42), ({}, 911), (None, 911)],
)
def test_missing_serialized_rng_uses_manifest_then_default(seeds, expected):
    np.testing.assert_array_equal(
        resolve_scm_random_state(None, seeds).uniform(size=12),
        np.random.RandomState(expected).uniform(size=12),
    )


def test_malformed_serialized_rng_does_not_silently_fall_back():
    with pytest.raises(ValueError):
        resolve_scm_random_state("invalid JSON", {"scm": 150})
