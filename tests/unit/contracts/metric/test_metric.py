"""The Causality Game - Metric Tests."""

from typing import Any

import pytest

from TheCausalityGame.core.contracts.metric import Metric
from TheCausalityGame.core.contracts.specs.metric import MetricSpec
from TheCausalityGame.core.infrastructure.registry import (
    build_from_spec,
    load_subclasses_from_path,
)
from TheCausalityGame.core.lib.utils.tests import (
    assert_dicts_equal,
)

# Search Classes
base_path = "TheCausalityGame/metric"
classes = load_subclasses_from_path(Metric, base_path)


# Tests
@pytest.fixture(params=classes, scope="module")
def metric_instance(request: Any) -> None:  # noqa: ANN401
    """Test class construction."""
    cls = request.param
    # We assume a no-arg constructor given they have a `mount` method.
    return cls()


def test_metric_serialization_roundtrip(metric_instance: Any) -> None:  # noqa: ANN401
    """Test serialization roundtrip."""
    spec = metric_instance.to_spec()
    assert isinstance(spec, MetricSpec)
    metric2 = build_from_spec(spec)
    assert_dicts_equal(metric_instance.to_dict(), metric2.to_dict())
