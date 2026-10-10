"""Resource boundaries are specified independently of floating-point rounding."""

import pytest

from TheCausalityGame.core.contracts.dto.environment import Experiment
from TheCausalityGame.core.contracts.specs.acquisition import AcquisitionPricing
from TheCausalityGame.core.contracts.specs.budget import BudgetSpec
from TheCausalityGame.core.lib.errors.budget import SamplesBudgetExceededError
from TheCausalityGame.core.managers.budget import (
    AcquisitionBudgetExceededError,
    BudgetManager,
)


@pytest.mark.parametrize("chunks", [[3], [1, 1, 1], [1, 2]])
def test_decimal_cost_exhaustion_is_batch_independent(chunks):
    manager = BudgetManager(
        BudgetSpec(rounds=10, cost=0.3, pricing=AcquisitionPricing(per_sample=0.1))
    )
    for n in chunks:
        experiments = [Experiment(n=n, treatment=None)]
        cost = manager.preflight(experiments)
        manager.charge_acquisition(experiments, cost)
    assert manager.samples_used == 3
    assert manager.cost_used == 0.3
    assert manager.stop_reason() == "cost"
    with pytest.raises(AcquisitionBudgetExceededError):
        manager.preflight([Experiment(n=1, treatment=None)])


def test_rejected_acquisition_leaves_every_counter_unchanged():
    manager = BudgetManager(BudgetSpec(rounds=10, samples=2, experiments=1, cost=1))
    before = (manager.samples_used, manager.experiments_used, manager.cost_used)
    with pytest.raises(SamplesBudgetExceededError):
        manager.preflight([Experiment(n=3, treatment=None)])
    assert (manager.samples_used, manager.experiments_used, manager.cost_used) == before


def test_unlimited_samples_are_still_accounted():
    manager = BudgetManager(BudgetSpec(rounds=10))
    experiments = [
        Experiment(n=7, treatment=None),
        Experiment(n=11, treatment={"X": 1}),
    ]
    manager.charge_acquisition(experiments, manager.preflight(experiments))
    assert manager.samples_used == 18
    assert manager.experiments_used == 2
    assert manager.snapshot().samples_left is None
