"""Tests for the Budget management utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from TheCausalityGame.core.contracts.dto.environment import BudgetSnapshot
from TheCausalityGame.core.contracts.specs.budget import BudgetSpec
from TheCausalityGame.core.lib.errors.budget import (
    MemoryBudgetExceededError,
    RoundsBudgetExceededError,
    SamplesBudgetExceededError,
    TimeBudgetExceededError,
)
from TheCausalityGame.core.managers.budget import BudgetManager


def simulate_dataframe_memory(rows: int = 100, cols: int = 5) -> pd.DataFrame:
    """Generate a DataFrame with `rows x cols` of float64s to simulate memory use."""
    df = pd.DataFrame(
        np.random.default_rng().random((rows, cols)),
        columns=[f"col{i}" for i in range(cols)],
    )
    return df


class _PerfCounter:
    """Deterministic perf counter replacement for time-dependent tests."""

    def __init__(self, start: float = 1_000.0) -> None:
        self._current = start

    def advance(self, delta: float) -> None:
        self._current += delta

    def __call__(self) -> float:
        return self._current


@pytest.fixture
def perf_counter(monkeypatch: pytest.MonkeyPatch) -> _PerfCounter:
    """Fixture to mock time.perf_counter for deterministic time tests."""
    perf = _PerfCounter()
    monkeypatch.setattr("TheCausalityGame.core.managers.budget.time.perf_counter", perf)
    return perf


def test_simulate_dataframe_memory_shape_and_dtype() -> None:
    """Test that the simulated DataFrame has the correct shape and dtypes."""
    df = simulate_dataframe_memory(rows=10, cols=3)

    assert df.shape == (10, 3)
    assert list(df.columns) == ["col0", "col1", "col2"]
    assert (df.dtypes == "float64").all()  # type: ignore


def test_charge_samples_respects_limits() -> None:
    """Test that charging samples respects the budget limits."""
    manager = BudgetManager(BudgetSpec(rounds=1, samples=5))

    manager.charge_samples(2)
    assert manager.samples_used == 2  # noqa: PLR2004
    assert manager.samples_left() == 3  # noqa: PLR2004

    manager.charge_samples(0)
    manager.charge_samples(-4)
    assert manager.samples_used == 2  # noqa: PLR2004

    manager.charge_samples(3)
    assert manager.samples_used == 5  # noqa: PLR2004

    with pytest.raises(SamplesBudgetExceededError):
        manager.charge_samples(1)


def test_charge_memory_respects_limits() -> None:
    """Test that charging memory respects the budget limits."""
    manager = BudgetManager(BudgetSpec(rounds=1, memory_mb=0.002))

    manager.charge_memory(0.001 * manager.BYTES_PER_MB)
    assert manager.memory_used_mb == pytest.approx(0.001)  # type: ignore
    assert manager.memory_mb_left() == pytest.approx(0.001, rel=1e-3, abs=1e-6)  # type: ignore

    manager.charge_memory(0)
    manager.charge_memory(-128)
    assert manager.memory_used_mb == pytest.approx(0.001)  # type: ignore

    manager.charge_memory(0.001 * manager.BYTES_PER_MB)
    assert manager.memory_mb_left() == pytest.approx(0.0, abs=1e-9)  # type: ignore

    with pytest.raises(MemoryBudgetExceededError):
        manager.charge_memory(1)


def test_tick_round_respects_limit() -> None:
    """Test that ticking rounds respects the budget limits."""
    manager = BudgetManager(BudgetSpec(rounds=2))

    manager.tick_round()
    assert manager.rounds_used == 1
    assert manager.rounds_left() == 1

    manager.tick_round()
    assert manager.rounds_used == 2  # noqa: PLR2004
    assert manager.rounds_left() == 0

    with pytest.raises(RoundsBudgetExceededError):
        manager.tick_round()


def test_check_time_respects_pause_and_resume(perf_counter: _PerfCounter) -> None:
    """Test that time checking respects pause and resume functionality."""
    manager = BudgetManager(BudgetSpec(rounds=1, time_s=2.0))
    manager.start_time()

    perf_counter.advance(0.9)
    manager.check_time()

    manager.pause_time()
    perf_counter.advance(10.0)
    manager.check_time()

    manager.resume_time()
    perf_counter.advance(1.2)
    with pytest.raises(TimeBudgetExceededError):
        manager.check_time()


def test_budget_manager_flow_enforces_limits(perf_counter: _PerfCounter) -> None:
    """Test the overall flow of BudgetManager and enforcement of limits."""
    spec = BudgetSpec(rounds=2, time_s=1.0, samples=5, memory_mb=0.001)
    manager = BudgetManager(spec)
    manager.start_time()

    snapshot = manager.snapshot()
    assert isinstance(snapshot, BudgetSnapshot)
    assert snapshot.rounds_left == 2  # noqa: PLR2004
    assert snapshot.samples_left == 5  # noqa: PLR2004
    assert snapshot.memory_mb_left == pytest.approx(0.001)  # type: ignore
    assert snapshot.time_s_left == pytest.approx(1.0)  # type: ignore

    manager.tick_round()
    snapshot = manager.snapshot()
    assert snapshot.rounds_left == 1

    manager.charge_samples(3)
    assert manager.samples_left() == 2  # noqa: PLR2004
    assert manager.snapshot().samples_left == 2  # noqa: PLR2004

    half_limit_bytes = 0.0005 * manager.BYTES_PER_MB
    manager.charge_memory(half_limit_bytes)
    assert manager.memory_mb_left() == pytest.approx(0.0005, rel=1e-3, abs=1e-6)  # type: ignore

    with pytest.raises(MemoryBudgetExceededError):
        manager.charge_memory(0.0006 * manager.BYTES_PER_MB)

    perf_counter.advance(0.5)
    manager.check_time()

    perf_counter.advance(0.6)
    with pytest.raises(TimeBudgetExceededError):
        manager.check_time()

    with pytest.raises(SamplesBudgetExceededError):
        manager.charge_samples(3)

    manager.tick_round()
    assert manager.rounds_left() == 0
    with pytest.raises(RoundsBudgetExceededError):
        manager.tick_round()

    final_snapshot = manager.snapshot()
    assert final_snapshot.rounds_left == 0
    assert final_snapshot.samples_left == 0
    assert final_snapshot.memory_mb_left == 0.0
    assert final_snapshot.time_s_left == 0.0
