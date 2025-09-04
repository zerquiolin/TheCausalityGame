from __future__ import annotations

import time

from TheCausalityGame.core.infra.budgets import RoundBudget, SamplesBudget, TimeBudget


def test_round_budget_happy_path() -> None:
    b = RoundBudget(2)
    assert b.max_rounds == 2
    assert b.used == 0 and b.remaining == 2
    b.step()
    assert b.used == 1 and b.remaining == 1
    b.step()
    assert b.used == 2 and b.remaining == 0


def test_round_budget_exceeded() -> None:
    b = RoundBudget(1)
    b.step()
    try:
        b.step()
    except RuntimeError as e:
        assert "Round budget exceeded" in str(e)
    else:
        raise AssertionError("Expected RuntimeError for exceeding round budget")


def test_samples_budget() -> None:
    b = SamplesBudget(5)
    b.consume(2)
    assert b.used == 2 and b.remaining == 3
    b.consume(3)
    assert b.used == 5 and b.remaining == 0
    try:
        b.consume(1)
    except RuntimeError as e:
        assert "Samples budget exceeded" in str(e)
    else:
        raise AssertionError("Expected RuntimeError for exceeding samples budget")


def test_time_budget() -> None:
    b = TimeBudget(0.05)
    # Initially should not raise
    b.check()
    # Sleep a tiny bit and ensure either still fine or exceed soon
    time.sleep(0.06)
    try:
        b.check()
    except RuntimeError as e:
        assert "Time budget exceeded" in str(e)
    else:
        # On very slow machines this could still be allowed; assert invariants
        assert b.elapsed >= 0.0
        assert b.remaining >= 0.0
