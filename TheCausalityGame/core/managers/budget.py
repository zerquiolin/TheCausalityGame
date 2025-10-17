"""The Causality Game - Budget Manager."""

from __future__ import annotations

import time

from TheCausalityGame.core.contracts.dto.environment import BudgetSnapshot
from TheCausalityGame.core.contracts.specs.budget import BudgetSpec
from TheCausalityGame.core.lib.errors.budget import (
    MemoryBudgetExceededError,
    RoundsBudgetExceededError,
    SamplesBudgetExceededError,
    TimeBudgetExceededError,
)


class BudgetManager:
    """Manages and enforces runtime budgets (time, rounds, samples, memory).

    Usage
    -----
    manager = BudgetManager(budget_spec)
    manager.start_time()

    for _ in range(rounds):
        manager.tick_round()
        manager.check_time()
        ...
        manager.charge_samples(n)
        manager.charge_memory(bytes_used)

    Snapshot
    --------
    manager.snapshot() -> BudgetSnapshot
    """

    BYTES_PER_MB = 1024.0 * 1024.0

    def __init__(self, budget_spec: BudgetSpec) -> None:
        # Hard limits
        self.rounds_limit: int = budget_spec.rounds
        self.time_limit_s: float | None = budget_spec.time_s
        self.sample_limit: int | None = budget_spec.samples
        self.memory_mb_limit: float | None = budget_spec.memory_mb

        # Counters
        self.rounds_used: int = 0
        self.samples_used: int = 0
        self.memory_used_mb: float = 0.0

        # Timing
        self._t0: float = time.perf_counter()
        self._t1: float | None = None
        self._paused_s: float = 0.0

    # ---- Time control ----
    def start_time(self) -> None:
        """Reset internal timer to current time."""
        self._t0 = time.perf_counter()
        self._t1 = None
        self._paused_s = 0.0

    def pause_time(self) -> None:
        """Pause the internal timer."""
        if self._t1 is None:
            self._t1 = time.perf_counter()

    def resume_time(self) -> None:
        """Resume internal timer after pause."""
        if self._t1 is not None:
            self._paused_s += time.perf_counter() - self._t1
            self._t1 = None

    # ---- Budget calculations ----
    def seconds_elapsed(self) -> float:
        """Return elapsed wall-clock seconds since timer start."""
        paused = self._paused_s
        if self._t1 is not None:
            paused += time.perf_counter() - self._t1
        return time.perf_counter() - self._t0 - paused

    def seconds_left(self) -> float | None:
        """Return remaining seconds (None if unlimited)."""
        if self.time_limit_s is None:
            return None
        return max(0.0, self.time_limit_s - self.seconds_elapsed())

    def rounds_left(self) -> int:
        """Return remaining rounds (non-negative)."""
        return max(0, self.rounds_limit - self.rounds_used)

    def samples_left(self) -> int | None:
        """Return remaining sample allowance (None if unlimited)."""
        if self.sample_limit is None:
            return None
        return max(0, self.sample_limit - self.samples_used)

    def memory_mb_left(self) -> float | None:
        """Return remaining memory allowance in MB (None if unlimited)."""
        if self.memory_mb_limit is None:
            return None
        return max(0.0, self.memory_mb_limit - self.memory_used_mb)

    # ---- Enforcement ----
    def check_time(self) -> None:
        """Raise if the time budget is exhausted."""
        if self.time_limit_s is not None and self.seconds_elapsed() > self.time_limit_s:
            raise TimeBudgetExceededError(
                used=self.seconds_elapsed(), allowed=self.time_limit_s
            )

    def tick_round(self) -> None:
        """Consume one round; raise if beyond limit."""
        self.rounds_used += 1
        if self.rounds_used > self.rounds_limit:
            raise RoundsBudgetExceededError(
                used=self.rounds_used, allowed=self.rounds_limit
            )

    def charge_samples(self, n: int) -> None:
        """
        Accrue sample usage; raise if beyond limit.

        Parameters
        ----------
        n : int
            Number of samples to charge.
        """
        if n <= 0:
            return
        self.samples_used += n
        if self.sample_limit is not None and self.samples_used > self.sample_limit:
            raise SamplesBudgetExceededError(
                used=self.samples_used, allowed=self.sample_limit
            )

    def charge_memory(self, bytes_used: int | float) -> None:
        """
        Accrue memory usage (in bytes); raise if beyond limit.

        Parameters
        ----------
        bytes_used : int | float
            Number of bytes to charge.
        """
        if bytes_used <= 0:
            return
        self.memory_used_mb += float(bytes_used) / self.BYTES_PER_MB
        if (
            self.memory_mb_limit is not None
            and self.memory_used_mb > self.memory_mb_limit
        ):
            raise MemoryBudgetExceededError(
                used_mb=self.memory_used_mb, allowed_mb=self.memory_mb_limit
            )

    # ---- Snapshot ----
    def snapshot(self) -> BudgetSnapshot:
        """Return a JSON-safe snapshot of the current budget state."""
        return BudgetSnapshot(
            rounds_left=self.rounds_left(),
            time_s_left=self.seconds_left(),
            samples_left=self.samples_left(),
            memory_mb_left=self.memory_mb_left(),
        )
