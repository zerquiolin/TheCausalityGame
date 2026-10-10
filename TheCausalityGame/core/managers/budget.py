"""The Causality Game - Budget Manager."""

from __future__ import annotations

import time
from decimal import Decimal

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
        self.spec = budget_spec
        self.experiments_used = 0
        self.cost_used = 0.0
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
        if self.sample_limit is not None and self.samples_used + n > self.sample_limit:
            raise SamplesBudgetExceededError(
                used=self.samples_used + n, allowed=self.sample_limit
            )
        self.samples_used += n

    def preflight(self, experiments: list) -> float:
        """Validate a whole acquisition atomically before drawing any samples."""
        n = sum(experiment.n for experiment in experiments)
        if self.sample_limit is not None and self.samples_used + n > self.sample_limit:
            raise SamplesBudgetExceededError(self.samples_used + n, self.sample_limit)
        if (
            self.spec.experiments is not None
            and self.experiments_used + len(experiments) > self.spec.experiments
        ):
            raise AcquisitionBudgetExceededError("experiments")
        cost = sum(
            (
                Decimal(
                    str(self.spec.pricing.quote(experiment.treatment, experiment.n))
                )
                for experiment in experiments
            ),
            Decimal(0),
        )
        if self.spec.cost is not None and Decimal(str(self.cost_used)) + cost > Decimal(
            str(self.spec.cost)
        ):
            raise AcquisitionBudgetExceededError("cost")
        return float(cost)

    def charge_acquisition(self, experiments: list, cost: float) -> None:
        """Commit a preflighted acquisition after successful sampling."""
        self.charge_samples(sum(experiment.n for experiment in experiments))
        self.experiments_used += len(experiments)
        self.cost_used = float(Decimal(str(self.cost_used)) + Decimal(str(cost)))

    def stop_reason(self) -> str | None:
        """Return the first exhausted hard resource, then evaluate optional stopping."""
        used = {
            "rounds": self.rounds_used,
            "samples": self.samples_used,
            "experiments": self.experiments_used,
            "cost": self.cost_used,
            "time_s": self.seconds_elapsed(),
            "memory_mb": self.memory_used_mb,
        }
        limits = {
            "rounds": self.rounds_limit,
            "samples": self.sample_limit,
            "experiments": self.spec.experiments,
            "cost": self.spec.cost,
            "time": self.time_limit_s,
            "memory": self.memory_mb_limit,
        }
        for resource, limit in limits.items():
            key = {"time": "time_s", "memory": "memory_mb"}.get(resource, resource)
            if limit is not None and used[key] >= limit:
                return resource
        if self.spec.stopping is not None and self.spec.stopping.reached(used):
            return "stopping"
        return None

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
            experiments_left=(
                None
                if self.spec.experiments is None
                else max(0, self.spec.experiments - self.experiments_used)
            ),
            cost_left=(
                None
                if self.spec.cost is None
                else max(0.0, self.spec.cost - self.cost_used)
            ),
        )


class AcquisitionBudgetExceededError(RuntimeError):
    """A requested acquisition would exceed an experiment or financial limit."""

    def __init__(self, resource: str) -> None:
        self.resource = resource
        super().__init__(f"Acquisition exceeds {resource} budget.")
