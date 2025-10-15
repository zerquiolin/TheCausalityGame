from __future__ import annotations

import time
from dataclasses import dataclass, field

from TheCausalityGame.core.contracts.dto.environment import BudgetSnapshot
from TheCausalityGame.core.contracts.specs.budget import BudgetSpec


@dataclass(slots=True)
class BudgetState:
    """Mutable runtime budget state.

    Attributes
    ----------
        hard_round_limit: Maximum number of rounds allowed.
        time_limit_s: Optional wall-clock time limit (seconds).
        sample_limit: Optional cap on cumulative samples generated.
        memory_mb_limit: Optional cap on cumulative memory used by produced samples (MB).

        rounds_used: Running count of rounds consumed.
        samples_used: Running count of samples generated.
        memory_used_mb: Running total of memory (in MB) charged against the budget.
    """

    rounds_limit: int
    time_limit_s: float | None = None
    sample_limit: int | None = None
    memory_mb_limit: float | None = None

    # live counters
    rounds_used: int = 0
    samples_used: int = 0
    memory_used_mb: float = 0.0

    # timing
    _t0: float = field(default_factory=time.perf_counter, init=False, repr=False)
    _t1: float | None = field(default=None, init=False, repr=False)
    _t2: float | None = field(default=None, init=False, repr=False)
    _paused_s: float = 0.0

    def start_time(self) -> None:
        """Reset the internal timer to now."""
        self._t0 = time.perf_counter()

    def pause_time(self) -> None:
        """Pause the internal timer."""
        self._t1 = time.perf_counter()

    def resume_time(self) -> None:
        """Resume the internal timer."""
        if self._t1 is not None:
            self._paused_s += time.perf_counter() - self._t1
            self._t1 = None

    # ---- derived views ----
    def seconds_elapsed(self) -> float:
        """Return seconds elapsed since state creation."""
        return (
            time.perf_counter()
            - self._t0
            - (
                self._paused_s
                if self._t1 is None
                else (self._paused_s + (time.perf_counter() - self._t1))
            )
        )

    def seconds_left(self) -> float | None:
        """Return seconds left (None if unlimited)."""
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


class BudgetExceededError(RuntimeError):
    """Raised when a hard budget is exceeded."""


class BudgetEnforcer:
    """Enforces time, round, sample, and memory budgets.

    Usage pattern in the environment:
      - call `tick_round()` at the *start* of each new round
      - call `check_time()` opportunistically (before/after actions)
      - call `charge_samples(n)` after experiments that produce rows
      - call `charge_memory(bytes_used)` after estimating dataset memory
    """

    BYTES_PER_MB = 1024.0 * 1024.0

    def __init__(self, budget_spec: BudgetSpec) -> None:
        self._s = BudgetState(
            rounds_limit=budget_spec.rounds,
            time_limit_s=budget_spec.time_s,
            sample_limit=budget_spec.samples,
            memory_mb_limit=budget_spec.memory_mb,
        )

    @property
    def rounds_limit(self) -> int:
        """Return the hard round limit."""
        return self._s.rounds_limit

    # ---- resets ----
    def start_time(self) -> None:
        """Reset the internal timer to now."""
        self._s.start_time()

    def pause_time(self) -> None:
        """Pause the internal timer."""
        self._s.pause_time()

    def resume_time(self) -> None:
        """Resume the internal timer."""
        self._s.resume_time()

    # ---- checks & charges ----
    def check_time(self) -> None:
        """Raise if time budget is exhausted."""
        if self._s.time_limit_s is None:
            return
        if self._s.seconds_elapsed() > self._s.time_limit_s:
            raise BudgetExceededError("Time budget exhausted")

    def tick_round(self) -> None:
        """Consume one round; raise if beyond limit."""
        self._s.rounds_used += 1
        if self._s.rounds_used > self._s.rounds_limit:
            raise BudgetExceededError("Round budget exhausted")

    def charge_samples(self, n: int) -> None:
        """Accrue sample usage; raise if beyond limit."""
        if n <= 0:
            return
        self._s.samples_used += n
        if (
            self._s.sample_limit is not None
            and self._s.samples_used > self._s.sample_limit
        ):
            raise BudgetExceededError("Sample budget exhausted")

    def charge_memory(self, bytes_used: int | float) -> None:
        """Accrue memory usage (bytes); raise if beyond limit."""
        if bytes_used <= 0:
            return
        self._s.memory_used_mb += float(bytes_used) / self.BYTES_PER_MB
        if (
            self._s.memory_mb_limit is not None
            and self._s.memory_used_mb > self._s.memory_mb_limit
        ):
            raise BudgetExceededError("Memory budget exhausted")

    # ---- snapshots ----
    def snapshot(self) -> BudgetSnapshot:
        """Return a JSON-safe snapshot for RoundInfo/bindings."""
        return BudgetSnapshot(
            rounds_left=self._s.rounds_left(),
            time_s_left=self._s.seconds_left(),
            samples_left=self._s.samples_left(),
            memory_mb_left=self._s.memory_mb_left(),
        )
