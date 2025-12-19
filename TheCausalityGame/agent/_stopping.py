"""Utilities for configurable stopping criteria."""

from __future__ import annotations

from dataclasses import dataclass, field

from TheCausalityGame.core.contracts.dto.environment import RoundInfo


@dataclass
class StoppingPolicy:
    """
    Track and evaluate stopping criteria for agents.

    Parameters
    ----------
    max_rounds : int | None
        Maximum round index after which the agent should answer.
    target_score : float | None
        Score threshold that triggers an early stop once met.
    patience : int | None
        Number of consecutive rounds without improvement before stopping.
    tolerance : float
        Numerical tolerance used when comparing floating-point improvements.
    """

    max_rounds: int | None = None
    target_score: float | None = None
    patience: int | None = None
    tolerance: float = 1e-6

    _best_score: float | None = field(default=None, init=False)
    _rounds_since_improvement: int = field(default=0, init=False)
    _target_met: bool = field(default=False, init=False)
    _patience_exceeded: bool = field(default=False, init=False)

    def should_stop_on_round(self, round_info: RoundInfo) -> bool:
        """Return True when the round-based limit has been reached."""
        return self.max_rounds is not None and round_info.round >= self.max_rounds

    def register_progress(self, score: float | None = None) -> bool:
        """
        Update internal state from an optional progress score and return stop signal.
        """
        if score is not None:
            if self.target_score is not None and score >= self.target_score - self.tolerance:
                self._target_met = True

            if self.patience is not None:
                if self._best_score is None or score > self._best_score + self.tolerance:
                    self._best_score = score
                    self._rounds_since_improvement = 0
                else:
                    self._rounds_since_improvement += 1
                    if self._rounds_since_improvement >= self.patience:
                        self._patience_exceeded = True

        return self.should_stop()

    def should_stop(self) -> bool:
        """Return True if any stopping trigger has been activated."""
        return self._target_met or self._patience_exceeded

    def to_params(self) -> dict[str, int | float | None]:
        """Serialize configuration parameters for specs."""
        return {
            "max_rounds": self.max_rounds,
            "target_score": self.target_score,
            "patience": self.patience,
            "tolerance": self.tolerance,
        }
