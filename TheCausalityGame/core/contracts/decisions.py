# TheCausalityGame/core/contracts/decisions.py
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Union

__all__ = ["ExperimentSpec", "Decision", "ExperimentLike"]


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    """One experiment: (interventions, n).

    Attributes
    ----------
        interventions: Mapping[var, value] for do()-style interventions.
            None or {} means OBSERVATIONAL.
        n: Positive integer number of samples for this experiment.
    """

    interventions: Mapping[str, Any] | None
    n: int

    def __post_init__(self) -> None:
        if not isinstance(self.n, int) or self.n <= 0:
            raise ValueError(f"'n' must be a positive int, got {self.n!r}")
        if self.interventions is not None and not isinstance(
            self.interventions, Mapping
        ):
            raise TypeError("'interventions' must be a mapping or None")

    @property
    def is_observational(self) -> bool:
        return not self.interventions or len(self.interventions) == 0


ExperimentTuple = tuple[Mapping[str, Any] | None, int]
ExperimentLike = Union[ExperimentSpec, ExperimentTuple]


@dataclass(frozen=True, slots=True)
class Decision:
    """Agent decision with a two-action surface.

    kind:
        - "experiment" -> carries a tuple of ExperimentSpec (immutable).
        - "answer"     -> no extra fields; env will call agent.answer().

    Notes
    -----
        - Seeds are *not* part of the decision; the environment derives them.
        - Immutable by design for safety & reproducibility.
    """

    kind: Literal["experiment", "answer"]
    experiments: tuple[ExperimentSpec, ...] = field(default_factory=tuple)

    # ---------- factories ----------

    @classmethod
    def experiment(
        cls, *items: ExperimentLike | Iterable[ExperimentLike]
    ) -> Decision:
        """Create an experiment decision from heterogeneous inputs.

        Accepts any mix of:
          - ExperimentSpec
          - (interventions, n) tuples
          - Iterables containing either of the above

        Examples
        --------
            Decision.experiment((None, 500))
            Decision.experiment(ExperimentSpec({'X':1}, 200))
            Decision.experiment((None, 100), ({'X':1}, 100))
            Decision.experiment([({'X':1}, 50), (None, 50)])
        """
        specs: list[ExperimentSpec] = []
        for item in items:
            if isinstance(item, ExperimentSpec):
                specs.append(item)
            elif _is_tuple_like_experiment(item):
                iv, n = item  # type: ignore[misc]
                specs.append(ExperimentSpec(interventions=iv, n=n))
            # Allow nested iterables
            elif isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
                for sub in item:
                    if isinstance(sub, ExperimentSpec):
                        specs.append(sub)
                    elif _is_tuple_like_experiment(sub):
                        iv2, n2 = sub  # type: ignore[misc]
                        specs.append(ExperimentSpec(interventions=iv2, n=n2))
                    else:
                        raise TypeError(_bad_type_msg(sub))
            else:
                raise TypeError(_bad_type_msg(item))

        if not specs:
            raise ValueError("kind='experiment' requires at least one experiment spec")

        return cls(kind="experiment", experiments=tuple(specs))

    @classmethod
    def answer(cls) -> Decision:
        """Create an answer decision (no additional fields)."""
        return cls(kind="answer")

    # ---------- immutable builder helpers ----------

    def add_experiment(
        self, interventions: Mapping[str, Any] | None, n: int
    ) -> Decision:
        """Return a *new* decision with one additional experiment appended."""
        if self.kind != "experiment":
            raise ValueError("Can only add experiments when kind='experiment'")
        new_spec = ExperimentSpec(interventions=interventions, n=n)
        return replace(self, experiments=self.experiments + (new_spec,))

    def extend(self, more: Iterable[ExperimentLike]) -> Decision:
        """Return a *new* decision with multiple experiments appended."""
        if self.kind != "experiment":
            raise ValueError("Can only extend experiments when kind='experiment'")
        extra: list[ExperimentSpec] = []
        for item in more:
            if isinstance(item, ExperimentSpec):
                extra.append(item)
            elif _is_tuple_like_experiment(item):
                iv, n = item  # type: ignore[misc]
                extra.append(ExperimentSpec(interventions=iv, n=n))
            else:
                raise TypeError(_bad_type_msg(item))
        if not extra:
            return self
        return replace(self, experiments=self.experiments + tuple(extra))

    # ---------- predicates ----------

    @property
    def is_experiment(self) -> bool:
        return self.kind == "experiment"

    @property
    def is_answer(self) -> bool:
        return self.kind == "answer"


# ---------- helpers ----------


def _is_tuple_like_experiment(x: object) -> bool:
    if not isinstance(x, tuple) or len(x) != 2:
        return False
    iv, n = x
    if iv is not None and not isinstance(iv, Mapping):
        return False
    return isinstance(n, int) and n > 0


def _bad_type_msg(obj: object) -> str:
    return (
        "Experiment inputs must be ExperimentSpec, "
        "(interventions: Mapping[str, Any] | None, n: int>0) tuples, "
        "or iterables of those; got "
        f"{type(obj).__name__!s}"
    )
