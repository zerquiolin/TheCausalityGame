"""The Causality Game - Decision Infrastructure."""

from __future__ import annotations

from typing import Any

from TheCausalityGame.core.contracts.dto.environment import Experiment
from TheCausalityGame.core.lib.enum.environment import ActionKind
from TheCausalityGame.core.lib.errors.environment import DecisionMismatchError

__all__ = ["Decision", "ExperimentLike"]

# An experiment can be passed as either an Experiment object
# or as a tuple of (treatment_dict, sample_count)
ExperimentTuple = tuple[dict[str, int | float | str] | None, int]
ExperimentLike = Experiment | ExperimentTuple


class Decision:
    """
    Represents a decision taken by the agent, either to run experiments or submit an answer.

    Attributes
    ----------
    kind : ActionKind
        The type of action ('experiment' or 'answer').
    experiments : list[Experiment]
        A list of experiments to perform (only relevant if kind == 'experiment').

    Notes
    -----
    - Designed to be immutable in usage (though not enforced at the language level).
    - Does not store random seeds; the environment handles reproducibility.
    """

    kind: ActionKind
    experiments: list[Experiment]

    def __init__(
        self, *, kind: ActionKind, experiments: list[Experiment] | None = None
    ) -> None:
        self.kind = kind
        self.experiments = experiments or []

    # ---------- Factory Constructors ----------

    @classmethod
    def experiment(cls) -> Decision:
        """
        Construct a new Decision of kind 'experiment'.

        Returns
        -------
        Decision
            An empty experiment decision.
        """
        return cls(kind=ActionKind.EXPERIMENT)

    @classmethod
    def answer(cls) -> Decision:
        """
        Construct a new Decision of kind 'answer'.

        Returns
        -------
        Decision
            An answer decision (with no experiments).
        """
        return cls(kind=ActionKind.ANSWER)

    # ---------- Builder Methods ----------

    def add_experiment(self, treatment: dict[str, Any] | None, n: int) -> Decision:
        """
        Add a single experiment to the decision.

        Parameters
        ----------
        treatment : dict[str, Any] | None
            A dictionary of variable names to values, or None for observational.
        n : int
            Number of samples to request for the experiment.

        Returns
        -------
        Decision
            Self, to allow method chaining.

        Raises
        ------
        DecisionMismatchError
            If the decision is not of kind 'experiment'.
        """
        if not self.is_experiment:
            raise DecisionMismatchError()
        self.experiments.append(Experiment(treatment=treatment, n=n))
        return self

    def extend(self, experiments: list[ExperimentLike]) -> Decision:
        """
        Add multiple experiments to the decision.

        Parameters
        ----------
        experiments : list[Experiment | tuple[dict[str, Any] | None, int]]
            A list of experiments, either as objects or tuples.

        Returns
        -------
        Decision
            Self, to allow method chaining.

        Raises
        ------
        DecisionMismatchError
            If the decision is not of kind 'experiment'.
        """
        if not self.is_experiment:
            raise DecisionMismatchError()

        for item in experiments:
            if isinstance(item, Experiment):
                self.experiments.append(item)
            else:
                treatment, n = item
                self.experiments.append(Experiment(treatment=treatment, n=n))

        return self

    # ---------- Properties ----------

    @property
    def is_experiment(self) -> bool:
        """
        Check if the decision is an experiment.

        Returns
        -------
        bool
            Whether the decision is an experiment.
        """
        return self.kind == ActionKind.EXPERIMENT

    @property
    def is_answer(self) -> bool:
        """
        Check if the decision is an answer.

        Returns
        -------
        bool
            Whether the decision is an answer.
        """
        return self.kind == ActionKind.ANSWER

    def __repr__(self) -> str:
        return f"<Decision kind={self.kind!r}, experiments={len(self.experiments)}>"
