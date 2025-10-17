"""The Causality Game - Metric contract."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from TheCausalityGame.core.contracts.dto.transcript import Transcript
from TheCausalityGame.core.contracts.scm import SCM
from TheCausalityGame.core.contracts.serializable import Serializable


class Metric(Serializable):
    """
    Base class for all metrics used in The Causality Game.

    This abstract class defines the interface for evaluating agent performance.
    Subclasses must implement `mount()` and `evaluate()` methods.

    Attributes
    ----------
    name : str
        Human-readable name of the metric.
    description : str
        Description of what the metric measures.
    is_mounted : bool
        Indicates whether the metric has been mounted with an SCM.
    """

    name: str
    description: str
    is_mounted = False

    @abstractmethod
    def mount(self, scm: SCM) -> None:
        """
        Prepare the metric for evaluation (e.g., store SCM context).

        Called once after deserialization.

        Parameters
        ----------
        scm : SCM
            The structural causal model used during the evaluation.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, *args: Any, **kwargs: Any) -> float:  # noqa :ANN401
        """
        Compute the metric score.

        Returns
        -------
        float
            Numeric score representing agent performance.
        """
        raise NotImplementedError


class BehaviorMetric(Metric):
    """
    Abstract metric that evaluates agent behavior (e.g., efficiency, compliance).

    Subclasses must implement `evaluate(transcript)`.
    """

    @abstractmethod
    def evaluate(self, transcript: Transcript) -> float:
        """
        Compute the behavioral score from the full game transcript.

        Parameters
        ----------
        transcript : Transcript
            The full record of the game session.

        Returns
        -------
        float
            Behavior score.
        """
        raise NotImplementedError


class ResultMetric(Metric):
    """
    Abstract metric that evaluates the final result or answer of an agent.

    Supports multiple evaluation kinds (e.g., 'mse', 'loglikelihood').

    Attributes
    ----------
    kinds : list of str
        List of supported evaluation modes.
    """

    kinds: list[str]

    @abstractmethod
    def evaluate(self, kind: str, result: Any) -> float:  # noqa :ANN401
        """
        Compute a result-based score.

        Parameters
        ----------
        kind : str
            The type of result evaluation (e.g., 'mse').
        result : Any
            The agent's output or prediction to score.

        Returns
        -------
        float
            Result score.
        """
        raise NotImplementedError
