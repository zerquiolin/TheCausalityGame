"""The Causality Game - Mission contract."""

from __future__ import annotations

from abc import abstractmethod

from TheCausalityGame.core.contracts.dto.transcript import Transcript
from TheCausalityGame.core.contracts.metric import BehaviorMetric, ResultMetric
from TheCausalityGame.core.contracts.result_validator import ResultValidator
from TheCausalityGame.core.contracts.scm import SCM
from TheCausalityGame.core.contracts.serializable import Serializable


class Mission(Serializable):
    """
    Contract for defining a mission in The Causality Game.

    A mission defines:
      - the behavior and result metrics to evaluate agents,
      - a result validator to enforce output compatibility,
      - and the evaluation logic for scoring agents.

    Attributes
    ----------
    id : str
        Unique identifier for the mission.
    name : str
        Human-readable mission name.
    description : str
        Description of the mission's purpose.
    is_mounted : bool
        Whether the mission has been initialized with an SCM.
    behavior_metric : BehaviorMetric
        Metric used to assess agent behavior during the run.
    result_metric : ResultMetric
        Metric used to score the agent's final output.
    result_validator : ResultValidator
        Validator to check agent outputs before evaluation.
    """

    id: str
    name: str
    description: str

    is_mounted: bool = False

    def __init__(
        self,
        behavior_metric: BehaviorMetric,
        result_metric: ResultMetric,
        result_validator: ResultValidator,
    ) -> None:
        """Initialize the mission with specified metrics and validator."""
        self.behavior_metric = behavior_metric
        self.result_metric = result_metric
        self.result_validator = result_validator

        # Ensure compatibility between the result metric and validator
        assert self.result_validator.kind in self.result_metric.kinds, (
            f"Result Metric kinds {self.result_metric.kinds} "
            f"are not compatible with Result Validator kind {self.result_validator.kind}."
        )

    def mount(self, scm: SCM) -> None:
        """
        Initialize the mission by attaching a structural causal model (SCM).

        This also mounts the behavior and result metrics.

        Parameters
        ----------
        scm : SCM
            The structural causal model used for evaluating the agent.
        """
        self.behavior_metric.mount(scm)
        self.result_metric.mount(scm)
        self.is_mounted = True

    @abstractmethod
    def evaluate(self, transcript: Transcript) -> tuple[float, float]:
        """
        Evaluate the agent using the provided transcript.

        Parameters
        ----------
        transcript : Transcript
            The complete interaction history from a single agent run.

        Returns
        -------
        tuple[float, float]
            A tuple of (behavior_score, result_score).



        Raises
        ------
        NotImplementedError
            If the method is not overridden in a concrete subclass.
        """
        raise NotImplementedError(
            "The evaluate method must be implemented in the derived class."
        )
