"""The Causality Game - Mission contract."""

from __future__ import annotations

from abc import abstractmethod

from TheCausalityGame.core.contracts.metric import BehaviorMetric, ResultMetric
from TheCausalityGame.core.contracts.result_validator import ResultValidator
from TheCausalityGame.core.contracts.scm import SCM
from TheCausalityGame.core.contracts.serializable import Serializable
from TheCausalityGame.core.dto.transcript import Transcript


class Mission(Serializable):
    """Mission contract."""

    name: str
    description: str

    is_mounted: bool = False  # True if the mission is mounted (not just a reference)

    def __init__(
        self,
        behavior_metric: BehaviorMetric,
        result_metric: ResultMetric,
        result_validator: ResultValidator,
    ):
        self.behavior_metric = behavior_metric
        self.result_metric = result_metric
        self.result_validator = result_validator
        # Assert the result metric is compatible with the result validator kind.
        assert self.result_validator.kind in self.result_metric.kinds, (
            f"Result Metric kinds {self.result_metric.kinds} "
            f"are not compatible with Result Validator kind {self.result_validator.kind}."
        )

    def mount(self, scm: SCM) -> None:
        """Prepare the mission to be mounted to a given SCM.

        Args:
            scm (SCM): Structural Causal Model to mount the mission to.

        """
        # Mount Behavior Metric
        self.behavior_metric.mount(scm)
        # Mount Deliverable Metric
        self.result_metric.mount(scm)
        # Update the mounted state
        self.is_mounted = True

    @abstractmethod
    def evaluate(self, transcript: Transcript):
        """Evaluate the mission on a given SCM and history.

        Args:
            transcript (Transcript): Transcript containing the history of actions and observations.

        Returns
        -------
            dict: Evaluation results.

        """
        raise NotImplementedError(
            "The evaluate method must be implemented in the derived class."
        )
