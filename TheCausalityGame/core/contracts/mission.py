"""The Causality Game - Mission contract."""

from __future__ import annotations

from abc import abstractmethod

from TheCausalityGame.core.contracts.dto.transcript import Transcript
from TheCausalityGame.core.contracts.metric import BehaviorMetric, ResultMetric
from TheCausalityGame.core.contracts.scm import SCM
from TheCausalityGame.core.contracts.serializable import Serializable


class Mission(Serializable):
    """Mission contract."""

    name: str
    description: str

    is_mounted: bool = False  # True if the mission is mounted (not just a reference)

    def __init__(
        self, behavior_metric: BehaviorMetric, deliverable_metric: ResultMetric
    ):
        self.behavior_metric = behavior_metric
        self.deliverable_metric = deliverable_metric

    def mount(self, scm: SCM) -> None:
        """Prepare the mission to be mounted to a given SCM.

        Args:
            scm (SCM): Structural Causal Model to mount the mission to.

        """
        # Mount Behavior Metric
        self.behavior_metric.mount(scm)
        # Mount Deliverable Metric
        self.deliverable_metric.mount(scm)
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
