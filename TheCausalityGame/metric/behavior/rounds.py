"""The Causality Game - Rounds Behavior Metric."""

from typing import override

from TheCausalityGame.core.contracts.dto.transcript import Transcript
from TheCausalityGame.core.contracts.metric import BehaviorMetric
from TheCausalityGame.core.contracts.scm import SCM
from TheCausalityGame.core.contracts.specs.metric import MetricSpec
from TheCausalityGame.core.infrastructure.registry import get_class_path
from TheCausalityGame.core.lib.enum.environment import ActionKind
from TheCausalityGame.core.lib.errors.metric import AttributeOutOfBoundsError
from TheCausalityGame.core.lib.utils.metrics import log_penalty


class RoundsBehaviorMetric(BehaviorMetric):
    """
    A behavior metric that penalizes the number of rounds taken before submitting an answer.

    This metric applies a logarithmic penalty to the number of rounds in which the agent
    has not submitted a final answer. The final round is excluded from the penalty if it ends
    with an `ActionKind.ANSWER`.
    """

    name: str = "Rounds Metric"
    description: str = (
        "Behavior metric that penalizes the number of rounds taken before stopping with an answer."
        "It applies a logarithmic penalty to the number of non-terminal rounds."
    )

    def __init__(self, alpha: float = 0.10) -> None:
        """
        Initialize the metric with a decay parameter for the penalty.

        Parameters
        ----------
        alpha : float
            Decay rate for the log penalty function. Must be between 0 and 1.

        Raises
        ------
        ValueError
            If alpha is not in the open interval (0, 1).
        """
        if not 0.0 < alpha < 1.0:
            raise AttributeOutOfBoundsError(attribute_name="alpha", value=alpha, domain=[0.0, 1.0])
        self.alpha = alpha

    @override
    def mount(self, scm: SCM) -> None:
        pass

    @override
    def evaluate(self, transcript: Transcript) -> float:
        """
        Compute the behavior penalty from the transcript.

        Parameters
        ----------
        transcript : Transcript
            The transcript containing all round entries.

        Returns
        -------
        float
            Logarithmic penalty based on the number of rounds before submitting an answer.
        """
        # If the final action is ANSWER, exclude that round from the penalty
        n_rounds = len(transcript.entries)
        decision = transcript.entries[-1].decision
        if decision and decision.kind == ActionKind.ANSWER:
            n_rounds -= 1

        return log_penalty(n_rounds, alpha=self.alpha)

    @override
    def to_spec(self) -> MetricSpec:
        return MetricSpec(
            class_=get_class_path(self.__class__),
            params={"alpha": self.alpha},
        )

    @classmethod
    @override
    def from_spec(cls, spec: MetricSpec) -> "RoundsBehaviorMetric":
        if spec.params is None:
            return cls()
        return cls(**spec.params)
