from TheCausalityGame.core.contracts.dto.transcript import Transcript
from TheCausalityGame.core.contracts.metric import BehaviorMetric
from TheCausalityGame.core.contracts.scm import SCM
from TheCausalityGame.core.contracts.specs.metric import MetricSpec
from TheCausalityGame.core.infrastructure.registry import get_class_path
from TheCausalityGame.core.lib.enum.environment import ActionKind
from TheCausalityGame.core.lib.utils.metrics import log_penalty


class RoundsBehaviorMetric(BehaviorMetric):
    """
    Compute a penalty based on the number of rounds taken before stopping with an answer.

    This metric counts all actions except the final 'stop_with_answer' and applies a logarithmic
    penalty function to the count.
    """

    name: str = "Rounds Metric"
    description: str = (
        "A behavior metric that penalizes the number of rounds taken before stopping with an answer."
        "It counts all actions except the final 'stop_with_answer' and applies a logarithmic penalty."
    )

    def __init__(self, alpha: float = 0.10) -> None:
        """
        Initialize the metric with a penalty decay parameter.

        Args:
            alpha (float): Decay rate for the log_penalty function; must be in (0, 1).

        Raises:
            ValueError: If alpha is not between 0 and 1 (exclusive).
        """
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
        self.alpha = alpha

    def mount(self, scm: SCM) -> None:  # unused
        pass

    def evaluate(self, transcript: Transcript) -> float:
        """
                Calculate the behavior metric for a given interaction history.
        This method counts the number of rounds in which the action was not 'stop_with_answer',
                excluding the final round, and applies the log_penalty function.

                Args:
                    transcript (Transcript): The transcript containing the interaction history.

                Returns:
                    float: The computed penalty metric.

                Raises:
                    TypeError: If history is not a pandas DataFrame.
                    KeyError: If the 'action' column is missing from the DataFrame.
        """
        # Check if the last action is 'asnwer'
        if transcript.entries[-1].decision == ActionKind.ANSWER:
            return log_penalty(len(transcript.entries) - 1, alpha=self.alpha)

        return log_penalty(len(transcript.entries), alpha=self.alpha)

    def to_spec(self) -> MetricSpec:
        return MetricSpec(
            class_=get_class_path(self.__class__),
            params={"alpha": self.alpha},
        )

    @classmethod
    def from_spec(cls, spec: MetricSpec) -> "RoundsBehaviorMetric":
        return RoundsBehaviorMetric(**spec.params)
