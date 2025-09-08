# outputs/dummy_output.py
from TheCausalityGame.core.contracts.dto import MetricResult
from TheCausalityGame.core.contracts.output import OutputContract


class DummyOutput(OutputContract):
    """A minimal output formatter."""

    def save(self, result: MetricResult):
        print(
            f"[DummyOutput] Result saved: success={result.success}, score={result.score}"
        )
