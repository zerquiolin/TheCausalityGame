# Abstract
# Distributions
from scipy.stats import uniform

from TheCausalityGame.core.contracts.noise import NoiseDistribution
from TheCausalityGame.core.contracts.specs.noise import NoiseDistributionSpec
from TheCausalityGame.core.infra.registry import get_class_path


class UniformNoiseDistribution(NoiseDistribution):
    def __init__(self, low: float = -1.0, high: float = 1.0):
        self.low = low
        self.high = high

    def generate(self, size: int, random_state: int = 911) -> float:
        return uniform.rvs(
            loc=self.low,
            scale=self.high - self.low,
            size=size,
            random_state=random_state,
        )

    def to_spec(self) -> NoiseDistributionSpec:
        return NoiseDistributionSpec(
            class_=get_class_path(self.__class__),
            params={"low": self.low, "high": self.high},
        )

    @classmethod
    def from_spec(cls, spec: NoiseDistributionSpec) -> "UniformNoiseDistribution":
        return cls(**spec.params)  # TODO: This might raise an error is params is None
