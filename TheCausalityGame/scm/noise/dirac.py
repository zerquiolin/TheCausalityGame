import numpy as np

from TheCausalityGame.core.contracts.noise import NoiseDistribution
from TheCausalityGame.core.contracts.specs.noise import NoiseDistributionSpec
from TheCausalityGame.core.infra.registry import get_class_path


class DiracNoiseDistribution(NoiseDistribution):

    def __init__(self, val: int | float):
        self.val = val

    def generate(self, size, random_state=911) -> float:
        return self.val * np.ones(size)

    def to_spec(self):
        return NoiseDistributionSpec(
            class_=get_class_path(self.__class__),
            params={"val": self.val},
        )

    @classmethod
    def from_spec(cls, spec: NoiseDistributionSpec) -> "DiracNoiseDistribution":
        return DiracNoiseDistribution(**spec.params)
