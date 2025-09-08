"""The Causality Game - Noise contract."""

from TheCausalityGame.core.contracts.serializable import Serializable


class NoiseDistribution(Serializable):
    """Abstract base for noise distributions."""

    def generate(self, size: int, random_state: int | None = 911) -> float:
        """Generate a noise value using the provided random state.

        Args:
            size (int): The size of the noise to generate.
            random_state (int, optional): Seed for random number generation. Defaults to 911.

        Returns
        -------
            float: A generated noise value.

        """
        raise NotImplementedError
