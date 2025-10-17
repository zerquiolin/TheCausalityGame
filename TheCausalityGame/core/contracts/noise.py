"""The Causality Game - Noise contract."""

from TheCausalityGame.core.contracts.serializable import Serializable


class NoiseDistribution(Serializable):
    """
    Abstract base class for noise distributions.

    Subclasses must implement the `generate` method to produce noise values,
    typically used in structural causal models.
    """

    def generate(self, size: int, random_state: int | None = 911) -> float:
        """
        Generate a noise value using the given size and random seed.

        Parameters
        ----------
        size : int
            Number of values to generate (can be used as a vectorized shape).
        random_state : int, optional
            Seed for the noise generator. Defaults to 911.

        Returns
        -------
        float
            A generated noise value.

        Raises
        ------
        NotImplementedError
            If the method is not overridden in a subclass.
        """
        raise NotImplementedError
