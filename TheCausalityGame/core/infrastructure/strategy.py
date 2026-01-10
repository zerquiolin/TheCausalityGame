"""The Causality Game - Strategy Base Class."""

from TheCausalityGame.core.contracts.dto.environment import SamplesCollection


class Strategy:
    """Base class for strategies in The Causality Game."""

    _is_initialized: bool = False

    @property
    def is_initialized(self) -> bool:
        """Indicate whether the strategy has been initialized."""
        return self._is_initialized

    def initialize(self) -> None:
        """Initialize the strategy."""
        raise NotImplementedError

    def learn(self, samples: SamplesCollection) -> None:
        """Learn from new samples."""
        raise NotImplementedError

    def answer(self) -> None:
        """Provide an answer based on learned data."""
        raise NotImplementedError
