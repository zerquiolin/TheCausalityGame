from TheCausalityGame.core.contracts.dto.environment import SamplesCollection


class Strategy:

    _is_initialized: bool = False

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    def initialize(self) -> None:
        raise NotImplementedError

    def learn(self, samples: SamplesCollection) -> None:
        raise NotImplementedError

    def answer(self) -> None:
        raise NotImplementedError
