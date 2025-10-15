"""The Causality Game - Result Validator contract."""

from abc import abstractmethod

from TheCausalityGame.core.contracts.serializable import Serializable


class ResultValidator(Serializable):
    """Abstract base for noise distributions."""

    _kind: str

    @property
    def kind(self) -> str:
        """Get the kind of the noise distribution."""
        return self._kind

    @abstractmethod
    def validate(self, result: any) -> any:
        """Validate and process the result.

        Args:
            result (any): The result to validate.

        Returns
        -------
            any | None: The processed result if valid, else None.

        Raises
        ------
            NotImplementedError: If the method is not implemented.
            ValueError: If the result is invalid.

        """
        raise NotImplementedError
