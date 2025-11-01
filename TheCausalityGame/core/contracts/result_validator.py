"""The Causality Game - Result Validator contract."""

from abc import abstractmethod
from typing import Any

from TheCausalityGame.core.contracts.serializable import Serializable


class ResultValidator(Serializable):
    """
    Abstract base class for validating agent results.

    Used to enforce constraints or canonical formats for outputs
    returned by the agent's `answer()` method.
    """

    _kind: str
    _spec: str = (
        "TheCausalityGame.core.contracts.specs.result_validator:ResultValidatorSpec"
    )

    @property
    def kind(self) -> str:
        """
        Return the kind identifier for the result validator.

        Returns
        -------
        str
            The type of result this validator accepts (used to match against metrics).
        """
        return self._kind

    @abstractmethod
    def validate(self, result: Any) -> Any:  # noqa :ANN401
        """
        Validate and optionally transform the agent's result.

        Parameters
        ----------
        result : Any
            The raw result returned by the agent.

        Returns
        -------
        Any or None
            A cleaned/standardized result, or None if validation fails.

        Raises
        ------
        ValueError
            If the result is invalid or incompatible.
        """
        raise NotImplementedError
