"""The Causality Game - Metric contract."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from TheCausalityGame.core.contracts.dto.transcript import Transcript
from TheCausalityGame.core.contracts.scm import SCM
from TheCausalityGame.core.contracts.serializable import Serializable


class Metric(Serializable):
    """Metric contract + serializable base."""

    @abstractmethod
    def mount(self, scm: SCM) -> None:
        """Perform any setup required before evaluation.

        This is called once per metric instance, after deserialization.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> float:
        """Compute a score for this run.

        Returns:
            float score value.

        """
        raise NotImplementedError


class BehaviorMetric(Metric):
    """Behavior Metric contract + serializable base."""

    @abstractmethod
    def evaluate(self, transcript: Transcript) -> float:
        """Compute a score for this run.

        Returns:
            float score value.

        """
        raise NotImplementedError


class ResultMetric(Metric):
    """Result Metric contract + serializable base."""

    @abstractmethod
    def evaluate(self, result: Any) -> float:
        """Compute a score for this run.

        Returns:
            float score value.

        """
        raise NotImplementedError
