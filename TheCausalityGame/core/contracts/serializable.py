"""The Causality Game - Serializable Contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypeVar

from TheCausalityGame.core.infrastructure.serialization import dumps, loads

T = TypeVar("T", bound="Serializable")


class Serializable(ABC):
    """Abstract base for objects that can be represented as a spec & JSON.

    The canonical spec we use across the framework is:
        {"class": "<module>:<ClassName>", "config": {...}}
    """

    _spec: str

    @abstractmethod
    def to_spec(self) -> Any:  # noqa :ANN401
        """Return a canonical spec for this instance."""

    @classmethod
    @abstractmethod
    def from_spec(cls, spec: Any) -> Any:  # noqa :ANN401
        """Create an instance from a canonical spec."""

    # ----- JSON helpers (backed by strict JSON) -----

    def to_dict(self) -> dict[str, Any]:
        """Strict dict dump of the canonical spec (for persistence/sharing)."""
        return self.to_spec().model_dump(exclude_none=True)

    def to_json(self) -> str:
        """Strict JSON dump of the canonical spec (for persistence/sharing)."""
        return dumps(self.to_dict(), indent=None)

    @classmethod
    def from_json(cls: type[T], s: str) -> T:
        """Strict JSON load (for persistence/sharing)."""
        spec = loads(s)
        return cls.from_spec(spec)
