"""The Causality Game - Serialization Errors."""

from typing import Any


class ObjectNotSerializableError(Exception):
    """Raised when an object cannot be serialized."""

    def __init__(self, obj: Any) -> None:  # noqa :ANN401
        """Initialize the error."""
        super().__init__(f"Object {obj} cannot be serialized.")


class ObjectNotDeserializableError(Exception):
    """Raised when an object cannot be deserialized."""

    def __init__(self, obj: Any) -> None:  # noqa :ANN401
        """Initialize the error."""
        super().__init__(f"Object {obj} cannot be deserialized.")
