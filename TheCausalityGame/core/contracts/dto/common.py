"""The Causality Game - Common Data Transfer Object (DTO) base class."""

from pydantic import BaseModel, ConfigDict


class CommonDTO(BaseModel):
    """
    Base class for all immutable DTOs in The Causality Game.

    Configures DTOs to:
      - Ignore extra fields when parsing.
      - Be immutable (frozen) after creation.
    """

    model_config = ConfigDict(
        extra="ignore",  # Ignore unexpected fields when loading
        frozen=True,  # Make instances immutable
    )
