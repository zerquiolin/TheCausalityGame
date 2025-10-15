"""The Causality Game - Common DTO."""

from pydantic import BaseModel, ConfigDict, Field


class CommonDTO(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
