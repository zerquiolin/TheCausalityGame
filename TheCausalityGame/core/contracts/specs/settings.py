from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RuntimeSettingsSpec(BaseModel):
    mode: Literal["restricted", "dev"] = Field(
        default="restricted",
        description="Mode of operation. Can be either 'restricted' or 'dev'.",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode with more verbose logging and relaxed checks.",
    )
    trusted: bool = Field(
        default=False,
        description="Allow callable deliverables, etc. (e.g., for testing).",
    )
