from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from TheCausalityGame.core.lib.enum.runtime import RuntimeDebugLevel, RuntimeMode


class RuntimeSettingsSpec(BaseModel):
    mode: RuntimeMode = Field(
        default=RuntimeMode.PROD,
        description="Enables logging on console and debug features.",
    )
    debug_level: RuntimeDebugLevel = Field(
        default=RuntimeDebugLevel.INFO,
        description="Enables saving all ",
    )
