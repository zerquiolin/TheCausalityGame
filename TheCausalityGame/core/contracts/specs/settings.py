from __future__ import annotations
from typing import Literal
from pydantic import BaseModel


class RuntimeSettingsSpec(BaseModel):
    mode: Literal["restricted", "dev"] = "restricted"
    debug: bool | None = None
    trusted: bool | None = None
