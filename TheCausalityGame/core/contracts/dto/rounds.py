from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RoundInfo(BaseModel):
    round_index: int
    remaining_rounds: int
    budgets_snapshot: dict[str, Any] = Field(
        default_factory=dict,
        description="e.g., {'time_s_left': 12.3, 'samples_left': 500, 'memory_mb_left': None}",
    )
