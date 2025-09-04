from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ExperimentSpace(BaseModel):
    variables: Dict[str, List[Any]]
    max_n: Optional[int] = None


class AvailableActions(BaseModel):
    experiment: ExperimentSpace
    submit: Dict[str, Any] = Field(default_factory=dict)
