from typing import Tuple
from contracts.types.common import JsonDict
from pydantic import BaseModel, ConfigDict, Field


class MetricSpec(BaseModel):
    """Specification for constructing a metric."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    class_: str = Field(alias="class")
    params: JsonDict = {}


class MetricSpecs(BaseModel):
    """Canonical metric pair for each mission."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    behavior: MetricSpec
    result: MetricSpec
    custom: Tuple[MetricSpec, ...] = ()
