"""The Causality Game - Common Spec."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from TheCausalityGame.core.contracts.types.common import JsonDict
from TheCausalityGame.core.infra.registry import (
    get_class_path,
)


class CommonSpec(BaseModel):
    """Specification for constructing a DAG.

    Attributes
    ----------
        class_: Import path 'module:Class' (aliased from 'class' in JSON).
        nodes: List of node identifiers.
        edges: List of (source, target) edge tuples.
    """

    model_config = ConfigDict(extra="ignore", frozen=True, populate_by_name=True)

    class_: str = Field(alias="class")
    spec_: str | None = Field(default=None, description="Spec class path.")
    params: JsonDict | None = Field(
        default=None,
        description="Optional class configuration payload.",
    )

    @model_validator(mode="after")
    def _set_spec_default(cls, values):  # type: ignore[override]
        # `frozen=True` prevents normal assignment; use object.__setattr__.
        if values.spec_ is None:
            # # Lazy import here to avoid circular import at module import time
            # from TheCausalityGame.core.infra.registry import (
            #     get_class_path,
            # )

            object.__setattr__(values, "spec_", get_class_path(values.__class__))
        return values
