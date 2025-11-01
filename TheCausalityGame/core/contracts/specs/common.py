"""The Causality Game - Common Specification."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CommonSpec(BaseModel):
    """
    Base specification for dynamically instantiable components.

    This spec includes a reference to the object's implementation class (`class_`),
    an optional secondary specification class (`spec_`), and an optional parameter dictionary
    for configuration.

    Attributes
    ----------
    class_ : str
        Full import path of the implementation class (e.g., "module.submodule:ClassName").
    params : dict[str, Any]
        Optional dictionary of parameters to configure the component.
    """

    model_config = ConfigDict(
        extra="ignore",  # Ignore unexpected fields during parsing
        frozen=True,  # Make instances immutable (hashable)
        populate_by_name=True,  # Allow aliasing (e.g., `class_` -> `class`)
    )

    class_: str = Field(..., description="Class import path (module:Class).")
    params: dict[str, Any] | None = Field(
        default=None,
        description="Optional configuration parameters for the component.",
    )
