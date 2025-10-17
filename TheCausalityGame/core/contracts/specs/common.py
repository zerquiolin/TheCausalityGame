"""The Causality Game - Common Specification."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from TheCausalityGame.core.infrastructure.registry import get_class_path


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
    spec_ : str | None
        Optional reference to the spec class (used during deserialization).
    params : dict[str, Any]
        Optional dictionary of parameters to configure the component.
    """

    model_config = ConfigDict(
        extra="ignore",  # Ignore unexpected fields during parsing
        frozen=True,  # Make instances immutable (hashable)
        populate_by_name=True,  # Allow aliasing (e.g., `class_` -> `class`)
    )

    class_: str = Field(
        ..., alias="class", description="Class import path (module:Class)."
    )
    spec_: str | None = Field(
        default=None,
        description="Optional spec class import path. Auto-populated if not provided.",
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional configuration parameters for the component.",
    )

    @model_validator(mode="after")
    def _set_spec_default(self) -> "CommonSpec":
        """Automatically populate `spec_` with the current spec class path if not provided."""
        if self.spec_ is None:
            object.__setattr__(self, "spec_", get_class_path(self.__class__))
        return self
