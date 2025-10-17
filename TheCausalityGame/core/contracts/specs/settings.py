"""The Causality Game - Runtime Settings Specification."""

from __future__ import annotations

from pydantic import BaseModel, Field

from TheCausalityGame.core.lib.enum.runtime import RuntimeDebugLevel, RuntimeMode


class RuntimeSettingsSpec(BaseModel):
    """
    Specification for runtime execution settings.

    Attributes
    ----------
    mode : RuntimeMode
        Execution mode. When set to 'PROD', disables debug logging and features.
        When 'DEV', enables verbose logging and debug tooling.
    debug_level : RuntimeDebugLevel
        Level of debug information to capture (e.g., INFO, DEBUG).
        Controls verbosity of logs, output capture, and diagnostics.
    """

    mode: RuntimeMode = Field(
        default=RuntimeMode.PROD,
        description="Runtime execution mode: PROD (production) or DEV (development), where DEV enables console logging and game/round plots.",  # noqa: E501
    )
    debug_level: RuntimeDebugLevel = Field(
        default=RuntimeDebugLevel.INFO,
        description=(
            "Granularity of debug information to capture. Higher levels include more logs "
            "and internal diagnostic data (e.g., DEBUG, TRACE)."
        ),
    )
