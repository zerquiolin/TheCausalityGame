from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeSettings:
    """Global toggles influencing runtime behavior."""

    mode: str  # "restricted" | "dev"
    debug: bool  # enables stacktraces and DEBUG logs
    trusted: bool  # allows callable deliverables, etc.

    @classmethod
    def from_sources(
        cls,
        *,
        mode: str | None = None,
        debug: bool | None = None,
        trusted: bool | None = None,
    ) -> "RuntimeSettings":
        # environment overrides (prefixed to avoid collisions)
        env_mode = os.getenv("TCG_MODE")
        env_debug = os.getenv("TCG_DEBUG")
        env_trusted = os.getenv("TCG_TRUSTED")

        if mode is not None and mode.lower() == "restricted":
            debug = False

        mode_val = (mode or env_mode or "restricted").lower()
        if mode_val not in ("restricted", "dev"):
            mode_val = "restricted"

        # Defaults derived from mode if not explicitly set
        debug_val = (
            debug
            if debug is not None
            else (
                (env_debug is not None and env_debug != "0")
                if env_debug is not None
                else (mode_val == "dev")
            )
        )

        trusted_val = (
            trusted
            if trusted is not None
            else (
                (env_trusted is not None and env_trusted != "0")
                if env_trusted is not None
                else (mode_val == "dev")
            )
        )

        return cls(mode=mode_val, debug=bool(debug_val), trusted=bool(trusted_val))
