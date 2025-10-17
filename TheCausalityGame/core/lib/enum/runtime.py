"""The Causality Game - Runtime Configuration Enums."""

from enum import Enum


class RuntimeMode(str, Enum):
    """Defines the overall runtime environment mode.

    - PROD: Production mode with minimal logging and optimizations.
    - DEV: Development mode with extended debugging and logging.
    """

    PROD = "prod"
    """Production mode — optimized and minimal debug output."""

    DEV = "dev"
    """Development mode — includes debug output and additional checks."""


class RuntimeDebugLevel(int, Enum):
    """Logging levels aligned with standard Python logging.

    These levels control the verbosity of runtime diagnostics.

    - DEBUG: Detailed information for diagnostics.
    - INFO: General runtime events.
    - WARNING: Indicative of potential issues.
    - ERROR: Serious problems that prevent parts of the system from functioning.
    - CRITICAL: Severe errors causing complete failure.
    """

    DEBUG = 10
    """Detailed debugging messages."""

    INFO = 20
    """Informational messages that highlight the progress."""

    WARNING = 30
    """Indications of potential problems or non-critical issues."""

    ERROR = 40
    """Error events that may disrupt specific operations."""

    CRITICAL = 50
    """Critical issues causing complete system failure."""
