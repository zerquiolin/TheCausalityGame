"""Provenance collection utilities."""

from __future__ import annotations

import os
import platform
import sys


def collect() -> dict:
    """Collect and return a provenance snapshot."""
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "env": {"TZ": os.environ.get("TZ", "UTC")},
    }
