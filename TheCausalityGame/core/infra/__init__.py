"""Infra utilities: serialization, artifacts, logging, registry, budgets, settings.

Public surface (stable):
- serialization: dumps, loads, get_class_path
- artifacts: ensure_run_dir, write_json, append_jsonl, snapshot_provenance
- logging_: configure_logging, get_logger, log_json, bind
- registry: load_class, build_from_spec
- budgets: RoundBudget, TimeBudget
- settings: RuntimeSettings
"""

from __future__ import annotations

# Artifacts
from .artifacts import append_jsonl, ensure_run_dir, snapshot_provenance, write_json

# Budgets
from .budgets import RoundBudget, TimeBudget

# Logging
from .logging_ import bind, configure_logging, get_logger, log_json

# Registry
from .registry import build_from_spec, load_class

# Serialization
from .serialization import dumps, get_class_path, loads

# Settings (dev/restricted toggle)
from .settings import RuntimeSettings

__all__ = [
    # serialization
    "dumps",
    "loads",
    "get_class_path",
    # artifacts
    "ensure_run_dir",
    "write_json",
    "append_jsonl",
    "snapshot_provenance",
    # logging
    "configure_logging",
    "get_logger",
    "log_json",
    "bind",
    # registry
    "load_class",
    "build_from_spec",
    # budgets
    "RoundBudget",
    "TimeBudget",
    # settings
    "RuntimeSettings",
]
