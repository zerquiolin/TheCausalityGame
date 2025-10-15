"""Typed error hierarchy for the core framework."""

from __future__ import annotations


class TCGError(Exception):
    """Base error for TheCausalityGame."""


class ConfigurationError(TCGError):
    """Configuration-related error."""


class RegistryError(TCGError):
    """Component registry-related error."""


class DiscoveryError(TCGError):
    """Component discovery/registry-related error."""


class LoadError(TCGError):
    """Dynamic import or plugin loading error."""


class InvalidAction(TCGError):
    """Invalid action submitted to a mission/environment."""


class BudgetExceededError(TCGError):
    """Sample budget exceeded."""


class TimeoutExceeded(TCGError):
    """Time budget exceeded."""


class SecurityViolation(TCGError):
    """Security policy violation (e.g., network access in restricted mode)."""


class SerializationError(TCGError):
    """Error during serialization/deserialization of data structures."""


class AgentError(TCGError):
    """Error related to agent operations, such as action generation or observation handling."""
