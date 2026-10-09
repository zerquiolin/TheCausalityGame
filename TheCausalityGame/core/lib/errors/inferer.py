"""Errors raised by authoritative inferer adapters."""


class InfererBackendError(RuntimeError):
    """Base error for an authoritative inferer backend."""


class InfererDependencyError(InfererBackendError):
    """Raised when an explicitly selected authoritative backend is unavailable."""


class UnsupportedInfererInputError(InfererBackendError, ValueError):
    """Raised when data violate the selected upstream algorithm's contract."""


class ObsoleteInfererParameterError(InfererBackendError, ValueError):
    """Raised when a parameter belongs to the removed empirical implementation."""
