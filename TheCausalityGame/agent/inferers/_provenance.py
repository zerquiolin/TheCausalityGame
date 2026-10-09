"""Provenance records for authoritative inferer adapters."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class BackendProvenance:
    """Immutable identity of code delegated to by an inferer adapter."""

    project: str
    algorithm: str
    version: str
    source: str
    citation: str
    license: str
    adapter_version: str = "1"
    implementation_kind: str = "authoritative"

    def to_dict(self) -> dict[str, str]:
        """Return JSON-friendly metadata for belief snapshots and reports."""
        return asdict(self)
