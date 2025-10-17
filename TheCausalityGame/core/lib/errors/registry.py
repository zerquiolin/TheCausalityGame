"""The Causality Game - Registry Errors."""


class RegistryError(Exception):
    """Component registry-related error."""

    def __init__(self, component_name: str | None = None) -> None:
        """Initialize a new RegistryError."""
        message = "Registry error occurred"
        if component_name:
            message += f" for component '{component_name}'"
        super().__init__(message)


class DiscoveryError(Exception):
    """Component discovery/registry-related error."""

    def __init__(self, component_name: str | None = None) -> None:
        """Initialize a new DiscoveryError."""
        message = "Discovery error occurred"
        if component_name:
            message += f" for component '{component_name}'"
        super().__init__(message)
