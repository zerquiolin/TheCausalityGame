"""The Causality Game - Metric Errors."""


class NotInitializedError(Exception):
    """Raised when a metric is not initialized."""

    def __init__(self, metric_name: str) -> None:
        """Initialize the error with the metric name."""
        super().__init__(f"The metric '{metric_name}' is not initialized.")


class AttributeOutOfBoundsError(Exception):
    """Raised when an attribute value is out of bounds."""

    def __init__(
        self, attribute_name: str, value: float, domain: list[float | str]
    ) -> None:
        """Initialize the error with the attribute name, value, and domain."""
        self.attribute_name = attribute_name
        self.value = value
        self.domain = domain


class UnsupportedMetricTypeError(Exception):
    """Raised when an unsupported metric type is encountered."""

    def __init__(self, metric_type: str) -> None:
        """Initialize the error with the metric type."""
        super().__init__(f"The metric type '{metric_type}' is unsupported.")
