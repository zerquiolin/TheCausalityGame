"""The Causality Game - Registry Errors."""


class ClassPathError(ValueError):
    """Class path related error."""

    def __init__(self, class_path: str) -> None:
        message = "Invalid class path"
        if class_path:
            message += f": '{class_path}'."
        super().__init__(message)


class NotAllowedByPolicyError(PermissionError):
    """Class path not allowed by policy error."""

    def __init__(self, class_path: str) -> None:
        message = "Class path not allowed by policy."
        if class_path:
            message += f": '{class_path}'"
        super().__init__(message)


class LoadError(ImportError):
    """Class loading related error."""

    def __init__(self, class_path: str) -> None:
        base_message = "Failed to load class."
        if class_path:
            base_message += f": {class_path}"
        super().__init__(base_message)


class PathFormatError(ValueError):
    """
    Path format related error.

    Correct format is 'module:Class'.
    """

    def __init__(self, path: str) -> None:
        message = "Invalid path format"
        if path:
            message += f": '{path}'."
        super().__init__(message)


class DeriveClassPathError(ValueError):
    """Class path related error."""

    def __init__(self, class_path: str) -> None:
        message = "Cannot derive class path"
        if class_path:
            message += f": '{class_path}'."
        super().__init__(message)


class InvalidSpecFormatError(ValueError):
    """Invalid spec format error."""

    def __init__(self) -> None:
        message = "Invalid spec format."
        super().__init__(message)


class MissingAttributeError(AttributeError):
    """Missing attribute error."""

    def __init__(self, attribute_name: str) -> None:
        message = "Missing attribute"
        if attribute_name:
            message += f":  '{attribute_name}'."
        super().__init__(message)


class MissingMethodError(AttributeError):
    """Missing method error."""

    def __init__(self, method_name: str) -> None:
        message = "Missing method"
        if method_name:
            message += f": '{method_name}'."
        super().__init__(message)
