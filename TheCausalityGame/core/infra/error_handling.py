from __future__ import annotations

import logging
import traceback
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, TypeVar

# Import your canonical error classes. If names differ, map below.
from TheCausalityGame.core.contracts.errors import (
    BudgetExceededError,  # time/round/sample budget exceeded
    ConfigurationError,  # config/manifest/DTO issues
    RegistryError,  # dynamic loading/lookup fails
    # If you have these in your project, import them too; if not, the
    # handler will still work without:
    # InvalidAction, LoadError, TimeoutExceeded, SecurityViolation
    SerializationError,  # JSON (de)serialization failures
    TCGError,  # base
)

# Optional: define soft references for classes that may or may not exist.
try:
    from TheCausalityGame.core.contracts.errors import InvalidAction  # type: ignore
except Exception:  # pragma: no cover

    class InvalidAction(TCGError):  # type: ignore[misc,override]
        pass


try:
    from TheCausalityGame.core.contracts.errors import LoadError  # type: ignore
except Exception:  # pragma: no cover

    class LoadError(TCGError):  # type: ignore[misc,override]
        pass


try:
    from TheCausalityGame.core.contracts.errors import TimeoutExceeded  # type: ignore
except Exception:  # pragma: no cover

    class TimeoutExceeded(TCGError):  # type: ignore[misc,override]
        pass


try:
    from TheCausalityGame.core.contracts.errors import SecurityViolation  # type: ignore
except Exception:  # pragma: no cover

    class SecurityViolation(TCGError):  # type: ignore[misc,override]
        pass


@dataclass(frozen=True)
class UserMessage:
    """A small, safe message to present to users/UIs and store in transcripts."""

    level: str  # "error" | "warning" | "info"
    title: str  # short headline
    message: str  # concise description
    code: str  # stable machine-readable code, e.g. "CONFIG_INVALID"
    details: dict[str, Any]  # extra info safe to share (no stack traces)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Stable error codes used across UI, CLI and logs.
# Keep codes short and immutable; they are part of your public surface.
_ERROR_MAP: list[tuple[type[BaseException], str, str, str]] = [
    # (ExceptionType, code, level, default title)
    (ConfigurationError, "CONFIG_INVALID", "error", "Configuration error"),
    (SerializationError, "SERDE_ERROR", "error", "Serialization error"),
    (RegistryError, "REGISTRY_ERROR", "error", "Component lookup failed"),
    (LoadError, "LOAD_ERROR", "error", "Failed to load component"),
    (InvalidAction, "INVALID_ACTION", "warning", "Invalid action"),
    (BudgetExceededError, "BUDGET_EXCEEDED", "warning", "Budget exceeded"),
    (TimeoutExceeded, "TIMEOUT_EXCEEDED", "warning", "Operation timed out"),
    (SecurityViolation, "SECURITY_VIOLATION", "error", "Security policy violation"),
    (TCGError, "GAME_ERROR", "error", "Game error"),
    (Exception, "UNEXPECTED_ERROR", "error", "Unexpected error"),
]


def _classify(exc: BaseException) -> tuple[str, str, str]:
    """Return (code, level, title) for a given exception type."""
    for etype, code, level, title in _ERROR_MAP:
        if isinstance(exc, etype):
            return code, level, title
    return "UNEXPECTED_ERROR", "error", "Unexpected error"


def format_user_message(
    exc: BaseException,
    *,
    debug: bool = False,
    context: dict[str, Any] | None = None,
) -> UserMessage:
    """Create a safe, user-facing message from an exception.

    Args:
        exc: The exception raised.
        debug: If True, include a short stack trace in details (for developers).
        context: Optional context labels (agent_id, round_idx, file, etc.).

    Returns
    -------
        UserMessage suitable for console/CLI/JSON artifacts.
    """
    code, level, title = _classify(exc)
    details: dict[str, Any] = {}

    # Always include the exception message; keep it short and neutral.
    message = str(exc).strip() or title

    # Attach small safe context (no PII/large blobs).
    if context:
        details["context"] = {k: v for k, v in context.items() if _is_json_safe(v)}

    # Include trace in debug mode only.
    if debug:
        details["traceback"] = _trim_traceback(traceback.format_exc(limit=10))

    return UserMessage(
        level=level, title=title, message=message, code=code, details=details
    )


def log_exception(
    logger: logging.Logger,
    exc: BaseException,
    *,
    context: dict[str, Any] | None = None,
    level_override: int | None = None,
    debug: bool = False,
) -> None:
    """Log an exception in a structured and consistent way."""
    code, level, title = _classify(exc)
    log_level = level_override or (_to_log_level(level))
    payload = {
        "event": "exception",
        "code": code,
        "title": title,
        "message": str(exc).strip(),
        "context": context or {},
    }
    if debug:
        payload["traceback"] = _trim_traceback(traceback.format_exc(limit=10))
    # Log as single line JSON-like dict (works with std handlers and JSON formatters).
    logger.log(log_level, "exception=%s payload=%s", code, payload, exc_info=debug)


def emit_hook(
    emitter: Callable[[str, dict[str, Any]], None] | None,
    event: str,
    data: dict[str, Any],
) -> None:
    """Emit a hook event if an emitter is provided, ignore otherwise."""
    try:
        if emitter:
            emitter(event, data)
    except Exception:  # hooks must not crash the runtime
        pass


# --------------------------
# Convenience decorator & CM
# --------------------------

F = TypeVar("F", bound=Callable[..., Any])


def catch_errors(
    *,
    logger: logging.Logger,
    user_notify: Callable[[UserMessage], None] | None = None,
    hook_emitter: Callable[[str, dict[str, Any]], None] | None = None,
    debug: bool = False,
    reraise: bool = False,
    context_provider: Callable[[], dict[str, Any]] | None = None,
) -> Callable[[F], F]:
    """Decorator to catch, log, emit hook, and surface a user message.

    Usage:
        @catch_errors(logger=my_logger, user_notify=ui.send, hook_emitter=hooks.emit)
        def run_agent(...): ...
    """

    def decorator(fn: F) -> F:
        def wrapper(*args: Any, **kwargs: Any):  # type: ignore[misc]
            context = context_provider() if context_provider else {}
            try:
                return fn(*args, **kwargs)
            except BaseException as exc:  # noqa: BLE001
                log_exception(logger, exc, context=context, debug=debug)
                msg = format_user_message(exc, debug=debug, context=context)
                emit_hook(
                    hook_emitter,
                    "on_component_error",
                    {"code": msg.code, "details": msg.to_dict()},
                )
                if user_notify:
                    user_notify(msg)
                if reraise:
                    raise
                return None

        return wrapper  # type: ignore[return-value]

    return decorator


@contextmanager
def error_boundary(
    *,
    logger: logging.Logger,
    user_notify: Callable[[UserMessage], None] | None = None,
    hook_emitter: Callable[[str, dict[str, Any]], None] | None = None,
    debug: bool = False,
    reraise: bool = False,
    context: dict[str, Any] | None = None,
) -> Iterable[None]:
    """Context manager variant of catch_errors for ad-hoc blocks."""
    try:
        yield
    except BaseException as exc:  # noqa: BLE001
        log_exception(logger, exc, context=context, debug=debug)
        msg = format_user_message(exc, debug=debug, context=context)
        emit_hook(
            hook_emitter,
            "on_component_error",
            {"code": msg.code, "details": msg.to_dict()},
        )
        if user_notify:
            user_notify(msg)
        if reraise:
            raise


# --------------------------
# Helpers
# --------------------------


def _to_log_level(level: str) -> int:
    return {
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }.get(level, logging.ERROR)


def _is_json_safe(value: Any) -> bool:
    try:
        import json

        json.dumps(value)
        return True
    except Exception:
        return False


def _trim_traceback(tb: str, *, max_chars: int = 4000) -> str:
    if len(tb) <= max_chars:
        return tb
    head = tb[: max_chars // 2]
    tail = tb[-max_chars // 2 :]
    return head + "\n...\n" + tail
