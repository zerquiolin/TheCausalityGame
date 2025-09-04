from __future__ import annotations

import logging
from typing import List

from TheCausalityGame.core.infra import RuntimeSettings, configure_logging, get_logger


class ListHandler(logging.Handler):
    """Collect log records in-memory for assertions."""

    def __init__(self, level: int = logging.NOTSET) -> None:
        super().__init__(level)
        self.records: List[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        self.records.append(record)


def _prepare_logger(name: str) -> logging.Logger:
    """Make logger cooperate with root-level gating."""
    logger = get_logger(name)
    # Let root decide effective level; ensure records bubble up
    logger.setLevel(logging.NOTSET)
    logger.propagate = True
    return logger


def _attach_capture_handler() -> ListHandler:
    """Attach a fresh capture handler to root and return it."""
    root = logging.getLogger()
    cap = ListHandler(level=logging.DEBUG)  # capture everything root emits
    root.addHandler(cap)
    return cap


def _detach_capture_handler(h: ListHandler) -> None:
    root = logging.getLogger()
    try:
        root.removeHandler(h)
    except Exception:
        pass


def test_configure_logging_and_emit() -> None:
    """When debug=False, INFO should be visible while DEBUG should not."""
    # Configure runtime (this should set the root level)
    configure_logging(debug=False)

    # Attach capture AFTER configuration so we observe effective root behavior
    cap = _attach_capture_handler()
    try:
        logger = _prepare_logger("tcg.test")
        logger.debug("debug-hidden")
        logger.info("hello-info")

        msgs = [r.getMessage() for r in cap.records]
        assert "hello-info" in msgs
        assert "debug-hidden" not in msgs
    finally:
        _detach_capture_handler(cap)


def test_dev_mode_enables_debug() -> None:
    """In dev mode, debug=True → DEBUG and INFO are both visible."""
    settings = RuntimeSettings.from_sources(mode="dev")
    assert settings.debug is True

    configure_logging(debug=settings.debug)
    cap = _attach_capture_handler()
    try:
        logger = _prepare_logger("tcg.devtest")
        logger.debug("dev-debug-on")
        logger.info("dev-info-on")

        msgs = [r.getMessage() for r in cap.records]
        assert "dev-debug-on" in msgs
        assert "dev-info-on" in msgs
    finally:
        _detach_capture_handler(cap)


def test_restricted_mode_defaults_info_level() -> None:
    """In restricted mode, debug=False → INFO visible, DEBUG hidden."""
    settings = RuntimeSettings.from_sources(mode="restricted")
    assert settings.debug is False

    configure_logging(debug=settings.debug)
    cap = _attach_capture_handler()
    try:
        logger = _prepare_logger("tcg.restricted")
        logger.debug("restricted-debug-hidden")
        logger.info("restricted-info-visible")

        msgs = [r.getMessage() for r in cap.records]
        assert "restricted-info-visible" in msgs
        assert "restricted-debug-hidden" not in msgs
    finally:
        _detach_capture_handler(cap)
