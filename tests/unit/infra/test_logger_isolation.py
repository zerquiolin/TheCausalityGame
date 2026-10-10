"""Run-specific logging configuration must not mutate other run outputs."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from TheCausalityGame.core.infrastructure.logger import Logger


def test_same_display_name_keeps_distinct_run_files_isolated(tmp_path: Path) -> None:
    """Runs with an identical display name must not share handlers or files."""
    left = Logger(name="same-agent", log_dir=tmp_path / "left", log_to_console=False)
    right = Logger(name="same-agent", log_dir=tmp_path / "right", log_to_console=False)

    left.info("left-only-marker")
    right.info("right-only-marker")

    left_text = (tmp_path / "left" / "same-agent.log").read_text(encoding="utf-8")
    right_text = (tmp_path / "right" / "same-agent.log").read_text(encoding="utf-8")
    assert "left-only-marker" in left_text
    assert "right-only-marker" not in left_text
    assert "right-only-marker" in right_text
    assert "left-only-marker" not in right_text


def test_identical_configuration_does_not_duplicate_handlers(tmp_path: Path) -> None:
    """Reusing a destination should not cause duplicate log records."""
    first = Logger(name="repeat-agent", log_dir=tmp_path, log_to_console=False)
    initial_count = len(first.logger.handlers)
    second = Logger(name="repeat-agent", log_dir=tmp_path, log_to_console=False)

    assert second.logger is first.logger
    assert len(second.logger.handlers) == initial_count == 1
    second.info("written-once")
    text = (tmp_path / "repeat-agent.log").read_text(encoding="utf-8")
    assert text.count("written-once") == 1


def test_loggers_remain_registered_for_multiprocessing_pickling(tmp_path: Path) -> None:
    """Destination isolation should preserve standard logger registration."""
    instance = Logger(name="pickle-agent", log_dir=tmp_path, log_to_console=False)
    assert logging.getLogger(instance.logger.name) is instance.logger


def test_same_destination_rejects_conflicting_handler_configuration(tmp_path: Path) -> None:
    """A shared rotating file must not acquire incompatible handler policies."""
    Logger(name="config-agent", log_dir=tmp_path, log_to_console=False, level=logging.INFO)
    with pytest.raises(ValueError, match="conflicting handler settings"):
        Logger(
            name="config-agent",
            log_dir=tmp_path,
            log_to_console=False,
            level=logging.DEBUG,
        )


def test_shared_logger_closes_handlers_after_last_owner(tmp_path: Path) -> None:
    """One wrapper cannot close a handler still owned by another wrapper."""
    first = Logger(name="close-agent", log_dir=tmp_path, log_to_console=False)
    second = Logger(name="close-agent", log_dir=tmp_path, log_to_console=False)
    handler = first.logger.handlers[0]

    first.close()
    second.info("still-open")
    assert not handler.stream.closed

    second.close()
    assert handler.stream is None
    assert not first.logger.handlers
    with pytest.raises(RuntimeError, match="closed logger"):
        first.__enter__()

    reopened = Logger(name="close-agent", log_dir=tmp_path, log_to_console=False)
    reopened.info("reopened")
    reopened.close()
    text = (tmp_path / "close-agent.log").read_text(encoding="utf-8")
    assert "still-open" in text and "reopened" in text
