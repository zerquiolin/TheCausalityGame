"""The Causality Game - Logging Infrastructure."""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, ClassVar

__all__ = ["Logger"]


class Logger:
    """Configurable logging utility with optional file rotation and color output."""

    _lifecycle_lock: ClassVar[threading.RLock] = threading.RLock()
    _active_instances: ClassVar[dict[str, int]] = {}

    def __init__(  # noqa: PLR0913
        self,
        *,
        name: str = "app",
        log_dir: Path | None = None,
        log_to_console: bool = True,
        level: int = logging.INFO,
        max_bytes: int = 5 * 1024 * 1024,
        backup_count: int = 3,
        colorize: bool = True,
    ) -> None:
        self.name = name
        self.log_dir = log_dir
        self.log_to_console = log_to_console
        self.level = level
        self.colorize = colorize
        self._closed = False

        # ``logging.getLogger(name)`` is process-global. A second run using the
        # same display name used to clear the first run's handlers, and runs
        # writing to different directories could silently redirect each
        # other's output. Give each effective destination/configuration its
        # own registered logger so it remains multiprocessing-pickleable.
        destination = str(Path(log_dir).resolve()) if log_dir is not None else None
        identity = repr((name, destination, log_to_console))
        suffix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
        self.logger = logging.getLogger(f"{name}.{suffix}")
        configuration = (level, max_bytes, backup_count, colorize)
        prior_configuration = getattr(self.logger, "_tcg_configuration", None)
        if prior_configuration is not None and prior_configuration != configuration:
            raise ValueError(
                "A logger destination cannot be reused with conflicting handler settings."
            )
        self.logger.setLevel(level)
        self.logger.propagate = False

        # Identical destinations share one registered logger. Configure it
        # once; never remove another Logger instance's handlers.
        if prior_configuration is None:
            self.logger._tcg_configuration = configuration  # type: ignore[attr-defined]
            self._setup_handlers(max_bytes, backup_count)
        with self._lifecycle_lock:
            self._active_instances[self.logger.name] = (
                self._active_instances.get(self.logger.name, 0) + 1
            )

    def close(self) -> None:
        """Release this wrapper and close shared handlers after the last user."""
        with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
            remaining = self._active_instances.get(self.logger.name, 1) - 1
            if remaining > 0:
                self._active_instances[self.logger.name] = remaining
                return
            self._active_instances.pop(self.logger.name, None)
            for handler in self.logger.handlers[:]:
                self.logger.removeHandler(handler)
                handler.flush()
                handler.close()
            if hasattr(self.logger, "_tcg_configuration"):
                delattr(self.logger, "_tcg_configuration")

    def __enter__(self) -> Logger:
        """Return this logger for use as a context manager."""
        if self._closed:
            raise RuntimeError("A closed logger cannot be reopened")
        return self

    def __exit__(self, *_: object) -> None:
        """Close this logger when leaving its context."""
        self.close()

    def _setup_handlers(self, max_bytes: int, backup_count: int) -> None:
        """Configure console and file handlers."""
        base_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"

        plain_formatter = logging.Formatter(base_format, datefmt)
        formatter = (
            self._get_color_formatter(base_format, datefmt)
            if self.colorize
            else plain_formatter
        )

        if self.log_to_console:
            console = logging.StreamHandler()
            console.setLevel(self.level)
            console.setFormatter(formatter)
            self.logger.addHandler(console)

        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            log_path = (
                self.log_dir / f"{self.name}.log"
                if ".log" not in str(self.log_dir)
                else self.log_dir
            )
            file_handler = RotatingFileHandler(
                log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
            )
            file_handler.setLevel(self.level)
            file_handler.setFormatter(plain_formatter)
            self.logger.addHandler(file_handler)

    def _get_color_formatter(self, base_format: str, datefmt: str) -> logging.Formatter:
        """Return ANSI color formatter for console output."""
        colors = {
            "DEBUG": "\033[36m",
            "INFO": "\033[32m",
            "WARNING": "\033[33m",
            "ERROR": "\033[31m",
            "CRITICAL": "\033[41m",
        }
        reset = "\033[0m"

        class ColorFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                output = super().format(record)
                level = record.levelname
                color = colors.get(level)
                if not color:
                    return output
                token = f"[{level}]"
                colored = f"[{color}{level}{reset}]"
                return output.replace(token, colored, 1)

        return ColorFormatter(base_format, datefmt)

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        """Log a debug message."""
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        """Log an info message."""
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        """Log a warning message."""
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        """Log an error message."""
        self.logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        """Log a critical message."""
        self.logger.critical(msg, *args, **kwargs)
