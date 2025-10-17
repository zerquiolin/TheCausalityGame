"""The Causality Game - Logging Infrastructure."""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

__all__ = ["Logger"]


class Logger:
    """Configurable logging utility with optional file rotation and color output."""

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

        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False

        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        self._setup_handlers(max_bytes, backup_count)

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
            log_path = self.log_dir / f"{self.name}.log"
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
                colored = f"{color}{token}{reset}"
                return output.replace(token, colored, 1)

        return ColorFormatter(base_format, datefmt)

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        self.logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        self.logger.critical(msg, *args, **kwargs)

    def set_level(self, level: int) -> None:
        self.logger.setLevel(level)
        for handler in self.logger.handlers:
            handler.setLevel(level)

    def get_logger(self) -> logging.Logger:
        return self.logger
