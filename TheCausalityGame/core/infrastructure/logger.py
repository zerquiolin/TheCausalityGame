import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


class Logger:
    """
    A professional, configurable logging utility for Python applications.

    Features:
      - Console + file logging (optional)
      - Rotating log files with size limit
      - Timestamped, structured formatting
      - Optional colorized console output
      - Supports info, debug, warning, error, critical
    """

    def __init__(
        self,
        name: str = "app",
        log_dir: Path | None = None,
        log_to_console: bool = True,
        level: int = logging.INFO,
        max_bytes: int = 5 * 1024 * 1024,  # 5 MB per file
        backup_count: int = 3,  # keep 3 rotations
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

        # Clear existing handlers to avoid duplicates
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        self._setup_handlers(max_bytes, backup_count)

    # ----------------------------------------------------
    def _setup_handlers(self, max_bytes: int, backup_count: int) -> None:
        """Configure console and file handlers."""
        base_format = "[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"

        plain_formatter = logging.Formatter(base_format, datefmt)
        color_formatter = (
            self._get_formatter(base_format, datefmt)
            if self.colorize
            else plain_formatter
        )

        # Console handler
        if self.log_to_console:
            ch = logging.StreamHandler()
            ch.setLevel(self.level)
            ch.setFormatter(color_formatter)
            self.logger.addHandler(ch)

        # File handler (if directory provided)
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            log_file = os.path.join(self.log_dir, f"{self.name}.log")
            fh = RotatingFileHandler(
                log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
            )
            fh.setLevel(self.level)
            fh.setFormatter(plain_formatter)  # no colors
            self.logger.addHandler(fh)

    # ----------------------------------------------------
    def _get_formatter(self, base_format: str, datefmt: str) -> logging.Formatter:
        """Return colorized formatter for console output only."""

        colors = {
            "DEBUG": "\033[36m",  # Cyan
            "INFO": "\033[32m",  # Green
            "WARNING": "\033[33m",  # Yellow
            "ERROR": "\033[31m",  # Red
            "CRITICAL": "\033[41m",  # Red background
        }
        reset = "\033[0m"

        class SafeColorFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                # Make a shallow copy to avoid mutating shared record
                record_copy = logging.LogRecord(
                    name=record.name,
                    level=record.levelno,
                    pathname=record.pathname,
                    lineno=record.lineno,
                    msg=record.msg,
                    args=record.args,
                    exc_info=record.exc_info,
                )
                record_copy.created = record.created
                record_copy.msecs = record.msecs
                record_copy.relativeCreated = record.relativeCreated
                record_copy.thread = record.thread
                record_copy.process = record.process
                record_copy.levelname = record.levelname
                record_copy.levelno = record.levelno

                # Apply color only for this formatter (console)
                level_color = colors.get(record_copy.levelname, "")
                record_copy.levelname = f"{level_color}{record_copy.levelname}{reset}"

                return super().format(record_copy)

        return SafeColorFormatter(base_format, datefmt)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.logger.info(msg, *args, **kwargs)

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.logger.debug(msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.logger.critical(msg, *args, **kwargs)

    # ----------------------------------------------------
    def set_level(self, level: int) -> None:
        """Dynamically change logging level."""
        self.logger.setLevel(level)
        for handler in self.logger.handlers:
            handler.setLevel(level)

    def get_logger(self) -> logging.Logger:
        """Return the underlying logging.Logger instance."""
        return self.logger
