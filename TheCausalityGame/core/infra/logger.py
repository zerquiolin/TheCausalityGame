import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler


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
        log_dir: str | None = None,
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

        # Avoid duplicate handlers if reinitialized
        if not self.logger.handlers:
            self._setup_handlers(max_bytes, backup_count)

    # ----------------------------------------------------
    def _setup_handlers(self, max_bytes: int, backup_count: int) -> None:
        """Configure console and file handlers."""
        formatter = self._get_formatter()

        # Console handler
        if self.log_to_console:
            ch = logging.StreamHandler()
            ch.setLevel(self.level)
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

        # File handler (if directory provided)
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            log_file = os.path.join(
                self.log_dir, f"{self.name}_{datetime.now().strftime('%Y%m%d')}.log"
            )
            fh = RotatingFileHandler(
                log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
            )
            fh.setLevel(self.level)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    # ----------------------------------------------------
    def _get_formatter(self) -> logging.Formatter:
        """Return colorized or plain formatter."""
        base_format = "[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"

        if not self.colorize:
            return logging.Formatter(base_format, datefmt)

        # ANSI color codes
        colors = {
            "DEBUG": "\033[36m",  # Cyan
            "INFO": "\033[32m",  # Green
            "WARNING": "\033[33m",  # Yellow
            "ERROR": "\033[31m",  # Red
            "CRITICAL": "\033[41m",  # Red background
        }
        reset = "\033[0m"

        class ColorFormatter(logging.Formatter):
            def format(self, record):
                level_color = colors.get(record.levelname, "")
                record.levelname = f"{level_color}{record.levelname}{reset}"
                return super().format(record)

        return ColorFormatter(base_format, datefmt)

    # ----------------------------------------------------
    def info(self, msg: str, *args, **kwargs) -> None:
        self.logger.info(msg, *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs) -> None:
        self.logger.debug(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        self.logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs) -> None:
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
