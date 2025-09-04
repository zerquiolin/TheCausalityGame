from __future__ import annotations

import json
import logging
import sys
from typing import Any, Mapping


def _json(msg: str, **fields: Any) -> str:
    payload: dict[str, Any] = {"message": msg}
    payload.update(fields)
    return json.dumps(payload, ensure_ascii=False)


def configure_logging(*, debug: bool) -> None:
    # Remove all existing handlers (important for tests + caplog)
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    # Pick level based on debug flag
    level = logging.DEBUG if debug else logging.INFO

    # Configure root logger with a simple format
    logging.basicConfig(
        level=level,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    # Set root logger level
    root.setLevel(level)

    # Configure root logger with a simple format
    logging.basicConfig(
        level=level,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    # Set root logger level
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_json(logger: logging.Logger, level: int, msg: str, **fields: Any) -> None:
    logger.log(level, _json(msg, **fields))


def bind(logger: logging.Logger, **ctx: Any) -> logging.LoggerAdapter:
    return logging.LoggerAdapter(logger, extra=ctx)  # preserves base logger methods


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover
        base = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            base["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(base, ensure_ascii=False)
