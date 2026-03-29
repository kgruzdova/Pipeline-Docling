"""Loguru + перенаправление stdlib logging в loguru."""

from __future__ import annotations

import logging
import sys

from loguru import logger


def setup_loguru() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        level="INFO",
        colorize=True,
    )


class _InterceptLoggingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            logger.opt(depth=6, exception=record.exc_info).log(record.levelno, record.getMessage())
        except Exception:
            self.handleError(record)


def setup_stdlib_logging_bridge() -> None:
    logging.captureWarnings(True)
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(_InterceptLoggingHandler())
    root.setLevel(logging.INFO)


def init_logging() -> None:
    setup_loguru()
    setup_stdlib_logging_bridge()
