from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from newsagent.config import LoggingConfig


def setup_logging(config: LoggingConfig) -> None:
    log_level = getattr(logging, config.level.upper(), logging.INFO)

    if config.format == "json":
        processors = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ]
    else:
        processors = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )


def configure_logging(level: str, format: str) -> None:
    config = LoggingConfig(level=level, format=format)
    setup_logging(config)


def get_logger(name: str) -> structlog.BoundLogger:
    return structlog.get_logger(name)


class LogContext:
    def __init__(self, logger: structlog.BoundLogger, **kwargs: Any):
        self.logger = logger
        self.kwargs = kwargs
        self.prev_context = None

    def __enter__(self) -> structlog.BoundLogger:
        self.prev_context = structlog.contextvars.get_merged_contextvars()
        structlog.contextvars.bind_contextvars(**self.kwargs)
        return self.logger

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        structlog.contextvars.unbind_contextvars(*self.kwargs.keys())
        if self.prev_context:
            structlog.contextvars.bind_contextvars(**self.prev_context)


def bind_context(**kwargs: Any) -> LogContext:
    logger = structlog.get_logger()
    return LogContext(logger, **kwargs)