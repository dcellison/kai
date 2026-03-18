"""Shared test fixtures."""

import logging

import structlog


def _configure_structlog_for_tests():
    """Configure structlog to work with pytest's caplog fixture.

    In production, structlog uses ProcessorFormatter which renders events
    as JSON/console output. For tests, we use a simpler pipeline that
    renders to plain strings compatible with caplog.text assertions.
    """
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,  # Allow reconfig between tests
    )

    # Add a simple formatter to the root logger so caplog captures readable text.
    # structlog's ProcessorFormatter wraps records for its own formatters;
    # we install one on a NullHandler so the record's msg gets rendered
    # before caplog sees it.
    root = logging.getLogger()
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer(colors=False),
        ],
    )
    # Apply formatter to caplog's handler if present
    for h in root.handlers:
        if isinstance(h, logging.StreamHandler):
            h.setFormatter(formatter)


_configure_structlog_for_tests()
