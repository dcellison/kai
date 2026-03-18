"""
Tests for setup_logging() configuration.

The main() and _init_and_run() functions orchestrate the full application
lifecycle and are impractical to unit test. setup_logging() is testable
in isolation since it just configures the root logger.
"""

import logging
from logging.handlers import TimedRotatingFileHandler

import pytest

from kai.logging import setup_logging

# ── setup_logging() ──────────────────────────────────────────────────


class TestSetupLogging:
    @pytest.fixture(autouse=True)
    def _restore_root_logger(self):
        """
        Restore root logger state after each test.

        setup_logging() modifies the global root logger by adding handlers
        and setting levels. Without cleanup, handlers accumulate across
        tests and can cause file handle leaks.
        """
        root = logging.getLogger()
        original_handlers = root.handlers[:]
        original_level = root.level
        yield
        # Close any file handlers we added (prevents open file warnings)
        for h in root.handlers:
            if h not in original_handlers and hasattr(h, "close"):
                h.close()
        root.handlers = original_handlers
        root.level = original_level

    def test_creates_log_directory(self, tmp_path):
        """Creates the log directory."""
        log_dir = tmp_path / "logs"
        setup_logging(log_dir)
        assert log_dir.is_dir()

    def test_adds_file_handler(self, tmp_path):
        """Adds TimedRotatingFileHandlers to the root logger (kai.log + audit.log)."""
        setup_logging(tmp_path / "logs")
        root = logging.getLogger()
        file_handlers = [h for h in root.handlers if isinstance(h, TimedRotatingFileHandler)]
        assert len(file_handlers) >= 2  # kai.log + audit.log

    def test_adds_stream_handler(self, tmp_path):
        """Adds a StreamHandler to the root logger."""
        setup_logging(tmp_path / "logs")
        root = logging.getLogger()
        stream_handlers = [
            h
            for h in root.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, TimedRotatingFileHandler)
        ]
        assert len(stream_handlers) >= 1

    def test_root_level_info(self, tmp_path):
        """Sets root logger to INFO level."""
        setup_logging(tmp_path / "logs")
        assert logging.getLogger().level == logging.INFO

    def test_httpx_level_warning(self, tmp_path):
        """Sets httpx logger to WARNING to silence per-request HTTP logs."""
        setup_logging(tmp_path / "logs")
        assert logging.getLogger("httpx").level == logging.WARNING

    def test_apscheduler_level_warning(self, tmp_path):
        """Sets apscheduler logger to WARNING to silence tick logs."""
        setup_logging(tmp_path / "logs")
        assert logging.getLogger("apscheduler.executors.default").level == logging.WARNING
