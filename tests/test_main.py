"""
Tests for main.py - setup_logging() and _bootstrap_memory().

The main() and _init_and_run() functions orchestrate the full application
lifecycle and are impractical to unit test. setup_logging() and
_bootstrap_memory() are testable in isolation.
"""

import logging
from logging.handlers import TimedRotatingFileHandler
from unittest.mock import patch

import pytest

from kai.main import _bootstrap_memory, setup_logging

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
        """Creates the logs/ directory under DATA_DIR."""
        with patch("kai.main.DATA_DIR", tmp_path):
            setup_logging()
        assert (tmp_path / "logs").is_dir()

    def test_adds_file_handler(self, tmp_path):
        """Adds a TimedRotatingFileHandler to the root logger."""
        with patch("kai.main.DATA_DIR", tmp_path):
            setup_logging()
        root = logging.getLogger()
        file_handlers = [h for h in root.handlers if isinstance(h, TimedRotatingFileHandler)]
        assert len(file_handlers) >= 1

    def test_adds_stream_handler(self, tmp_path):
        """Adds a StreamHandler to the root logger."""
        with patch("kai.main.DATA_DIR", tmp_path):
            setup_logging()
        root = logging.getLogger()
        stream_handlers = [
            h
            for h in root.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, TimedRotatingFileHandler)
        ]
        assert len(stream_handlers) >= 1

    def test_root_level_info(self, tmp_path):
        """Sets root logger to INFO level."""
        with patch("kai.main.DATA_DIR", tmp_path):
            setup_logging()
        assert logging.getLogger().level == logging.INFO

    def test_httpx_level_warning(self, tmp_path):
        """Sets httpx logger to WARNING to silence per-request HTTP logs."""
        with patch("kai.main.DATA_DIR", tmp_path):
            setup_logging()
        assert logging.getLogger("httpx").level == logging.WARNING

    def test_apscheduler_level_warning(self, tmp_path):
        """Sets apscheduler logger to WARNING to silence tick logs."""
        with patch("kai.main.DATA_DIR", tmp_path):
            setup_logging()
        assert logging.getLogger("apscheduler.executors.default").level == logging.WARNING


# ── _bootstrap_memory() ──────────────────────────────────────────────


class TestBootstrapMemory:
    def test_from_example_template(self, tmp_path, monkeypatch):
        """Creates MEMORY.md from example template when missing."""
        data_dir = tmp_path / "data"
        project_root = tmp_path / "project"
        example_dir = project_root / "home" / ".claude"
        example_dir.mkdir(parents=True)
        (example_dir / "MEMORY.md.example").write_text("# Memory\n\n## About the User\n")

        monkeypatch.setattr("kai.main.DATA_DIR", data_dir)
        monkeypatch.setattr("kai.main.PROJECT_ROOT", project_root)

        _bootstrap_memory()

        memory_file = data_dir / "memory" / "MEMORY.md"
        assert memory_file.exists()
        assert "About the User" in memory_file.read_text()

    def test_no_example_creates_minimal(self, tmp_path, monkeypatch):
        """Creates a minimal MEMORY.md when no example template exists."""
        data_dir = tmp_path / "data"
        project_root = tmp_path / "project"
        (project_root / "home" / ".claude").mkdir(parents=True)

        monkeypatch.setattr("kai.main.DATA_DIR", data_dir)
        monkeypatch.setattr("kai.main.PROJECT_ROOT", project_root)

        _bootstrap_memory()

        memory_file = data_dir / "memory" / "MEMORY.md"
        assert memory_file.exists()
        assert memory_file.read_text() == "# Memory\n"

    def test_skips_existing(self, tmp_path, monkeypatch):
        """Does not overwrite an existing MEMORY.md."""
        data_dir = tmp_path / "data"
        memory_dir = data_dir / "memory"
        memory_dir.mkdir(parents=True)
        memory_file = memory_dir / "MEMORY.md"
        memory_file.write_text("User prefers dry humor.")

        monkeypatch.setattr("kai.main.DATA_DIR", data_dir)
        monkeypatch.setattr("kai.main.PROJECT_ROOT", tmp_path / "project")

        _bootstrap_memory()

        assert memory_file.read_text() == "User prefers dry humor."
