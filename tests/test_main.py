"""
Tests for main.py - setup_logging(), _bootstrap_memory(), and _file_age/_file_cleanup_loop.

The main() and _init_and_run() functions orchestrate the full application
lifecycle and are impractical to unit test. The helper functions are
testable in isolation.
"""

import asyncio
import logging
from datetime import UTC, datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from unittest.mock import patch

import pytest

from kai.main import _bootstrap_memory, _file_age, _file_cleanup_loop, setup_logging

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


# ── _file_age() ──────────────────────────────────────────────────────


class TestFileAge:
    def test_parses_valid_timestamp(self):
        """Extracts datetime from YYYYMMDD_HHMMSS prefix."""
        path = Path("20260228_084059_615331_photo_abc.jpg")
        result = _file_age(path)
        assert result is not None
        assert result.year == 2026
        assert result.month == 2
        assert result.day == 28
        assert result.hour == 8
        assert result.minute == 40
        assert result.second == 59
        assert result.tzinfo == UTC

    def test_returns_none_for_no_prefix(self):
        """Files without timestamp prefix return None."""
        assert _file_age(Path("readme.txt")) is None
        assert _file_age(Path("photo.jpg")) is None

    def test_returns_none_for_malformed_timestamp(self):
        """Malformed timestamps (invalid date) return None."""
        assert _file_age(Path("99991301_999999_file.txt")) is None

    def test_returns_none_for_partial_match(self):
        """Partial matches (missing microsecond separator) return None."""
        assert _file_age(Path("20260228_084059.jpg")) is None


# ── _file_cleanup_loop() ────────────────────────────────────────────


class TestFileCleanupLoop:
    @pytest.mark.asyncio
    async def test_deletes_old_files(self, tmp_path, monkeypatch):
        """Files older than retention cutoff are deleted."""
        monkeypatch.setattr("kai.main.DATA_DIR", tmp_path)
        monkeypatch.setattr("kai.main._CLEANUP_STARTUP_DELAY", 0)
        monkeypatch.setattr("kai.main._CLEANUP_INTERVAL", 0)

        files_dir = tmp_path / "files"
        files_dir.mkdir()
        # Old file (60 days ago)
        old_file = files_dir / "20260101_120000_000000_photo.jpg"
        old_file.write_bytes(b"old")
        # New file (today-ish)
        now = datetime.now(UTC)
        ts = now.strftime("%Y%m%d_%H%M%S")
        new_file = files_dir / f"{ts}_000000_photo.jpg"
        new_file.write_bytes(b"new")

        # Run one iteration then cancel
        call_count = 0

        async def mock_sleep(duration):
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise asyncio.CancelledError

        with patch("kai.main.asyncio.sleep", side_effect=mock_sleep):
            try:
                await _file_cleanup_loop(30)
            except asyncio.CancelledError:
                pass

        assert not old_file.exists()
        assert new_file.exists()

    @pytest.mark.asyncio
    async def test_preserves_files_without_timestamp(self, tmp_path, monkeypatch):
        """Files without timestamp prefix are never deleted."""
        monkeypatch.setattr("kai.main.DATA_DIR", tmp_path)
        monkeypatch.setattr("kai.main._CLEANUP_STARTUP_DELAY", 0)
        monkeypatch.setattr("kai.main._CLEANUP_INTERVAL", 0)

        files_dir = tmp_path / "files"
        files_dir.mkdir()
        manual_file = files_dir / "readme.txt"
        manual_file.write_text("keep me")

        call_count = 0

        async def mock_sleep(duration):
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise asyncio.CancelledError

        with patch("kai.main.asyncio.sleep", side_effect=mock_sleep):
            try:
                await _file_cleanup_loop(1)
            except asyncio.CancelledError:
                pass

        assert manual_file.exists()

    @pytest.mark.asyncio
    async def test_removes_empty_user_directories(self, tmp_path, monkeypatch):
        """Empty per-user directories are removed after cleanup."""
        monkeypatch.setattr("kai.main.DATA_DIR", tmp_path)
        monkeypatch.setattr("kai.main._CLEANUP_STARTUP_DELAY", 0)
        monkeypatch.setattr("kai.main._CLEANUP_INTERVAL", 0)

        files_dir = tmp_path / "files"
        user_dir = files_dir / "12345"
        user_dir.mkdir(parents=True)
        old_file = user_dir / "20260101_120000_000000_photo.jpg"
        old_file.write_bytes(b"old")

        call_count = 0

        async def mock_sleep(duration):
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise asyncio.CancelledError

        with patch("kai.main.asyncio.sleep", side_effect=mock_sleep):
            try:
                await _file_cleanup_loop(30)
            except asyncio.CancelledError:
                pass

        assert not old_file.exists()
        assert not user_dir.exists()

    @pytest.mark.asyncio
    async def test_leaves_nonempty_user_directories(self, tmp_path, monkeypatch):
        """Non-empty per-user directories are left intact."""
        monkeypatch.setattr("kai.main.DATA_DIR", tmp_path)
        monkeypatch.setattr("kai.main._CLEANUP_STARTUP_DELAY", 0)
        monkeypatch.setattr("kai.main._CLEANUP_INTERVAL", 0)

        files_dir = tmp_path / "files"
        user_dir = files_dir / "12345"
        user_dir.mkdir(parents=True)
        # One old (deleted), one without timestamp (kept)
        old_file = user_dir / "20260101_120000_000000_photo.jpg"
        old_file.write_bytes(b"old")
        manual_file = user_dir / "keep_me.txt"
        manual_file.write_text("important")

        call_count = 0

        async def mock_sleep(duration):
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise asyncio.CancelledError

        with patch("kai.main.asyncio.sleep", side_effect=mock_sleep):
            try:
                await _file_cleanup_loop(30)
            except asyncio.CancelledError:
                pass

        assert not old_file.exists()
        assert manual_file.exists()
        assert user_dir.exists()

    @pytest.mark.asyncio
    async def test_handles_missing_files_directory(self, tmp_path, monkeypatch):
        """Missing files/ directory is handled gracefully."""
        monkeypatch.setattr("kai.main.DATA_DIR", tmp_path)
        monkeypatch.setattr("kai.main._CLEANUP_STARTUP_DELAY", 0)
        monkeypatch.setattr("kai.main._CLEANUP_INTERVAL", 0)
        # No files/ directory created

        call_count = 0

        async def mock_sleep(duration):
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise asyncio.CancelledError

        with patch("kai.main.asyncio.sleep", side_effect=mock_sleep):
            try:
                await _file_cleanup_loop(30)
            except asyncio.CancelledError:
                pass
        # Should not raise

    @pytest.mark.asyncio
    async def test_handles_unlink_oserror(self, tmp_path, monkeypatch):
        """OSError during unlink is counted but doesn't crash the loop."""
        monkeypatch.setattr("kai.main.DATA_DIR", tmp_path)
        monkeypatch.setattr("kai.main._CLEANUP_STARTUP_DELAY", 0)
        monkeypatch.setattr("kai.main._CLEANUP_INTERVAL", 0)

        files_dir = tmp_path / "files"
        files_dir.mkdir()
        old_file = files_dir / "20260101_120000_000000_photo.jpg"
        old_file.write_bytes(b"old")

        call_count = 0

        async def mock_sleep(duration):
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise asyncio.CancelledError

        with (
            patch("kai.main.asyncio.sleep", side_effect=mock_sleep),
            patch.object(Path, "unlink", side_effect=OSError("permission denied")),
        ):
            try:
                await _file_cleanup_loop(30)
            except asyncio.CancelledError:
                pass
        # Should not raise - error is counted, not propagated

    @pytest.mark.asyncio
    async def test_rglob_exception_does_not_kill_loop(self, tmp_path, monkeypatch):
        """PermissionError from rglob is logged and the loop continues."""
        monkeypatch.setattr("kai.main.DATA_DIR", tmp_path)
        monkeypatch.setattr("kai.main._CLEANUP_STARTUP_DELAY", 0)
        monkeypatch.setattr("kai.main._CLEANUP_INTERVAL", 0)
        (tmp_path / "files").mkdir()

        call_count = 0

        async def mock_sleep(duration):
            nonlocal call_count
            call_count += 1
            if call_count > 2:
                raise asyncio.CancelledError

        with (
            patch("kai.main.asyncio.sleep", side_effect=mock_sleep),
            patch.object(Path, "rglob", side_effect=PermissionError("denied")),
            patch("kai.main.logging.exception") as mock_log,
        ):
            try:
                await _file_cleanup_loop(30)
            except asyncio.CancelledError:
                pass

        # Loop ran twice (not terminated after first exception)
        assert call_count == 3
        # Error was logged
        mock_log.assert_called()
        # Should not raise - error is counted, not propagated
