"""
Tests for main.py - setup_logging() and _file_age / _file_cleanup_loop.

The main() and _init_and_run() functions orchestrate the full application
lifecycle and are impractical to unit test. The helper functions are
testable in isolation.

MEMORY.md bootstrapping moved to the backend (per-user, lazy) in #347;
coverage for that path now lives in tests/test_backend.py.
"""

import asyncio
import logging
from datetime import UTC, datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from kai.config import UserConfig
from kai.main import (
    _file_age,
    _file_cleanup_loop,
    _warn_if_compatibility_jobs_are_dormant,
    _workshop_bootstrap_humans,
    _workshop_bootstrap_notification_channels,
    setup_logging,
)
from tests.workshop_profiles import profile_id, profile_registry


class TestWorkshopBootstrapMapping:
    def test_only_interactive_user_identity_becomes_a_channel_binding(self):
        config = SimpleNamespace(
            user_configs={
                101: UserConfig(
                    telegram_id=101,
                    name="Admin",
                    role="admin",
                    github_notify_chat_id=-999,
                )
            }
        )

        humans = _workshop_bootstrap_humans(config, profile_registry(101))

        assert len(humans) == 1
        assert humans[0].external_subject == "101"
        assert humans[0].external_channel_id == "101"
        assert humans[0].role == "admin"
        assert humans[0].runtime_profile_id == profile_id(101)
        assert "-999" not in repr(humans)

    async def test_effective_negative_destinations_become_deduplicated_notification_channels(self):
        config = SimpleNamespace(
            user_configs={
                101: UserConfig(telegram_id=101, name="Admin", role="admin"),
                202: UserConfig(telegram_id=202, name="Second"),
                303: UserConfig(telegram_id=303, name="Direct"),
            }
        )

        async def effective(user_id, _config):
            return {"notify_chat_id": -999 if user_id in {101, 202} else 303}

        with patch("kai.main.sessions.resolve_github_settings", new=AsyncMock(side_effect=effective)):
            channels = await _workshop_bootstrap_notification_channels(config)

        assert len(channels) == 1
        assert channels[0].transport == "telegram"
        assert channels[0].external_channel_id == "-999"
        assert channels[0].member_external_subjects == ("101", "202")


class TestCompatibilityScheduleWarning:
    async def test_warns_when_active_jobs_are_dormant_without_telegram(self, caplog):
        config = SimpleNamespace(telegram_enabled=False)
        with (
            patch(
                "kai.main.sessions.get_all_active_jobs",
                new=AsyncMock(return_value=[{"id": 1}, {"id": 2}]),
            ),
            caplog.at_level(logging.WARNING),
        ):
            count = await _warn_if_compatibility_jobs_are_dormant(config)

        assert count == 2
        assert "2 active compatibility scheduled job(s) are dormant" in caplog.text

    async def test_does_not_query_jobs_when_telegram_is_enabled(self):
        config = SimpleNamespace(telegram_enabled=True)
        query = AsyncMock()
        with patch("kai.main.sessions.get_all_active_jobs", new=query):
            count = await _warn_if_compatibility_jobs_are_dormant(config)

        assert count == 0
        query.assert_not_awaited()


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


# ── main() startup-error choke point ─────────────────────────────────


class TestMainStartupErrorLogging:
    """main()'s SystemExit choke point mirrors fail-closed gate
    messages into the standard logger. Without it the message reaches
    only stderr, which launchd redirects away from the main log, so a
    gate exit reads as a hang to anyone tailing kai.log."""

    def test_gate_message_logged_critical(self, caplog, monkeypatch):
        from kai.main import main

        monkeypatch.setattr("kai.main.setup_logging", lambda: None)
        monkeypatch.setattr(
            "kai.main.load_config",
            lambda: (_ for _ in ()).throw(SystemExit("codex binary unreachable")),
        )

        with caplog.at_level(logging.CRITICAL), pytest.raises(SystemExit, match="codex binary unreachable"):
            main()

        assert any(
            r.levelno == logging.CRITICAL and "codex binary unreachable" in r.getMessage() for r in caplog.records
        )

    @pytest.mark.parametrize("code", [None, 0, 2])
    def test_clean_or_numeric_exits_stay_silent(self, caplog, monkeypatch, code):
        """SystemExit with no message (clean shutdowns, numeric exit
        codes) must not produce a CRITICAL line; only the string-
        message gate exits carry a diagnostic worth mirroring."""
        from kai.main import main

        monkeypatch.setattr("kai.main.setup_logging", lambda: None)
        monkeypatch.setattr(
            "kai.main.load_config",
            lambda: (_ for _ in ()).throw(SystemExit(code)),
        )

        with caplog.at_level(logging.CRITICAL), pytest.raises(SystemExit):
            main()

        assert not [r for r in caplog.records if r.levelno == logging.CRITICAL]


class TestStartCrashExit:
    """Unexpected runtime crashes must not look like clean process exits."""

    @staticmethod
    def _patch_minimal_config(monkeypatch):
        monkeypatch.setattr(
            "kai.main.load_config",
            lambda: SimpleNamespace(
                default_model="gpt-test",
                allowed_user_ids={123},
                telegram_webhook_url=None,
                session_db_path=":memory:",
            ),
        )

    @staticmethod
    def _patch_asyncio_crash(monkeypatch):
        def fail_run(coro):
            coro.close()
            raise RuntimeError("event loop died")

        monkeypatch.setattr("kai.main.asyncio.run", fail_run)

    def test_unexpected_asyncio_crash_exits_nonzero(self, caplog, monkeypatch):
        from kai.main import _start

        self._patch_minimal_config(monkeypatch)
        monkeypatch.setattr("kai.main._read_protected_file", lambda _path: "")
        monkeypatch.setattr("kai.main.services.load_services", lambda _path: {})
        self._patch_asyncio_crash(monkeypatch)

        with caplog.at_level(logging.ERROR), pytest.raises(SystemExit) as excinfo:
            _start()

        assert excinfo.value.code == 1
        assert any(r.levelno == logging.ERROR and "Kai crashed" in r.getMessage() for r in caplog.records)

    def test_empty_protected_services_yaml_disables_services_without_local_fallback(self, monkeypatch):
        """An empty readable /etc/kai/services.yaml is authoritative.

        Protected installs must not fall back to PROJECT_ROOT/services.yaml
        merely because the protected file is empty.
        """
        from kai.main import _start

        calls: list[tuple[str, str]] = []
        self._patch_minimal_config(monkeypatch)
        monkeypatch.setattr("kai.main._read_protected_file", lambda _path: "")
        monkeypatch.setattr(
            "kai.main.services.load_services_from_string",
            lambda text: calls.append(("protected", text)) or {},
        )
        monkeypatch.setattr(
            "kai.main.services.load_services",
            lambda path: calls.append(("local", str(path))) or {},
        )
        self._patch_asyncio_crash(monkeypatch)

        with pytest.raises(SystemExit):
            _start()

        assert calls == [("protected", "")]

    def test_unreadable_protected_services_yaml_uses_local_fallback(self, monkeypatch):
        """None from the protected reader means absent/unreadable, so dev
        fallback to PROJECT_ROOT/services.yaml is still preserved.
        """
        from kai.main import _start

        calls: list[tuple[str, str]] = []
        self._patch_minimal_config(monkeypatch)
        monkeypatch.setattr("kai.main._read_protected_file", lambda _path: None)
        monkeypatch.setattr(
            "kai.main.services.load_services_from_string",
            lambda text: calls.append(("protected", text)) or {},
        )
        monkeypatch.setattr(
            "kai.main.services.load_services",
            lambda path: calls.append(("local", str(path))) or {},
        )
        self._patch_asyncio_crash(monkeypatch)

        with pytest.raises(SystemExit):
            _start()

        assert calls == [("local", str(Path(__file__).resolve().parents[1] / "services.yaml"))]
