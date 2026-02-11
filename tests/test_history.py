"""Tests for history.py message logging and retrieval."""

import json
from datetime import UTC, datetime, timedelta

import pytest

from kai import history
from kai.history import get_recent_history, log_message


@pytest.fixture(autouse=True)
def _log_dir(monkeypatch, tmp_path):
    """Redirect history log dir to a temp directory."""
    monkeypatch.setattr(history, "_LOG_DIR", tmp_path)
    return tmp_path


# ── log_message ──────────────────────────────────────────────────────


class TestLogMessage:
    def test_creates_jsonl_file(self, _log_dir):
        log_message(direction="user", chat_id=1, text="hello")
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        path = _log_dir / f"{today}.jsonl"
        assert path.exists()

    def test_appends_multiple_records(self, _log_dir):
        log_message(direction="user", chat_id=1, text="first")
        log_message(direction="assistant", chat_id=1, text="second")
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        lines = (_log_dir / f"{today}.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2

    def test_record_fields(self, _log_dir):
        log_message(direction="user", chat_id=42, text="hi", media={"type": "photo"})
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        line = (_log_dir / f"{today}.jsonl").read_text().strip()
        record = json.loads(line)
        assert record["dir"] == "user"
        assert record["chat_id"] == 42
        assert record["text"] == "hi"
        assert record["media"] == {"type": "photo"}
        assert "ts" in record

    def test_media_defaults_to_none(self, _log_dir):
        log_message(direction="user", chat_id=1, text="text only")
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        line = (_log_dir / f"{today}.jsonl").read_text().strip()
        record = json.loads(line)
        assert record["media"] is None


# ── get_recent_history ───────────────────────────────────────────────


class TestGetRecentHistory:
    def test_empty_when_no_files(self):
        assert get_recent_history() == ""

    def test_formats_messages(self, _log_dir):
        log_message(direction="user", chat_id=1, text="hello")
        log_message(direction="assistant", chat_id=1, text="hi there")
        result = get_recent_history()
        assert "You: hello" in result
        assert "Kai: hi there" in result

    def test_truncates_long_messages(self, _log_dir):
        long_text = "x" * 600
        log_message(direction="user", chat_id=1, text=long_text)
        result = get_recent_history()
        # _MAX_CHARS_PER_MESSAGE = 500, truncated with "..."
        assert "x" * 500 + "..." in result
        assert "x" * 501 not in result

    def test_limits_to_max_recent(self, _log_dir, monkeypatch):
        monkeypatch.setattr(history, "_MAX_RECENT_MESSAGES", 3)
        for i in range(5):
            log_message(direction="user", chat_id=1, text=f"msg{i}")
        result = get_recent_history()
        # Only last 3 messages
        assert "msg2" in result
        assert "msg3" in result
        assert "msg4" in result
        assert "msg0" not in result
        assert "msg1" not in result

    def test_reads_yesterday_file(self, _log_dir):
        yesterday = (datetime.now(UTC) - timedelta(days=1)).strftime("%Y-%m-%d")
        record = {
            "ts": f"{yesterday}T23:00:00+00:00",
            "dir": "user",
            "chat_id": 1,
            "text": "yesterday msg",
            "media": None,
        }
        (_log_dir / f"{yesterday}.jsonl").write_text(
            json.dumps(record) + "\n"
        )
        # Also add a today message
        log_message(direction="assistant", chat_id=1, text="today msg")
        result = get_recent_history()
        assert "yesterday msg" in result
        assert "today msg" in result
