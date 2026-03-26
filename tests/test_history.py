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
    def test_creates_per_user_directory(self, _log_dir):
        """Log creates a per-user subdirectory and writes the file there."""
        log_message(direction="user", chat_id=1, text="hello")
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        path = _log_dir / "1" / f"{today}.jsonl"
        assert path.exists()
        assert (_log_dir / "1").is_dir()

    def test_appends_multiple_records(self, _log_dir):
        log_message(direction="user", chat_id=1, text="first")
        log_message(direction="assistant", chat_id=1, text="second")
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        lines = (_log_dir / "1" / f"{today}.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2

    def test_record_fields(self, _log_dir):
        log_message(direction="user", chat_id=42, text="hi", media={"type": "photo"})
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        line = (_log_dir / "42" / f"{today}.jsonl").read_text().strip()
        record = json.loads(line)
        assert record["dir"] == "user"
        assert record["chat_id"] == 42
        assert record["text"] == "hi"
        assert record["media"] == {"type": "photo"}
        assert "ts" in record

    def test_media_defaults_to_none(self, _log_dir):
        log_message(direction="user", chat_id=1, text="text only")
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        line = (_log_dir / "1" / f"{today}.jsonl").read_text().strip()
        record = json.loads(line)
        assert record["media"] is None

    def test_different_users_get_separate_directories(self, _log_dir):
        """Messages from different users go to different subdirectories."""
        log_message(direction="user", chat_id=111, text="from alice")
        log_message(direction="user", chat_id=222, text="from bob")
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        assert (_log_dir / "111" / f"{today}.jsonl").exists()
        assert (_log_dir / "222" / f"{today}.jsonl").exists()
        # Each file has exactly one record
        alice_lines = (_log_dir / "111" / f"{today}.jsonl").read_text().strip().splitlines()
        bob_lines = (_log_dir / "222" / f"{today}.jsonl").read_text().strip().splitlines()
        assert len(alice_lines) == 1
        assert len(bob_lines) == 1

    def test_no_flat_file_created(self, _log_dir):
        """New writes go to per-user dirs, not the flat _LOG_DIR root."""
        log_message(direction="user", chat_id=1, text="hello")
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        # No flat file at the root
        assert not (_log_dir / f"{today}.jsonl").exists()
        # File is in the per-user subdirectory
        assert (_log_dir / "1" / f"{today}.jsonl").exists()


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

    def test_reads_older_files(self, _log_dir):
        """History should scan back beyond yesterday to find messages."""
        yesterday = (datetime.now(UTC) - timedelta(days=1)).strftime("%Y-%m-%d")
        record = {
            "ts": f"{yesterday}T23:00:00+00:00",
            "dir": "user",
            "chat_id": 1,
            "text": "yesterday msg",
            "media": None,
        }
        (_log_dir / f"{yesterday}.jsonl").write_text(json.dumps(record) + "\n")
        # Also add a today message
        log_message(direction="assistant", chat_id=1, text="today msg")
        result = get_recent_history()
        assert "yesterday msg" in result
        assert "today msg" in result

    def test_scans_back_multiple_days(self, _log_dir):
        """Messages from several days ago should still be found."""
        # Write a message from 5 days ago — old code would miss this entirely
        old_date = (datetime.now(UTC) - timedelta(days=5)).strftime("%Y-%m-%d")
        record = {
            "ts": f"{old_date}T10:00:00+00:00",
            "dir": "user",
            "chat_id": 1,
            "text": "five days ago",
            "media": None,
        }
        (_log_dir / f"{old_date}.jsonl").write_text(json.dumps(record) + "\n")
        result = get_recent_history()
        assert "five days ago" in result

    def test_chronological_order_across_days(self, _log_dir):
        """Messages from multiple days should appear oldest-first."""
        # Create files for 3 days ago, 1 day ago, and today
        for days_back, msg in [(3, "three days"), (1, "one day"), (0, "today")]:
            date = (datetime.now(UTC) - timedelta(days=days_back)).strftime("%Y-%m-%d")
            record = {
                "ts": f"{date}T12:00:00+00:00",
                "dir": "user",
                "chat_id": 1,
                "text": msg,
                "media": None,
            }
            (_log_dir / f"{date}.jsonl").write_text(json.dumps(record) + "\n")
        result = get_recent_history()
        # All three should be present, in chronological order
        assert "three days" in result
        assert "one day" in result
        assert "today" in result
        assert result.index("three days") < result.index("one day") < result.index("today")

    def test_stops_scanning_when_enough_messages(self, _log_dir, monkeypatch):
        """Should stop reading older files once enough messages are collected."""
        monkeypatch.setattr(history, "_MAX_RECENT_MESSAGES", 3)
        # Write 2 messages today and 2 messages from 5 days ago
        for msg in ["today1", "today2"]:
            log_message(direction="user", chat_id=1, text=msg)
        old_date = (datetime.now(UTC) - timedelta(days=5)).strftime("%Y-%m-%d")
        for msg in ["old1", "old2"]:
            record = {
                "ts": f"{old_date}T12:00:00+00:00",
                "dir": "user",
                "chat_id": 1,
                "text": msg,
                "media": None,
            }
            with open(_log_dir / f"{old_date}.jsonl", "a") as f:
                f.write(json.dumps(record) + "\n")
        result = get_recent_history()
        # With max=3, should get the last 3: old2, today1, today2
        assert "today1" in result
        assert "today2" in result
        # old2 should be included (it's in the last 3 of the 4 total)
        assert "old2" in result
        # old1 should be excluded (it's the 4th oldest, beyond the cap)
        assert "old1" not in result


def test_log_dir_uses_data_dir():
    """Verify history module imports DATA_DIR, not PROJECT_ROOT."""
    import inspect

    source = inspect.getsource(__import__("kai.history", fromlist=["_LOG_DIR"]))
    # The module should use DATA_DIR for _LOG_DIR, not PROJECT_ROOT
    assert "DATA_DIR" in source
    assert '_LOG_DIR = DATA_DIR / "history"' in source


# ── Per-user history isolation ───────────────────────────────────────


class TestPerUserHistory:
    def test_reads_only_target_user(self, _log_dir):
        """get_recent_history(chat_id=X) returns only X's messages."""
        log_message(direction="user", chat_id=111, text="alice msg")
        log_message(direction="user", chat_id=222, text="bob msg")

        result = get_recent_history(chat_id=111)
        assert "alice msg" in result
        assert "bob msg" not in result

    def test_excludes_other_users(self, _log_dir):
        """Messages from user Y are not in user X's history."""
        log_message(direction="user", chat_id=111, text="from alice")
        log_message(direction="user", chat_id=222, text="from bob")

        result = get_recent_history(chat_id=222)
        assert "from bob" in result
        assert "from alice" not in result

    def test_none_chat_id_returns_all(self, _log_dir):
        """get_recent_history(chat_id=None) returns messages from all users."""
        log_message(direction="user", chat_id=111, text="alice")
        log_message(direction="user", chat_id=222, text="bob")

        result = get_recent_history(chat_id=None)
        assert "alice" in result
        assert "bob" in result

    def test_legacy_flat_files_included(self, _log_dir):
        """Legacy flat files (pre-per-user) are included in reads."""
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        # Simulate a legacy flat file at _LOG_DIR root
        legacy_record = {
            "ts": datetime.now(UTC).isoformat(),
            "dir": "user",
            "chat_id": 111,
            "text": "old message",
        }
        legacy_path = _log_dir / f"{today}.jsonl"
        legacy_path.write_text(json.dumps(legacy_record) + "\n")

        result = get_recent_history(chat_id=111)
        assert "old message" in result

    def test_new_user_empty_history(self, _log_dir):
        """A user with no history directory gets an empty string."""
        result = get_recent_history(chat_id=999)
        # No legacy files either, so should be empty
        assert result == ""
