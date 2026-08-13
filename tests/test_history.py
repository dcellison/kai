"""Tests for history.py message logging and retrieval."""

import json
import stat
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, call

import pytest

from kai import history
from kai.history import get_recent_history, get_recent_pairs, log_message
from kai.workshop.domain import ChannelId
from kai.workshop.storage_namespaces import (
    WorkshopChannelHistoryNamespace,
    WorkshopChannelHistoryRegistry,
)


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

    def test_creates_private_history_tree(self, _log_dir):
        log_message(direction="user", chat_id=1, text="private")
        today = datetime.now(UTC).strftime("%Y-%m-%d")

        assert stat.S_IMODE(_log_dir.stat().st_mode) == 0o711
        assert stat.S_IMODE((_log_dir / "1").stat().st_mode) == 0o700
        assert stat.S_IMODE((_log_dir / "1" / f"{today}.jsonl").stat().st_mode) == 0o600

    def test_grants_mapped_reader_on_new_directory_and_file(self, _log_dir, monkeypatch):
        grant = MagicMock()
        monkeypatch.setattr(history, "grant_named_read_access", grant)

        result = log_message(
            direction="user",
            chat_id=1,
            text="private",
            reader_user="daniel",
        )
        today = datetime.now(UTC).strftime("%Y-%m-%d")

        assert result is not None
        assert grant.call_args_list == [
            call(_log_dir / "1", "daniel", directory=True),
            call(_log_dir / "1" / f"{today}.jsonl", "daniel", directory=False),
        ]

    def test_existing_file_does_not_accumulate_duplicate_acl(self, _log_dir, monkeypatch):
        grant = MagicMock()
        monkeypatch.setattr(history, "grant_named_read_access", grant)

        log_message(direction="user", chat_id=1, text="first", reader_user="daniel")
        grant.reset_mock()
        log_message(direction="user", chat_id=1, text="second", reader_user="daniel")

        grant.assert_not_called()

    def test_acl_failure_keeps_file_private_and_returns_none(self, _log_dir, monkeypatch):
        def fail_for_file(_path, _reader, *, directory):
            if not directory:
                raise OSError("acl failed")

        monkeypatch.setattr(history, "grant_named_read_access", fail_for_file)

        result = log_message(direction="user", chat_id=1, text="private", reader_user="daniel")
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        path = _log_dir / "1" / f"{today}.jsonl"

        assert result is None
        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    def test_refuses_symlinked_daily_file(self, _log_dir):
        user_dir = _log_dir / "1"
        user_dir.mkdir()
        target = _log_dir / "target"
        target.write_text("unchanged")
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        (user_dir / f"{today}.jsonl").symlink_to(target)

        result = log_message(direction="user", chat_id=1, text="attack")

        assert result is None
        assert target.read_text() == "unchanged"


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


# ── get_recent_pairs (issue #392) ────────────────────────────────────


class TestGetRecentPairs:
    """get_recent_pairs feeds the windowed episode classifier in
    memory_extraction (issue #392). It walks JSONL newest-first
    internally but returns oldest-first so callers can render
    PRIOR USER 1, 2, 3 in chronological order. Strict filtering
    matters here in a way it doesn't for get_recent_history: a
    botched-exchange placeholder in prior context distorts the
    classifier's closure heuristic.
    """

    @staticmethod
    def _write_records(log_dir, chat_id: int, day_offset: int, records: list[dict]) -> None:
        """Write raw JSONL records to a chat_id subdirectory for a
        specific date offset (0 = today, 1 = yesterday, ...).

        Bypasses log_message so tests can stage synthetic edge cases
        (missing chat_id field, synthetic markers, empty text) that
        the public API would never produce. day_offset > 0 is used
        by the cross-day chronological-order test.
        """
        date = (datetime.now(UTC) - timedelta(days=day_offset)).strftime("%Y-%m-%d")
        user_dir = log_dir / str(chat_id)
        user_dir.mkdir(parents=True, exist_ok=True)
        path = user_dir / f"{date}.jsonl"
        with path.open("a") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def test_returns_chronological_order_across_days(self, _log_dir):
        """Walking files newest-first internally must reverse to
        oldest-first on output. Three pairs spread across two days:
        the day-1 pair must appear BEFORE the day-0 pairs in the
        returned list."""
        # Day -1 (yesterday): one pair
        self._write_records(
            _log_dir,
            chat_id=1,
            day_offset=1,
            records=[
                {"dir": "user", "chat_id": 1, "text": "yesterday user"},
                {"dir": "assistant", "chat_id": 1, "text": "yesterday asst"},
            ],
        )
        # Day 0 (today): two pairs
        self._write_records(
            _log_dir,
            chat_id=1,
            day_offset=0,
            records=[
                {"dir": "user", "chat_id": 1, "text": "today user 1"},
                {"dir": "assistant", "chat_id": 1, "text": "today asst 1"},
                {"dir": "user", "chat_id": 1, "text": "today user 2"},
                {"dir": "assistant", "chat_id": 1, "text": "today asst 2"},
            ],
        )

        pairs = get_recent_pairs(chat_id=1, n=10)

        # All three pairs returned, oldest-first.
        assert pairs == [
            ("yesterday user", "yesterday asst"),
            ("today user 1", "today asst 1"),
            ("today user 2", "today asst 2"),
        ]

    def test_caps_at_n_keeping_most_recent(self, _log_dir):
        """When more pairs exist than requested, return the N most
        recent in chronological order. Pin the slice direction:
        slicing from the tail (rather than the head) is what makes
        'window of recent prior turns' work correctly."""
        records: list[dict] = []
        for i in range(1, 6):
            records.append({"dir": "user", "chat_id": 1, "text": f"u{i}"})
            records.append({"dir": "assistant", "chat_id": 1, "text": f"a{i}"})
        self._write_records(_log_dir, chat_id=1, day_offset=0, records=records)

        pairs = get_recent_pairs(chat_id=1, n=3)

        assert pairs == [("u3", "a3"), ("u4", "a4"), ("u5", "a5")]

    def test_skips_other_users(self, _log_dir):
        """User-partition filter: records belonging to a different
        chat_id must not leak into the returned pairs. Defends against
        a backup/restore that mis-placed a JSONL into the wrong
        subdirectory."""
        # Stage records for user 1, then write rogue records for user
        # 2 into user 1's directory directly (mimicking a misplaced
        # backup file). The chat_id field on each record is the
        # source of truth, not the directory location.
        date = datetime.now(UTC).strftime("%Y-%m-%d")
        user_dir = _log_dir / "1"
        user_dir.mkdir(parents=True, exist_ok=True)
        path = user_dir / f"{date}.jsonl"
        with path.open("w") as f:
            f.write(json.dumps({"dir": "user", "chat_id": 1, "text": "u1 mine"}) + "\n")
            f.write(json.dumps({"dir": "user", "chat_id": 2, "text": "u2 stranger"}) + "\n")
            f.write(json.dumps({"dir": "assistant", "chat_id": 2, "text": "a2 stranger"}) + "\n")
            f.write(json.dumps({"dir": "assistant", "chat_id": 1, "text": "a1 mine"}) + "\n")

        pairs = get_recent_pairs(chat_id=1, n=10)

        # Stranger's records are filtered out before pairing. The
        # remaining records pair u1 (mine) with a1 (mine) - the
        # stranger's u2/a2 do not interpose between them.
        assert pairs == [("u1 mine", "a1 mine")]

    def test_skips_synthetic_assistant_markers(self, _log_dir):
        """The failure-path placeholders written by bot.py
        (`[stopped by user]`, `[no response]`, `[error: ...]`) are
        not real conversation; pairing them into a windowed payload
        would feed the classifier a botched-exchange prior turn.
        Verify all three shapes are filtered out AND that an
        anchored-regex false-positive case (a real user message that
        quotes one of the markers as prose) is NOT mis-skipped."""
        self._write_records(
            _log_dir,
            chat_id=1,
            day_offset=0,
            records=[
                # Pair 1: real exchange (kept).
                {"dir": "user", "chat_id": 1, "text": "ask 1"},
                {"dir": "assistant", "chat_id": 1, "text": "real reply 1"},
                # Pair 2: assistant `[stopped by user]` synthetic - assistant filtered, leaves orphan user "stop test"
                {"dir": "user", "chat_id": 1, "text": "stop test"},
                {"dir": "assistant", "chat_id": 1, "text": "[stopped by user]"},
                # Pair 3: assistant `[error: TimeoutError]` synthetic - same shape
                {"dir": "user", "chat_id": 1, "text": "timeout test"},
                {"dir": "assistant", "chat_id": 1, "text": "[error: TimeoutError]"},
                # Pair 4: full-line-match false-positive case. A USER
                # message asking about an error literal is NOT an
                # assistant synthetic marker; even if it were on the
                # assistant side, the surrounding prose ("what does
                # ... mean?") means the FULL string does not match
                # the marker shape, so re.fullmatch correctly rejects.
                {"dir": "user", "chat_id": 1, "text": "what does [error: foo] mean?"},
                {"dir": "assistant", "chat_id": 1, "text": "real reply 2"},
                # Pair 5: assistant `[no response]` synthetic - same shape
                {"dir": "user", "chat_id": 1, "text": "noresp test"},
                {"dir": "assistant", "chat_id": 1, "text": "[no response]"},
            ],
        )

        pairs = get_recent_pairs(chat_id=1, n=10)

        # The synthetic-marker assistants are filtered out. Their
        # preceding user records become orphans (no following real
        # assistant before the next user), so the next user message
        # OVERWRITES the pending_user slot per the documented
        # multi-user-before-assistant collapse. Final pair shape:
        # ("ask 1", "real reply 1") and ("what does [error: foo] mean?", "real reply 2").
        # Note that pair 2 turns into "stop test" -> overwritten by
        # "timeout test" -> overwritten by "what does [error: foo] mean?"
        # which finally pairs with "real reply 2". Verifies that the
        # quoted-error user message survives the assistant-only
        # synthetic filter.
        assert pairs == [
            ("ask 1", "real reply 1"),
            ("what does [error: foo] mean?", "real reply 2"),
        ]

    def test_skips_empty_text(self, _log_dir):
        """Image-only and voice-only records can land in JSONL with
        empty `text` (the media field carries the payload metadata).
        The classifier path drops these because pairing produces a
        meaningless prior turn. A user record with empty text
        followed by a real assistant produces an orphan assistant
        (the orphan handler drops it)."""
        self._write_records(
            _log_dir,
            chat_id=1,
            day_offset=0,
            records=[
                # A real pair before, so the test asserts the empty-text
                # filter does not silently consume good records.
                {"dir": "user", "chat_id": 1, "text": "real q"},
                {"dir": "assistant", "chat_id": 1, "text": "real a"},
                # Empty user text followed by real assistant: filter
                # drops the empty user, leaving the assistant orphan.
                {"dir": "user", "chat_id": 1, "text": "", "media": {"type": "photo"}},
                {"dir": "assistant", "chat_id": 1, "text": "orphan asst"},
            ],
        )

        pairs = get_recent_pairs(chat_id=1, n=10)

        # Only the real pair survives. The empty-user-then-orphan-assistant
        # sequence produces zero pairs because the orphan assistant is dropped.
        assert pairs == [("real q", "real a")]

    def test_skips_records_without_chat_id_field(self, _log_dir):
        """Greenfield divergence from get_recent_history: pre-Phase-2
        legacy records (no chat_id field) are dropped entirely. The
        classifier path is sensitive to mis-attributed prior turns
        in a way the display path is not."""
        self._write_records(
            _log_dir,
            chat_id=1,
            day_offset=0,
            records=[
                # Legacy record (no chat_id field) - filtered out.
                {"dir": "user", "text": "legacy q"},
                {"dir": "assistant", "text": "legacy a"},
                # Modern record (has chat_id) - kept.
                {"dir": "user", "chat_id": 1, "text": "modern q"},
                {"dir": "assistant", "chat_id": 1, "text": "modern a"},
            ],
        )

        pairs = get_recent_pairs(chat_id=1, n=10)

        # Legacy pair filtered out; only the modern pair survives.
        assert pairs == [("modern q", "modern a")]

    def test_handles_unpaired_assistant(self, _log_dir):
        """An assistant record with no prior pending user (operator
        restored from backup, partial JSONL truncation, etc.) is
        dropped silently. The next real user/assistant pair then
        reads cleanly."""
        self._write_records(
            _log_dir,
            chat_id=1,
            day_offset=0,
            records=[
                # Orphan assistant at start of file (no prior user) -
                # dropped silently.
                {"dir": "assistant", "chat_id": 1, "text": "orphan"},
                # Real pair survives intact.
                {"dir": "user", "chat_id": 1, "text": "real q"},
                {"dir": "assistant", "chat_id": 1, "text": "real a"},
            ],
        )

        pairs = get_recent_pairs(chat_id=1, n=10)

        assert pairs == [("real q", "real a")]

    def test_multiple_users_before_assistant_collapse_to_last(self, _log_dir):
        """Telegram supports rapid-fire user messages before getting a
        reply. Pair against the LAST user message: that's the one the
        assistant is actually responding to, and it carries the most
        immediate intent. Pin this so a future change to use the FIRST
        user (or to emit multiple pairs per assistant) breaks the test
        rather than silently changing classifier semantics."""
        self._write_records(
            _log_dir,
            chat_id=1,
            day_offset=0,
            records=[
                {"dir": "user", "chat_id": 1, "text": "u1 first"},
                {"dir": "user", "chat_id": 1, "text": "u2 follow up"},
                {"dir": "user", "chat_id": 1, "text": "u3 final"},
                {"dir": "assistant", "chat_id": 1, "text": "asst reply"},
            ],
        )

        pairs = get_recent_pairs(chat_id=1, n=10)

        # Only the last user pairs with the assistant; u1 and u2 are
        # dropped as orphans.
        assert pairs == [("u3 final", "asst reply")]

    def test_no_history_returns_empty(self, _log_dir):
        """A user with no history directory returns [] without
        raising. Brand-new users hit this path on every extraction;
        the helper must handle it cleanly."""
        pairs = get_recent_pairs(chat_id=999, n=10)

        assert pairs == []

    def test_zero_n_returns_empty(self, _log_dir):
        """Defensive bound: `n=0` is the disable path that
        bot.py uses when EPISODE_CLASSIFIER_CONTEXT_TURNS=0. The
        helper short-circuits without touching the disk so the
        kill-switch path stays cheap."""
        # Stage real records that would otherwise return.
        self._write_records(
            _log_dir,
            chat_id=1,
            day_offset=0,
            records=[
                {"dir": "user", "chat_id": 1, "text": "u"},
                {"dir": "assistant", "chat_id": 1, "text": "a"},
            ],
        )

        pairs = get_recent_pairs(chat_id=1, n=0)

        assert pairs == []


# ── Transcript provenance: LogEntry contract ────────────────────────


class _Provenance:
    """Lightweight stand-in for kai.memory.TranscriptProvenance.

    Lets the helper tests construct lookup inputs without dragging
    Mem0 into the test surface; the real resolver is exercised in
    test_memory.py. Field names mirror the dataclass exactly.
    """

    def __init__(
        self,
        *,
        present=True,
        chat_id=1,
        date=None,
        user_ts=None,
        user_text_sha256=None,
        assistant_ts=None,
        date_end=None,
    ):
        self.present = present
        self.chat_id = chat_id
        self.date = date
        self.user_ts = user_ts
        self.user_text_sha256 = user_text_sha256
        self.assistant_ts = assistant_ts
        self.date_end = date_end


class TestLogEntryContract:
    def test_returns_log_entry_with_matching_fields(self, _log_dir):
        entry = log_message(direction="user", chat_id=42, text="hello")
        assert entry is not None
        assert entry.chat_id == 42
        assert entry.direction == "user"
        assert entry.text == "hello"
        # ts and date derive from the same now() call; the date must
        # match the filename the line landed in.
        assert (_log_dir / "42" / f"{entry.date}.jsonl").exists()
        # sha256 fingerprints the exact text byte sequence.
        import hashlib

        assert entry.sha256 == hashlib.sha256(b"hello").hexdigest()

    def test_failure_paths_still_return_entries(self, _log_dir):
        # Synthetic placeholder writes (the assistant failure markers)
        # also receive populated LogEntry returns; extraction does not
        # run on those paths, so the returns are simply unused.
        entry = log_message(direction="assistant", chat_id=1, text="[stopped by user]")
        assert entry is not None
        assert entry.text == "[stopped by user]"

    def test_oserror_returns_none(self, monkeypatch, _log_dir):
        """An OSError during the JSONL append yields None, not a
        populated entry that would point at a never-written line."""
        monkeypatch.setattr(history.os, "open", MagicMock(side_effect=OSError("disk full")))
        entry = log_message(direction="user", chat_id=1, text="lost")
        assert entry is None


class TestFetchTranscriptContext:
    """Black-box tests over fetch_transcript_context using the local
    `_Provenance` stub so the helper's behaviour is exercised without
    dragging the kai.memory resolver into the test inputs."""

    def _write_jsonl(self, _log_dir, chat_id, date, records):
        path = _log_dir / str(chat_id) / f"{date}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        return path

    def test_legacy_returns_legacy_reason(self):
        result = history.fetch_transcript_context(_Provenance(present=False))
        assert result.reason == "legacy"
        assert result.context is None

    def test_happy_path_returns_target_and_window(self, _log_dir):
        import hashlib

        self._write_jsonl(
            _log_dir,
            1,
            "2026-06-13",
            [
                {"ts": "2026-06-13T08:00:00+00:00", "dir": "user", "chat_id": 1, "text": "prev user"},
                {"ts": "2026-06-13T08:00:10+00:00", "dir": "assistant", "chat_id": 1, "text": "prev assistant"},
                {"ts": "2026-06-13T09:00:00+00:00", "dir": "user", "chat_id": 1, "text": "target user"},
                {"ts": "2026-06-13T09:00:30+00:00", "dir": "assistant", "chat_id": 1, "text": "target assistant"},
                {"ts": "2026-06-13T09:30:00+00:00", "dir": "user", "chat_id": 1, "text": "next user"},
            ],
        )
        provenance = _Provenance(
            chat_id=1,
            date="2026-06-13",
            user_ts="2026-06-13T09:00:00+00:00",
            user_text_sha256=hashlib.sha256(b"target user").hexdigest(),
            assistant_ts="2026-06-13T09:00:30+00:00",
        )
        result = history.fetch_transcript_context(provenance, before=2, after=1)
        assert result.reason == "ok"
        assert result.context.target_user.text == "target user"
        assert result.context.target_assistant.text == "target assistant"
        assert [t.text for t in result.context.before] == ["prev user", "prev assistant"]
        assert [t.text for t in result.context.after] == ["next user"]

    def test_file_missing_logs_drift(self, _log_dir, caplog):
        provenance = _Provenance(
            chat_id=99,
            date="2026-01-01",
            user_ts="2026-01-01T00:00:00+00:00",
            user_text_sha256="abc",
        )
        with caplog.at_level("INFO", logger="kai.history"):
            result = history.fetch_transcript_context(provenance, memory_id="mem-1")
        assert result.reason == "file_missing"
        # Drift log fired with the row id and reason.
        drift = next(r for r in caplog.records if "memory.provenance.drift" in r.message)
        assert '"memory_id":"mem-1"' in drift.message
        assert '"reason":"file_missing"' in drift.message

    def test_ts_not_found_user_side(self, _log_dir, caplog):
        self._write_jsonl(
            _log_dir,
            1,
            "2026-06-13",
            [{"ts": "2026-06-13T01:00:00+00:00", "dir": "user", "chat_id": 1, "text": "x"}],
        )
        provenance = _Provenance(
            chat_id=1,
            date="2026-06-13",
            user_ts="2026-06-13T09:00:00+00:00",
            user_text_sha256="any",
        )
        with caplog.at_level("INFO", logger="kai.history"):
            result = history.fetch_transcript_context(provenance, memory_id="m")
        assert result.reason == "ts_not_found"
        drift = next(r for r in caplog.records if "memory.provenance.drift" in r.message)
        assert '"side":"user"' in drift.message

    def test_hash_mismatch_returns_reason_and_logs(self, _log_dir, caplog):
        self._write_jsonl(
            _log_dir,
            1,
            "2026-06-13",
            [{"ts": "2026-06-13T09:00:00+00:00", "dir": "user", "chat_id": 1, "text": "actual"}],
        )
        provenance = _Provenance(
            chat_id=1,
            date="2026-06-13",
            user_ts="2026-06-13T09:00:00+00:00",
            user_text_sha256="wrong" * 12,
        )
        with caplog.at_level("INFO", logger="kai.history"):
            result = history.fetch_transcript_context(provenance, memory_id="m")
        assert result.reason == "hash_mismatch"
        assert result.context is None
        assert any("hash_mismatch" in r.message for r in caplog.records)

    def test_assistant_missing_returns_ts_not_found(self, _log_dir, caplog):
        """The drift case where user matches but the named assistant
        ts is absent from JSONL: drift-not-ok, side=assistant."""
        import hashlib

        self._write_jsonl(
            _log_dir,
            1,
            "2026-06-13",
            [{"ts": "2026-06-13T09:00:00+00:00", "dir": "user", "chat_id": 1, "text": "user msg"}],
        )
        provenance = _Provenance(
            chat_id=1,
            date="2026-06-13",
            user_ts="2026-06-13T09:00:00+00:00",
            user_text_sha256=hashlib.sha256(b"user msg").hexdigest(),
            assistant_ts="2026-06-13T09:00:30+00:00",
        )
        with caplog.at_level("INFO", logger="kai.history"):
            result = history.fetch_transcript_context(provenance, memory_id="m")
        assert result.reason == "ts_not_found"
        drift = next(r for r in caplog.records if "memory.provenance.drift" in r.message)
        assert '"side":"assistant"' in drift.message

    def test_legitimately_no_assistant_returns_ok_with_none(self, _log_dir):
        """A row whose stored provenance has no assistant_ts at all
        (None) returns ok with target_assistant=None."""
        import hashlib

        self._write_jsonl(
            _log_dir,
            1,
            "2026-06-13",
            [{"ts": "2026-06-13T09:00:00+00:00", "dir": "user", "chat_id": 1, "text": "u"}],
        )
        provenance = _Provenance(
            chat_id=1,
            date="2026-06-13",
            user_ts="2026-06-13T09:00:00+00:00",
            user_text_sha256=hashlib.sha256(b"u").hexdigest(),
            assistant_ts=None,
        )
        result = history.fetch_transcript_context(provenance)
        assert result.reason == "ok"
        assert result.context.target_assistant is None

    def test_synthetic_markers_excluded_from_context(self, _log_dir):
        import hashlib

        self._write_jsonl(
            _log_dir,
            1,
            "2026-06-13",
            [
                {"ts": "2026-06-13T07:00:00+00:00", "dir": "user", "chat_id": 1, "text": "real user"},
                {"ts": "2026-06-13T07:00:10+00:00", "dir": "assistant", "chat_id": 1, "text": "[stopped by user]"},
                {"ts": "2026-06-13T09:00:00+00:00", "dir": "user", "chat_id": 1, "text": "target"},
                {"ts": "2026-06-13T09:00:30+00:00", "dir": "assistant", "chat_id": 1, "text": "answer"},
            ],
        )
        provenance = _Provenance(
            chat_id=1,
            date="2026-06-13",
            user_ts="2026-06-13T09:00:00+00:00",
            user_text_sha256=hashlib.sha256(b"target").hexdigest(),
            assistant_ts="2026-06-13T09:00:30+00:00",
        )
        result = history.fetch_transcript_context(provenance, before=5, after=0)
        # Synthetic assistant placeholder is skipped; only the real
        # prior user turn appears in `before`.
        assert [t.text for t in result.context.before] == ["real user"]

    def test_assistant_in_next_day_file(self, _log_dir):
        """Episode midnight cross: user on day D, assistant on day D+1."""
        import hashlib

        self._write_jsonl(
            _log_dir,
            1,
            "2026-06-12",
            [{"ts": "2026-06-12T23:59:00+00:00", "dir": "user", "chat_id": 1, "text": "u"}],
        )
        self._write_jsonl(
            _log_dir,
            1,
            "2026-06-13",
            [{"ts": "2026-06-13T00:00:30+00:00", "dir": "assistant", "chat_id": 1, "text": "a"}],
        )
        provenance = _Provenance(
            chat_id=1,
            date="2026-06-12",
            user_ts="2026-06-12T23:59:00+00:00",
            user_text_sha256=hashlib.sha256(b"u").hexdigest(),
            assistant_ts="2026-06-13T00:00:30+00:00",
            date_end="2026-06-13",
        )
        result = history.fetch_transcript_context(provenance, after=0)
        assert result.reason == "ok"
        assert result.context.target_assistant.text == "a"

    def test_before_window_walks_to_previous_file(self, _log_dir):
        import hashlib

        self._write_jsonl(
            _log_dir,
            1,
            "2026-06-12",
            [{"ts": "2026-06-12T23:00:00+00:00", "dir": "user", "chat_id": 1, "text": "yesterday"}],
        )
        self._write_jsonl(
            _log_dir,
            1,
            "2026-06-13",
            [{"ts": "2026-06-13T00:30:00+00:00", "dir": "user", "chat_id": 1, "text": "today target"}],
        )
        provenance = _Provenance(
            chat_id=1,
            date="2026-06-13",
            user_ts="2026-06-13T00:30:00+00:00",
            user_text_sha256=hashlib.sha256(b"today target").hexdigest(),
            assistant_ts=None,
        )
        result = history.fetch_transcript_context(provenance, before=2, after=0)
        # `before` walks one file back and pulls the yesterday user
        # turn rather than returning a short window.
        assert [t.text for t in result.context.before] == ["yesterday"]


class TestFetchTranscriptContextChatOwnership:
    """Cross-chat pointer protection: when expected_chat_id disagrees
    with provenance.chat_id, the helper refuses to dereference."""

    def test_mismatch_returns_chat_mismatch_before_disk(self, _log_dir, caplog):
        import hashlib

        # Write a file for chat 2 that DOES contain the target line.
        # The helper must not read it on behalf of expected_chat_id=1.
        path = _log_dir / "2" / "2026-06-13.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"ts": "2026-06-13T09:00:00+00:00", "dir": "user", "chat_id": 2, "text": "secret"}) + "\n"
        )
        provenance = _Provenance(
            chat_id=2,  # the row's source pointer (bad data on a chat-1 row)
            date="2026-06-13",
            user_ts="2026-06-13T09:00:00+00:00",
            user_text_sha256=hashlib.sha256(b"secret").hexdigest(),
        )
        with caplog.at_level("INFO", logger="kai.history"):
            result = history.fetch_transcript_context(provenance, expected_chat_id=1, memory_id="m")
        assert result.reason == "chat_mismatch"
        assert result.context is None
        # The drift log fired with the new reason.
        assert any('"reason":"chat_mismatch"' in r.message for r in caplog.records)

    def test_match_proceeds(self, _log_dir):
        import hashlib

        path = _log_dir / "1" / "2026-06-13.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"ts": "2026-06-13T09:00:00+00:00", "dir": "user", "chat_id": 1, "text": "ok"}) + "\n"
        )
        provenance = _Provenance(
            chat_id=1,
            date="2026-06-13",
            user_ts="2026-06-13T09:00:00+00:00",
            user_text_sha256=hashlib.sha256(b"ok").hexdigest(),
        )
        result = history.fetch_transcript_context(provenance, expected_chat_id=1)
        assert result.reason == "ok"

    def test_no_expected_chat_id_skips_check(self, _log_dir):
        """Admin callers (cross-chat scans) get the old behaviour when
        they omit the gate."""
        import hashlib

        path = _log_dir / "5" / "2026-06-13.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"ts": "2026-06-13T09:00:00+00:00", "dir": "user", "chat_id": 5, "text": "x"}) + "\n"
        )
        provenance = _Provenance(
            chat_id=5,
            date="2026-06-13",
            user_ts="2026-06-13T09:00:00+00:00",
            user_text_sha256=hashlib.sha256(b"x").hexdigest(),
        )
        result = history.fetch_transcript_context(provenance)
        assert result.reason == "ok"


class TestFetchTranscriptContextMidnightCrossAfterWindow:
    """The after-window walks into the next-day file when the assistant
    target itself lives there: a follow-up turn on day D+1 must appear
    in `lookup.context.after`."""

    def test_after_window_includes_next_day_turn(self, _log_dir):
        import hashlib

        prev = _log_dir / "1" / "2026-06-12.jsonl"
        nextp = _log_dir / "1" / "2026-06-13.jsonl"
        prev.parent.mkdir(parents=True, exist_ok=True)
        prev.write_text(
            json.dumps({"ts": "2026-06-12T23:59:00+00:00", "dir": "user", "chat_id": 1, "text": "u"}) + "\n"
        )
        nextp.write_text(
            json.dumps({"ts": "2026-06-13T00:00:30+00:00", "dir": "assistant", "chat_id": 1, "text": "a"})
            + "\n"
            + json.dumps({"ts": "2026-06-13T00:05:00+00:00", "dir": "user", "chat_id": 1, "text": "follow-up"})
            + "\n"
        )
        provenance = _Provenance(
            chat_id=1,
            date="2026-06-12",
            user_ts="2026-06-12T23:59:00+00:00",
            user_text_sha256=hashlib.sha256(b"u").hexdigest(),
            assistant_ts="2026-06-13T00:00:30+00:00",
            date_end="2026-06-13",
        )
        result = history.fetch_transcript_context(provenance, before=0, after=1)
        assert result.reason == "ok"
        assert [t.text for t in result.context.after] == ["follow-up"]


class TestLogMessageMkdirFailure:
    def test_mkdir_oserror_returns_none(self, monkeypatch, _log_dir):
        """Directory creation failures now share the LogEntry | None
        contract with append failures (P3 boundary fix)."""
        from pathlib import Path

        original_mkdir = Path.mkdir

        def boom(self, *args, **kwargs):
            if str(self).startswith(str(_log_dir)):
                raise OSError("read-only filesystem")
            return original_mkdir(self, *args, **kwargs)

        monkeypatch.setattr(Path, "mkdir", boom)
        entry = log_message(direction="user", chat_id=99, text="x")
        assert entry is None


class TestCanonicalChannelHistoryNamespace:
    @staticmethod
    def _registry() -> WorkshopChannelHistoryRegistry:
        return WorkshopChannelHistoryRegistry(
            (
                WorkshopChannelHistoryNamespace(
                    ChannelId("chn_" + "a" * 32),
                    101,
                ),
            )
        )

    def test_new_writes_use_only_canonical_channel_directory(
        self,
        _log_dir,
        monkeypatch,
    ):
        monkeypatch.setattr(history, "_CHANNEL_HISTORY_REGISTRY", self._registry())

        entry = log_message(direction="user", chat_id=101, text="canonical")

        assert entry is not None
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        assert (_log_dir / ("chn_" + "a" * 32) / f"{today}.jsonl").is_file()
        assert not (_log_dir / "101").exists()

    def test_reads_merge_legacy_and_canonical_same_day_chronologically(
        self,
        _log_dir,
        monkeypatch,
    ):
        monkeypatch.setattr(history, "_CHANNEL_HISTORY_REGISTRY", self._registry())
        legacy = _log_dir / "101" / "2026-08-13.jsonl"
        canonical = _log_dir / ("chn_" + "a" * 32) / "2026-08-13.jsonl"
        legacy.parent.mkdir(parents=True)
        canonical.parent.mkdir(parents=True)
        legacy.write_text(
            json.dumps(
                {
                    "ts": "2026-08-13T09:00:00+00:00",
                    "dir": "user",
                    "chat_id": 101,
                    "text": "before cutover",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        canonical.write_text(
            json.dumps(
                {
                    "ts": "2026-08-13T09:01:00+00:00",
                    "dir": "assistant",
                    "chat_id": 101,
                    "text": "after cutover",
                }
            )
            + "\n",
            encoding="utf-8",
        )

        assert get_recent_pairs(101, 1) == [("before cutover", "after cutover")]
        recent = get_recent_history(chat_id=101)
        assert recent.index("before cutover") < recent.index("after cutover")

    def test_unknown_chat_does_not_recreate_numeric_namespace(
        self,
        _log_dir,
        monkeypatch,
    ):
        monkeypatch.setattr(history, "_CHANNEL_HISTORY_REGISTRY", self._registry())

        assert log_message(direction="user", chat_id=202, text="refuse") is None
        assert not (_log_dir / "202").exists()
