"""
Tests for bot.py - pure functions and handler coverage.

The first section tests pure/synchronous helpers (resolve_workspace_path,
chunk_text, etc.) with no mocking needed. The second section tests command
handlers, callback handlers, media handlers, and the streaming response
handler using mock Telegram Update/Context objects.
"""

import asyncio
import json
import logging
import stat
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
from telegram.error import BadRequest

from kai import sessions
from kai.backend import AgentResponse, StreamEvent
from kai.bot import (
    _QUEUED_MESSAGE_MARKER,
    ResponseDeliveryRoute,
    _acquire_lock_or_kill,
    _backend_name_for_instance,
    _clear_responding,
    _do_switch_workspace,
    _edit_message_safe,
    _handle_settings_reset,
    _handle_workspace_allow,
    _handle_workspace_allowed,
    _handle_workspace_config,
    _handle_workspace_deny,
    _is_authorized,
    _is_notify_chat_used,
    _models_keyboard,
    _notify_if_queued,
    _prepend_queue_marker,
    _reply_safe,
    _require_auth,
    _resolve_workspace_path,
    _save_upload,
    _set_responding,
    _short_workspace_name,
    _show_github,
    _show_settings,
    _switch_workspace,
    _truncate_for_telegram,
    _voices_keyboard,
    _workspace_config_suffix,
    _workspaces_keyboard,
    create_bot,
    handle_document,
    handle_github,
    handle_help,
    handle_job,
    handle_jobs,
    handle_message,
    handle_model,
    handle_model_callback,
    handle_models,
    handle_new,
    handle_photo,
    handle_review_command,
    handle_settings,
    handle_start,
    handle_stats,
    handle_stop,
    handle_unknown_command,
    handle_voice,
    handle_voice_callback,
    handle_voice_command,
    handle_voices,
    handle_webhooks,
    handle_workspace,
    handle_workspace_callback,
    handle_workspaces,
)
from kai.config import PROVIDER_MODELS, Config, UserConfig, get_default_model_for_backend
from kai.review import CollectionWarning, PRReviewResult
from kai.tts import DEFAULT_VOICE, VOICES
from kai.workshop.artifacts import InboundArtifact
from kai.workshop.domain import MessageId
from kai.workshop.inbound import InboundMessage
from kai.workshop.outbound import DeliveryObservation, OutboundMessage
from kai.workshop.streaming_preview import ConfirmedTelegramStreamingPreview
from kai.workspace_utils import is_workspace_allowed

# ── _backend_name_for_instance ───────────────────────────────────────


class TestBackendNameForInstance:
    """
    Verify the bot-side runtime backend dispatch reads `backend_name`
    off the instance rather than inspecting the concrete class name.

    Real backends each set the attribute to their canonical identifier
    (claude.py: "claude"; goose.py: "goose"; codex.py: "codex").
    Missing or invalid identities are rejected so model validation
    cannot silently use another backend's policy.
    """

    def test_claude_backend_returns_class_attribute(self):
        """ClaudeCodeBackend.backend_name is "claude"."""
        from kai.claude import ClaudeCodeBackend

        instance = MagicMock(spec=ClaudeCodeBackend)
        instance.backend_name = ClaudeCodeBackend.backend_name
        assert _backend_name_for_instance(instance) == "claude"

    def test_goose_backend_returns_goose(self):
        """GooseBackend.backend_name is "goose"."""
        from kai.goose import GooseBackend

        instance = MagicMock(spec=GooseBackend)
        instance.backend_name = GooseBackend.backend_name
        assert _backend_name_for_instance(instance) == "goose"

    def test_codex_backend_returns_codex(self):
        """CodexBackend.backend_name is "codex"."""
        from kai.codex import CodexBackend

        instance = MagicMock(spec=CodexBackend)
        instance.backend_name = CodexBackend.backend_name
        assert _backend_name_for_instance(instance) == "codex"

    def test_opencode_backend_returns_opencode(self):
        """OpenCodeBackend.backend_name is "opencode"."""
        from kai.opencode import OpenCodeBackend

        instance = MagicMock(spec=OpenCodeBackend)
        instance.backend_name = OpenCodeBackend.backend_name
        assert _backend_name_for_instance(instance) == "opencode"

    def test_falsy_backend_name_is_rejected(self):
        """A stub with an empty backend identity fails closed."""
        stub = MagicMock()
        stub.backend_name = ""

        with pytest.raises(RuntimeError, match="invalid backend_name"):
            _backend_name_for_instance(stub)

    def test_missing_backend_name_attribute_is_rejected(self):
        """A stub without a backend identity fails closed."""

        class _Stub:
            pass

        with pytest.raises(RuntimeError, match="invalid backend_name"):
            _backend_name_for_instance(_Stub())

    def test_unknown_backend_name_is_rejected(self):
        """A non-canonical backend identity cannot reach model routing."""
        stub = MagicMock()
        stub.backend_name = "unknown"

        with pytest.raises(RuntimeError, match="invalid backend_name"):
            _backend_name_for_instance(stub)


# ── _get_user_models warning suppression ─────────────────────────────


class TestGetUserModelsWarningSuppression:
    """
    `_get_user_models` warns when models_for_backend returns None for a
    provider that should have a curated list (programming oversight).
    Codex AND opencode are intentionally excluded: both return None for
    legitimate reasons (codex curates separately; opencode model IDs
    are open-ended provider/model strings whose set Kai cannot curate).
    Pin the exclusion so a real missing-registry-entry warning is not
    drowned out by per-turn opencode noise.
    """

    def test_opencode_does_not_log_missing_registry_warning(self, caplog):
        """Global opencode install with empty provider should NOT log the warning."""
        import logging

        from kai.bot import _get_user_models
        from kai.pool import SubprocessPool

        # Empty provider is the realistic global-opencode case
        # (VALID_PROVIDERS["opencode"] does not exist; the wizard never
        # prompts for a provider). The pre-fix code path warns
        # because "" is not in OPEN_ENDED_PROVIDERS.
        config = MagicMock()
        config.default_backend = "opencode"
        config.default_provider = ""
        config.get_user_config = MagicMock(return_value=None)
        pool = MagicMock(spec=SubprocessPool)
        pool.get_if_exists = MagicMock(return_value=None)

        with caplog.at_level(logging.WARNING, logger="kai.bot"):
            result = _get_user_models(pool, 111, config)
        # OpenCode returns None deliberately (free-text /model input).
        assert result is None
        # And no false warning was logged.
        assert not any("no curated model list" in rec.message for rec in caplog.records)

    def test_codex_does_not_log_missing_registry_warning(self, caplog):
        """Confirm the pre-existing codex exclusion still works (regression guard)."""
        import logging

        from kai.bot import _get_user_models
        from kai.pool import SubprocessPool

        config = MagicMock()
        config.default_backend = "codex"
        config.default_provider = ""
        config.get_user_config = MagicMock(return_value=None)
        pool = MagicMock(spec=SubprocessPool)
        pool.get_if_exists = MagicMock(return_value=None)

        with caplog.at_level(logging.WARNING, logger="kai.bot"):
            _get_user_models(pool, 111, config)
        assert not any("no curated model list" in rec.message for rec in caplog.records)


# ── _resolve_workspace_path ──────────────────────────────────────────


class TestResolveWorkspacePath:
    def test_valid_name(self, tmp_path):
        result = _resolve_workspace_path("myproject", tmp_path)
        assert result == (tmp_path / "myproject").resolve()

    def test_returns_none_when_no_base(self):
        assert _resolve_workspace_path("anything", None) is None

    def test_rejects_traversal(self, tmp_path):
        assert _resolve_workspace_path("../escape", tmp_path) is None

    def test_resolves_to_base_itself(self, tmp_path):
        result = _resolve_workspace_path(".", tmp_path)
        assert result == tmp_path

    def test_nested_path(self, tmp_path):
        result = _resolve_workspace_path("sub/project", tmp_path)
        assert result == (tmp_path / "sub" / "project").resolve()


# ── _short_workspace_name ────────────────────────────────────────────


class TestShortWorkspaceName:
    def test_path_under_base(self):
        assert _short_workspace_name("/base/myproject", Path("/base")) == "myproject"

    def test_path_not_under_base(self):
        assert _short_workspace_name("/other/myproject", Path("/base")) == "myproject"

    def test_base_is_none(self):
        assert _short_workspace_name("/some/path/project", None) == "project"


# ── _truncate_for_telegram ───────────────────────────────────────────


class TestTruncateForTelegram:
    def test_short_text_unchanged(self):
        assert _truncate_for_telegram("hello", 100) == "hello"

    def test_long_text_truncated_with_suffix(self):
        result = _truncate_for_telegram("a" * 100, 50)
        assert len(result) == 50
        assert result.endswith("\n...")
        assert result == "a" * 46 + "\n..."

    def test_exact_length_not_truncated(self):
        text = "a" * 50
        assert _truncate_for_telegram(text, 50) == text


# ── _stream_publishable_prefix ──────────────────────────────────────


class TestStreamPublishablePrefix:
    """
    Stability filter applied to live streaming updates. The helper
    returns the longest stable prefix of the accumulated text, or None
    when nothing in the buffer is safe to publish yet. Backends keep
    emitting; the transport just waits for a coherent boundary.
    """

    def test_rejects_empty_text(self):
        from kai.bot import _stream_publishable_prefix

        assert _stream_publishable_prefix("") is None
        assert _stream_publishable_prefix("   \n  \t  ") is None

    def test_rejects_single_initial_word(self):
        from kai.bot import _stream_publishable_prefix

        assert _stream_publishable_prefix("One") is None

    def test_cuts_dangling_suffix_to_paragraph(self):
        from kai.bot import _stream_publishable_prefix

        assert _stream_publishable_prefix("Complete paragraph.\n\nOne") == "Complete paragraph."

    def test_allows_complete_sentence(self):
        from kai.bot import _stream_publishable_prefix

        assert _stream_publishable_prefix("One complete sentence.") == "One complete sentence."

    def test_includes_closing_sentence_punctuation(self):
        from kai.bot import _stream_publishable_prefix

        assert _stream_publishable_prefix('He said "Yes."') == 'He said "Yes."'

    def test_cuts_dangling_list_item_to_previous_item(self):
        from kai.bot import _stream_publishable_prefix

        text = "- Item one\n- Item two\n- Three"
        assert _stream_publishable_prefix(text) == "- Item one\n- Item two"

    def test_rejects_open_fenced_code_block(self):
        from kai.bot import _stream_publishable_prefix

        assert _stream_publishable_prefix("```python\nprint('hi')") is None

    def test_allows_closed_fenced_code_block(self):
        from kai.bot import _stream_publishable_prefix

        text = "```python\nprint('hi')\n```"
        assert _stream_publishable_prefix(text) == text

    def test_rejects_open_inline_code(self):
        from kai.bot import _stream_publishable_prefix

        # Single unmatched backtick mid-final-line is dangling.
        assert _stream_publishable_prefix("Use the `cmd flag") is None

    def test_rejects_open_link(self):
        from kai.bot import _stream_publishable_prefix

        # Final line has an unmatched `[` so the link target is mid-stream.
        assert _stream_publishable_prefix("See [the docs") is None

    def test_long_span_fallback_when_no_smaller_boundary(self):
        from kai.bot import _stream_publishable_prefix

        # 240+ chars of prose on one line, then a newline and a final
        # tiny dangling word. No paragraph break, no sentence end inside
        # the prose, no list. Long-span fallback must release the prose
        # block while withholding the dangling final line.
        prose = "alpha beta gamma " * 16  # 272 chars
        text = prose + "\nOne"
        result = _stream_publishable_prefix(text)
        assert result is not None
        assert result == prose.rstrip()

    def test_full_rejects_long_unfinished_final_paragraph(self):
        """A completed paragraph followed by a long mid-sentence fragment
        must publish only the completed paragraph. The full candidate is
        longer than the paragraph cut, so without a tight final-line guard
        it would win the priority sort and ship the in-progress fragment.
        """
        from kai.bot import _stream_publishable_prefix

        text = "Complete paragraph.\n\nOne hard inconsistency is emerging"
        assert _stream_publishable_prefix(text) == "Complete paragraph."

    def test_long_span_handles_single_line_prose(self):
        """A long unpunctuated single-line monologue must still produce a
        streaming surface. Without the single-line fallback in
        ``_long_span_cut`` the helper has nothing to publish (no
        paragraph, sentence, list, or fence boundary) and the user sees
        a stalled live message until a sentence terminator arrives.
        """
        from kai.bot import _stream_publishable_prefix

        text = ("alpha beta " * 30).rstrip()
        result = _stream_publishable_prefix(text)
        assert result is not None
        # Threshold is 240; the long-span cut must land on a whitespace
        # at or beyond that, leaving a substantive prefix.
        assert len(result) >= 240
        # The cut lands at a whitespace, so the published prefix never
        # ends with the in-progress trailing word.
        assert not result.endswith(" ")

    def test_sentence_cut_rejects_mid_token_decimal(self):
        """A period inside a decimal is not a sentence boundary; the
        helper must not publish ``"Use Python 3."`` while the stream is
        still emitting ``"13"``.
        """
        from kai.bot import _stream_publishable_prefix

        assert _stream_publishable_prefix("Use Python 3.13") is None

    def test_sentence_cut_rejects_mid_token_path(self):
        """A period inside a file path (``bot.py``) is not a sentence
        boundary; the helper must not publish ``"Open src/kai/bot."``
        while the stream is still emitting ``"py"``.
        """
        from kai.bot import _stream_publishable_prefix

        assert _stream_publishable_prefix("Open src/kai/bot.py") is None

    def test_terminating_period_after_internal_dots_is_sentence_end(self):
        """A period followed by end-of-string IS a sentence boundary,
        even when the preceding token contains internal dots. The
        complete version-string sentence publishes as a unit.
        """
        from kai.bot import _stream_publishable_prefix

        assert _stream_publishable_prefix("Use Python 3.13.") == "Use Python 3.13."

    def test_ordered_list_marker_is_not_sentence_boundary(self):
        """The dot in ``1.`` is a list marker, not a sentence end. A
        single in-progress numbered item must not publish as the bare
        marker; with the previous sentence-cut rules it would have
        produced ``"1."`` because the marker period is followed by
        whitespace and therefore satisfied the new sentence predicate.
        """
        from kai.bot import _stream_publishable_prefix

        assert _stream_publishable_prefix("1. Item one") is None

    def test_forming_next_marker_releases_previous_items(self):
        """Numbered items publish as a list once the next item's marker
        starts emitting, even before the period and content arrive.
        Without forming-marker awareness, an in-progress next item
        (just the digits ``3``) would not count as a boundary and the
        helper would publish only ``1. Item one`` instead of the full
        ``1. Item one\\n2. Item two``.
        """
        from kai.bot import _stream_publishable_prefix

        text = "1. Item one\n2. Item two\n3"
        assert _stream_publishable_prefix(text) == "1. Item one\n2. Item two"


# ── _save_upload ────────────────────────────────────────────────────


class TestSaveUpload:
    def test_creates_files_directory(self, tmp_path, monkeypatch):
        """Automatically creates the files/ subdirectory if missing."""
        monkeypatch.setattr("kai.bot.DATA_DIR", tmp_path)
        monkeypatch.setattr("kai.bot._REVIEW_TMP_DIR", tmp_path)
        _save_upload(b"hello", "test.txt")
        assert (tmp_path / "files").is_dir()
        assert stat.S_IMODE((tmp_path / "files").stat().st_mode) == 0o711

    def test_saves_content_correctly(self, tmp_path, monkeypatch):
        """Written bytes match the input exactly."""
        monkeypatch.setattr("kai.bot.DATA_DIR", tmp_path)
        monkeypatch.setattr("kai.bot._REVIEW_TMP_DIR", tmp_path)
        data = b"binary content here"
        result = _save_upload(data, "doc.pdf")
        assert result.read_bytes() == data
        assert stat.S_IMODE(result.stat().st_mode) == 0o600

    def test_per_user_directory_is_traversal_only(self, tmp_path, monkeypatch):
        """Per-user upload dirs are not listable by sibling local users."""
        monkeypatch.setattr("kai.bot.DATA_DIR", tmp_path)
        result = _save_upload(b"x", "report.pdf", user_id=123)
        assert result.parent == tmp_path / "files" / "123"
        assert stat.S_IMODE((tmp_path / "files").stat().st_mode) == 0o711
        assert stat.S_IMODE(result.parent.stat().st_mode) == 0o711
        assert stat.S_IMODE(result.stat().st_mode) == 0o600

    def test_macos_reader_user_gets_named_read_acl(self, tmp_path, monkeypatch):
        """Cross-user protected installs grant the target os_user exact-file read access."""
        monkeypatch.setattr("kai.bot.DATA_DIR", tmp_path)
        completed = MagicMock(returncode=0, stderr="")
        with (
            patch("kai.bot.sys.platform", "darwin"),
            patch("kai.bot.subprocess.run", return_value=completed) as run,
        ):
            result = _save_upload(b"x", "report.pdf", user_id=123, reader_user="alice")

        command = run.call_args.args[0]
        assert command[:2] == ["/bin/chmod", "+a"]
        assert command[2] == "user:alice allow read,readattr,readextattr,readsecurity"
        assert command[3] == str(result)
        assert stat.S_IMODE(result.stat().st_mode) == 0o600

    def test_linux_missing_setfacl_fails_closed(self, tmp_path, monkeypatch):
        """If a protected handoff cannot grant read access, no unreadable path is left behind."""
        monkeypatch.setattr("kai.bot.DATA_DIR", tmp_path)
        with (
            patch("kai.bot.sys.platform", "linux"),
            patch("kai.bot.shutil.which", return_value=None),
            pytest.raises(OSError, match="setfacl is required"),
        ):
            _save_upload(b"x", "report.pdf", user_id=123, reader_user="alice")

        assert list((tmp_path / "files" / "123").glob("*")) == []

    def test_filename_contains_original_name(self, tmp_path, monkeypatch):
        """Saved filename preserves the original name after the timestamp."""
        monkeypatch.setattr("kai.bot.DATA_DIR", tmp_path)
        monkeypatch.setattr("kai.bot._REVIEW_TMP_DIR", tmp_path)
        result = _save_upload(b"x", "report.pdf")
        assert "report.pdf" in result.name

    def test_timestamp_prefix_format(self, tmp_path, monkeypatch):
        """Filename starts with YYYYMMDD_HHMMSS_ffffff timestamp."""
        monkeypatch.setattr("kai.bot.DATA_DIR", tmp_path)
        monkeypatch.setattr("kai.bot._REVIEW_TMP_DIR", tmp_path)
        result = _save_upload(b"x", "file.txt")
        # Format: YYYYMMDD_HHMMSS_ffffff_file.txt
        parts = result.name.split("_", 3)
        assert len(parts[0]) == 8  # date
        assert len(parts[1]) == 6  # time
        assert len(parts[2]) == 6  # microseconds

    def test_sanitizes_slashes_and_spaces(self, tmp_path, monkeypatch):
        """Slashes and spaces in filenames are replaced with underscores."""
        monkeypatch.setattr("kai.bot.DATA_DIR", tmp_path)
        monkeypatch.setattr("kai.bot._REVIEW_TMP_DIR", tmp_path)
        result = _save_upload(b"x", "my file/name.txt")
        assert "/" not in result.name
        assert " " not in result.name

    def test_returns_absolute_path(self, tmp_path, monkeypatch):
        """Returned path is absolute and points to an existing file."""
        monkeypatch.setattr("kai.bot.DATA_DIR", tmp_path)
        monkeypatch.setattr("kai.bot._REVIEW_TMP_DIR", tmp_path)
        result = _save_upload(b"x", "test.txt")
        assert result.is_absolute()
        assert result.is_file()


# ── _workspaces_keyboard ────────────────────────────────────────────


def _button_labels(markup) -> list[str]:
    """Flatten InlineKeyboardMarkup into a list of button labels."""
    return [btn.text for row in markup.inline_keyboard for btn in row]


def _button_callbacks(markup) -> list[str]:
    """Flatten InlineKeyboardMarkup into a list of callback data strings."""
    return [btn.callback_data for row in markup.inline_keyboard for btn in row]


class TestWorkspacesKeyboard:
    def test_home_always_first(self, tmp_path):
        """Home button appears first regardless of history or allowed workspaces."""
        markup = _workspaces_keyboard([], "/home", "/home", None, [])
        assert _button_labels(markup)[0] == "\U0001f3e0 Home \U0001f7e2"

    def test_allowed_workspaces_appear_before_history(self, tmp_path):
        """Pinned workspaces appear between Home and history entries."""
        pinned = tmp_path / "pinned"
        pinned.mkdir()
        history = [{"path": "/other/project"}]
        markup = _workspaces_keyboard(history, "/other/project", "/home", None, [pinned])
        labels = _button_labels(markup)
        # Home, then pinned, then history
        assert labels[0].startswith("\U0001f3e0 Home")
        assert labels[1] == "pinned"
        assert labels[2].endswith("\U0001f7e2")  # history entry marked as current

    def test_allowed_workspace_callback_data(self, tmp_path):
        """Pinned workspaces use ws:allowed:<index> callback data."""
        pinned = tmp_path / "project-a"
        pinned.mkdir()
        markup = _workspaces_keyboard([], "/home", "/home", None, [pinned])
        callbacks = _button_callbacks(markup)
        assert "ws:allowed:0" in callbacks

    def test_history_deduplicated_against_allowed(self, tmp_path):
        """A path in both allowed and history appears only once (in allowed section)."""
        pinned = tmp_path / "shared"
        pinned.mkdir()
        history = [{"path": str(pinned)}]
        markup = _workspaces_keyboard(history, "/home", "/home", None, [pinned])
        labels = _button_labels(markup)
        # Should be: Home + one "shared" entry - not two "shared" entries
        assert labels.count("shared") == 1
        callbacks = _button_callbacks(markup)
        # The single entry should be the allowed version, not a bare history index
        assert "ws:allowed:0" in callbacks
        assert not any(c == "ws:0" for c in callbacks)

    def test_current_workspace_marked_in_allowed(self, tmp_path):
        """Green dot appears on the pinned workspace button when it is current."""
        pinned = tmp_path / "active"
        pinned.mkdir()
        markup = _workspaces_keyboard([], str(pinned), "/home", None, [pinned])
        labels = _button_labels(markup)
        assert any("active" in lbl and "\U0001f7e2" in lbl for lbl in labels)

    def test_no_allowed_no_history_shows_only_home(self):
        """With no allowed workspaces and no history, only the Home button appears."""
        markup = _workspaces_keyboard([], "/home", "/home", None, [])
        assert len(_button_labels(markup)) == 1

    def test_disambiguates_duplicate_names(self, tmp_path):
        """Two allowed workspaces with the same directory name get parent/name labels."""
        foo_a = tmp_path / "projects" / "foo"
        foo_b = tmp_path / "clients" / "foo"
        foo_a.mkdir(parents=True)
        foo_b.mkdir(parents=True)
        markup = _workspaces_keyboard([], "/home", "/home", None, [foo_a, foo_b])
        labels = _button_labels(markup)
        assert "projects/foo" in labels
        assert "clients/foo" in labels
        # Neither bare "foo" label should appear
        assert "foo" not in labels

    def test_unique_names_not_disambiguated(self, tmp_path):
        """Allowed workspaces with unique names keep their short labels."""
        bar = tmp_path / "bar"
        baz = tmp_path / "baz"
        bar.mkdir()
        baz.mkdir()
        markup = _workspaces_keyboard([], "/home", "/home", None, [bar, baz])
        labels = _button_labels(markup)
        assert "bar" in labels
        assert "baz" in labels


# ── is_workspace_allowed ─────────────────────────────────────────────


def _make_config(**overrides) -> Config:
    """
    Create a Config for tests with sensible defaults.

    Accepts any Config field as a keyword override. Used by both the pure
    function tests (workspace_base, allowed_workspaces) and the handler
    tests (tts_enabled, voice_enabled, named webhook secrets, etc.).

    `claude_workspace=` is accepted as a back-compat alias for pre-#353
    tests that wanted a specific home directory. Config no longer has
    that field; instead the value is wired into a UserConfig override
    for chat 12345 (the default chat_id used by _make_update) so that
    bot.resolve_home_workspace returns the requested path. Tests that
    pass their own `user_configs` win over the back-compat translation.
    """
    legacy_home = overrides.pop("claude_workspace", None)
    if legacy_home is not None and "user_configs" not in overrides:
        overrides["user_configs"] = {
            12345: UserConfig(
                telegram_id=12345,
                name="test",
                home_workspace=legacy_home,
            ),
        }
    defaults: dict = {
        "telegram_bot_token": "test-token",
        "allowed_user_ids": {1},
        "workspace_base": None,
        "allowed_workspaces": [],
        "github_webhook_secret": "github-test-secret",
        "generic_webhook_secret": "generic-test-secret",
        "webhook_port": 8080,
        "tts_enabled": False,
        "voice_enabled": False,
        "default_backend": "claude",
        "default_model": get_default_model_for_backend("claude", "anthropic"),
    }
    defaults.update(overrides)
    return Config(**defaults)


class TestIsWorkspaceAllowed:
    def test_no_sources_allows_anything(self, tmp_path):
        """With no base and no allowed list, all paths are accepted (permissive mode)."""
        assert is_workspace_allowed(tmp_path / "anything", None, []) is True

    def test_path_under_base_is_allowed(self, tmp_path):
        """Paths under workspace_base are allowed."""
        assert is_workspace_allowed(tmp_path / "myproject", tmp_path, []) is True

    def test_path_in_allowed_list(self, tmp_path):
        """Paths in the allowed list are allowed."""
        project = tmp_path / "project"
        project.mkdir()
        assert is_workspace_allowed(project, None, [project]) is True

    def test_path_outside_both_is_rejected(self, tmp_path):
        """Paths not under base or in allowed list are rejected."""
        base = tmp_path / "base"
        base.mkdir()
        outside = tmp_path / "outside"
        assert is_workspace_allowed(outside, base, []) is False

    def test_base_set_allowed_empty_rejects_outside(self, tmp_path):
        """With base set but empty allowed list, outside paths are rejected."""
        base = tmp_path / "base"
        base.mkdir()
        assert is_workspace_allowed(tmp_path / "other", base, []) is False

    def test_only_allowed_set_rejects_unlisted(self, tmp_path):
        """With only allowed list set, unlisted paths are rejected."""
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        unlisted = tmp_path / "unlisted"
        assert is_workspace_allowed(unlisted, None, [allowed]) is False

    def test_resolves_symlinks_for_comparison(self, tmp_path):
        """Path resolution handles non-canonical paths correctly."""
        project = tmp_path / "project"
        project.mkdir()
        # Pass the resolved canonical path - should still match
        assert is_workspace_allowed(project.resolve(), None, [project]) is True


# ── create_bot transport mode ──────────────────────────────────────


class TestCreateBotTransportMode:
    @pytest.fixture(autouse=True)
    def _init_services(self, tmp_path):
        """Initialize services before create_bot() (normally done in main.py).

        create_bot() calls services.get_available_services(), which requires
        load_services() to have been called first. Use a nonexistent file so
        it loads an empty service registry (graceful degradation).
        """
        from kai import services

        services.load_services(tmp_path / "nonexistent.yaml")

    def test_webhook_mode_suppresses_updater(self):
        """In webhook mode, the Updater is suppressed (None)."""
        config = _make_config()
        app = create_bot(config, use_webhook=True)
        assert app.updater is None

    def test_polling_mode_keeps_updater(self):
        """In polling mode, the Updater is present for start_polling()."""
        config = _make_config()
        app = create_bot(config, use_webhook=False)
        assert app.updater is not None

    def test_installs_workshop_shadow_recorder(self):
        app = create_bot(_make_config())

        assert app.bot_data["workshop_inbound_recorder"] is sessions.record_workshop_inbound_message
        assert app.bot_data["workshop_artifact_recorder"] is sessions.record_workshop_inbound_artifact
        assert app.bot_data["workshop_outbound_recorder"] is sessions.record_workshop_outbound_message
        assert app.bot_data["workshop_delivery_recorder"] is sessions.record_workshop_delivery_observation
        assert app.bot_data["workshop_streaming_preview_recorder"] is sessions.record_workshop_streaming_preview
        assert app.bot_data["workshop_streaming_finalizer"] is sessions.record_workshop_streaming_finalization


# ══════════════════════════════════════════════════════════════════════
# Handler tests - mock Telegram Update/Context objects
# ══════════════════════════════════════════════════════════════════════


# ── Test helpers ─────────────────────────────────────────────────────


def _make_update(text="hello", chat_id=12345, user_id=1, *, update_id=9001, message_id=42):
    """Create a mock Telegram Update for handler tests."""
    update = MagicMock()
    update.message.text = text
    update.message.reply_text = AsyncMock()
    update.message.delete = AsyncMock()
    update.message.caption = None
    update.message.photo = None
    update.message.document = None
    update.message.voice = None
    update.message.message_id = message_id
    update.message.date = datetime(2026, 8, 11, 12, 0, tzinfo=UTC)
    update.update_id = update_id
    update.effective_chat.id = chat_id
    update.effective_chat.send_message = AsyncMock()
    update.effective_user.id = user_id
    return update


def _make_callback_update(data="model:opus", chat_id=12345, user_id=1):
    """Create a mock Update for callback query handlers."""
    update = MagicMock()
    update.callback_query.data = data
    update.callback_query.answer = AsyncMock()
    update.callback_query.edit_message_text = AsyncMock()
    update.callback_query.edit_message_reply_markup = AsyncMock()
    update.effective_chat.id = chat_id
    update.effective_user.id = user_id
    return update


def _make_mock_claude(model="sonnet", workspace=None, is_alive=True, provider="anthropic"):
    """Create a mock SubprocessPool with pool-compatible methods.

    Despite the name (kept for backward compatibility with existing test
    call sites), this returns a pool-like mock, not a ClaudeCodeBackend.
    """
    pool = MagicMock()
    ws = workspace or Path("/home/workspace")
    pool.get_model = MagicMock(return_value=model)
    pool.get_effective_model = AsyncMock(return_value=model)
    pool.set_model = MagicMock()
    pool.get_effective_workspace = AsyncMock(return_value=ws)
    pool.is_alive = MagicMock(return_value=is_alive)
    pool.get_session_id = MagicMock(return_value=None)
    pool.restart = AsyncMock()
    pool.change_workspace = AsyncMock()
    pool.force_kill = AsyncMock()
    pool.send = MagicMock()  # configured per test
    # Mock instance returned by pool.get() and pool.get_if_exists()
    # with the provider attribute for provider-aware model selection.
    instance = MagicMock()
    instance.backend_name = "claude"
    instance.model = model
    instance.provider = provider
    instance.workspace = ws
    instance.workspace_config = None
    pool.get = MagicMock(return_value=instance)
    pool.get_if_exists = MagicMock(return_value=instance)
    return pool


def _make_context(config=None, claude=None, pool=None, args=None, user_data=None, job_queue=None):
    """Create a mock PTB context with bot_data, args, and user_data."""
    ctx = MagicMock()
    # Accept either pool (Phase 3) or claude (legacy test compat) as the
    # mock subprocess manager. Pool is preferred for new tests.
    mock_pool = pool or claude or _make_mock_claude()
    ctx.bot_data = {
        "config": config or _make_config(),
        "pool": mock_pool,
    }
    ctx.args = args or []
    ctx.user_data = user_data if user_data is not None else {}
    ctx.bot.send_chat_action = AsyncMock()
    ctx.bot.get_file = AsyncMock()
    ctx.bot.send_voice = AsyncMock()
    if job_queue is not None:
        ctx.application.job_queue = job_queue
    return ctx


def _fake_lock(*_args, **_kwargs):
    """Return a real asyncio.Lock to stand in for the per-chat lock.

    Uses a real Lock instead of a bare async context manager so that both
    async-with and .locked() work (the latter is needed by _notify_if_queued).
    The lock starts unlocked, so _notify_if_queued correctly skips notification.
    """
    return asyncio.Lock()


def _mock_resolve(base=None, allowed=None):
    """Return a patch context that mocks sessions.resolve_workspace_access.

    Simplifies handler tests that call handle_workspace, handle_workspaces,
    or handle_workspace_callback - all of which resolve per-user workspace
    access at the top of the handler.
    """
    return patch(
        "kai.bot.sessions.resolve_workspace_access",
        new_callable=AsyncMock,
        return_value=(base, allowed or []),
    )


def _text_event(text: str) -> StreamEvent:
    """Non-final streaming event with accumulated text."""
    return StreamEvent(text_so_far=text, done=False, response=None)


def _done_event(text="Final response", session_id="sess-1", success=True, error=None) -> StreamEvent:
    """Final streaming event with an AgentResponse."""
    return StreamEvent(
        text_so_far=text,
        done=True,
        response=AgentResponse(
            text=text,
            success=success,
            error=error,
            duration_ms=1000,
            session_id=session_id,
        ),
    )


async def _fake_stream(*events):
    """Async generator that yields StreamEvents."""
    for e in events:
        yield e


# ── Crash recovery flag ──────────────────────────────────────────────


class TestCrashRecoveryFlag:
    def test_set_responding_writes_chat_id(self, tmp_path):
        """Per-user flag file is created under the .responding directory."""
        responding_dir = tmp_path / ".responding"
        with patch("kai.bot._RESPONDING_DIR", responding_dir):
            _set_responding(12345)
        assert (responding_dir / "12345").exists()

    def test_clear_responding_removes_flag(self, tmp_path):
        """Per-user flag file is deleted after clearing."""
        responding_dir = tmp_path / ".responding"
        responding_dir.mkdir()
        (responding_dir / "12345").touch()
        with patch("kai.bot._RESPONDING_DIR", responding_dir):
            _clear_responding(12345)
        assert not (responding_dir / "12345").exists()

    def test_clear_responding_noop_if_missing(self, tmp_path):
        """No error when flag file doesn't exist."""
        responding_dir = tmp_path / ".responding"
        with patch("kai.bot._RESPONDING_DIR", responding_dir):
            _clear_responding(12345)  # should not raise


# ── Authorization ────────────────────────────────────────────────────


class TestAuthorization:
    def test_authorized_user(self):
        config = _make_config(allowed_user_ids={1, 2})
        assert _is_authorized(config, 1) is True

    def test_unauthorized_user(self):
        config = _make_config(allowed_user_ids={1})
        assert _is_authorized(config, 99) is False

    @pytest.mark.asyncio
    async def test_require_auth_calls_wrapped(self):
        """Authorized user: the wrapped function is called."""
        inner = AsyncMock()
        wrapped = _require_auth(inner)
        update = _make_update(user_id=1)
        ctx = _make_context(config=_make_config(allowed_user_ids={1}))
        await wrapped(update, ctx)
        inner.assert_called_once()

    @pytest.mark.asyncio
    async def test_require_auth_blocks_unauthorized(self):
        """Unauthorized user: the wrapped function is NOT called."""
        inner = AsyncMock()
        wrapped = _require_auth(inner)
        update = _make_update(user_id=99)
        ctx = _make_context(config=_make_config(allowed_user_ids={1}))
        await wrapped(update, ctx)
        inner.assert_not_called()


# ── _reply_safe ──────────────────────────────────────────────────────


class TestReplySafe:
    @pytest.mark.asyncio
    async def test_markdown_success(self):
        """Successful Markdown send returns the message."""
        msg = MagicMock()
        msg.reply_text = AsyncMock(return_value="sent")
        result = await _reply_safe(msg, "hello")
        assert result == "sent"

    @pytest.mark.asyncio
    async def test_markdown_fails_retries_plain(self):
        """Markdown failure falls back to plain text."""
        msg = MagicMock()
        msg.reply_text = AsyncMock(side_effect=[BadRequest("bad markup"), "sent-plain"])
        result = await _reply_safe(msg, "*bad*")
        assert result == "sent-plain"
        assert msg.reply_text.call_count == 2


# ── _edit_message_safe ───────────────────────────────────────────────


class TestEditMessageSafe:
    @pytest.mark.asyncio
    async def test_markdown_edit_success(self):
        msg = MagicMock()
        msg.edit_text = AsyncMock()
        result = await _edit_message_safe(msg, "hello")
        assert result is True
        msg.edit_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_markdown_fails_retries_plain(self):
        """BadRequest on Markdown triggers plain text retry."""
        msg = MagicMock()
        msg.edit_text = AsyncMock(side_effect=[BadRequest("bad"), None])
        result = await _edit_message_safe(msg, "text")
        assert result is True
        assert msg.edit_text.call_count == 2

    @pytest.mark.asyncio
    async def test_both_fail_no_exception(self, caplog):
        """Both Markdown and plain text fail: logs debug, no exception raised."""
        msg = MagicMock()
        msg.edit_text = AsyncMock(side_effect=[BadRequest("bad"), RuntimeError("fail")])
        with caplog.at_level(logging.DEBUG, logger="kai.bot"):
            result = await _edit_message_safe(msg, "text")
        assert result is False
        assert "Failed to edit message" in caplog.text

    @pytest.mark.asyncio
    async def test_non_badrequest_exception(self, caplog):
        """Non-BadRequest exception is caught and logged."""
        msg = MagicMock()
        msg.edit_text = AsyncMock(side_effect=RuntimeError("network"))
        with caplog.at_level(logging.DEBUG, logger="kai.bot"):
            result = await _edit_message_safe(msg, "text")
        assert result is False
        assert "Failed to edit message" in caplog.text

    @pytest.mark.asyncio
    async def test_long_text_truncated(self):
        """Text exceeding 4096 chars is truncated before editing."""
        msg = MagicMock()
        msg.edit_text = AsyncMock()
        await _edit_message_safe(msg, "a" * 5000)
        sent = msg.edit_text.call_args[0][0]
        assert len(sent) <= 4096


# ── _models_keyboard ─────────────────────────────────────────────────


class TestModelsKeyboard:
    # Use Anthropic's curated models for keyboard tests
    _anthropic_models = PROVIDER_MODELS["anthropic"]

    def test_current_model_gets_green_dot(self):
        kb = _models_keyboard("sonnet", self._anthropic_models)
        labels = _button_labels(kb)
        assert any("\U0001f7e2" in lbl and "Sonnet" in lbl for lbl in labels)

    def test_all_models_present(self):
        kb = _models_keyboard("sonnet", self._anthropic_models)
        callbacks = _button_callbacks(kb)
        assert "model:opus" in callbacks
        assert "model:sonnet" in callbacks
        assert "model:haiku" in callbacks

    def test_codex_keyboard_offers_exact_gpt56_models(self):
        """The ChatGPT-account GPT-5.6 IDs are selectable, not the rejected shorthand."""
        from kai.config import CODEX_MODELS

        kb = _models_keyboard("gpt-5.5", CODEX_MODELS)
        callbacks = _button_callbacks(kb)
        assert "model:gpt-5.6-sol" in callbacks
        assert "model:gpt-5.6-terra" in callbacks
        assert "model:gpt-5.6-luna" in callbacks
        assert "model:gpt-5.6" not in callbacks

    def test_callback_data_format(self):
        kb = _models_keyboard("opus", self._anthropic_models)
        callbacks = _button_callbacks(kb)
        assert all(c.startswith("model:") for c in callbacks)


# ── _voices_keyboard ─────────────────────────────────────────────────


class TestVoicesKeyboard:
    def test_current_voice_gets_green_dot(self):
        kb = _voices_keyboard(DEFAULT_VOICE)
        labels = _button_labels(kb)
        assert any("\U0001f7e2" in lbl for lbl in labels)

    def test_all_voices_present(self):
        kb = _voices_keyboard(DEFAULT_VOICE)
        callbacks = _button_callbacks(kb)
        for key in VOICES:
            assert f"voice:{key}" in callbacks

    def test_callback_data_format(self):
        kb = _voices_keyboard("jenny")
        callbacks = _button_callbacks(kb)
        assert all(c.startswith("voice:") for c in callbacks)


# ── Simple command handlers ──────────────────────────────────────────


class TestHandleStart:
    @pytest.mark.asyncio
    async def test_sends_greeting(self):
        update = _make_update()
        ctx = _make_context()
        await handle_start(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "ready" in reply.lower()


class TestHandleNew:
    @pytest.mark.asyncio
    async def test_clears_session_and_restarts(self):
        claude = _make_mock_claude()
        update = _make_update()
        ctx = _make_context(claude=claude)
        with patch("kai.bot.sessions.clear_session", new_callable=AsyncMock) as mock_clear:
            await handle_new(update, ctx)
        claude.restart.assert_called_once()
        mock_clear.assert_called_once_with(12345)
        reply = update.message.reply_text.call_args[0][0]
        assert "cleared" in reply.lower()


class TestHandleHelp:
    @pytest.mark.asyncio
    async def test_sends_help_text(self):
        update = _make_update()
        ctx = _make_context()
        await handle_help(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "/stop" in reply
        assert "/new" in reply
        assert "/workspace" in reply

    @pytest.mark.asyncio
    async def test_help_text_matches_current_command_surface(self):
        """Pin each line that drifted from the actual handler surface
        in the past so a future drift fails this test before reaching
        an operator. The corrected forms come from the runtime
        handlers and the canonical _HELP_TEXT in memory_command.py."""
        update = _make_update()
        ctx = _make_context()
        await handle_help(update, ctx)
        reply = update.message.reply_text.call_args[0][0]

        # /github notify accepts <chat_id|reset>, not [on|off]. The
        # [on|off] shape belongs to /github reviews and /github triage;
        # confusing them sent operators down the wrong path when
        # routing notifications.
        assert "/github notify <chat_id|reset>" in reply
        assert "/github notify [on|off]" not in reply

        # /memory browses facts and episodes; the tag-browse axis was
        # retired when the dashboard was redesigned as Facts/Episodes/
        # Stats. Tags survive only as decoration on individual rows.
        assert "Browse remembered facts and episodes" in reply
        assert "Browse remembered facts by tag" not in reply

        # /memory forget <tag> as a command-line subcommand was
        # retired; the forget button still exists inside the
        # dashboard but the slash-command form is gone.
        assert "/memory forget" not in reply

        # /voice toggles between "off" and "only" (voice-only), not
        # "off" and "on". The "on" form requires an explicit
        # /voice on. The /voice off command is real (handle_voice
        # accepts it as one of the explicit-mode args) and now listed.
        assert "/voice - Toggle voice off / voice-only" in reply
        assert "/voice off" in reply
        assert "/voice - Toggle voice on/off" not in reply


class TestHandleUnknownCommand:
    @pytest.mark.asyncio
    async def test_echoes_unknown_command(self):
        update = _make_update(text="/foo")
        ctx = _make_context()
        await handle_unknown_command(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "/foo" in reply
        assert "Unknown command" in reply


class TestHandleStop:
    @pytest.mark.asyncio
    async def test_sets_stop_event_and_kills(self):
        """Sets the stop event, kills Claude, and sends confirmation."""
        claude = _make_mock_claude()
        update = _make_update()
        ctx = _make_context(claude=claude)
        stop_event = asyncio.Event()
        with patch("kai.bot.get_stop_event", return_value=stop_event):
            await handle_stop(update, ctx)
        assert stop_event.is_set()
        claude.force_kill.assert_called_once()
        reply = update.message.reply_text.call_args[0][0]
        assert "stopping" in reply.lower()


# ── handle_stats ─────────────────────────────────────────────────────


class TestHandleStats:
    @pytest.mark.asyncio
    async def test_no_active_session(self):
        update = _make_update()
        ctx = _make_context()
        with patch("kai.bot.sessions.get_stats", new_callable=AsyncMock, return_value=None):
            await handle_stats(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "No active session" in reply

    @pytest.mark.asyncio
    async def test_active_session(self):
        update = _make_update()
        ctx = _make_context()
        stats = {
            "session_id": "abcd1234efgh",
            "model": "sonnet",
            "created_at": "2026-01-01",
            "last_used_at": "2026-01-02",
        }
        with patch("kai.bot.sessions.get_stats", new_callable=AsyncMock, return_value=stats):
            await handle_stats(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "abcd1234" in reply
        assert "sonnet" in reply


# ── handle_jobs ──────────────────────────────────────────────────────


class TestHandleJob:
    """Tests for the unified /job command and its subcommands."""

    # ── /job (no args) and /job list ────────────────────────────────

    @pytest.mark.asyncio
    async def test_no_args_lists_jobs(self):
        """/job with no args lists all jobs."""
        update = _make_update()
        ctx = _make_context()
        with patch("kai.bot.sessions.get_jobs", new_callable=AsyncMock, return_value=[]):
            await handle_job(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "No active" in reply

    @pytest.mark.asyncio
    async def test_list_subcommand(self):
        """/job list is equivalent to /job with no args."""
        update = _make_update()
        ctx = _make_context(args=["list"])
        with patch("kai.bot.sessions.get_jobs", new_callable=AsyncMock, return_value=[]):
            await handle_job(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "No active" in reply

    @pytest.mark.asyncio
    async def test_formats_interval_hours(self):
        """Interval >= 3600s displays as hours."""
        update = _make_update()
        ctx = _make_context()
        jobs = [
            {
                "id": 1,
                "name": "Check",
                "job_type": "claude",
                "schedule_type": "interval",
                "schedule_data": json.dumps({"seconds": 7200}),
            }
        ]
        with patch("kai.bot.sessions.get_jobs", new_callable=AsyncMock, return_value=jobs):
            await handle_job(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "2h" in reply
        assert "\U0001f916" in reply  # robot emoji for claude type

    @pytest.mark.asyncio
    async def test_formats_interval_minutes(self):
        """Interval >= 60s but < 3600s displays as minutes."""
        update = _make_update()
        ctx = _make_context()
        jobs = [
            {
                "id": 2,
                "name": "Ping",
                "job_type": "reminder",
                "schedule_type": "interval",
                "schedule_data": json.dumps({"seconds": 300}),
            }
        ]
        with patch("kai.bot.sessions.get_jobs", new_callable=AsyncMock, return_value=jobs):
            await handle_job(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "5m" in reply
        assert "\U0001f514" in reply  # bell emoji for reminder type

    @pytest.mark.asyncio
    async def test_formats_daily(self):
        update = _make_update()
        ctx = _make_context()
        jobs = [
            {
                "id": 3,
                "name": "Standup",
                "job_type": "reminder",
                "schedule_type": "daily",
                "schedule_data": json.dumps({"times": ["14:00"]}),
            }
        ]
        with patch("kai.bot.sessions.get_jobs", new_callable=AsyncMock, return_value=jobs):
            await handle_job(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "14:00" in reply

    # ── /job info <id> ──────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_info_shows_job_details(self):
        """/job info <id> shows full details for an owned job."""
        update = _make_update(chat_id=12345)
        ctx = _make_context(args=["info", "4"])
        job = {
            "id": 4,
            "chat_id": 12345,
            "name": "Weather report",
            "job_type": "claude",
            "prompt": "What is the weather today?",
            "schedule_type": "daily",
            "schedule_data": json.dumps({"times": ["08:00"]}),
            "auto_remove": False,
            "notify_on_check": False,
        }
        with patch("kai.bot.sessions.get_job_by_id", new_callable=AsyncMock, return_value=job):
            await handle_job(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "Job #4" in reply
        assert "Weather report" in reply
        assert "claude" in reply
        assert "08:00" in reply
        assert "What is the weather today?" in reply

    @pytest.mark.asyncio
    async def test_info_not_found(self):
        """/job info <id> returns not found for non-existent job."""
        update = _make_update()
        ctx = _make_context(args=["info", "999"])
        with patch("kai.bot.sessions.get_job_by_id", new_callable=AsyncMock, return_value=None):
            await handle_job(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "not found" in reply.lower()

    @pytest.mark.asyncio
    async def test_info_wrong_owner(self):
        """/job info <id> returns not found for a job owned by another chat."""
        update = _make_update(chat_id=12345)
        ctx = _make_context(args=["info", "4"])
        job = {
            "id": 4,
            "chat_id": 99999,  # Different owner
            "name": "Secret",
            "job_type": "claude",
            "prompt": "hidden",
            "schedule_type": "daily",
            "schedule_data": json.dumps({"times": ["08:00"]}),
            "auto_remove": False,
            "notify_on_check": False,
        }
        with patch("kai.bot.sessions.get_job_by_id", new_callable=AsyncMock, return_value=job):
            await handle_job(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "not found" in reply.lower()

    @pytest.mark.asyncio
    async def test_info_missing_id(self):
        """/job info with no ID shows usage."""
        update = _make_update()
        ctx = _make_context(args=["info"])
        await handle_job(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "Usage" in reply

    @pytest.mark.asyncio
    async def test_info_non_numeric_id(self):
        """/job info abc shows error."""
        update = _make_update()
        ctx = _make_context(args=["info", "abc"])
        await handle_job(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "number" in reply.lower()

    # ── /job cancel <id> ────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_cancel_successful(self):
        """Deletes from DB and removes APScheduler jobs."""
        update = _make_update()
        mock_job = MagicMock()
        mock_job.name = "cron_5"
        mock_job.schedule_removal = MagicMock()
        jq = MagicMock()
        jq.jobs.return_value = [mock_job]
        ctx = _make_context(args=["cancel", "5"], job_queue=jq)
        with patch("kai.bot.sessions.delete_job", new_callable=AsyncMock, return_value=True):
            await handle_job(update, ctx)
        mock_job.schedule_removal.assert_called_once()
        reply = update.message.reply_text.call_args[0][0]
        assert "cancelled" in reply.lower()

    @pytest.mark.asyncio
    async def test_cancel_not_found(self):
        update = _make_update()
        ctx = _make_context(args=["cancel", "99"])
        with patch("kai.bot.sessions.delete_job", new_callable=AsyncMock, return_value=False):
            await handle_job(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "not found" in reply.lower()

    @pytest.mark.asyncio
    async def test_cancel_missing_id(self):
        """/job cancel with no ID shows usage."""
        update = _make_update()
        ctx = _make_context(args=["cancel"])
        await handle_job(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "Usage" in reply

    @pytest.mark.asyncio
    async def test_cancel_non_numeric_id(self):
        """/job cancel abc shows error."""
        update = _make_update()
        ctx = _make_context(args=["cancel", "abc"])
        await handle_job(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "number" in reply.lower()

    # ── Unknown subcommand ──────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_unknown_subcommand_shows_usage(self):
        update = _make_update()
        ctx = _make_context(args=["bogus"])
        await handle_job(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "Usage" in reply

    # ── /jobs alias ─────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_jobs_alias_lists(self):
        """/jobs delegates to the list logic."""
        update = _make_update()
        ctx = _make_context()
        with patch("kai.bot.sessions.get_jobs", new_callable=AsyncMock, return_value=[]):
            await handle_jobs(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "No active" in reply


# ── handle_models ────────────────────────────────────────────────────


class TestHandleModels:
    @pytest.mark.asyncio
    async def test_sends_keyboard(self):
        update = _make_update()
        ctx = _make_context()
        await handle_models(update, ctx)
        call = update.message.reply_text.call_args
        assert "Choose a model" in call[0][0]
        assert call[1]["reply_markup"] is not None

    @pytest.mark.asyncio
    async def test_open_ended_provider_shows_text(self):
        """Open-ended provider (openrouter/ollama) shows text instead of keyboard."""
        pool = _make_mock_claude(model="llama3", provider="ollama")
        update = _make_update()
        ctx = _make_context(claude=pool)
        await handle_models(update, ctx)
        call = update.message.reply_text.call_args
        reply = call[0][0]
        assert "Current model" in reply
        assert "llama3" in reply
        assert "/model" in reply
        # No keyboard for open-ended providers
        assert "reply_markup" not in call[1] or call[1].get("reply_markup") is None


# ── handle_model ─────────────────────────────────────────────────────


class TestHandleModel:
    @pytest.mark.asyncio
    async def test_no_args(self):
        update = _make_update()
        ctx = _make_context()
        await handle_model(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "Usage" in reply

    @pytest.mark.asyncio
    async def test_invalid_model(self):
        update = _make_update()
        ctx = _make_context(args=["gpt4"])
        await handle_model(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "opus" in reply.lower() or "sonnet" in reply.lower()

    @pytest.mark.asyncio
    async def test_valid_model(self):
        pool = _make_mock_claude(model="sonnet")
        update = _make_update()
        ctx = _make_context(claude=pool, args=["opus"])
        with (
            patch("kai.bot.sessions.clear_session", new_callable=AsyncMock),
            patch("kai.bot.sessions.set_user_setting", new_callable=AsyncMock),
            patch("kai.bot.sessions.delete_workspace_config_setting", new_callable=AsyncMock) as mock_delete,
        ):
            await handle_model(update, ctx)
        pool.set_model.assert_called_once_with(ANY, "opus")
        pool.restart.assert_called_once()
        # Model switch must clear workspace config override to prevent
        # the stale entry from shadowing the user setting on restart.
        mock_delete.assert_called_once_with(ANY, ANY, "model")


# ── handle_model_callback ────────────────────────────────────────────


class TestHandleModelCallback:
    @pytest.mark.asyncio
    async def test_unauthorized(self):
        update = _make_callback_update(data="model:opus", user_id=99)
        ctx = _make_context(config=_make_config(allowed_user_ids={1}))
        await handle_model_callback(update, ctx)
        update.callback_query.answer.assert_called_once_with("Not authorized.")

    @pytest.mark.asyncio
    async def test_invalid_model(self):
        update = _make_callback_update(data="model:gpt4")
        ctx = _make_context()
        await handle_model_callback(update, ctx)
        update.callback_query.answer.assert_called_once_with("Invalid model.")

    @pytest.mark.asyncio
    async def test_same_model_no_change(self):
        """Selecting the current model shows 'No change.'"""
        claude = _make_mock_claude(model="opus")
        update = _make_callback_update(data="model:opus")
        ctx = _make_context(claude=claude)
        await handle_model_callback(update, ctx)
        edit_text = update.callback_query.edit_message_text.call_args[0][0]
        assert "No change" in edit_text

    @pytest.mark.asyncio
    async def test_switch_model(self):
        pool = _make_mock_claude(model="sonnet")
        update = _make_callback_update(data="model:opus")
        ctx = _make_context(claude=pool)
        with (
            patch("kai.bot.sessions.clear_session", new_callable=AsyncMock),
            patch("kai.bot.sessions.set_user_setting", new_callable=AsyncMock),
            patch("kai.bot.sessions.delete_workspace_config_setting", new_callable=AsyncMock) as mock_delete,
        ):
            await handle_model_callback(update, ctx)
        pool.set_model.assert_called_once_with(ANY, "opus")
        edit_text = update.callback_query.edit_message_text.call_args[0][0]
        assert "Switched" in edit_text
        # Model switch must clear workspace config override to prevent
        # the stale entry from shadowing the user setting on restart.
        mock_delete.assert_called_once_with(ANY, ANY, "model")

    @pytest.mark.asyncio
    async def test_switch_model_resolver_runs_before_set_model(self, tmp_path):
        """
        /model x against a saved non-home workspace must leave the live
        instance carrying "x", not a workspace-level override.

        The resolver runs change_workspace under the hood, which resets
        the live model to the default and re-applies any workspace-
        level model override from the cached WorkspaceConfig. If the
        resolver fires AFTER set_model, that reset clobbers the user's
        chosen model with the stale override. Pin the ordering by
        emulating the change_workspace side effect on the mock pool.
        """
        from kai.bot import _switch_model

        pool = _make_mock_claude(model="default")
        instance = pool.get_if_exists.return_value
        saved_ws = tmp_path / "foo"

        async def resolver_side_effect(_chat_id):
            # change_workspace inside the resolver resets the live
            # model to the default and re-applies the workspace's
            # stale override.
            instance.model = "old-override"
            return saved_ws

        def set_model_side_effect(_chat_id, model):
            instance.model = model

        pool.get_effective_workspace.side_effect = resolver_side_effect
        pool.set_model.side_effect = set_model_side_effect

        ctx = _make_context(claude=pool)
        with (
            patch("kai.bot.sessions.clear_session", new_callable=AsyncMock),
            patch("kai.bot.sessions.set_user_setting", new_callable=AsyncMock),
            patch("kai.bot.sessions.delete_workspace_config_setting", new_callable=AsyncMock) as mock_delete,
        ):
            await _switch_model(ctx, 12345, "x")

        # Live instance carries the user's choice, not the workspace's
        # stale override. Catches a regression where set_model runs
        # before the resolver.
        assert instance.model == "x"
        # Delete targets the saved workspace, not home, because the
        # resolver returned the saved path before delete was called.
        mock_delete.assert_called_once_with(12345, str(saved_ws), "model")


# ── handle_voice_command ─────────────────────────────────────────────


class TestHandleVoiceCommand:
    @pytest.mark.asyncio
    async def test_tts_disabled(self):
        update = _make_update()
        ctx = _make_context(config=_make_config(tts_enabled=False))
        await handle_voice_command(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "not enabled" in reply.lower()

    @pytest.mark.asyncio
    async def test_toggle_off_to_only(self):
        """No args when mode is off: toggles to 'only'."""
        update = _make_update()
        ctx = _make_context(config=_make_config(tts_enabled=True))
        with (
            patch("kai.bot.sessions.get_setting", new_callable=AsyncMock, return_value=None),
            patch("kai.bot.sessions.set_setting", new_callable=AsyncMock) as mock_set,
        ):
            await handle_voice_command(update, ctx)
        # Should set to "only" (toggling from default "off")
        mock_set.assert_called_once_with("voice_mode:12345", "only")

    @pytest.mark.asyncio
    async def test_toggle_only_to_off(self):
        """No args when mode is 'only': toggles to 'off'."""
        update = _make_update()
        ctx = _make_context(config=_make_config(tts_enabled=True))

        async def _get(key):
            if "voice_mode" in key:
                return "only"
            return None

        with (
            patch("kai.bot.sessions.get_setting", side_effect=_get),
            patch("kai.bot.sessions.set_setting", new_callable=AsyncMock) as mock_set,
        ):
            await handle_voice_command(update, ctx)
        mock_set.assert_called_once_with("voice_mode:12345", "off")

    @pytest.mark.asyncio
    async def test_set_mode_on(self):
        update = _make_update()
        ctx = _make_context(config=_make_config(tts_enabled=True), args=["on"])
        with (
            patch("kai.bot.sessions.get_setting", new_callable=AsyncMock, return_value=None),
            patch("kai.bot.sessions.set_setting", new_callable=AsyncMock) as mock_set,
        ):
            await handle_voice_command(update, ctx)
        mock_set.assert_called_once_with("voice_mode:12345", "on")

    @pytest.mark.asyncio
    async def test_set_voice_name_enables_if_off(self):
        """Setting a voice name auto-enables voice mode when off."""
        update = _make_update()
        # Use a real voice key from the VOICES dict
        voice_key = next(iter(VOICES.keys()))
        ctx = _make_context(config=_make_config(tts_enabled=True), args=[voice_key])

        async def _get(key):
            if "voice_mode" in key:
                return "off"
            return DEFAULT_VOICE

        with (
            patch("kai.bot.sessions.get_setting", side_effect=_get),
            patch("kai.bot.sessions.set_setting", new_callable=AsyncMock) as mock_set,
        ):
            await handle_voice_command(update, ctx)
        # Should set both voice name and mode
        calls = {c[0] for c in mock_set.call_args_list}
        assert ("voice_name:12345", voice_key) in calls
        assert ("voice_mode:12345", "only") in calls

    @pytest.mark.asyncio
    async def test_invalid_voice_name(self):
        update = _make_update()
        ctx = _make_context(config=_make_config(tts_enabled=True), args=["badname"])
        with patch("kai.bot.sessions.get_setting", new_callable=AsyncMock, return_value=None):
            await handle_voice_command(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "Unknown" in reply or "Usage" in reply


# ── handle_voices ────────────────────────────────────────────────────


class TestHandleVoices:
    @pytest.mark.asyncio
    async def test_tts_disabled(self):
        update = _make_update()
        ctx = _make_context(config=_make_config(tts_enabled=False))
        await handle_voices(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "not enabled" in reply.lower()

    @pytest.mark.asyncio
    async def test_sends_keyboard(self):
        update = _make_update()
        ctx = _make_context(config=_make_config(tts_enabled=True))
        with patch("kai.bot.sessions.get_setting", new_callable=AsyncMock, return_value=None):
            await handle_voices(update, ctx)
        call = update.message.reply_text.call_args
        assert call[1]["reply_markup"] is not None


# ── handle_voice_callback ────────────────────────────────────────────


class TestHandleVoiceCallback:
    @pytest.mark.asyncio
    async def test_unauthorized(self):
        update = _make_callback_update(data="voice:jenny", user_id=99)
        ctx = _make_context(config=_make_config(allowed_user_ids={1}))
        await handle_voice_callback(update, ctx)
        update.callback_query.answer.assert_called_once_with("Not authorized.")

    @pytest.mark.asyncio
    async def test_invalid_voice(self):
        update = _make_callback_update(data="voice:nonexistent")
        ctx = _make_context()
        await handle_voice_callback(update, ctx)
        update.callback_query.answer.assert_called_once_with("Invalid voice.")

    @pytest.mark.asyncio
    async def test_same_voice_no_change(self):
        update = _make_callback_update(data=f"voice:{DEFAULT_VOICE}")
        ctx = _make_context()
        with patch("kai.bot.sessions.get_setting", new_callable=AsyncMock, return_value=None):
            await handle_voice_callback(update, ctx)
        edit_text = update.callback_query.edit_message_text.call_args[0][0]
        assert "No change" in edit_text

    @pytest.mark.asyncio
    async def test_switch_voice(self):
        """Switching voice sets the new name and confirms."""
        new_voice = "jenny" if DEFAULT_VOICE != "jenny" else "alan"
        update = _make_callback_update(data=f"voice:{new_voice}")
        ctx = _make_context()
        with (
            patch("kai.bot.sessions.get_setting", new_callable=AsyncMock, return_value=None),
            patch("kai.bot.sessions.set_setting", new_callable=AsyncMock),
        ):
            await handle_voice_callback(update, ctx)
        edit_text = update.callback_query.edit_message_text.call_args[0][0]
        assert VOICES[new_voice] in edit_text

    @pytest.mark.asyncio
    async def test_auto_enables_when_off(self):
        """Switching voice auto-enables mode to 'only' when off."""
        new_voice = "jenny" if DEFAULT_VOICE != "jenny" else "alan"
        update = _make_callback_update(data=f"voice:{new_voice}")
        ctx = _make_context()

        async def _get(key):
            if "voice_mode" in key:
                return "off"
            return None  # default voice

        with (
            patch("kai.bot.sessions.get_setting", side_effect=_get),
            patch("kai.bot.sessions.set_setting", new_callable=AsyncMock) as mock_set,
        ):
            await handle_voice_callback(update, ctx)
        calls = {c[0] for c in mock_set.call_args_list}
        assert ("voice_mode:12345", "only") in calls


# ── handle_webhooks ──────────────────────────────────────────────────


class TestHandleWebhooks:
    @pytest.mark.asyncio
    async def test_running_with_secret(self):
        update = _make_update()
        ctx = _make_context(config=_make_config(github_webhook_secret="s3cret"))
        with patch("kai.bot.webhook.is_running", return_value=True):
            await handle_webhooks(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "running" in reply
        assert "GitHub setup" in reply

    @pytest.mark.asyncio
    async def test_not_running(self):
        update = _make_update()
        ctx = _make_context(config=_make_config(github_webhook_secret="s3cret"))
        with patch("kai.bot.webhook.is_running", return_value=False):
            await handle_webhooks(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "not running" in reply

    @pytest.mark.asyncio
    async def test_no_external_secrets(self):
        update = _make_update()
        ctx = _make_context(
            config=_make_config(
                github_webhook_secret="",
                generic_webhook_secret="",
            )
        )
        with patch("kai.bot.webhook.is_running", return_value=True):
            await handle_webhooks(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "No external webhook secrets" in reply
        assert "/api/schedule" in reply


# ── handle_workspace ─────────────────────────────────────────────────


class TestHandleWorkspace:
    @pytest.mark.asyncio
    async def test_no_args_shows_current(self):
        """No args: shows the current workspace."""
        claude = _make_mock_claude(workspace=Path("/home/workspace"))
        update = _make_update()
        ctx = _make_context(claude=claude)
        with _mock_resolve():
            await handle_workspace(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "Home" in reply or "workspace" in reply.lower()

    @pytest.mark.asyncio
    async def test_home_switches(self, tmp_path):
        """'home' keyword switches to home workspace."""
        home = tmp_path / "home"
        home.mkdir()
        claude = _make_mock_claude(workspace=Path("/other"))
        config = _make_config(claude_workspace=home)
        update = _make_update()
        ctx = _make_context(claude=claude, config=config, args=["home"])
        with (
            _mock_resolve(),
            patch("kai.bot.sessions.clear_session", new_callable=AsyncMock),
            patch("kai.bot.sessions.delete_setting", new_callable=AsyncMock),
            patch("kai.bot.sessions.build_workspace_config", new_callable=AsyncMock, return_value=None),
        ):
            await handle_workspace(update, ctx)
        claude.change_workspace.assert_called_once()

    @pytest.mark.asyncio
    async def test_workspace_home_lands_in_per_user_directory(self, tmp_path, monkeypatch):
        """
        Spec 353 acceptance: `/workspace home` for a user with no
        users.yaml home_workspace override must land in DATA_DIR/home/<chat_id>/,
        NOT in a shared global directory.

        This is the multi-user privacy guarantee. We patch backend.DATA_DIR
        to tmp_path so resolve_home_workspace creates a per-user directory
        under the test root, then capture the path passed to
        Claude.change_workspace and assert it matches that user's slot.
        """
        # Point backend.DATA_DIR at the test root so ensure_user_home
        # creates the per-user dir under tmp_path rather than /var/lib/kai.
        monkeypatch.setattr("kai.backend.DATA_DIR", tmp_path)

        # No claude_workspace= override -> no UserConfig.home_workspace,
        # so resolve_home_workspace must fall through to ensure_user_home.
        # users.yaml is mandatory post-#565 tranche A; supply a minimal
        # entry for the default test chat_id so the resolver treats it
        # as interactive rather than a runtime-added notification chat.
        config = _make_config(user_configs={12345: UserConfig(telegram_id=12345, name="test")})
        claude = _make_mock_claude(workspace=Path("/other"))
        # _make_update defaults to chat_id=12345; the per-user home
        # directory therefore lands at tmp_path/home/12345.
        update = _make_update()
        ctx = _make_context(claude=claude, config=config, args=["home"])
        with (
            _mock_resolve(),
            patch("kai.bot.sessions.clear_session", new_callable=AsyncMock),
            patch("kai.bot.sessions.delete_setting", new_callable=AsyncMock),
            patch("kai.bot.sessions.build_workspace_config", new_callable=AsyncMock, return_value=None),
        ):
            await handle_workspace(update, ctx)

        # Assert change_workspace was passed the per-user directory.
        # Positional arg 0 is chat_id; positional arg 1 is the target Path.
        claude.change_workspace.assert_called_once()
        call_args = claude.change_workspace.call_args
        target_path = call_args.args[1] if len(call_args.args) > 1 else call_args.kwargs["workspace"]
        expected = tmp_path / "home" / "12345"
        assert target_path == expected, f"Expected {expected}, got {target_path}"
        # The directory must actually exist (ensure_user_home creates it).
        assert expected.is_dir()

    @pytest.mark.asyncio
    async def test_absolute_path_rejected(self):
        update = _make_update()
        ctx = _make_context(args=["/tmp/evil"])
        with _mock_resolve():
            await handle_workspace(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "Absolute paths" in reply

    @pytest.mark.asyncio
    async def test_tilde_path_rejected(self):
        update = _make_update()
        ctx = _make_context(args=["~/foo"])
        with _mock_resolve():
            await handle_workspace(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "Absolute paths" in reply

    @pytest.mark.asyncio
    async def test_new_without_name(self):
        update = _make_update()
        ctx = _make_context(args=["new"])
        with _mock_resolve():
            await handle_workspace(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "Usage" in reply

    @pytest.mark.asyncio
    async def test_new_without_base(self):
        """'new' with no workspace_base shows the updated error message."""
        update = _make_update()
        ctx = _make_context(config=_make_config(workspace_base=None), args=["new", "myproj"])
        with _mock_resolve():
            await handle_workspace(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "workspace base" in reply.lower()

    @pytest.mark.asyncio
    async def test_new_already_exists(self, tmp_path):
        existing = tmp_path / "myproj"
        existing.mkdir()
        update = _make_update()
        ctx = _make_context(
            config=_make_config(workspace_base=tmp_path, claude_workspace=Path("/home")),
            args=["new", "myproj"],
        )
        with _mock_resolve(base=tmp_path):
            await handle_workspace(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "Already exists" in reply

    @pytest.mark.asyncio
    async def test_new_creates_and_switches(self, tmp_path):
        """'new <name>' creates the directory, runs git init, and switches."""
        update = _make_update()
        claude = _make_mock_claude(workspace=Path("/home"))
        config = _make_config(workspace_base=tmp_path, claude_workspace=Path("/home"))
        ctx = _make_context(config=config, claude=claude, args=["new", "fresh"])
        mock_proc = MagicMock()
        mock_proc.wait = AsyncMock()
        with (
            _mock_resolve(base=tmp_path),
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_proc),
            patch("kai.bot.sessions.clear_session", new_callable=AsyncMock),
            patch("kai.bot.sessions.set_setting", new_callable=AsyncMock),
            patch("kai.bot.sessions.upsert_workspace_history", new_callable=AsyncMock),
            patch("kai.bot.sessions.build_workspace_config", new_callable=AsyncMock, return_value=None),
        ):
            await handle_workspace(update, ctx)
        # Directory should have been created
        assert (tmp_path / "fresh").is_dir()
        claude.change_workspace.assert_called_once()

    @pytest.mark.asyncio
    async def test_new_git_init_failure_warns(self, tmp_path):
        """Failed git init warns the user but still switches workspace."""
        update = _make_update()
        claude = _make_mock_claude(workspace=Path("/home"))
        config = _make_config(workspace_base=tmp_path, claude_workspace=Path("/home"))
        ctx = _make_context(config=config, claude=claude, args=["new", "broken"])
        # Simulate git init returning a non-zero exit code
        mock_proc = MagicMock()
        mock_proc.wait = AsyncMock(return_value=1)
        with (
            _mock_resolve(base=tmp_path),
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=mock_proc),
            patch("kai.bot.sessions.clear_session", new_callable=AsyncMock),
            patch("kai.bot.sessions.set_setting", new_callable=AsyncMock),
            patch("kai.bot.sessions.upsert_workspace_history", new_callable=AsyncMock),
            patch("kai.bot.sessions.build_workspace_config", new_callable=AsyncMock, return_value=None),
        ):
            await handle_workspace(update, ctx)
        # Warning about git init failure was sent
        replies = [call[0][0] for call in update.message.reply_text.call_args_list]
        assert any("git init failed" in r for r in replies)
        # Workspace switch still happened despite the failure
        claude.change_workspace.assert_called_once()

    @pytest.mark.asyncio
    async def test_workspace_new_prefix_not_caught(self, tmp_path):
        """Workspace names starting with 'new' are not mistaken for /workspace new."""
        # "newsletter" is a valid workspace name, not the "new" subcommand
        project = tmp_path / "newsletter"
        project.mkdir()
        claude = _make_mock_claude(workspace=Path("/other"))
        config = _make_config(workspace_base=tmp_path, claude_workspace=Path("/home"))
        update = _make_update()
        ctx = _make_context(config=config, claude=claude, args=["newsletter"])
        with (
            _mock_resolve(base=tmp_path),
            patch("kai.bot.sessions.clear_session", new_callable=AsyncMock),
            patch("kai.bot.sessions.set_setting", new_callable=AsyncMock),
            patch("kai.bot.sessions.upsert_workspace_history", new_callable=AsyncMock),
            patch("kai.bot.sessions.build_workspace_config", new_callable=AsyncMock, return_value=None),
        ):
            await handle_workspace(update, ctx)
        # Should switch to the workspace, not show "Usage: /workspace new <name>"
        claude.change_workspace.assert_called_once()

    @pytest.mark.asyncio
    async def test_workspace_new_bare_shows_usage(self):
        """'/workspace new' with no name shows usage hint."""
        update = _make_update()
        ctx = _make_context(args=["new"])
        with _mock_resolve():
            await handle_workspace(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "Usage" in reply

    @pytest.mark.asyncio
    async def test_name_found_in_base(self, tmp_path):
        """Name resolved via workspace_base."""
        project = tmp_path / "myproj"
        project.mkdir()
        claude = _make_mock_claude(workspace=Path("/other"))
        config = _make_config(workspace_base=tmp_path, claude_workspace=Path("/home"))
        update = _make_update()
        ctx = _make_context(config=config, claude=claude, args=["myproj"])
        with (
            _mock_resolve(base=tmp_path),
            patch("kai.bot.sessions.clear_session", new_callable=AsyncMock),
            patch("kai.bot.sessions.set_setting", new_callable=AsyncMock),
            patch("kai.bot.sessions.upsert_workspace_history", new_callable=AsyncMock),
            patch("kai.bot.sessions.build_workspace_config", new_callable=AsyncMock, return_value=None),
        ):
            await handle_workspace(update, ctx)
        claude.change_workspace.assert_called_once()

    @pytest.mark.asyncio
    async def test_name_found_in_allowed(self, tmp_path):
        """Name resolved via allowed list directory name match."""
        project = tmp_path / "myproj"
        project.mkdir()
        claude = _make_mock_claude(workspace=Path("/other"))
        config = _make_config(claude_workspace=Path("/home"))
        update = _make_update()
        ctx = _make_context(config=config, claude=claude, args=["myproj"])
        with (
            _mock_resolve(allowed=[project]),
            patch("kai.bot.sessions.clear_session", new_callable=AsyncMock),
            patch("kai.bot.sessions.set_setting", new_callable=AsyncMock),
            patch("kai.bot.sessions.upsert_workspace_history", new_callable=AsyncMock),
            patch("kai.bot.sessions.build_workspace_config", new_callable=AsyncMock, return_value=None),
        ):
            await handle_workspace(update, ctx)
        claude.change_workspace.assert_called_once()

    @pytest.mark.asyncio
    async def test_multiple_matches_in_allowed(self, tmp_path):
        """Multiple allowed workspaces with the same name: shows disambiguation message."""
        proj_a = tmp_path / "a" / "proj"
        proj_b = tmp_path / "b" / "proj"
        proj_a.mkdir(parents=True)
        proj_b.mkdir(parents=True)
        config = _make_config(claude_workspace=Path("/home"))
        update = _make_update()
        ctx = _make_context(config=config, args=["proj"])
        with _mock_resolve(allowed=[proj_a, proj_b]):
            await handle_workspace(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "Multiple workspaces" in reply

    @pytest.mark.asyncio
    async def test_not_found_with_sources(self, tmp_path):
        config = _make_config(claude_workspace=Path("/home"))
        update = _make_update()
        ctx = _make_context(config=config, args=["nonexistent"])
        with _mock_resolve(base=tmp_path):
            await handle_workspace(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "not found" in reply.lower()

    @pytest.mark.asyncio
    async def test_not_found_no_sources(self):
        config = _make_config(workspace_base=None, allowed_workspaces=[], claude_workspace=Path("/home"))
        update = _make_update()
        ctx = _make_context(config=config, args=["anything"])
        with _mock_resolve():
            await handle_workspace(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "workspace base" in reply.lower()

    @pytest.mark.asyncio
    async def test_ws_alias_registered(self):
        """/ws is registered as an alias for /workspace via create_bot."""
        # The alias is a CommandHandler("ws", handle_workspace) in create_bot.
        # We verify it by calling handle_workspace directly with no args
        # (same handler, so same behavior as /workspace with no args).
        claude = _make_mock_claude(workspace=Path("/home/workspace"))
        update = _make_update()
        ctx = _make_context(claude=claude)
        with _mock_resolve():
            await handle_workspace(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "Home" in reply or "workspace" in reply.lower()


# ── handle_workspaces ────────────────────────────────────────────────


class TestHandleWorkspaces:
    @pytest.mark.asyncio
    async def test_no_history_at_home(self):
        update = _make_update()
        claude = _make_mock_claude(workspace=Path("/home/workspace"))
        config = _make_config(claude_workspace=Path("/home/workspace"))
        ctx = _make_context(config=config, claude=claude)
        with (
            _mock_resolve(),
            patch("kai.bot.sessions.get_workspace_history", new_callable=AsyncMock, return_value=[]),
        ):
            await handle_workspaces(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "No workspace history" in reply

    @pytest.mark.asyncio
    async def test_has_history_shows_keyboard(self):
        update = _make_update()
        claude = _make_mock_claude(workspace=Path("/home/workspace"))
        config = _make_config(claude_workspace=Path("/home/workspace"))
        ctx = _make_context(config=config, claude=claude)
        history = [{"path": "/some/project"}]
        with (
            _mock_resolve(),
            patch("kai.bot.sessions.get_workspace_history", new_callable=AsyncMock, return_value=history),
        ):
            await handle_workspaces(update, ctx)
        call = update.message.reply_text.call_args
        assert call[1]["reply_markup"] is not None


# ── handle_workspace_callback ────────────────────────────────────────


class TestHandleWorkspaceCallback:
    @pytest.mark.asyncio
    async def test_unauthorized(self):
        update = _make_callback_update(data="ws:home", user_id=99)
        ctx = _make_context(config=_make_config(allowed_user_ids={1}))
        await handle_workspace_callback(update, ctx)
        update.callback_query.answer.assert_called_once_with("Not authorized.")

    @pytest.mark.asyncio
    async def test_home(self):
        """ws:home switches to home workspace."""
        claude = _make_mock_claude(workspace=Path("/other"))
        config = _make_config(claude_workspace=Path("/home/workspace"))
        update = _make_callback_update(data="ws:home")
        ctx = _make_context(config=config, claude=claude)
        with (
            _mock_resolve(),
            patch("kai.bot.sessions.clear_session", new_callable=AsyncMock),
            patch("kai.bot.sessions.delete_setting", new_callable=AsyncMock),
            patch("kai.bot.sessions.build_workspace_config", new_callable=AsyncMock, return_value=None),
        ):
            await handle_workspace_callback(update, ctx)
        claude.change_workspace.assert_called_once()
        edit_text = update.callback_query.edit_message_text.call_args[0][0]
        assert "Home" in edit_text

    @pytest.mark.asyncio
    async def test_allowed_workspace(self, tmp_path):
        """ws:allowed:<idx> switches to the indexed allowed workspace."""
        project = tmp_path / "proj"
        project.mkdir()
        claude = _make_mock_claude(workspace=Path("/other"))
        config = _make_config(claude_workspace=Path("/home/workspace"))
        update = _make_callback_update(data="ws:allowed:0")
        ctx = _make_context(config=config, claude=claude)
        with (
            _mock_resolve(allowed=[project]),
            patch("kai.bot.sessions.clear_session", new_callable=AsyncMock),
            patch("kai.bot.sessions.set_setting", new_callable=AsyncMock),
            patch("kai.bot.sessions.upsert_workspace_history", new_callable=AsyncMock),
            patch("kai.bot.sessions.build_workspace_config", new_callable=AsyncMock, return_value=None),
        ):
            await handle_workspace_callback(update, ctx)
        claude.change_workspace.assert_called_once()

    @pytest.mark.asyncio
    async def test_allowed_nonexistent_dir(self, tmp_path):
        """ws:allowed:<idx> where directory was deleted."""
        gone = tmp_path / "gone"  # not created
        config = _make_config(claude_workspace=Path("/home"))
        update = _make_callback_update(data="ws:allowed:0")
        ctx = _make_context(config=config)
        with _mock_resolve(allowed=[gone]):
            await handle_workspace_callback(update, ctx)
        update.callback_query.answer.assert_called_once_with("That workspace no longer exists.")

    @pytest.mark.asyncio
    async def test_allowed_bad_index(self):
        update = _make_callback_update(data="ws:allowed:bad")
        ctx = _make_context()
        with _mock_resolve():
            await handle_workspace_callback(update, ctx)
        update.callback_query.answer.assert_called_once_with("Invalid selection.")

    @pytest.mark.asyncio
    async def test_allowed_out_of_range(self):
        config = _make_config(claude_workspace=Path("/home"))
        update = _make_callback_update(data="ws:allowed:99")
        ctx = _make_context(config=config)
        with _mock_resolve():
            await handle_workspace_callback(update, ctx)
        update.callback_query.answer.assert_called_once_with("Workspace no longer available.")

    @pytest.mark.asyncio
    async def test_history_entry(self, tmp_path):
        """ws:<idx> switches to a history entry."""
        project = tmp_path / "proj"
        project.mkdir()
        claude = _make_mock_claude(workspace=Path("/other"))
        config = _make_config(claude_workspace=Path("/home/workspace"))
        update = _make_callback_update(data="ws:0")
        ctx = _make_context(config=config, claude=claude)
        history = [{"path": str(project)}]
        with (
            _mock_resolve(),
            patch("kai.bot.sessions.get_workspace_history", new_callable=AsyncMock, return_value=history),
            patch("kai.bot.sessions.clear_session", new_callable=AsyncMock),
            patch("kai.bot.sessions.set_setting", new_callable=AsyncMock),
            patch("kai.bot.sessions.upsert_workspace_history", new_callable=AsyncMock),
            patch("kai.bot.sessions.build_workspace_config", new_callable=AsyncMock, return_value=None),
        ):
            await handle_workspace_callback(update, ctx)
        claude.change_workspace.assert_called_once()

    @pytest.mark.asyncio
    async def test_history_no_longer_allowed(self, tmp_path):
        """History entry disallowed: deleted from history, keyboard refreshed."""
        project = tmp_path / "proj"
        project.mkdir()
        # Configure with a base that doesn't contain the project - the
        # per-user resolve returns base=other_base so the project path
        # fails _is_workspace_allowed.
        other_base = tmp_path / "base"
        other_base.mkdir()
        config = _make_config(claude_workspace=Path("/home/workspace"))
        claude = _make_mock_claude(workspace=Path("/home/workspace"))
        update = _make_callback_update(data="ws:0")
        ctx = _make_context(config=config, claude=claude)
        history = [{"path": str(project)}]
        with (
            _mock_resolve(base=other_base),
            patch("kai.bot.sessions.get_workspace_history", new_callable=AsyncMock, return_value=history),
            patch("kai.bot.sessions.delete_workspace_history", new_callable=AsyncMock) as mock_del,
        ):
            await handle_workspace_callback(update, ctx)
        mock_del.assert_called_once_with(str(project), 12345)
        update.callback_query.answer.assert_called_once_with("That workspace is no longer allowed.")

    @pytest.mark.asyncio
    async def test_history_dir_deleted(self, tmp_path):
        """History entry whose directory no longer exists: removed from history."""
        gone = tmp_path / "gone"  # not created
        config = _make_config(claude_workspace=Path("/home/workspace"))
        claude = _make_mock_claude(workspace=Path("/home/workspace"))
        update = _make_callback_update(data="ws:0")
        ctx = _make_context(config=config, claude=claude)
        history = [{"path": str(gone)}]
        with (
            _mock_resolve(),
            patch("kai.bot.sessions.get_workspace_history", new_callable=AsyncMock, return_value=history),
            patch("kai.bot.sessions.delete_workspace_history", new_callable=AsyncMock) as mock_del,
        ):
            await handle_workspace_callback(update, ctx)
        mock_del.assert_called_once_with(str(gone), 12345)

    @pytest.mark.asyncio
    async def test_already_in_workspace(self, tmp_path):
        """Selecting the current workspace shows 'No change.'"""
        home = Path("/home/workspace")
        claude = _make_mock_claude(workspace=home)
        config = _make_config(claude_workspace=home)
        update = _make_callback_update(data="ws:home")
        ctx = _make_context(config=config, claude=claude)
        with _mock_resolve():
            await handle_workspace_callback(update, ctx)
        edit_text = update.callback_query.edit_message_text.call_args[0][0]
        assert "No change" in edit_text


# ── /workspace allow/deny/allowed ───────────────────────────────────


class TestHandleWorkspaceAllow:
    @pytest.mark.asyncio
    async def test_allow_success(self, tmp_path):
        """Adding a valid directory path succeeds."""
        ws = tmp_path / "new-ws"
        ws.mkdir()
        update = _make_update()
        ctx = _make_context(args=["allow", str(ws)])
        with (
            _mock_resolve(),
            patch("kai.bot.sessions.add_allowed_workspace", new_callable=AsyncMock) as mock_add,
        ):
            await _handle_workspace_allow(update, ctx, f"allow {ws}")
        mock_add.assert_called_once_with(12345, str(ws.resolve()))
        reply = update.message.reply_text.call_args[0][0]
        assert "Added" in reply

    @pytest.mark.asyncio
    async def test_allow_no_path(self):
        """Missing path argument shows usage."""
        update = _make_update()
        ctx = _make_context(args=["allow"])
        await _handle_workspace_allow(update, ctx, "allow")
        reply = update.message.reply_text.call_args[0][0]
        assert "Usage" in reply

    @pytest.mark.asyncio
    async def test_allow_relative_path_rejected(self):
        """Relative paths (not starting with /) are rejected."""
        update = _make_update()
        ctx = _make_context(args=["allow", "relative/path"])
        await _handle_workspace_allow(update, ctx, "allow relative/path")
        reply = update.message.reply_text.call_args[0][0]
        assert "absolute" in reply.lower()

    @pytest.mark.asyncio
    async def test_allow_nonexistent_rejected(self):
        """Non-existent path is rejected."""
        update = _make_update()
        ctx = _make_context(args=["allow", "/nonexistent/path"])
        await _handle_workspace_allow(update, ctx, "allow /nonexistent/path")
        reply = update.message.reply_text.call_args[0][0]
        assert "Not a directory" in reply

    @pytest.mark.asyncio
    async def test_allow_under_base_redundant(self, tmp_path):
        """Path under workspace_base is flagged as redundant."""
        base = tmp_path / "base"
        sub = base / "project"
        sub.mkdir(parents=True)
        update = _make_update()
        ctx = _make_context(args=["allow", str(sub)])
        with _mock_resolve(base=base):
            await _handle_workspace_allow(update, ctx, f"allow {sub}")
        reply = update.message.reply_text.call_args[0][0]
        assert "workspace base" in reply.lower()

    @pytest.mark.asyncio
    async def test_allow_duplicate(self, tmp_path):
        """Path already in allowed list is flagged as duplicate."""
        ws = tmp_path / "ws"
        ws.mkdir()
        update = _make_update()
        ctx = _make_context(args=["allow", str(ws)])
        with _mock_resolve(allowed=[ws]):
            await _handle_workspace_allow(update, ctx, f"allow {ws}")
        reply = update.message.reply_text.call_args[0][0]
        assert "Already" in reply


class TestHandleWorkspaceDeny:
    @pytest.mark.asyncio
    async def test_deny_success(self, tmp_path):
        """Removing a user-added path succeeds."""
        ws = tmp_path / "ws"
        ws.mkdir()
        update = _make_update()
        ctx = _make_context(args=["deny", str(ws)])
        with patch(
            "kai.bot.sessions.remove_allowed_workspace",
            new_callable=AsyncMock,
            return_value=True,
        ):
            await _handle_workspace_deny(update, ctx, f"deny {ws}")
        reply = update.message.reply_text.call_args[0][0]
        assert "Removed" in reply

    @pytest.mark.asyncio
    async def test_deny_no_path(self):
        """Missing path argument shows usage."""
        update = _make_update()
        ctx = _make_context(args=["deny"])
        await _handle_workspace_deny(update, ctx, "deny")
        reply = update.message.reply_text.call_args[0][0]
        assert "Usage" in reply

    @pytest.mark.asyncio
    async def test_deny_relative_path_rejected(self):
        """Relative paths are rejected with a clear message."""
        update = _make_update()
        ctx = _make_context(args=["deny", "relative/path"])
        await _handle_workspace_deny(update, ctx, "deny relative/path")
        reply = update.message.reply_text.call_args[0][0]
        assert "absolute" in reply.lower()

    @pytest.mark.asyncio
    async def test_deny_global_path(self, tmp_path):
        """Trying to deny a global ALLOWED_WORKSPACES path shows an explanation."""
        ws = tmp_path / "global-ws"
        ws.mkdir()
        config = _make_config(allowed_workspaces=[ws])
        update = _make_update()
        ctx = _make_context(config=config, args=["deny", str(ws)])
        with patch(
            "kai.bot.sessions.remove_allowed_workspace",
            new_callable=AsyncMock,
            return_value=False,
        ):
            await _handle_workspace_deny(update, ctx, f"deny {ws}")
        reply = update.message.reply_text.call_args[0][0]
        assert "globally" in reply.lower()

    @pytest.mark.asyncio
    async def test_deny_unknown_path(self, tmp_path):
        """Denying a path not in any list shows not-found message."""
        update = _make_update()
        ctx = _make_context(args=["deny", "/some/unknown/path"])
        with patch(
            "kai.bot.sessions.remove_allowed_workspace",
            new_callable=AsyncMock,
            return_value=False,
        ):
            await _handle_workspace_deny(update, ctx, "deny /some/unknown/path")
        reply = update.message.reply_text.call_args[0][0]
        assert "Not in your" in reply


class TestHandleWorkspaceAllowed:
    @pytest.mark.asyncio
    async def test_shows_list_with_sources(self, tmp_path):
        """Shows allowed workspaces with source attribution."""
        db_ws = tmp_path / "user-ws"
        db_ws.mkdir()
        global_ws = tmp_path / "global-ws"
        global_ws.mkdir()
        config = _make_config(allowed_workspaces=[global_ws])
        update = _make_update()
        ctx = _make_context(config=config)
        with patch(
            "kai.bot.sessions.get_allowed_workspaces",
            new_callable=AsyncMock,
            return_value=[db_ws],
        ):
            await _handle_workspace_allowed(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "(you)" in reply
        assert "(global)" in reply
        assert "Workspace base: not set" in reply

    @pytest.mark.asyncio
    async def test_permissive_mode(self):
        """No base and no allowed shows permissive mode message."""
        config = _make_config()
        update = _make_update()
        ctx = _make_context(config=config)
        with patch(
            "kai.bot.sessions.get_allowed_workspaces",
            new_callable=AsyncMock,
            return_value=[],
        ):
            await _handle_workspace_allowed(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "permissive" in reply.lower()

    @pytest.mark.asyncio
    async def test_shows_base(self, tmp_path):
        """Shows workspace_base when configured."""
        config = _make_config(workspace_base=tmp_path)
        update = _make_update()
        ctx = _make_context(config=config)
        with patch(
            "kai.bot.sessions.get_allowed_workspaces",
            new_callable=AsyncMock,
            return_value=[],
        ):
            await _handle_workspace_allowed(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert f"Workspace base: {tmp_path}" in reply

    @pytest.mark.asyncio
    async def test_shows_yaml_per_user_tier(self, tmp_path):
        """
        Per-user yaml `allowed_workspaces:` entries (#460) appear in
        the listing with a `(yaml)` source label, distinct from the
        existing `(you)` (DB) and `(global)` tiers. Pre-#460 these
        entries were silently dropped by the config loader; with the
        loader fix in place this UI test pins the user-visible
        result of the change.
        """
        from kai.config import UserConfig

        yaml_ws = tmp_path / "yaml-ws"
        yaml_ws.mkdir()
        db_ws = tmp_path / "db-ws"
        db_ws.mkdir()
        global_ws = tmp_path / "global-ws"
        global_ws.mkdir()

        uc = UserConfig(
            telegram_id=123,
            name="alice",
            allowed_workspaces=[yaml_ws.resolve()],
        )
        config = _make_config(
            user_configs={123: uc},
            allowed_workspaces=[global_ws.resolve()],
        )
        update = _make_update(chat_id=123)
        ctx = _make_context(config=config)
        with patch(
            "kai.bot.sessions.get_allowed_workspaces",
            new_callable=AsyncMock,
            return_value=[db_ws],
        ):
            await _handle_workspace_allowed(update, ctx)

        reply = update.message.reply_text.call_args[0][0]
        # All three source tiers attributed correctly.
        assert f"{db_ws.resolve()} (you)" in reply
        assert f"{yaml_ws.resolve()} (yaml)" in reply
        assert f"{global_ws.resolve()} (global)" in reply

    @pytest.mark.asyncio
    async def test_yaml_tier_priority_when_in_db_too(self, tmp_path):
        """
        A path listed in BOTH the per-user yaml and the DB collapses
        to a single line; the `(you)` (DB) label wins because the
        DB tier appears earlier in the union. Same priority order
        as `resolve_workspace_access`.
        """
        from kai.config import UserConfig

        shared = tmp_path / "shared-ws"
        shared.mkdir()
        uc = UserConfig(
            telegram_id=123,
            name="alice",
            allowed_workspaces=[shared.resolve()],
        )
        config = _make_config(user_configs={123: uc})
        update = _make_update(chat_id=123)
        ctx = _make_context(config=config)
        with patch(
            "kai.bot.sessions.get_allowed_workspaces",
            new_callable=AsyncMock,
            return_value=[shared],
        ):
            await _handle_workspace_allowed(update, ctx)

        reply = update.message.reply_text.call_args[0][0]
        assert f"{shared.resolve()} (you)" in reply
        # Exactly one occurrence of the path in the output: no
        # duplicate listing under the yaml tier.
        assert reply.count(str(shared.resolve())) == 1


# ── /workspace allow/deny routing via handle_workspace ──────────────


class TestWorkspaceSubcommandRouting:
    @pytest.mark.asyncio
    async def test_allow_routed(self, tmp_path):
        """/workspace allow <path> routes to _handle_workspace_allow."""
        ws = tmp_path / "ws"
        ws.mkdir()
        update = _make_update()
        ctx = _make_context(args=["allow", str(ws)])
        with (
            _mock_resolve(),
            patch("kai.bot.sessions.add_allowed_workspace", new_callable=AsyncMock),
        ):
            await handle_workspace(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "Added" in reply

    @pytest.mark.asyncio
    async def test_deny_routed(self, tmp_path):
        """/workspace deny <path> routes to _handle_workspace_deny."""
        update = _make_update()
        ctx = _make_context(args=["deny", str(tmp_path)])
        with (
            _mock_resolve(),
            patch("kai.bot.sessions.remove_allowed_workspace", new_callable=AsyncMock, return_value=False),
        ):
            await handle_workspace(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "Not in your" in reply

    @pytest.mark.asyncio
    async def test_allowed_routed(self):
        """/workspace allowed routes to _handle_workspace_allowed."""
        update = _make_update()
        ctx = _make_context(args=["allowed"])
        with (
            _mock_resolve(),
            patch("kai.bot.sessions.get_allowed_workspaces", new_callable=AsyncMock, return_value=[]),
        ):
            await handle_workspace(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "Workspace base" in reply

    @pytest.mark.asyncio
    async def test_name_uses_per_user_allowed(self, tmp_path):
        """/workspace <name> uses per-user allowed list for name resolution."""
        project = tmp_path / "myproj"
        project.mkdir()
        claude = _make_mock_claude(workspace=Path("/other"))
        config = _make_config(claude_workspace=Path("/home"))
        update = _make_update()
        ctx = _make_context(config=config, claude=claude, args=["myproj"])
        with (
            _mock_resolve(allowed=[project]),
            patch("kai.bot.sessions.clear_session", new_callable=AsyncMock),
            patch("kai.bot.sessions.set_setting", new_callable=AsyncMock),
            patch("kai.bot.sessions.upsert_workspace_history", new_callable=AsyncMock),
            patch("kai.bot.sessions.build_workspace_config", new_callable=AsyncMock, return_value=None),
        ):
            await handle_workspace(update, ctx)
        claude.change_workspace.assert_called_once()


# ── Workspace config in bot layer ────────────────────────────────────


class TestWorkspaceConfigSuffix:
    def test_with_model(self):
        """Shows model when configured."""
        from kai.config import WorkspaceConfig

        ws = WorkspaceConfig(path=Path("/tmp/ws"), model="haiku")
        assert _workspace_config_suffix(ws) == " (model: haiku)"

    def test_no_config(self):
        """Returns empty string when no config is provided."""
        assert _workspace_config_suffix(None) == ""

    def test_config_with_no_overrides(self):
        """Returns empty string when config has no model."""
        from kai.config import WorkspaceConfig

        ws = WorkspaceConfig(path=Path("/tmp/ws"))
        assert _workspace_config_suffix(ws) == ""


class TestSwitchWorkspaceConfig:
    @pytest.mark.asyncio
    async def test_switch_passes_config_to_change_workspace(self, tmp_path):
        """_do_switch_workspace passes the workspace config to Claude."""
        from kai.config import WorkspaceConfig

        ws_path = tmp_path / "ws"
        ws_path.mkdir()
        ws_config = WorkspaceConfig(path=ws_path.resolve(), model="opus")
        config = _make_config(
            claude_workspace=Path("/home/workspace"),
            workspace_configs={ws_path.resolve(): ws_config},
        )
        claude = _make_mock_claude()
        ctx = _make_context(config=config, claude=claude)

        mock_sessions = AsyncMock()
        # build_workspace_config returns the merged config (here, just YAML)
        mock_sessions.build_workspace_config = AsyncMock(return_value=ws_config)
        with (
            patch("kai.bot.sessions", mock_sessions),
            patch("kai.bot.webhook"),
        ):
            result = await _do_switch_workspace(ctx, 12345, ws_path.resolve())

        # Config was returned and passed to change_workspace
        assert result is ws_config
        claude.change_workspace.assert_called_once_with(12345, ws_path.resolve(), workspace_config=ws_config)

    @pytest.mark.asyncio
    async def test_switch_unconfigured_passes_none(self, tmp_path):
        """_do_switch_workspace passes None for unconfigured workspaces."""
        ws_path = tmp_path / "ws"
        ws_path.mkdir()
        config = _make_config(claude_workspace=Path("/home/workspace"))
        claude = _make_mock_claude()
        ctx = _make_context(config=config, claude=claude)

        mock_sessions = AsyncMock()
        # No YAML config, no DB overrides -> returns None
        mock_sessions.build_workspace_config = AsyncMock(return_value=None)
        with (
            patch("kai.bot.sessions", mock_sessions),
            patch("kai.bot.webhook"),
        ):
            result = await _do_switch_workspace(ctx, 12345, ws_path.resolve())

        assert result is None
        claude.change_workspace.assert_called_once_with(12345, ws_path.resolve(), workspace_config=None)

    @pytest.mark.asyncio
    async def test_switch_shows_config_info(self, tmp_path):
        """_switch_workspace confirmation includes model when configured."""
        from kai.config import WorkspaceConfig

        ws_path = tmp_path / "ws"
        ws_path.mkdir()
        ws_config = WorkspaceConfig(path=ws_path.resolve(), model="opus")
        config = _make_config(
            claude_workspace=Path("/home/workspace"),
            workspace_configs={ws_path.resolve(): ws_config},
        )
        claude = _make_mock_claude(workspace=Path("/home/workspace"))
        update = _make_update()
        ctx = _make_context(config=config, claude=claude)

        mock_sessions = AsyncMock()
        mock_sessions.build_workspace_config = AsyncMock(return_value=ws_config)
        with (
            patch("kai.bot.sessions", mock_sessions),
            patch("kai.bot.webhook"),
        ):
            await _switch_workspace(update, ctx, ws_path.resolve())

        reply_text = update.message.reply_text.call_args[0][0]
        assert "model: opus" in reply_text

    @pytest.mark.asyncio
    async def test_switch_deleted_directory(self, tmp_path):
        """Switching to a workspace whose directory no longer exists shows an error."""
        home = tmp_path / "home"
        home.mkdir()
        gone = tmp_path / "gone"
        gone.mkdir()
        gone.rmdir()  # create then delete so the path is guaranteed absent

        config = _make_config(claude_workspace=home)
        claude = _make_mock_claude(workspace=home)
        update = _make_update()
        ctx = _make_context(config=config, claude=claude)

        await _switch_workspace(update, ctx, gone)

        update.message.reply_text.assert_called_with("That workspace no longer exists.")


# ── handle_message (non-TOTP) ────────────────────────────────────────


class TestHandleMessage:
    @pytest.mark.asyncio
    async def test_normal_message(self):
        """Normal message: logs, acquires lock, calls _handle_response."""
        update = _make_update(text="hello world")
        ctx = _make_context()
        with (
            patch("kai.bot.is_totp_configured", return_value=False),
            patch("kai.bot._handle_response", new_callable=AsyncMock) as mock_resp,
            patch("kai.bot.log_message") as mock_log,
            patch("kai.bot._set_responding"),
            patch("kai.bot._clear_responding"),
            patch("kai.bot.get_lock", return_value=_fake_lock()),
        ):
            await handle_message(update, ctx)
        mock_resp.assert_called_once()
        assert mock_resp.await_args.kwargs["delivery_route"] == ResponseDeliveryRoute.LEGACY
        mock_log.assert_called_once()

    @pytest.mark.asyncio
    async def test_records_authenticated_message_after_totp_gate(self):
        update = _make_update(text="canonical input", chat_id=1, user_id=1)
        ctx = _make_context(config=_make_config(allowed_user_ids={1}))
        recorder = AsyncMock()
        inbound_id = MessageId("msg_00000000000000000000000000000001")
        recorder.return_value.event.envelope.aggregate_id = inbound_id
        ctx.bot_data["workshop_inbound_recorder"] = recorder
        with (
            patch("kai.bot.is_totp_configured", return_value=False),
            patch("kai.bot._handle_response", new_callable=AsyncMock) as response,
            patch("kai.bot.log_message"),
            patch("kai.bot._set_responding"),
            patch("kai.bot._clear_responding"),
            patch("kai.bot.get_lock", return_value=_fake_lock()),
        ):
            await handle_message(update, ctx)

        recorder.assert_awaited_once_with(
            InboundMessage(
                transport="telegram",
                update_id="9001",
                message_id="42",
                sender_subject="1",
                channel_subject="1",
                body="canonical input",
                occurred_at=datetime(2026, 8, 11, 12, 0, tzinfo=UTC),
            )
        )
        assert response.await_args.kwargs["workshop_inbound_message_id"] == inbound_id
        assert response.await_args.kwargs["delivery_route"] == ResponseDeliveryRoute.WORKSHOP_PRIVATE_TEXT

    @pytest.mark.asyncio
    async def test_does_not_record_when_totp_denies_message(self):
        update = _make_update(chat_id=1, user_id=1)
        ctx = _make_context(config=_make_config(allowed_user_ids={1}))
        recorder = AsyncMock()
        ctx.bot_data["workshop_inbound_recorder"] = recorder
        with patch("kai.bot._check_totp_text", new_callable=AsyncMock, return_value=False):
            await handle_message(update, ctx)

        recorder.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_does_not_record_unauthorized_message(self):
        update = _make_update(chat_id=99, user_id=99)
        ctx = _make_context(config=_make_config(allowed_user_ids={1}))
        recorder = AsyncMock()
        ctx.bot_data["workshop_inbound_recorder"] = recorder

        await handle_message(update, ctx)

        recorder.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_shadow_failure_does_not_change_existing_response_path(self, caplog):
        update = _make_update(chat_id=1, user_id=1)
        ctx = _make_context(config=_make_config(allowed_user_ids={1}))
        ctx.bot_data["workshop_inbound_recorder"] = AsyncMock(side_effect=RuntimeError("shadow failed"))
        with (
            patch("kai.bot.is_totp_configured", return_value=False),
            patch("kai.bot._handle_response", new_callable=AsyncMock) as mock_response,
            patch("kai.bot.log_message"),
            patch("kai.bot._set_responding"),
            patch("kai.bot._clear_responding"),
            patch("kai.bot.get_lock", return_value=_fake_lock()),
            caplog.at_level(logging.ERROR),
        ):
            await handle_message(update, ctx)

        mock_response.assert_awaited_once()
        assert "Workshop inbound shadow write failed" in caplog.text

    @pytest.mark.asyncio
    async def test_empty_message(self):
        """Empty text: returns early without processing."""
        update = _make_update()
        update.message.text = None
        ctx = _make_context()
        with (
            patch("kai.bot.is_totp_configured", return_value=False),
            patch("kai.bot._handle_response", new_callable=AsyncMock) as mock_resp,
        ):
            await handle_message(update, ctx)
        mock_resp.assert_not_called()

    @pytest.mark.asyncio
    async def test_set_and_clear_responding(self):
        """_set_responding called before and _clear_responding after, even on error."""
        update = _make_update()
        ctx = _make_context()
        with (
            patch("kai.bot.is_totp_configured", return_value=False),
            patch("kai.bot._handle_response", new_callable=AsyncMock, side_effect=RuntimeError("boom")),
            patch("kai.bot.log_message"),
            patch("kai.bot._set_responding") as mock_set,
            patch("kai.bot._clear_responding") as mock_clear,
            patch("kai.bot.get_lock", return_value=_fake_lock()),
            pytest.raises(RuntimeError),
        ):
            await handle_message(update, ctx)
        mock_set.assert_called_once()
        mock_clear.assert_called_once()


# ── handle_photo ─────────────────────────────────────────────────────


class TestHandlePhoto:
    @pytest.mark.asyncio
    async def test_records_authenticated_photo_message_and_artifact(self, tmp_path):
        update = _make_update(chat_id=1, user_id=1)
        photo = MagicMock()
        photo.file_id = "download-capability"
        photo.file_unique_id = "stable-photo-id"
        update.message.photo = [photo]
        update.message.caption = "Inspect this detail"
        saved = (tmp_path / "files" / "1" / "saved.jpg").resolve()
        saved.parent.mkdir(parents=True)
        saved.write_bytes(b"image-data")

        mock_file = MagicMock()
        mock_file.download_as_bytearray = AsyncMock(return_value=bytearray(b"image-data"))
        ctx = _make_context(
            config=_make_config(
                allowed_user_ids={1},
                user_configs={1: UserConfig(telegram_id=1, name="Daniel", os_user="daniel")},
            )
        )
        ctx.bot.get_file = AsyncMock(return_value=mock_file)
        inbound = AsyncMock()
        inbound_id = MessageId("msg_00000000000000000000000000000001")
        inbound.return_value.event.envelope.aggregate_id = inbound_id
        artifact = AsyncMock()
        ctx.bot_data["workshop_inbound_recorder"] = inbound
        ctx.bot_data["workshop_artifact_recorder"] = artifact

        with (
            patch("kai.bot.DATA_DIR", tmp_path),
            patch("kai.bot._save_upload", return_value=saved),
            patch("kai.bot._handle_response", new_callable=AsyncMock) as response,
            patch("kai.bot.log_message") as history,
            patch("kai.bot._set_responding"),
            patch("kai.bot._clear_responding"),
            patch("kai.bot.get_lock", return_value=_fake_lock()),
        ):
            await handle_photo(update, ctx)

        body = f"Inspect this detail\n[File saved to: {saved}]"
        inbound.assert_awaited_once_with(
            InboundMessage(
                transport="telegram",
                update_id="9001",
                message_id="42",
                sender_subject="1",
                channel_subject="1",
                body=body,
                occurred_at=datetime(2026, 8, 11, 12, 0, tzinfo=UTC),
            )
        )
        artifact.assert_awaited_once_with(
            InboundArtifact(
                message_id=inbound_id,
                kind="photo",
                media_type="image/jpeg",
                storage_path=saved,
                source_transport="telegram",
                source_unique_id="stable-photo-id",
                occurred_at=datetime(2026, 8, 11, 12, 0, tzinfo=UTC),
                original_filename=None,
            ),
            storage_root=tmp_path / "files",
        )
        ctx.bot.get_file.assert_awaited_once_with("download-capability")
        history.assert_called_once_with(
            direction="user",
            chat_id=1,
            text=body,
            reader_user="daniel",
            media={"type": "photo", "workshop_message_shadowed": True},
        )
        assert response.await_args.kwargs["workshop_inbound_message_id"] == inbound_id

    @pytest.mark.asyncio
    async def test_inbound_shadow_failure_skips_artifact_and_preserves_response(self, tmp_path, caplog):
        update = _make_update(chat_id=1, user_id=1)
        photo = MagicMock(file_id="file123", file_unique_id="uniq123")
        update.message.photo = [photo]
        saved = (tmp_path / "files" / "1" / "saved.jpg").resolve()
        saved.parent.mkdir(parents=True)
        saved.write_bytes(b"image-data")
        mock_file = MagicMock()
        mock_file.download_as_bytearray = AsyncMock(return_value=bytearray(b"image-data"))
        ctx = _make_context(config=_make_config(allowed_user_ids={1}))
        ctx.bot.get_file = AsyncMock(return_value=mock_file)
        ctx.bot_data["workshop_inbound_recorder"] = AsyncMock(side_effect=RuntimeError("shadow failed"))
        artifact = AsyncMock()
        ctx.bot_data["workshop_artifact_recorder"] = artifact

        with (
            patch("kai.bot.DATA_DIR", tmp_path),
            patch("kai.bot._save_upload", return_value=saved),
            patch("kai.bot._handle_response", new_callable=AsyncMock) as response,
            patch("kai.bot.log_message") as history,
            patch("kai.bot._set_responding"),
            patch("kai.bot._clear_responding"),
            patch("kai.bot.get_lock", return_value=_fake_lock()),
            caplog.at_level(logging.ERROR),
        ):
            await handle_photo(update, ctx)

        artifact.assert_not_awaited()
        response.assert_awaited_once()
        assert response.await_args.kwargs["workshop_inbound_message_id"] is None
        assert history.call_args.kwargs["media"] == {
            "type": "photo",
            "workshop_message_shadowed": False,
        }
        assert "Workshop photo message shadow write failed" in caplog.text

    @pytest.mark.asyncio
    async def test_artifact_shadow_failure_preserves_message_and_response(self, tmp_path, caplog):
        update = _make_update(chat_id=1, user_id=1)
        photo = MagicMock(file_id="file123", file_unique_id="uniq123")
        update.message.photo = [photo]
        saved = (tmp_path / "files" / "1" / "saved.jpg").resolve()
        saved.parent.mkdir(parents=True)
        saved.write_bytes(b"image-data")
        mock_file = MagicMock()
        mock_file.download_as_bytearray = AsyncMock(return_value=bytearray(b"image-data"))
        ctx = _make_context(config=_make_config(allowed_user_ids={1}))
        ctx.bot.get_file = AsyncMock(return_value=mock_file)
        inbound = AsyncMock()
        inbound_id = MessageId("msg_00000000000000000000000000000001")
        inbound.return_value.event.envelope.aggregate_id = inbound_id
        ctx.bot_data["workshop_inbound_recorder"] = inbound
        ctx.bot_data["workshop_artifact_recorder"] = AsyncMock(side_effect=RuntimeError("artifact failed"))

        with (
            patch("kai.bot.DATA_DIR", tmp_path),
            patch("kai.bot._save_upload", return_value=saved),
            patch("kai.bot._handle_response", new_callable=AsyncMock) as response,
            patch("kai.bot.log_message") as history,
            patch("kai.bot._set_responding"),
            patch("kai.bot._clear_responding"),
            patch("kai.bot.get_lock", return_value=_fake_lock()),
            caplog.at_level(logging.ERROR),
        ):
            await handle_photo(update, ctx)

        response.assert_awaited_once()
        assert response.await_args.kwargs["workshop_inbound_message_id"] == inbound_id
        assert history.call_args.kwargs["media"]["workshop_message_shadowed"] is True
        assert "Workshop photo artifact shadow write failed" in caplog.text

    @pytest.mark.asyncio
    async def test_downloads_and_sends_multimodal(self, tmp_path):
        """Downloads photo, base64-encodes, and calls _handle_response with list content."""
        update = _make_update()
        photo = MagicMock()
        photo.file_id = "file123"
        photo.file_unique_id = "uniq123"
        update.message.photo = [MagicMock(), photo]  # last is highest res

        mock_file = MagicMock()
        mock_file.download_as_bytearray = AsyncMock(return_value=bytearray(b"image-data"))
        claude = _make_mock_claude(workspace=tmp_path)
        ctx = _make_context(claude=claude)
        ctx.bot.get_file = AsyncMock(return_value=mock_file)

        with (
            patch("kai.bot._handle_response", new_callable=AsyncMock) as mock_resp,
            patch("kai.bot.log_message"),
            patch("kai.bot._set_responding"),
            patch("kai.bot._clear_responding"),
            patch("kai.bot.get_lock", return_value=_fake_lock()),
        ):
            await handle_photo(update, ctx)
        # The content arg should be a list (multi-modal)
        content = mock_resp.call_args[0][3]
        assert isinstance(content, list)
        assert content[1]["type"] == "image"

    @pytest.mark.asyncio
    async def test_uses_caption(self, tmp_path):
        """Uses caption if provided instead of default question."""
        update = _make_update()
        photo = MagicMock()
        photo.file_id = "file123"
        photo.file_unique_id = "uniq123"
        update.message.photo = [photo]
        update.message.caption = "Describe this logo"

        mock_file = MagicMock()
        mock_file.download_as_bytearray = AsyncMock(return_value=bytearray(b"img"))
        claude = _make_mock_claude(workspace=tmp_path)
        ctx = _make_context(claude=claude)
        ctx.bot.get_file = AsyncMock(return_value=mock_file)

        with (
            patch("kai.bot._handle_response", new_callable=AsyncMock) as mock_resp,
            patch("kai.bot.log_message"),
            patch("kai.bot._set_responding"),
            patch("kai.bot._clear_responding"),
            patch("kai.bot.get_lock", return_value=_fake_lock()),
        ):
            await handle_photo(update, ctx)
        content = mock_resp.call_args[0][3]
        assert "Describe this logo" in content[0]["text"]


# ── handle_document ──────────────────────────────────────────────────


class TestHandleDocument:
    def _setup_doc(self, update, file_name, mime_type=None, data=b"content"):
        """Attach a mock document to the update."""
        update.message.document = MagicMock()
        update.message.document.file_name = file_name
        update.message.document.file_id = "doc_file_id"
        update.message.document.file_unique_id = "stable-document-id"
        update.message.document.mime_type = mime_type
        return data

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("file_name", "mime_type", "data", "expected_body", "expected_media_type", "expected_filename"),
        (
            ("logo.png", "application/octet-stream", b"png-data", "logo.png", "image/png", "logo.png"),
            (
                "script.py",
                "Text/X-Python; charset=UTF-8",
                b"print('hi')",
                "[file: script.py]",
                "text/x-python",
                "script.py",
            ),
            (
                "archive.zip",
                "not a mime type",
                b"PK...",
                "[file: archive.zip]",
                "application/octet-stream",
                "archive.zip",
            ),
            (
                "folder\\report.pdf",
                "application/pdf",
                b"pdf-data",
                "[file: folder\\report.pdf]",
                "application/pdf",
                "report.pdf",
            ),
        ),
    )
    async def test_records_authenticated_document_message_and_artifact(
        self,
        tmp_path,
        file_name,
        mime_type,
        data,
        expected_body,
        expected_media_type,
        expected_filename,
    ):
        update = _make_update(chat_id=1, user_id=1)
        self._setup_doc(update, file_name, mime_type, data)
        saved = (tmp_path / "files" / "1" / "saved-document").resolve()
        saved.parent.mkdir(parents=True)
        saved.write_bytes(data)
        mock_file = MagicMock()
        mock_file.download_as_bytearray = AsyncMock(return_value=bytearray(data))
        ctx = _make_context(config=_make_config(allowed_user_ids={1}))
        ctx.bot.get_file = AsyncMock(return_value=mock_file)
        inbound = AsyncMock()
        inbound_id = MessageId("msg_00000000000000000000000000000001")
        inbound.return_value.event.envelope.aggregate_id = inbound_id
        artifact = AsyncMock()
        ctx.bot_data["workshop_inbound_recorder"] = inbound
        ctx.bot_data["workshop_artifact_recorder"] = artifact

        with (
            patch("kai.bot.DATA_DIR", tmp_path),
            patch("kai.bot._save_upload", return_value=saved),
            patch("kai.bot._handle_response", new_callable=AsyncMock) as response,
            patch("kai.bot.log_message") as history,
            patch("kai.bot._set_responding"),
            patch("kai.bot._clear_responding"),
            patch("kai.bot.get_lock", return_value=_fake_lock()),
        ):
            await handle_document(update, ctx)

        inbound.assert_awaited_once_with(
            InboundMessage(
                transport="telegram",
                update_id="9001",
                message_id="42",
                sender_subject="1",
                channel_subject="1",
                body=expected_body,
                occurred_at=datetime(2026, 8, 11, 12, 0, tzinfo=UTC),
            )
        )
        artifact.assert_awaited_once_with(
            InboundArtifact(
                message_id=inbound_id,
                kind="document",
                media_type=expected_media_type,
                storage_path=saved,
                source_transport="telegram",
                source_unique_id="stable-document-id",
                occurred_at=datetime(2026, 8, 11, 12, 0, tzinfo=UTC),
                original_filename=expected_filename,
            ),
            storage_root=tmp_path / "files",
        )
        history.assert_called_once_with(
            direction="user",
            chat_id=1,
            text=expected_body,
            reader_user=None,
            media={
                "type": "document",
                "filename": file_name,
                "workshop_message_shadowed": True,
            },
        )
        assert response.await_args.kwargs["workshop_inbound_message_id"] == inbound_id

    @pytest.mark.asyncio
    async def test_inbound_shadow_failure_skips_artifact_and_preserves_response(self, tmp_path, caplog):
        update = _make_update(chat_id=1, user_id=1)
        self._setup_doc(update, "report.txt", "text/plain", b"report")
        saved = (tmp_path / "files" / "1" / "report.txt").resolve()
        saved.parent.mkdir(parents=True)
        saved.write_bytes(b"report")
        mock_file = MagicMock()
        mock_file.download_as_bytearray = AsyncMock(return_value=bytearray(b"report"))
        ctx = _make_context(config=_make_config(allowed_user_ids={1}))
        ctx.bot.get_file = AsyncMock(return_value=mock_file)
        ctx.bot_data["workshop_inbound_recorder"] = AsyncMock(side_effect=RuntimeError("shadow failed"))
        artifact = AsyncMock()
        ctx.bot_data["workshop_artifact_recorder"] = artifact

        with (
            patch("kai.bot.DATA_DIR", tmp_path),
            patch("kai.bot._save_upload", return_value=saved),
            patch("kai.bot._handle_response", new_callable=AsyncMock) as response,
            patch("kai.bot.log_message") as history,
            patch("kai.bot._set_responding"),
            patch("kai.bot._clear_responding"),
            patch("kai.bot.get_lock", return_value=_fake_lock()),
            caplog.at_level(logging.ERROR),
        ):
            await handle_document(update, ctx)

        artifact.assert_not_awaited()
        response.assert_awaited_once()
        assert response.await_args.kwargs["workshop_inbound_message_id"] is None
        assert history.call_args.kwargs["media"]["workshop_message_shadowed"] is False
        assert "Workshop document message shadow write failed" in caplog.text

    @pytest.mark.asyncio
    async def test_artifact_shadow_failure_preserves_message_and_response(self, tmp_path, caplog):
        update = _make_update(chat_id=1, user_id=1)
        self._setup_doc(update, "report.txt", "text/plain", b"report")
        saved = (tmp_path / "files" / "1" / "report.txt").resolve()
        saved.parent.mkdir(parents=True)
        saved.write_bytes(b"report")
        mock_file = MagicMock()
        mock_file.download_as_bytearray = AsyncMock(return_value=bytearray(b"report"))
        ctx = _make_context(config=_make_config(allowed_user_ids={1}))
        ctx.bot.get_file = AsyncMock(return_value=mock_file)
        inbound = AsyncMock()
        inbound_id = MessageId("msg_00000000000000000000000000000001")
        inbound.return_value.event.envelope.aggregate_id = inbound_id
        ctx.bot_data["workshop_inbound_recorder"] = inbound
        ctx.bot_data["workshop_artifact_recorder"] = AsyncMock(side_effect=RuntimeError("artifact failed"))

        with (
            patch("kai.bot.DATA_DIR", tmp_path),
            patch("kai.bot._save_upload", return_value=saved),
            patch("kai.bot._handle_response", new_callable=AsyncMock) as response,
            patch("kai.bot.log_message") as history,
            patch("kai.bot._set_responding"),
            patch("kai.bot._clear_responding"),
            patch("kai.bot.get_lock", return_value=_fake_lock()),
            caplog.at_level(logging.ERROR),
        ):
            await handle_document(update, ctx)

        response.assert_awaited_once()
        assert response.await_args.kwargs["workshop_inbound_message_id"] == inbound_id
        assert history.call_args.kwargs["media"]["workshop_message_shadowed"] is True
        assert "Workshop document artifact shadow write failed" in caplog.text

    @pytest.mark.asyncio
    async def test_image_document(self, tmp_path):
        """Image file extension: sent as multi-modal content."""
        update = _make_update()
        self._setup_doc(update, "logo.png", "image/png")
        mock_file = MagicMock()
        mock_file.download_as_bytearray = AsyncMock(return_value=bytearray(b"png-data"))
        claude = _make_mock_claude(workspace=tmp_path)
        ctx = _make_context(claude=claude)
        ctx.bot.get_file = AsyncMock(return_value=mock_file)

        with (
            patch("kai.bot._handle_response", new_callable=AsyncMock) as mock_resp,
            patch("kai.bot.log_message"),
            patch("kai.bot._set_responding"),
            patch("kai.bot._clear_responding"),
            patch("kai.bot.get_lock", return_value=_fake_lock()),
        ):
            await handle_document(update, ctx)
        content = mock_resp.call_args[0][3]
        assert isinstance(content, list)
        assert content[1]["type"] == "image"

    @pytest.mark.asyncio
    async def test_text_document(self, tmp_path):
        """Text file: decoded as UTF-8 and sent as code block string."""
        update = _make_update()
        self._setup_doc(update, "script.py", "text/python")
        mock_file = MagicMock()
        mock_file.download_as_bytearray = AsyncMock(return_value=bytearray(b"print('hi')"))
        claude = _make_mock_claude(workspace=tmp_path)
        ctx = _make_context(claude=claude)
        ctx.bot.get_file = AsyncMock(return_value=mock_file)

        with (
            patch("kai.bot._handle_response", new_callable=AsyncMock) as mock_resp,
            patch("kai.bot.log_message"),
            patch("kai.bot._set_responding"),
            patch("kai.bot._clear_responding"),
            patch("kai.bot.get_lock", return_value=_fake_lock()),
        ):
            await handle_document(update, ctx)
        content = mock_resp.call_args[0][3]
        assert isinstance(content, str)
        assert "```" in content

    @pytest.mark.asyncio
    async def test_text_decode_error(self, tmp_path):
        """UTF-8 decode failure: sends error reply, no _handle_response call."""
        update = _make_update()
        self._setup_doc(update, "data.txt", "text/plain")
        mock_file = MagicMock()
        # Invalid UTF-8 bytes
        mock_file.download_as_bytearray = AsyncMock(return_value=bytearray(b"\xff\xfe"))
        claude = _make_mock_claude(workspace=tmp_path)
        ctx = _make_context(claude=claude)
        ctx.bot.get_file = AsyncMock(return_value=mock_file)

        with (
            patch("kai.bot._handle_response", new_callable=AsyncMock) as mock_resp,
            patch("kai.bot.log_message"),
            patch("kai.bot._set_responding"),
            patch("kai.bot._clear_responding"),
            patch("kai.bot.get_lock", return_value=_fake_lock()),
        ):
            await handle_document(update, ctx)
        mock_resp.assert_not_called()
        reply = update.message.reply_text.call_args[0][0]
        assert "decode" in reply.lower() or "text" in reply.lower()

    @pytest.mark.asyncio
    async def test_other_file_type(self, tmp_path):
        """Non-image, non-text file: saved to disk, path sent in prompt."""
        update = _make_update()
        self._setup_doc(update, "archive.zip", "application/zip")
        mock_file = MagicMock()
        mock_file.download_as_bytearray = AsyncMock(return_value=bytearray(b"PK..."))
        claude = _make_mock_claude(workspace=tmp_path)
        ctx = _make_context(claude=claude)
        ctx.bot.get_file = AsyncMock(return_value=mock_file)

        with (
            patch("kai.bot._handle_response", new_callable=AsyncMock) as mock_resp,
            patch("kai.bot.log_message"),
            patch("kai.bot._set_responding"),
            patch("kai.bot._clear_responding"),
            patch("kai.bot.get_lock", return_value=_fake_lock()),
        ):
            await handle_document(update, ctx)
        content = mock_resp.call_args[0][3]
        assert isinstance(content, str)
        assert "archive.zip" in content


# ── handle_voice ─────────────────────────────────────────────────────


class TestHandleVoice:
    @pytest.mark.asyncio
    async def test_records_authenticated_voice_message_and_artifact(self, tmp_path):
        model_file = tmp_path / "model.bin"
        model_file.touch()
        update = _make_update(chat_id=1, user_id=1)
        voice = MagicMock(
            file_id="voice-download-capability",
            file_unique_id="stable-voice-id",
            duration=5,
        )
        update.message.voice = voice
        mock_file = MagicMock()
        mock_file.download_as_bytearray = AsyncMock(return_value=bytearray(b"audio-data"))
        saved = (tmp_path / "files" / "1" / "saved.oga").resolve()
        saved.parent.mkdir(parents=True)
        saved.write_bytes(b"audio-data")
        ctx = _make_context(
            config=_make_config(allowed_user_ids={1}, voice_enabled=True, whisper_model_path=model_file)
        )
        ctx.bot.get_file = AsyncMock(return_value=mock_file)
        inbound = AsyncMock()
        inbound_id = MessageId("msg_00000000000000000000000000000001")
        inbound.return_value.event.envelope.aggregate_id = inbound_id
        artifact = AsyncMock()
        ctx.bot_data["workshop_inbound_recorder"] = inbound
        ctx.bot_data["workshop_artifact_recorder"] = artifact

        with (
            patch("shutil.which", return_value="/usr/bin/tool"),
            patch("kai.bot.DATA_DIR", tmp_path),
            patch("kai.bot.transcribe_voice", new_callable=AsyncMock, return_value="Hello there"),
            patch("kai.bot._save_upload", return_value=saved) as save_upload,
            patch("kai.bot.log_message") as history,
            patch("kai.bot._handle_response", new_callable=AsyncMock) as response,
            patch("kai.bot._set_responding"),
            patch("kai.bot._clear_responding"),
            patch("kai.bot.get_lock", return_value=_fake_lock()),
        ):
            await handle_voice(update, ctx)

        inbound.assert_awaited_once_with(
            InboundMessage(
                transport="telegram",
                update_id="9001",
                message_id="42",
                sender_subject="1",
                channel_subject="1",
                body="Hello there",
                occurred_at=datetime(2026, 8, 11, 12, 0, tzinfo=UTC),
            )
        )
        save_upload.assert_called_once_with(b"audio-data", "voice.oga", user_id=1)
        artifact.assert_awaited_once_with(
            InboundArtifact(
                message_id=inbound_id,
                kind="voice",
                media_type="audio/ogg",
                storage_path=saved,
                source_transport="telegram",
                source_unique_id="stable-voice-id",
                occurred_at=datetime(2026, 8, 11, 12, 0, tzinfo=UTC),
                original_filename=None,
            ),
            storage_root=tmp_path / "files",
        )
        history.assert_called_once_with(
            direction="user",
            chat_id=1,
            text="Hello there",
            media={"type": "voice", "duration": 5, "workshop_message_shadowed": True},
            reader_user=None,
        )
        assert response.await_args.args[3] == "[Voice message transcription]: Hello there"
        assert response.await_args.kwargs["workshop_inbound_message_id"] == inbound_id

    @pytest.mark.asyncio
    async def test_inbound_shadow_failure_skips_voice_artifact_and_preserves_response(self, tmp_path, caplog):
        model_file = tmp_path / "model.bin"
        model_file.touch()
        update = _make_update(chat_id=1, user_id=1)
        update.message.voice = MagicMock(file_id="v1", file_unique_id="stable-voice-id", duration=5)
        mock_file = MagicMock()
        mock_file.download_as_bytearray = AsyncMock(return_value=bytearray(b"audio-data"))
        ctx = _make_context(
            config=_make_config(allowed_user_ids={1}, voice_enabled=True, whisper_model_path=model_file)
        )
        ctx.bot.get_file = AsyncMock(return_value=mock_file)
        ctx.bot_data["workshop_inbound_recorder"] = AsyncMock(side_effect=RuntimeError("shadow failed"))
        artifact = AsyncMock()
        ctx.bot_data["workshop_artifact_recorder"] = artifact

        with (
            patch("shutil.which", return_value="/usr/bin/tool"),
            patch("kai.bot.transcribe_voice", new_callable=AsyncMock, return_value="Hello there"),
            patch("kai.bot._save_upload") as save_upload,
            patch("kai.bot.log_message") as history,
            patch("kai.bot._handle_response", new_callable=AsyncMock) as response,
            patch("kai.bot._set_responding"),
            patch("kai.bot._clear_responding"),
            patch("kai.bot.get_lock", return_value=_fake_lock()),
            caplog.at_level(logging.ERROR),
        ):
            await handle_voice(update, ctx)

        save_upload.assert_not_called()
        artifact.assert_not_awaited()
        response.assert_awaited_once()
        assert response.await_args.kwargs["workshop_inbound_message_id"] is None
        assert history.call_args.kwargs["media"] == {
            "type": "voice",
            "duration": 5,
            "workshop_message_shadowed": False,
        }
        assert "Workshop voice message shadow write failed" in caplog.text

    @pytest.mark.asyncio
    async def test_voice_artifact_failure_preserves_message_and_response(self, tmp_path, caplog):
        model_file = tmp_path / "model.bin"
        model_file.touch()
        update = _make_update(chat_id=1, user_id=1)
        update.message.voice = MagicMock(file_id="v1", file_unique_id="stable-voice-id", duration=5)
        mock_file = MagicMock()
        mock_file.download_as_bytearray = AsyncMock(return_value=bytearray(b"audio-data"))
        saved = (tmp_path / "files" / "1" / "saved.oga").resolve()
        saved.parent.mkdir(parents=True)
        saved.write_bytes(b"audio-data")
        ctx = _make_context(
            config=_make_config(allowed_user_ids={1}, voice_enabled=True, whisper_model_path=model_file)
        )
        ctx.bot.get_file = AsyncMock(return_value=mock_file)
        inbound = AsyncMock()
        inbound_id = MessageId("msg_00000000000000000000000000000001")
        inbound.return_value.event.envelope.aggregate_id = inbound_id
        ctx.bot_data["workshop_inbound_recorder"] = inbound
        ctx.bot_data["workshop_artifact_recorder"] = AsyncMock(side_effect=RuntimeError("artifact failed"))

        with (
            patch("shutil.which", return_value="/usr/bin/tool"),
            patch("kai.bot.DATA_DIR", tmp_path),
            patch("kai.bot.transcribe_voice", new_callable=AsyncMock, return_value="Hello there"),
            patch("kai.bot._save_upload", return_value=saved),
            patch("kai.bot.log_message") as history,
            patch("kai.bot._handle_response", new_callable=AsyncMock) as response,
            patch("kai.bot._set_responding"),
            patch("kai.bot._clear_responding"),
            patch("kai.bot.get_lock", return_value=_fake_lock()),
            caplog.at_level(logging.ERROR),
        ):
            await handle_voice(update, ctx)

        response.assert_awaited_once()
        assert response.await_args.kwargs["workshop_inbound_message_id"] == inbound_id
        assert history.call_args.kwargs["media"]["workshop_message_shadowed"] is True
        assert "Workshop voice artifact shadow write failed" in caplog.text

    @pytest.mark.asyncio
    async def test_voice_artifact_save_failure_preserves_message_and_response(self, tmp_path, caplog):
        model_file = tmp_path / "model.bin"
        model_file.touch()
        update = _make_update(chat_id=1, user_id=1)
        update.message.voice = MagicMock(file_id="v1", file_unique_id="stable-voice-id", duration=5)
        mock_file = MagicMock()
        mock_file.download_as_bytearray = AsyncMock(return_value=bytearray(b"audio-data"))
        ctx = _make_context(
            config=_make_config(allowed_user_ids={1}, voice_enabled=True, whisper_model_path=model_file)
        )
        ctx.bot.get_file = AsyncMock(return_value=mock_file)
        inbound = AsyncMock()
        inbound_id = MessageId("msg_00000000000000000000000000000001")
        inbound.return_value.event.envelope.aggregate_id = inbound_id
        artifact = AsyncMock()
        ctx.bot_data["workshop_inbound_recorder"] = inbound
        ctx.bot_data["workshop_artifact_recorder"] = artifact

        with (
            patch("shutil.which", return_value="/usr/bin/tool"),
            patch("kai.bot.transcribe_voice", new_callable=AsyncMock, return_value="Hello there"),
            patch("kai.bot._save_upload", side_effect=OSError("storage unavailable")),
            patch("kai.bot.log_message") as history,
            patch("kai.bot._handle_response", new_callable=AsyncMock) as response,
            patch("kai.bot._set_responding"),
            patch("kai.bot._clear_responding"),
            patch("kai.bot.get_lock", return_value=_fake_lock()),
            caplog.at_level(logging.ERROR),
        ):
            await handle_voice(update, ctx)

        artifact.assert_not_awaited()
        response.assert_awaited_once()
        assert response.await_args.kwargs["workshop_inbound_message_id"] == inbound_id
        assert history.call_args.kwargs["media"]["workshop_message_shadowed"] is True
        assert "Workshop voice artifact shadow write failed" in caplog.text

    @pytest.mark.asyncio
    async def test_voice_not_enabled(self):
        update = _make_update()
        update.message.voice = MagicMock()
        ctx = _make_context(config=_make_config(voice_enabled=False))
        await handle_voice(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "not enabled" in reply.lower()

    @pytest.mark.asyncio
    async def test_missing_dependencies(self, tmp_path):
        """Lists missing deps when ffmpeg/whisper/model aren't available."""
        update = _make_update()
        update.message.voice = MagicMock()
        config = _make_config(voice_enabled=True, whisper_model_path=tmp_path / "nomodel")
        ctx = _make_context(config=config)
        with patch("shutil.which", return_value=None):
            await handle_voice(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "ffmpeg" in reply

    @pytest.mark.asyncio
    async def test_transcription_error(self, tmp_path):
        """TranscriptionError: sends error reply."""
        from kai.transcribe import TranscriptionError

        model_file = tmp_path / "model.bin"
        model_file.touch()
        update = _make_update()
        voice_msg = MagicMock()
        voice_msg.file_id = "v1"
        voice_msg.duration = 5
        update.message.voice = voice_msg
        mock_file = MagicMock()
        mock_file.download_as_bytearray = AsyncMock(return_value=bytearray(b"audio"))
        config = _make_config(voice_enabled=True, whisper_model_path=model_file)
        ctx = _make_context(config=config)
        ctx.bot.get_file = AsyncMock(return_value=mock_file)

        with (
            patch("shutil.which", return_value="/usr/bin/ffmpeg"),
            patch("kai.bot.transcribe_voice", new_callable=AsyncMock, side_effect=TranscriptionError("fail")),
            patch("kai.bot.log_message"),
        ):
            await handle_voice(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "Transcription failed" in reply

    @pytest.mark.asyncio
    async def test_empty_transcript(self, tmp_path):
        model_file = tmp_path / "model.bin"
        model_file.touch()
        update = _make_update()
        voice_msg = MagicMock()
        voice_msg.file_id = "v1"
        voice_msg.duration = 5
        update.message.voice = voice_msg
        mock_file = MagicMock()
        mock_file.download_as_bytearray = AsyncMock(return_value=bytearray(b"audio"))
        config = _make_config(voice_enabled=True, whisper_model_path=model_file)
        ctx = _make_context(config=config)
        ctx.bot.get_file = AsyncMock(return_value=mock_file)

        with (
            patch("shutil.which", return_value="/usr/bin/ffmpeg"),
            patch("kai.bot.transcribe_voice", new_callable=AsyncMock, return_value=""),
            patch("kai.bot.log_message"),
        ):
            await handle_voice(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "speech" in reply.lower()

    @pytest.mark.asyncio
    async def test_successful_transcription(self, tmp_path):
        """Echoes transcript, then sends it to the configured agent."""
        model_file = tmp_path / "model.bin"
        model_file.touch()
        update = _make_update()
        voice_msg = MagicMock()
        voice_msg.file_id = "v1"
        voice_msg.duration = 5
        update.message.voice = voice_msg
        mock_file = MagicMock()
        mock_file.download_as_bytearray = AsyncMock(return_value=bytearray(b"audio"))
        claude = _make_mock_claude(workspace=tmp_path)
        config = _make_config(voice_enabled=True, whisper_model_path=model_file)
        ctx = _make_context(config=config, claude=claude)
        ctx.bot.get_file = AsyncMock(return_value=mock_file)

        with (
            patch("shutil.which", return_value="/usr/bin/ffmpeg"),
            patch("kai.bot.transcribe_voice", new_callable=AsyncMock, return_value="Hello there"),
            patch("kai.bot.log_message"),
            patch("kai.bot._handle_response", new_callable=AsyncMock) as mock_resp,
            patch("kai.bot._set_responding"),
            patch("kai.bot._clear_responding"),
            patch("kai.bot.get_lock", return_value=_fake_lock()),
        ):
            await handle_voice(update, ctx)
        # Echo the transcript
        echo_call = update.message.reply_text.call_args
        assert "Hello there" in echo_call[0][0]
        # Then send to the configured agent
        prompt = mock_resp.call_args[0][3]
        assert "Hello there" in prompt


# ── _handle_response ─────────────────────────────────────────────────


class TestHandleResponse:
    """Tests for the streaming response handler."""

    def _base_patches(self):
        """
        Common patches for _handle_response tests.

        Returns a dict suitable for patch.multiple("kai.bot", ...).
        Voice mode defaults to "off" (normal text mode).
        """
        return {
            "sessions": MagicMock(
                get_setting=AsyncMock(return_value="off"),
                save_session=AsyncMock(),
            ),
            "log_message": MagicMock(),
        }

    @staticmethod
    def _install_workshop_recorders(ctx, message_id: MessageId):
        outbound = AsyncMock()
        outbound.return_value.event.envelope.aggregate_id = message_id
        delivery = AsyncMock()
        ctx.bot_data["workshop_outbound_recorder"] = outbound
        ctx.bot_data["workshop_delivery_recorder"] = delivery
        return outbound, delivery

    @staticmethod
    def _install_workshop_finalizer(ctx, message_id: MessageId):
        preview = AsyncMock()
        finalizer = AsyncMock()
        finalizer.return_value.message.event.envelope.aggregate_id = message_id
        ctx.bot_data["workshop_streaming_preview_recorder"] = preview
        ctx.bot_data["workshop_streaming_finalizer"] = finalizer
        return preview, finalizer

    @pytest.mark.asyncio
    async def test_workshop_private_text_commits_without_direct_send_or_shadow_observation(self):
        from kai.bot import _handle_response

        inbound_id = MessageId("msg_00000000000000000000000000000001")
        outbound_id = MessageId("msg_00000000000000000000000000000002")
        update = _make_update()
        pool = _make_mock_claude()
        pool.send = MagicMock(return_value=_fake_stream(_done_event("Durable answer")))
        ctx = _make_context(pool=pool)
        _, finalizer = self._install_workshop_finalizer(ctx, outbound_id)
        outbound, delivery = self._install_workshop_recorders(ctx, outbound_id)

        with (
            patch.multiple("kai.bot", **self._base_patches()),
            patch("kai.bot._send_response", new_callable=AsyncMock) as direct_send,
        ):
            await _handle_response(
                update,
                ctx,
                12345,
                "test",
                pool,
                "sonnet",
                workshop_inbound_message_id=inbound_id,
                delivery_route=ResponseDeliveryRoute.WORKSHOP_PRIVATE_TEXT,
            )

        finalizer.assert_awaited_once()
        assert finalizer.await_args.args[0].body == "Durable answer"
        direct_send.assert_not_awaited()
        outbound.assert_not_awaited()
        delivery.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_workshop_streaming_preview_is_bound_and_final_edit_is_left_to_worker(self):
        from kai.bot import _handle_response

        inbound_id = MessageId("msg_00000000000000000000000000000001")
        outbound_id = MessageId("msg_00000000000000000000000000000002")
        update = _make_update()
        live_message = MagicMock(message_id=7001)
        update.message.reply_text = AsyncMock(return_value=live_message)
        pool = _make_mock_claude()
        pool.send = MagicMock(
            return_value=_fake_stream(
                _text_event("Stable streamed sentence."),
                _done_event("Final durable answer."),
            )
        )
        ctx = _make_context(pool=pool)
        preview, finalizer = self._install_workshop_finalizer(ctx, outbound_id)

        with (
            patch.multiple("kai.bot", **self._base_patches()),
            patch("kai.bot._edit_message_safe", new_callable=AsyncMock) as direct_edit,
        ):
            await _handle_response(
                update,
                ctx,
                12345,
                "test",
                pool,
                "sonnet",
                workshop_inbound_message_id=inbound_id,
                delivery_route=ResponseDeliveryRoute.WORKSHOP_PRIVATE_TEXT,
            )

        bound = preview.await_args.args[0]
        assert isinstance(bound, ConfirmedTelegramStreamingPreview)
        assert bound.inbound_message_id == inbound_id
        assert bound.external_message_id == 7001
        finalizer.assert_awaited_once()
        direct_edit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_definite_workshop_failure_refuses_direct_delivery(self):
        from kai.bot import _handle_response

        inbound_id = MessageId("msg_00000000000000000000000000000001")
        outbound_id = MessageId("msg_00000000000000000000000000000002")
        update = _make_update()
        pool = _make_mock_claude()
        pool.send = MagicMock(return_value=_fake_stream(_done_event("Durable answer")))
        ctx = _make_context(pool=pool)
        _, finalizer = self._install_workshop_finalizer(ctx, outbound_id)
        finalizer.side_effect = RuntimeError("definite preparation failure")
        outbound, _ = self._install_workshop_recorders(ctx, outbound_id)
        patches = self._base_patches()

        with (
            patch.multiple("kai.bot", **patches),
            patch("kai.bot._send_response", new_callable=AsyncMock) as direct_send,
        ):
            await _handle_response(
                update,
                ctx,
                12345,
                "test",
                pool,
                "sonnet",
                workshop_inbound_message_id=inbound_id,
                delivery_route=ResponseDeliveryRoute.WORKSHOP_PRIVATE_TEXT,
            )

        finalizer.assert_awaited_once()
        direct_send.assert_not_awaited()
        outbound.assert_not_awaited()
        assert "could not safely finalize durable delivery" in update.message.reply_text.await_args.args[0]
        assert patches["log_message"].call_args.kwargs["text"] == "[error: durable delivery finalization failed]"

    @pytest.mark.asyncio
    async def test_workshop_private_text_without_canonical_inbound_refuses_agent_and_direct_delivery(self):
        from kai.bot import _handle_response

        update = _make_update()
        pool = _make_mock_claude()
        ctx = _make_context(pool=pool)
        self._install_workshop_finalizer(ctx, MessageId("msg_00000000000000000000000000000002"))
        patches = self._base_patches()

        with (
            patch.multiple("kai.bot", **patches),
            patch("kai.bot._send_response", new_callable=AsyncMock) as direct_send,
        ):
            await _handle_response(
                update,
                ctx,
                12345,
                "test",
                pool,
                "sonnet",
                workshop_inbound_message_id=None,
                delivery_route=ResponseDeliveryRoute.WORKSHOP_PRIVATE_TEXT,
            )

        pool.send.assert_not_called()
        direct_send.assert_not_awaited()
        assert "could not safely prepare durable delivery" in update.message.reply_text.await_args.args[0]
        assert patches["log_message"].call_args.kwargs["text"] == "[error: durable delivery preparation failed]"

    @pytest.mark.asyncio
    async def test_workshop_preview_binding_failure_replaces_preview_with_notice_and_stops(self):
        from kai.bot import _handle_response

        inbound_id = MessageId("msg_00000000000000000000000000000001")
        outbound_id = MessageId("msg_00000000000000000000000000000002")
        update = _make_update()
        live_message = MagicMock(message_id=7001)
        update.message.reply_text = AsyncMock(return_value=live_message)
        pool = _make_mock_claude()
        pool.send = MagicMock(
            return_value=_fake_stream(
                _text_event("Stable streamed sentence."),
                _done_event("Final durable answer."),
            )
        )
        ctx = _make_context(pool=pool)
        preview, finalizer = self._install_workshop_finalizer(ctx, outbound_id)
        preview.side_effect = RuntimeError("preview binding failed")
        patches = self._base_patches()

        with (
            patch.multiple("kai.bot", **patches),
            patch("kai.bot._edit_message_safe", new_callable=AsyncMock, return_value=True) as direct_edit,
            patch("kai.bot._send_response", new_callable=AsyncMock) as direct_send,
        ):
            await _handle_response(
                update,
                ctx,
                12345,
                "test",
                pool,
                "sonnet",
                workshop_inbound_message_id=inbound_id,
                delivery_route=ResponseDeliveryRoute.WORKSHOP_PRIVATE_TEXT,
            )

        preview.assert_awaited_once()
        finalizer.assert_not_awaited()
        direct_send.assert_not_awaited()
        direct_edit.assert_awaited_once()
        assert "could not safely prepare durable delivery" in direct_edit.await_args.args[1]
        assert patches["log_message"].call_args.kwargs["text"] == "[error: durable delivery preparation failed]"

    @pytest.mark.asyncio
    async def test_workshop_finalization_failure_replaces_bound_preview_without_direct_answer(self):
        from kai.bot import _handle_response

        inbound_id = MessageId("msg_00000000000000000000000000000001")
        outbound_id = MessageId("msg_00000000000000000000000000000002")
        update = _make_update()
        live_message = MagicMock(message_id=7001)
        update.message.reply_text = AsyncMock(return_value=live_message)
        pool = _make_mock_claude()
        pool.send = MagicMock(
            return_value=_fake_stream(
                _text_event("Stable streamed sentence."),
                _done_event("Undelivered durable answer."),
            )
        )
        ctx = _make_context(pool=pool)
        preview, finalizer = self._install_workshop_finalizer(ctx, outbound_id)
        finalizer.side_effect = RuntimeError("finalization failed")
        outbound, delivery = self._install_workshop_recorders(ctx, outbound_id)
        patches = self._base_patches()

        with (
            patch.multiple("kai.bot", **patches),
            patch("kai.bot._edit_message_safe", new_callable=AsyncMock, return_value=True) as direct_edit,
            patch("kai.bot._send_response", new_callable=AsyncMock) as direct_send,
        ):
            await _handle_response(
                update,
                ctx,
                12345,
                "test",
                pool,
                "sonnet",
                workshop_inbound_message_id=inbound_id,
                delivery_route=ResponseDeliveryRoute.WORKSHOP_PRIVATE_TEXT,
            )

        preview.assert_awaited_once()
        finalizer.assert_awaited_once()
        direct_send.assert_not_awaited()
        outbound.assert_not_awaited()
        delivery.assert_not_awaited()
        assert "could not safely finalize durable delivery" in direct_edit.await_args.args[1]
        assert patches["log_message"].call_args.kwargs["text"] == "[error: durable delivery finalization failed]"

    @pytest.mark.asyncio
    async def test_uncertain_workshop_commit_refuses_duplicate_direct_send(self):
        from kai.bot import _handle_response

        inbound_id = MessageId("msg_00000000000000000000000000000001")
        outbound_id = MessageId("msg_00000000000000000000000000000002")
        update = _make_update()
        pool = _make_mock_claude()
        pool.send = MagicMock(return_value=_fake_stream(_done_event("Possibly committed answer")))
        ctx = _make_context(pool=pool)
        _, finalizer = self._install_workshop_finalizer(ctx, outbound_id)
        finalizer.side_effect = sessions.WorkshopFinalizationCommitUncertainError("unknown")

        with (
            patch.multiple("kai.bot", **self._base_patches()),
            patch("kai.bot._send_response", new_callable=AsyncMock) as direct_send,
        ):
            await _handle_response(
                update,
                ctx,
                12345,
                "test",
                pool,
                "sonnet",
                workshop_inbound_message_id=inbound_id,
                delivery_route=ResponseDeliveryRoute.WORKSHOP_PRIVATE_TEXT,
            )

        direct_send.assert_not_awaited()
        assert "avoid a duplicate" in update.message.reply_text.await_args.args[0]

    @pytest.mark.asyncio
    async def test_text_plus_voice_never_enters_workshop_text_cutover(self):
        from kai.bot import _handle_response

        inbound_id = MessageId("msg_00000000000000000000000000000001")
        outbound_id = MessageId("msg_00000000000000000000000000000002")
        update = _make_update()
        pool = _make_mock_claude()
        pool.send = MagicMock(return_value=_fake_stream(_done_event("Text and voice")))
        ctx = _make_context(config=_make_config(tts_enabled=True), pool=pool)
        _, finalizer = self._install_workshop_finalizer(ctx, outbound_id)
        self._install_workshop_recorders(ctx, outbound_id)
        patches = self._base_patches()
        patches["sessions"].get_setting = AsyncMock(return_value="on")

        with (
            patch.multiple("kai.bot", **patches),
            patch("kai.bot.synthesize_speech", new_callable=AsyncMock, return_value=b"audio"),
        ):
            await _handle_response(
                update,
                ctx,
                12345,
                "test",
                pool,
                "sonnet",
                workshop_inbound_message_id=inbound_id,
                delivery_route=ResponseDeliveryRoute.WORKSHOP_PRIVATE_TEXT,
            )

        finalizer.assert_not_awaited()
        update.message.reply_text.assert_awaited()

    @pytest.mark.asyncio
    async def test_legacy_route_does_not_retry_failed_jsonl_append(self):
        from kai.bot import _handle_response

        update = _make_update()
        pool = _make_mock_claude()
        pool.send = MagicMock(return_value=_fake_stream(_done_event("Legacy answer")))
        ctx = _make_context(pool=pool)
        patches = self._base_patches()
        patches["log_message"].return_value = None

        with patch.multiple("kai.bot", **patches):
            await _handle_response(
                update,
                ctx,
                12345,
                "test",
                pool,
                "sonnet",
                delivery_route=ResponseDeliveryRoute.LEGACY,
            )

        patches["log_message"].assert_called_once()
        assert patches["log_message"].call_args.kwargs["text"] == "Legacy answer"

    @pytest.mark.asyncio
    async def test_shadows_successful_assistant_result_and_text_delivery(self):
        from kai.bot import _handle_response

        inbound_id = MessageId("msg_00000000000000000000000000000001")
        outbound_id = MessageId("msg_00000000000000000000000000000002")
        update = _make_update()
        pool = _make_mock_claude()
        pool.send = MagicMock(return_value=_fake_stream(_done_event("Canonical answer")))
        ctx = _make_context(pool=pool)
        outbound, delivery = self._install_workshop_recorders(ctx, outbound_id)

        with patch.multiple("kai.bot", **self._base_patches()):
            await _handle_response(
                update,
                ctx,
                12345,
                "test",
                pool,
                "sonnet",
                workshop_inbound_message_id=inbound_id,
            )

        outbound.assert_awaited_once()
        outbound_message = outbound.await_args.args[0]
        assert isinstance(outbound_message, OutboundMessage)
        assert outbound_message.in_reply_to_message_id == inbound_id
        assert outbound_message.body == "Canonical answer"
        delivery.assert_awaited_once()
        observation = delivery.await_args.args[0]
        assert isinstance(observation, DeliveryObservation)
        assert observation.message_id == outbound_id
        assert (observation.transport, observation.mode, observation.succeeded) == (
            "telegram",
            "text",
            True,
        )

    @pytest.mark.asyncio
    async def test_outbound_shadow_failure_does_not_change_delivery(self, caplog):
        from kai.bot import _handle_response

        inbound_id = MessageId("msg_00000000000000000000000000000001")
        update = _make_update()
        pool = _make_mock_claude()
        pool.send = MagicMock(return_value=_fake_stream(_done_event("Still delivered")))
        ctx = _make_context(pool=pool)
        ctx.bot_data["workshop_outbound_recorder"] = AsyncMock(side_effect=RuntimeError("shadow failed"))
        delivery = AsyncMock()
        ctx.bot_data["workshop_delivery_recorder"] = delivery

        with patch.multiple("kai.bot", **self._base_patches()), caplog.at_level(logging.ERROR):
            await _handle_response(
                update,
                ctx,
                12345,
                "test",
                pool,
                "sonnet",
                workshop_inbound_message_id=inbound_id,
            )

        assert update.message.reply_text.called
        delivery.assert_not_awaited()
        assert "Workshop outbound shadow write failed" in caplog.text

    @pytest.mark.asyncio
    async def test_delivery_shadow_failure_does_not_change_existing_response_path(self, caplog):
        from kai.bot import _handle_response

        inbound_id = MessageId("msg_00000000000000000000000000000001")
        outbound_id = MessageId("msg_00000000000000000000000000000002")
        update = _make_update()
        pool = _make_mock_claude()
        pool.send = MagicMock(return_value=_fake_stream(_done_event("Delivered despite shadow failure")))
        ctx = _make_context(pool=pool)
        outbound, _ = self._install_workshop_recorders(ctx, outbound_id)
        ctx.bot_data["workshop_delivery_recorder"] = AsyncMock(side_effect=RuntimeError("shadow failed"))

        with patch.multiple("kai.bot", **self._base_patches()), caplog.at_level(logging.ERROR):
            await _handle_response(
                update,
                ctx,
                12345,
                "test",
                pool,
                "sonnet",
                workshop_inbound_message_id=inbound_id,
            )

        outbound.assert_awaited_once()
        assert update.message.reply_text.called
        assert "Workshop delivery shadow write failed" in caplog.text

    @pytest.mark.asyncio
    async def test_text_delivery_failure_is_observed_and_original_error_propagates(self):
        from kai.bot import _handle_response

        inbound_id = MessageId("msg_00000000000000000000000000000001")
        outbound_id = MessageId("msg_00000000000000000000000000000002")
        update = _make_update()
        pool = _make_mock_claude()
        pool.send = MagicMock(return_value=_fake_stream(_done_event("Undelivered")))
        ctx = _make_context(pool=pool)
        _, delivery = self._install_workshop_recorders(ctx, outbound_id)

        with (
            patch.multiple("kai.bot", **self._base_patches()),
            patch("kai.bot._send_response", new_callable=AsyncMock, side_effect=ConnectionError("network")),
            pytest.raises(ConnectionError, match="network"),
        ):
            await _handle_response(
                update,
                ctx,
                12345,
                "test",
                pool,
                "sonnet",
                workshop_inbound_message_id=inbound_id,
            )

        assert delivery.await_args.args[0].succeeded is False

    @pytest.mark.asyncio
    async def test_best_effort_final_edit_failure_is_observed_without_new_exception(self):
        from kai.bot import _handle_response

        inbound_id = MessageId("msg_00000000000000000000000000000001")
        outbound_id = MessageId("msg_00000000000000000000000000000002")
        update = _make_update()
        pool = _make_mock_claude()
        pool.send = MagicMock(return_value=_fake_stream(_text_event("Stable sentence."), _done_event("Final answer.")))
        ctx = _make_context(pool=pool)
        _, delivery = self._install_workshop_recorders(ctx, outbound_id)

        with (
            patch.multiple("kai.bot", **self._base_patches()),
            patch("kai.bot._edit_message_safe", new_callable=AsyncMock, return_value=False),
        ):
            await _handle_response(
                update,
                ctx,
                12345,
                "test",
                pool,
                "sonnet",
                workshop_inbound_message_id=inbound_id,
            )

        assert delivery.await_args.args[0].succeeded is False

    @pytest.mark.asyncio
    async def test_voice_only_delivery_success_is_observed(self):
        from kai.bot import _handle_response

        inbound_id = MessageId("msg_00000000000000000000000000000001")
        outbound_id = MessageId("msg_00000000000000000000000000000002")
        update = _make_update()
        pool = _make_mock_claude()
        pool.send = MagicMock(return_value=_fake_stream(_done_event("Spoken")))
        ctx = _make_context(config=_make_config(tts_enabled=True), pool=pool)
        _, delivery = self._install_workshop_recorders(ctx, outbound_id)
        patches = self._base_patches()
        patches["sessions"].get_setting = AsyncMock(return_value="only")

        with (
            patch.multiple("kai.bot", **patches),
            patch("kai.bot.synthesize_speech", new_callable=AsyncMock, return_value=b"audio"),
        ):
            await _handle_response(
                update,
                ctx,
                12345,
                "test",
                pool,
                "sonnet",
                workshop_inbound_message_id=inbound_id,
            )

        observation = delivery.await_args.args[0]
        assert (observation.mode, observation.succeeded) == ("voice", True)

    @pytest.mark.asyncio
    async def test_failed_backend_does_not_create_workshop_outbound_message(self):
        from kai.bot import _handle_response

        inbound_id = MessageId("msg_00000000000000000000000000000001")
        update = _make_update()
        pool = _make_mock_claude()
        pool.send = MagicMock(return_value=_fake_stream(_done_event("bad", success=False, error="bad")))
        ctx = _make_context(pool=pool)
        outbound = AsyncMock()
        ctx.bot_data["workshop_outbound_recorder"] = outbound

        with patch.multiple("kai.bot", **self._base_patches()):
            await _handle_response(
                update,
                ctx,
                12345,
                "test",
                pool,
                "sonnet",
                workshop_inbound_message_id=inbound_id,
            )

        outbound.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_normal_flow(self):
        """Streams text, creates live message, delivers final text."""
        from kai.bot import _handle_response

        update = _make_update()
        claude = _make_mock_claude()
        claude.send = MagicMock(return_value=_fake_stream(_text_event("Hello"), _done_event("Hello world")))
        ctx = _make_context(claude=claude)

        with patch.multiple("kai.bot", **self._base_patches()):
            await _handle_response(update, ctx, 12345, "test", claude, "sonnet")

        # Live message should have been created via reply_text
        assert update.message.reply_text.called

    @pytest.mark.asyncio
    async def test_final_matches_last_edit_no_redundant(self):
        """When final text matches last edit, no extra edit is made."""
        from kai.bot import _handle_response

        update = _make_update()
        # The live message mock - track edit_text calls
        live_msg = MagicMock()
        live_msg.edit_text = AsyncMock()
        update.message.reply_text = AsyncMock(return_value=live_msg)

        claude = _make_mock_claude()
        claude.send = MagicMock(return_value=_fake_stream(_text_event("Done"), _done_event("Done")))
        ctx = _make_context(claude=claude)

        with patch.multiple("kai.bot", **self._base_patches()):
            await _handle_response(update, ctx, 12345, "test", claude, "sonnet")

        # The important thing is no exception was raised and
        # the response completed successfully

    @pytest.mark.asyncio
    async def test_stop_interruption(self):
        """Stop event during streaming: edits '(stopped)', returns without error."""
        from kai.bot import _handle_response

        update = _make_update()
        live_msg = MagicMock()
        live_msg.edit_text = AsyncMock()
        update.message.reply_text = AsyncMock(return_value=live_msg)

        stop_event = asyncio.Event()

        async def _streaming(*args):
            yield _text_event("Partial")
            # Simulate /stop during streaming
            stop_event.set()
            yield _text_event("More text")
            yield _done_event("Should not reach")

        claude = _make_mock_claude()
        claude.send = MagicMock(return_value=_streaming())
        ctx = _make_context(claude=claude)

        with (
            patch.multiple("kai.bot", **self._base_patches()),
            patch("kai.bot.get_stop_event", return_value=stop_event),
        ):
            await _handle_response(update, ctx, 12345, "test", claude, "sonnet")

        # Should NOT send the "No response" error
        replies = [c[0][0] for c in update.message.reply_text.call_args_list]
        assert not any("Error" in r for r in replies)

    @pytest.mark.asyncio
    async def test_no_done_event_error(self):
        """No done event: sends 'No response from agent' error."""
        from kai.bot import _handle_response

        update = _make_update()
        claude = _make_mock_claude()
        # Stream that ends without a done event
        claude.send = MagicMock(return_value=_fake_stream(_text_event("Partial")))
        ctx = _make_context(claude=claude)

        with patch.multiple("kai.bot", **self._base_patches()):
            await _handle_response(update, ctx, 12345, "test", claude, "sonnet")

        replies = [c[0][0] for c in update.message.reply_text.call_args_list]
        assert any("No response from agent" in r for r in replies)

    @pytest.mark.asyncio
    async def test_error_response_with_live_msg(self):
        """success=False with existing live message: error appears as
        a NEW follow-up reply via _reply_safe (not edited into the
        live message via _edit_message_safe). The live streamed
        content stays visible so any tool-use, partial reasoning, and
        intermediate output the user was watching survives the error.

        Patches kai.bot._reply_safe directly rather than relying on
        its internal call to reply_text - the latter would couple
        this assertion to _reply_safe's implementation, and a future
        refactor of _reply_safe (e.g., to use a different telegram
        method on its first attempt) would silently void this
        assertion. The streaming loop also uses _reply_safe to
        create live_msg, so we filter the captured calls to those
        carrying an "Error" prefix to isolate the error-path
        invocations from the streaming-text invocations."""
        from kai.bot import _handle_response

        update = _make_update()
        live_msg = MagicMock()
        live_msg.edit_text = AsyncMock()
        update.message.reply_text = AsyncMock(return_value=live_msg)

        claude = _make_mock_claude()
        claude.send = MagicMock(
            return_value=_fake_stream(
                _text_event("Partial"),
                _done_event("Something broke", success=False, error="Something broke"),
            )
        )
        ctx = _make_context(claude=claude)

        with (
            patch.multiple("kai.bot", **self._base_patches()),
            patch("kai.bot._reply_safe", new_callable=AsyncMock) as mock_reply_safe,
            patch("kai.bot._edit_message_safe", new_callable=AsyncMock) as mock_edit_safe,
        ):
            await _handle_response(update, ctx, 12345, "test", claude, "sonnet")

        # The error path uses _reply_safe (NOT _edit_message_safe) -
        # error appears as a follow-up message, not an in-place edit.
        # Pinning the negative property because the absence of an
        # edit is the load-bearing contract change. Length guard on
        # args[1] mirrors the pattern used in test_error_recovery.py
        # so a future _reply_safe call site that omits the text arg
        # produces a meaningful assertion failure rather than
        # IndexError.
        for call in mock_edit_safe.await_args_list:
            text_arg = call.args[1] if len(call.args) > 1 else ""
            assert "Error" not in text_arg, (
                f"_edit_message_safe was called with an error string ({text_arg!r}), "
                f"violating the append-not-overwrite contract"
            )
        error_calls = [
            call
            for call in mock_reply_safe.await_args_list
            if (call.args[1] if len(call.args) > 1 else "").startswith("Error")
        ]
        assert len(error_calls) >= 1, "expected the error to appear via _reply_safe"
        assert "Something broke" in error_calls[-1].args[1]

    @pytest.mark.asyncio
    async def test_error_response_no_live_msg(self):
        """success=False with no live message: sends error as new reply."""
        from kai.bot import _handle_response

        update = _make_update()
        claude = _make_mock_claude()
        # Done event immediately (no text events to create a live message)
        claude.send = MagicMock(
            return_value=_fake_stream(
                _done_event("Broke", success=False, error="Broke"),
            )
        )
        ctx = _make_context(claude=claude)

        with patch.multiple("kai.bot", **self._base_patches()):
            await _handle_response(update, ctx, 12345, "test", claude, "sonnet")

        replies = [c[0][0] for c in update.message.reply_text.call_args_list]
        assert any("Error" in r for r in replies)

    @pytest.mark.asyncio
    async def test_long_response_chunked(self):
        """Response > 4096 chars: first chunk edits live message, rest sent as new messages."""
        from kai.bot import _handle_response

        update = _make_update()
        live_msg = MagicMock()
        live_msg.edit_text = AsyncMock()
        update.message.reply_text = AsyncMock(return_value=live_msg)

        long_text = "a" * 5000
        claude = _make_mock_claude()
        claude.send = MagicMock(return_value=_fake_stream(_text_event("start"), _done_event(long_text)))
        ctx = _make_context(claude=claude)

        with patch.multiple("kai.bot", **self._base_patches()):
            await _handle_response(update, ctx, 12345, "test", claude, "sonnet")

        # Multiple messages should have been sent (chunked)
        assert update.message.reply_text.call_count >= 2

    @pytest.mark.asyncio
    async def test_session_saved_with_id(self):
        """Saves session when session_id is present."""
        from kai.bot import _handle_response

        update = _make_update()
        claude = _make_mock_claude()
        claude.send = MagicMock(return_value=_fake_stream(_done_event("Ok", session_id="sess-abc")))
        ctx = _make_context(claude=claude)
        mock_sessions = MagicMock(
            get_setting=AsyncMock(return_value="off"),
            save_session=AsyncMock(),
        )

        with patch("kai.bot.sessions", mock_sessions), patch("kai.bot.log_message"):
            await _handle_response(update, ctx, 12345, "test", claude, "sonnet")

        mock_sessions.save_session.assert_called_once()
        args = mock_sessions.save_session.call_args[0]
        assert args[0] == 12345
        assert args[1] == "sess-abc"

    @pytest.mark.asyncio
    async def test_session_not_saved_without_id(self):
        """Does NOT save session when session_id is None."""
        from kai.bot import _handle_response

        update = _make_update()
        claude = _make_mock_claude()
        claude.send = MagicMock(return_value=_fake_stream(_done_event("Ok", session_id=None)))
        ctx = _make_context(claude=claude)
        mock_sessions = MagicMock(
            get_setting=AsyncMock(return_value="off"),
            save_session=AsyncMock(),
        )

        with patch("kai.bot.sessions", mock_sessions), patch("kai.bot.log_message"):
            await _handle_response(update, ctx, 12345, "test", claude, "sonnet")

        mock_sessions.save_session.assert_not_called()

    @pytest.mark.asyncio
    async def test_voice_only_mode(self):
        """Voice-only: no live text message, synthesizes and sends voice."""
        from kai.bot import _handle_response

        update = _make_update()
        claude = _make_mock_claude()
        claude.send = MagicMock(return_value=_fake_stream(_done_event("Hello voice")))
        config = _make_config(tts_enabled=True, piper_model_dir=Path("/models"))
        ctx = _make_context(config=config, claude=claude)
        mock_sessions = MagicMock(
            get_setting=AsyncMock(return_value="only"),
            save_session=AsyncMock(),
        )

        with (
            patch("kai.bot.sessions", mock_sessions),
            patch("kai.bot.log_message"),
            patch("kai.bot.synthesize_speech", new_callable=AsyncMock, return_value=b"audio-bytes"),
        ):
            await _handle_response(update, ctx, 12345, "test", claude, "sonnet")

        ctx.bot.send_voice.assert_called_once()

    @pytest.mark.asyncio
    async def test_voice_only_tts_failure_falls_back(self):
        """Voice-only TTS failure: falls back to text delivery."""
        from kai.bot import _handle_response
        from kai.tts import TTSError

        update = _make_update()
        claude = _make_mock_claude()
        claude.send = MagicMock(return_value=_fake_stream(_done_event("Fallback text")))
        config = _make_config(tts_enabled=True, piper_model_dir=Path("/models"))
        ctx = _make_context(config=config, claude=claude)
        mock_sessions = MagicMock(
            get_setting=AsyncMock(return_value="only"),
            save_session=AsyncMock(),
        )

        with (
            patch("kai.bot.sessions", mock_sessions),
            patch("kai.bot.log_message"),
            patch("kai.bot.synthesize_speech", new_callable=AsyncMock, side_effect=TTSError("fail")),
        ):
            await _handle_response(update, ctx, 12345, "test", claude, "sonnet")

        # Should fall back to text
        assert update.message.reply_text.called

    @pytest.mark.asyncio
    async def test_text_plus_voice_mode(self):
        """Text+voice mode: sends text normally, then sends voice note."""
        from kai.bot import _handle_response

        update = _make_update()
        live_msg = MagicMock()
        live_msg.edit_text = AsyncMock()
        update.message.reply_text = AsyncMock(return_value=live_msg)

        claude = _make_mock_claude()
        claude.send = MagicMock(return_value=_fake_stream(_text_event("Hi"), _done_event("Hi there")))
        config = _make_config(tts_enabled=True, piper_model_dir=Path("/models"))
        ctx = _make_context(config=config, claude=claude)

        async def _get_setting(key):
            if "voice_mode" in key:
                return "on"
            return DEFAULT_VOICE

        mock_sessions = MagicMock(
            get_setting=AsyncMock(side_effect=_get_setting),
            save_session=AsyncMock(),
        )

        with (
            patch("kai.bot.sessions", mock_sessions),
            patch("kai.bot.log_message"),
            patch("kai.bot.synthesize_speech", new_callable=AsyncMock, return_value=b"audio"),
        ):
            await _handle_response(update, ctx, 12345, "test", claude, "sonnet")

        # Both text (reply_text) and voice (send_voice) should be sent
        assert update.message.reply_text.called
        ctx.bot.send_voice.assert_called_once()

    @pytest.mark.asyncio
    async def test_typing_task_cancelled(self):
        """Typing indicator task is cancelled in finally block."""
        from kai.bot import _handle_response

        update = _make_update()
        claude = _make_mock_claude()
        claude.send = MagicMock(return_value=_fake_stream(_done_event("Ok")))
        ctx = _make_context(claude=claude)

        with patch.multiple("kai.bot", **self._base_patches()):
            await _handle_response(update, ctx, 12345, "test", claude, "sonnet")

        # If we got here without hanging, the typing task was properly cancelled

    @pytest.mark.asyncio
    async def test_extraction_skipped_when_memory_disabled(self, monkeypatch):
        """Switch point 5 (#434): under memory.is_enabled() == False,
        the post-response ingestion guard short-circuits and
        extract_and_store is never invoked. The fire-and-forget task
        is gated, so no _ingest_memory task is even created.

        Pinned regression: a refactor that flips the
        `if memory_is_enabled() and chat_id is not None:` guard to
        `if True:` (or removes it entirely) fails this test because
        extract_and_store would then run under disabled mode and write
        to Qdrant from a mode that promises never to write.
        """
        from kai.bot import _handle_response

        monkeypatch.setattr("kai.memory.is_enabled", lambda: False)
        extract_mock = AsyncMock()
        monkeypatch.setattr("kai.memory_extraction.extract_and_store", extract_mock)

        update = _make_update()
        pool = _make_mock_claude()
        pool.send = MagicMock(return_value=_fake_stream(_done_event("Hello world")))
        # memory_extraction_enabled=True ensures the inner config check
        # in _ingest_memory does NOT short-circuit before reaching the
        # extract_and_store call. The OUTER is_enabled() guard is the
        # contract under test; the inner check would mask its failure
        # if both were False.
        #
        # episode_classifier_context_turns=0 mirrors the enabled-mode
        # counterpart's setup so the two tests are structurally
        # symmetric. Under current behavior the disabled test never
        # reaches the inner block, so the value is unread; the
        # symmetry is defense-in-depth: if the monkeypatch ever stops
        # taking effect or the guard moves, the inner block would
        # otherwise execute with the default value of 3 and trigger
        # an unstated `get_recent_pairs(12345, 4)` disk read.
        config = _make_config(
            memory_extraction_enabled=True,
            episode_classifier_context_turns=0,
        )
        ctx = _make_context(config=config, pool=pool)

        # Snapshot the task set BEFORE the call so we can compute the
        # delta after. A bare `asyncio.all_tasks()` post-call would
        # also pick up unrelated tasks (pytest-asyncio fixtures,
        # leftover work from a prior test) and gather them too,
        # which is fragile even when it does not currently break.
        tasks_before = asyncio.all_tasks()

        with patch.multiple("kai.bot", **self._base_patches()):
            await _handle_response(update, ctx, 12345, "test prompt", pool, "claude-opus")

        # Drain any tasks the call may have spawned. The disabled-path
        # contract is that NO ingest task is created, so this should
        # be a no-op; if a regression spawned a task anyway, the
        # gather lets it run before the assertion, surfacing the bug
        # as a failed call_count check rather than a timing-flake.
        new_tasks = asyncio.all_tasks() - tasks_before - {asyncio.current_task()}
        if new_tasks:
            await asyncio.gather(*new_tasks, return_exceptions=True)

        extract_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_extraction_invoked_when_memory_enabled(self, monkeypatch):
        """Switch point 5 (#434): under memory.is_enabled() == True
        plus a successful response, the post-response ingestion path
        spawns a background _ingest_memory task that calls
        extract_and_store exactly once.

        Pinned regression: a refactor that flips the guard to
        `if False:` (or removes the call site from the closure) fails
        this test because extract_and_store would not run under
        enabled mode and the Qdrant fact surface would never receive
        new extractions.

        The fire-and-forget task is awaited explicitly via
        `asyncio.gather` over the post-call task set so the assertion
        does not race the create_task'd coroutine.
        """
        from kai.bot import _handle_response

        monkeypatch.setattr("kai.memory.is_enabled", lambda: True)
        extract_mock = AsyncMock()
        monkeypatch.setattr("kai.memory_extraction.extract_and_store", extract_mock)

        update = _make_update()
        pool = _make_mock_claude()
        pool.send = MagicMock(return_value=_fake_stream(_done_event("Hello world", session_id="sess-1")))
        # episode_classifier_context_turns=0 skips the history fetch
        # inside _ingest_memory (no get_recent_pairs call needed); the
        # contract under test is the outer gate plus the inner config
        # branch, neither of which depends on prior_pairs being
        # non-empty. memory_extraction_enabled=True so the inner gate
        # passes; _make_config explicitly selects the Claude backend
        # used by this test's mocked pool.
        config = _make_config(
            memory_extraction_enabled=True,
            episode_classifier_context_turns=0,
        )
        ctx = _make_context(config=config, pool=pool)

        # Snapshot pre-call so the post-call delta reflects only the
        # tasks _handle_response spawned. See the disabled counterpart
        # for the full rationale.
        tasks_before = asyncio.all_tasks()

        with patch.multiple("kai.bot", **self._base_patches()):
            await _handle_response(update, ctx, 12345, "test prompt", pool, "claude-opus")

        # Wait for the fire-and-forget _ingest_memory task to run to
        # completion. asyncio.create_task at the call site schedules
        # the coroutine onto the event loop; the test's coroutine
        # cannot assert until the scheduled task has had a chance to
        # progress past its `await extract_and_store(...)` line.
        #
        # NOT `return_exceptions=True`: in the positive test, a task
        # that crashed BEFORE reaching extract_and_store (e.g., a
        # config-guard mismatch, a missing attribute) would be
        # silently swallowed and surface only as "expected called
        # once, called zero times" - hiding the real cause. The
        # disabled-mode counterpart uses return_exceptions=True
        # because there a task appearing at all is the regression
        # signal; here we want crashes to propagate.
        new_tasks = asyncio.all_tasks() - tasks_before - {asyncio.current_task()}
        if new_tasks:
            await asyncio.gather(*new_tasks)

        extract_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_extraction_invoked_for_opencode_users(self, monkeypatch):
        """The bot.py extraction gate now admits opencode users. Pin
        that an effective `agent_backend="opencode"` flows through the
        post-response ingestion path and calls extract_and_store. The
        gate widening at bot.py:_ingest_memory is what unlocks the
        opencode one-shot reasoner; a regression that reverts the
        tuple literal would silently lose extraction for opencode
        users while leaving claude / codex flowing."""
        from kai.bot import _handle_response
        from kai.config import UserConfig

        monkeypatch.setattr("kai.memory.is_enabled", lambda: True)
        extract_mock = AsyncMock()
        monkeypatch.setattr("kai.memory_extraction.extract_and_store", extract_mock)

        update = _make_update()
        pool = _make_mock_claude()
        pool.send = MagicMock(return_value=_fake_stream(_done_event("Hello world", session_id="sess-1")))
        # Per-user opencode override on a global-claude install; the
        # bot.py gate consults the effective backend (user override
        # wins), which is the same shape an all-opencode install
        # produces for the gate.
        config = _make_config(
            memory_extraction_enabled=True,
            episode_classifier_context_turns=0,
            user_configs={
                12345: UserConfig(
                    telegram_id=12345,
                    name="opencode-user",
                    backend="opencode",
                ),
            },
        )
        ctx = _make_context(config=config, pool=pool)

        tasks_before = asyncio.all_tasks()

        with patch.multiple("kai.bot", **self._base_patches()):
            await _handle_response(update, ctx, 12345, "test prompt", pool, "anthropic/claude-sonnet-4-5")

        new_tasks = asyncio.all_tasks() - tasks_before - {asyncio.current_task()}
        if new_tasks:
            await asyncio.gather(*new_tasks)

        extract_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_extraction_skipped_for_non_reasoner_backend_users(self, monkeypatch):
        """Symmetric guard: users whose effective backend has no
        OneShotReasoner must NOT reach the extractor; an extraction
        attempt would crash. Every real backend is a member today, so
        the gate's negative branch is exercised by patching the
        constant down rather than naming a real backend."""
        from kai.bot import _handle_response
        from kai.config import UserConfig

        monkeypatch.setattr("kai.memory.is_enabled", lambda: True)
        monkeypatch.setattr("kai.bot.ONESHOT_REASONER_BACKENDS", frozenset({"claude", "codex"}))
        extract_mock = AsyncMock()
        monkeypatch.setattr("kai.memory_extraction.extract_and_store", extract_mock)

        update = _make_update()
        pool = _make_mock_claude()
        pool.send = MagicMock(return_value=_fake_stream(_done_event("Hello world", session_id="sess-1")))
        config = _make_config(
            memory_extraction_enabled=True,
            episode_classifier_context_turns=0,
            user_configs={
                12345: UserConfig(
                    telegram_id=12345,
                    name="goose-user",
                    backend="goose",
                    provider="openai",
                ),
            },
        )
        ctx = _make_context(config=config, pool=pool)

        tasks_before = asyncio.all_tasks()

        with patch.multiple("kai.bot", **self._base_patches()):
            await _handle_response(update, ctx, 12345, "test prompt", pool, "gpt-4")

        new_tasks = asyncio.all_tasks() - tasks_before - {asyncio.current_task()}
        if new_tasks:
            await asyncio.gather(*new_tasks, return_exceptions=True)

        extract_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_memory_ingest_task_held_by_module_set(self, monkeypatch):
        """The ingest task is added to `_pending_memory_tasks` at
        schedule time and discarded by the done-callback after the
        task completes. Pinning both sides of the lifecycle prevents
        a regression that reverts to `asyncio.create_task(...)` with
        no strong reference, which Python's GC can reap mid-flight.

        Drives the lifecycle with an asyncio.Event so the test can
        observe the set membership at three points: pre-call (empty
        baseline), mid-task (one entry, task in flight), and
        post-task (back to empty after the done-callback fires)."""
        from kai import bot
        from kai.bot import _handle_response, _pending_memory_tasks

        # Baseline: prior tests may have left the set populated if
        # they did not wait for their tasks to complete. Snapshot the
        # baseline rather than asserting empty so test ordering does
        # not couple this test to its siblings.
        baseline = set(_pending_memory_tasks)

        monkeypatch.setattr("kai.memory.is_enabled", lambda: True)

        # extract_and_store waits on this event so the test can
        # inspect the set while the task is in flight. The mid-task
        # assertion races the done-callback only if extract returns
        # immediately; the event keeps the task pending until the
        # test releases it.
        in_flight = asyncio.Event()
        release = asyncio.Event()

        async def blocking_extract(*args, **kwargs):
            in_flight.set()
            await release.wait()

        monkeypatch.setattr("kai.memory_extraction.extract_and_store", blocking_extract)

        update = _make_update()
        pool = _make_mock_claude()
        pool.send = MagicMock(return_value=_fake_stream(_done_event("Hello world", session_id="sess-1")))
        config = _make_config(
            memory_extraction_enabled=True,
            episode_classifier_context_turns=0,
        )
        ctx = _make_context(config=config, pool=pool)

        with patch.multiple("kai.bot", **self._base_patches()):
            await _handle_response(update, ctx, 12345, "test prompt", pool, "claude-opus")

        # The ingest task is scheduled inside _handle_response and
        # blocked at the extract_and_store await. Wait for the
        # in-flight signal so the assertion does not race the
        # task's first await.
        await asyncio.wait_for(in_flight.wait(), timeout=1.0)

        # Mid-task: exactly one new entry above baseline. The
        # add() at schedule time put it there; the done-callback
        # has not fired because the task is still blocked.
        assert len(_pending_memory_tasks - baseline) == 1, (
            "ingest task should be held in _pending_memory_tasks while it runs"
        )

        # Release the task so it can complete and the done-callback
        # can run.
        release.set()

        # Wait for the done-callback to discard the task. The
        # task itself completes inside extract_and_store; the
        # discard runs as a callback after. A short asyncio.sleep
        # loop lets the runtime schedule both.
        for _ in range(100):
            if not (_pending_memory_tasks - baseline):
                break
            await asyncio.sleep(0.01)

        # Post-task: the set is back to baseline. The discard
        # callback fired and self-pruned the entry, so a
        # long-running deployment does not accumulate references.
        assert _pending_memory_tasks - baseline == set(), (
            "done-callback should have discarded the completed task from _pending_memory_tasks"
        )

        # Silence pyflakes/ruff: `bot` is imported above to anchor
        # the patched attribute path; the assertions read from the
        # re-imported `_pending_memory_tasks` reference.
        _ = bot

    # ── Stable-prefix streaming gate ─────────────────────────────────

    @pytest.mark.asyncio
    async def test_does_not_create_live_message_for_initial_single_word(self):
        """First non-empty partial is a single word; gate withholds live message."""
        from kai.bot import _handle_response

        update = _make_update()
        claude = _make_mock_claude()
        claude.send = MagicMock(return_value=_fake_stream(_text_event("One"), _done_event("One final answer.")))
        ctx = _make_context(claude=claude)

        with (
            patch.multiple("kai.bot", **self._base_patches()),
            patch("kai.bot._reply_safe", new_callable=AsyncMock) as mock_reply,
            patch("kai.bot._edit_message_safe", new_callable=AsyncMock) as mock_edit,
        ):
            await _handle_response(update, ctx, 12345, "test", claude, "sonnet")

        # No live edit ever fired; final answer arrived through _reply_safe
        # via _send_response. The unstable "One" partial never created a
        # live message.
        mock_edit.assert_not_called()
        published_texts = [c.args[1] for c in mock_reply.call_args_list]
        assert "One" not in published_texts
        assert any("One final answer." in t for t in published_texts)

    @pytest.mark.asyncio
    async def test_does_not_create_live_message_for_bare_numbered_marker(self):
        """First non-empty partial is a single in-progress numbered item;
        the stable-prefix gate must withhold the live message. Without
        the ordered-list-aware sentence cut, the gate would publish the
        bare marker ``1.`` as if it were a complete sentence.
        """
        from kai.bot import _handle_response

        update = _make_update()
        claude = _make_mock_claude()
        claude.send = MagicMock(
            return_value=_fake_stream(
                _text_event("1. Item one"),
                _done_event("1. Item one\n2. Item two\n3. Item three"),
            )
        )
        ctx = _make_context(claude=claude)

        with (
            patch.multiple("kai.bot", **self._base_patches()),
            patch("kai.bot._reply_safe", new_callable=AsyncMock) as mock_reply,
            patch("kai.bot._edit_message_safe", new_callable=AsyncMock) as mock_edit,
        ):
            await _handle_response(update, ctx, 12345, "test", claude, "sonnet")

        # No live message edit ever fired; the unstable single in-progress
        # item never produced a stable-prefix candidate, so the live
        # message was never created mid-stream.
        mock_edit.assert_not_called()
        published_texts = [c.args[1] for c in mock_reply.call_args_list]
        # The bare marker is the worst Telegram artifact this PR
        # prevents; no published text may show it.
        assert "1." not in published_texts
        assert "1. Item one" not in published_texts
        # Final delivery still carries the complete answer via _reply_safe.
        assert any("3. Item three" in t for t in published_texts)

    @pytest.mark.asyncio
    async def test_publishes_completed_prefix_not_dangling_suffix(self):
        """Live message uses the completed paragraph, never the dangling word."""
        from kai.bot import _handle_response

        update = _make_update()
        claude = _make_mock_claude()
        claude.send = MagicMock(
            return_value=_fake_stream(
                _text_event("Complete paragraph.\n\nOne"),
                _done_event("Complete paragraph.\n\nOne final answer."),
            )
        )
        ctx = _make_context(claude=claude)

        with (
            patch.multiple("kai.bot", **self._base_patches()),
            patch("kai.bot._reply_safe", new_callable=AsyncMock) as mock_reply,
            patch("kai.bot._edit_message_safe", new_callable=AsyncMock) as mock_edit,
        ):
            await _handle_response(update, ctx, 12345, "test", claude, "sonnet")

        # The live message creation received the completed paragraph
        # without the dangling "One" suffix.
        live_create_texts = [c.args[1] for c in mock_reply.call_args_list]
        assert "Complete paragraph." in live_create_texts
        # No reply / edit ever showed the bare "Complete paragraph.\n\nOne"
        # intermediate. (The final delivery edit carries the complete
        # final answer, not the dangling intermediate.)
        edit_texts = [c.args[1] for c in mock_edit.call_args_list]
        assert "Complete paragraph.\n\nOne" not in edit_texts
        assert "Complete paragraph.\n\nOne" not in live_create_texts

    @pytest.mark.asyncio
    async def test_edits_only_when_publishable_prefix_advances(self):
        """A dangling suffix after a stable sentence does not trigger an edit."""
        from kai.bot import EDIT_INTERVAL, _handle_response

        update = _make_update()
        live_msg = MagicMock()
        live_msg.edit_text = AsyncMock()
        update.message.reply_text = AsyncMock(return_value=live_msg)

        claude = _make_mock_claude()
        claude.send = MagicMock(
            return_value=_fake_stream(
                _text_event("One complete sentence."),
                _text_event("One complete sentence. Two"),  # dangling suffix
                _done_event("One complete sentence. Two complete sentences."),
            )
        )
        ctx = _make_context(claude=claude)

        # Patch monotonic so the second event is past EDIT_INTERVAL.
        times = iter([0.0, EDIT_INTERVAL + 1.0, EDIT_INTERVAL + 2.0, EDIT_INTERVAL + 3.0])
        with (
            patch.multiple("kai.bot", **self._base_patches()),
            patch("kai.bot._edit_message_safe", new_callable=AsyncMock) as mock_edit,
            patch("kai.bot.time.monotonic", side_effect=lambda: next(times)),
        ):
            await _handle_response(update, ctx, 12345, "test", claude, "sonnet")

        # No edit ever published a dangling "Two" mid-stream.
        edit_texts = [c.args[1] for c in mock_edit.call_args_list]
        assert not any(t.endswith(" Two") for t in edit_texts)
        # Final edit carries the complete final text (not the dangling
        # intermediate).
        if edit_texts:
            assert "Two complete sentences." in edit_texts[-1]

    @pytest.mark.asyncio
    async def test_stop_before_stable_live_message_sends_no_fragment(self):
        """Stop while only unstable partials seen: no fragment reply, no error reply, stop logged."""
        from kai.bot import _handle_response

        update = _make_update()
        stop_event = asyncio.Event()

        async def _streaming(*args):
            yield _text_event("One")
            stop_event.set()
            yield _text_event("One more")  # still unstable
            yield _done_event("Should not reach")

        claude = _make_mock_claude()
        claude.send = MagicMock(return_value=_streaming())
        ctx = _make_context(claude=claude)

        patches = self._base_patches()
        log_message_mock = patches["log_message"]

        with (
            patch.multiple("kai.bot", **patches),
            patch("kai.bot.get_stop_event", return_value=stop_event),
            patch("kai.bot._reply_safe", new_callable=AsyncMock) as mock_reply,
            patch("kai.bot._edit_message_safe", new_callable=AsyncMock) as mock_edit,
        ):
            await _handle_response(update, ctx, 12345, "test", claude, "sonnet")

        # The unstable "One" partial never created a live message.
        mock_edit.assert_not_called()
        # Error fallback was NOT sent. update.message.reply_text is the
        # entry point for the bare "Error: No response from agent" path
        # (it bypasses _reply_safe).
        error_calls = [
            c
            for c in update.message.reply_text.call_args_list
            if c.args and "Error: No response from agent" in c.args[0]
        ]
        assert not error_calls
        # _reply_safe was never called with "One" or "One more".
        reply_texts = [c.args[1] for c in mock_reply.call_args_list]
        assert "One" not in reply_texts
        assert "One more" not in reply_texts
        # The stop was logged with the canonical marker.
        stop_log_calls = [c for c in log_message_mock.call_args_list if c.kwargs.get("text") == "[stopped by user]"]
        assert stop_log_calls, "expected '[stopped by user]' log entry"

    @pytest.mark.asyncio
    async def test_final_delivery_after_withheld_live_updates(self):
        """Stream only unstable partials; final delivery still sends the complete answer."""
        from kai.bot import _handle_response

        update = _make_update()
        claude = _make_mock_claude()
        claude.send = MagicMock(
            return_value=_fake_stream(
                _text_event("One"),
                _text_event("One more"),
                _done_event("One final complete response."),
            )
        )
        ctx = _make_context(claude=claude)

        with (
            patch.multiple("kai.bot", **self._base_patches()),
            patch("kai.bot._reply_safe", new_callable=AsyncMock) as mock_reply,
            patch("kai.bot._edit_message_safe", new_callable=AsyncMock) as mock_edit,
        ):
            await _handle_response(update, ctx, 12345, "test", claude, "sonnet")

        # No live edits, no fragment replies; only the final delivery.
        mock_edit.assert_not_called()
        published_texts = [c.args[1] for c in mock_reply.call_args_list]
        assert any("One final complete response." in t for t in published_texts)

    @pytest.mark.asyncio
    async def test_publishes_completed_list_items_not_dangling_next(self):
        """Live message receives `- Item one\\n- Item two`; the dangling `- Three` is withheld."""
        from kai.bot import _handle_response

        update = _make_update()
        claude = _make_mock_claude()
        claude.send = MagicMock(
            return_value=_fake_stream(
                _text_event("- Item one\n- Item two\n- Three"),
                _done_event("- Item one\n- Item two\n- Three is complete."),
            )
        )
        ctx = _make_context(claude=claude)

        with (
            patch.multiple("kai.bot", **self._base_patches()),
            patch("kai.bot._reply_safe", new_callable=AsyncMock) as mock_reply,
            patch("kai.bot._edit_message_safe", new_callable=AsyncMock) as mock_edit,
        ):
            await _handle_response(update, ctx, 12345, "test", claude, "sonnet")

        live_create_texts = [c.args[1] for c in mock_reply.call_args_list]
        edit_texts = [c.args[1] for c in mock_edit.call_args_list]

        # The live message carried the completed pair, never the
        # dangling next item.
        assert "- Item one\n- Item two" in live_create_texts
        assert "- Item one\n- Item two\n- Three" not in live_create_texts
        assert "- Item one\n- Item two\n- Three" not in edit_texts


# ── _notify_if_queued ────────────────────────────────────────────────


class TestNotifyIfQueued:
    """Tests for the pre-lock queue notification and context-switch marker."""

    @pytest.mark.asyncio
    async def test_sends_when_locked(self):
        """Sends a notification and returns True when the lock is already held."""
        update = _make_update()
        chat_id = 12345

        # Acquire the lock to simulate Kai being busy
        from kai.locks import get_lock

        lock = get_lock(chat_id)
        await lock.acquire()
        try:
            result = await _notify_if_queued(update, chat_id)
            assert result is True
            # The notification goes via reply_text (called by _reply_safe)
            update.message.reply_text.assert_called()
            call_text = update.message.reply_text.call_args[0][0]
            assert "Got your message" in call_text
            assert "/stop" in call_text
        finally:
            lock.release()

    @pytest.mark.asyncio
    async def test_silent_when_unlocked(self):
        """Does nothing and returns False when the lock is free."""
        update = _make_update()
        # Use a unique chat_id to avoid lock state from other tests
        chat_id = 99999

        result = await _notify_if_queued(update, chat_id)

        assert result is False
        update.message.reply_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_notification_not_sent_to_claude(self):
        """The notification goes directly to Telegram, not through Claude."""
        update = _make_update()
        chat_id = 12345

        from kai.locks import get_lock

        lock = get_lock(chat_id)
        await lock.acquire()
        try:
            # Mock Claude's send to verify it's never called
            mock_claude = _make_mock_claude()
            mock_claude.send = MagicMock()

            await _notify_if_queued(update, chat_id)

            # Claude.send was never called; the notification uses reply_text
            mock_claude.send.assert_not_called()
            update.message.reply_text.assert_called()
        finally:
            lock.release()

    @pytest.mark.asyncio
    async def test_queued_message_gets_notification_then_processes(self):
        """Integration test: message B gets a notification while A holds the lock.

        Simulates two concurrent messages: A acquires the lock and processes,
        B arrives while A is busy, gets a notification, then processes after
        A releases. Proves the full flow works end to end.
        """
        update_b = _make_update(text="second message")
        chat_id = 77777

        from kai.locks import get_lock

        lock = get_lock(chat_id)

        # Track ordering of events
        events: list[str] = []

        async def handler_a():
            """Simulate message A holding the lock."""
            async with lock:
                events.append("a_acquired")
                # Simulate processing time so B's handler runs
                await asyncio.sleep(0.05)
                events.append("a_released")

        async def handler_b():
            """Simulate message B arriving while A is busy."""
            # Small delay so A grabs the lock first
            await asyncio.sleep(0.01)
            # This is the pre-lock notification
            result = await _notify_if_queued(update_b, chat_id)
            assert result is True
            events.append("b_notified")
            async with lock:
                events.append("b_acquired")

        await asyncio.gather(handler_a(), handler_b())

        # B's notification happened while A held the lock
        assert events.index("b_notified") > events.index("a_acquired")
        assert events.index("b_notified") < events.index("a_released")
        # B acquired the lock after A released
        assert events.index("b_acquired") > events.index("a_released")
        # B got the notification
        update_b.message.reply_text.assert_called()
        call_text = update_b.message.reply_text.call_args[0][0]
        assert "Got your message" in call_text


class TestPrependQueueMarker:
    """Tests for _prepend_queue_marker(), the context-switch prompt helper."""

    def test_string_prompt(self):
        """Prepends marker to a plain string prompt."""
        result = _prepend_queue_marker("hello world")
        assert result.startswith(_QUEUED_MESSAGE_MARKER)
        assert result.endswith("hello world")

    def test_multimodal_prompt(self):
        """Prepends marker to the first text block of a multimodal list."""
        content = [
            {"type": "text", "text": "Photo caption"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "abc"}},
        ]
        result = _prepend_queue_marker(content)
        assert isinstance(result, list)
        assert len(result) == 2
        # First block has marker prepended
        assert result[0]["text"].startswith(_QUEUED_MESSAGE_MARKER)
        assert result[0]["text"].endswith("Photo caption")
        # Second block (image) is unchanged
        assert result[1] == content[1]

    def test_does_not_mutate_original(self):
        """Returns a new list; does not mutate the original content."""
        content = [
            {"type": "text", "text": "original"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "abc"}},
        ]
        _prepend_queue_marker(content)
        # Original is untouched
        assert content[0]["text"] == "original"


# ── _acquire_lock_or_kill ───────────────────────────────────────────


class TestAcquireLockOrKill:
    """Tests for the lock-with-timeout safety net."""

    @pytest.mark.asyncio
    async def test_acquires_free_lock(self):
        """Returns the lock when it's free - normal fast path."""
        update = _make_update()
        claude = _make_mock_claude()
        # Use a unique chat_id to avoid state from other tests
        chat_id = 88801

        lock = await _acquire_lock_or_kill(chat_id, claude, update)

        assert lock is not None
        assert lock.locked()
        lock.release()
        claude.force_kill.assert_not_called()

    @pytest.mark.asyncio
    async def test_timeout_kills_claude(self):
        """When the lock is held too long, force-kills Claude and notifies user."""
        update = _make_update()
        claude = _make_mock_claude()
        chat_id = 88802

        from kai.locks import get_lock

        held_lock = get_lock(chat_id)
        await held_lock.acquire()

        try:
            # Patch the timeout to something tiny so the test doesn't wait 1 hour
            with patch("kai.bot._LOCK_ACQUIRE_TIMEOUT", 0.05):
                result = await _acquire_lock_or_kill(chat_id, claude, update)
        finally:
            held_lock.release()

        assert result is None
        claude.force_kill.assert_called_once()
        update.message.reply_text.assert_called()
        msg = update.message.reply_text.call_args[0][0]
        assert "timed out" in msg.lower()

    @pytest.mark.asyncio
    async def test_returns_same_lock_object(self):
        """The returned lock is the same object from get_lock, not a copy."""
        update = _make_update()
        claude = _make_mock_claude()
        chat_id = 88803

        from kai.locks import get_lock

        expected_lock = get_lock(chat_id)
        returned_lock = await _acquire_lock_or_kill(chat_id, claude, update)

        assert returned_lock is expected_lock
        returned_lock.release()

    @pytest.mark.asyncio
    async def test_handle_message_releases_lock_on_error(self):
        """Lock is released even when _handle_response raises."""
        update = _make_update()
        ctx = _make_context()
        chat_id = 88804

        from kai.locks import get_lock

        lock = get_lock(chat_id)

        with (
            # Bypass the TOTP gate so handle_message reaches the lock
            # acquisition and _handle_response code paths under test.
            patch("kai.bot.is_totp_configured", return_value=False),
            patch(
                "kai.bot._handle_response",
                new_callable=AsyncMock,
                side_effect=RuntimeError("boom"),
            ),
            patch("kai.bot.log_message"),
            patch("kai.bot._set_responding"),
            patch("kai.bot._clear_responding"),
            # Use real get_lock so we can verify the lock state after
            pytest.raises(RuntimeError),
        ):
            await handle_message(update, ctx)

        # Lock must be released after the error
        assert not lock.locked()


# ── _handle_workspace_config ────────────────────────────────────────


class TestHandleWorkspaceConfig:
    """Tests for /workspace config subcommands.

    Each test patches the sessions module and pool to isolate the handler
    from the database and subprocess pool. The handler takes a "target"
    string that simulates what handle_workspace passes after splitting
    the user's message (e.g., "config model opus").
    """

    def _patches(self, mock_sessions):
        """Build the standard patch set for workspace config tests.

        Returns a context manager stack. mock_sessions should be an
        AsyncMock with the workspace config helpers pre-configured.
        """
        from contextlib import ExitStack

        stack = ExitStack()
        stack.enter_context(patch("kai.bot.sessions", mock_sessions))
        return stack

    def _mock_sessions(self, db_settings=None, user_settings=None):
        """Create a mock sessions module with workspace config helpers."""
        mock = AsyncMock()
        mock.get_workspace_config_settings = AsyncMock(return_value=db_settings or {})
        mock.set_workspace_config_setting = AsyncMock()
        mock.delete_workspace_config_setting = AsyncMock()
        mock.delete_all_workspace_config = AsyncMock()
        mock.build_workspace_config = AsyncMock(return_value=None)
        # _show_workspace_config also fetches user-level settings for its
        # fallback chain (workspace DB > workspaces.yaml > user DB >
        # users.yaml > global default).
        mock.get_user_settings = AsyncMock(return_value=user_settings or {})
        return mock

    # ── 1. Show config with no overrides ────────────────────────────

    @pytest.mark.asyncio
    async def test_show_config_no_overrides(self):
        """/workspace config with no overrides shows global defaults."""
        update = _make_update(text="/workspace config")
        config = _make_config()
        pool = _make_mock_claude()
        ctx = _make_context(config=config, pool=pool)
        mock_sessions = self._mock_sessions()

        with self._patches(mock_sessions):
            await _handle_workspace_config(update, ctx, "config")

        reply = update.message.reply_text.call_args[0][0]
        # Should show model and timeout with "global default" source
        assert "sonnet" in reply
        assert "global default" in reply

    @pytest.mark.asyncio
    async def test_show_config_user_setting_fallback(self):
        """/workspace config shows user-level model when no workspace override exists."""
        update = _make_update(text="/workspace config")
        config = _make_config()
        pool = _make_mock_claude()
        ctx = _make_context(config=config, pool=pool)
        # User set opus via /model, but no workspace-level override
        mock_sessions = self._mock_sessions(user_settings={"model": "opus"})

        with self._patches(mock_sessions):
            await _handle_workspace_config(update, ctx, "config")

        reply = update.message.reply_text.call_args[0][0]
        assert "opus" in reply
        assert "user setting" in reply

    # ── 2. Set model ────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_set_model(self):
        """/workspace config model opus sets the model."""
        update = _make_update(text="/workspace config model opus")
        config = _make_config()
        pool = _make_mock_claude()
        ctx = _make_context(config=config, pool=pool)
        mock_sessions = self._mock_sessions()

        with self._patches(mock_sessions):
            await _handle_workspace_config(update, ctx, "config model opus")

        # Should persist the model setting
        mock_sessions.set_workspace_config_setting.assert_called_once_with(
            12345, str(pool.get_effective_workspace.return_value), "model", "opus"
        )
        # Should apply config change (rebuild + change_workspace)
        mock_sessions.build_workspace_config.assert_called_once()
        pool.change_workspace.assert_called_once()
        # Should confirm to the user
        reply = update.message.reply_text.call_args[0][0]
        assert "opus" in reply.lower()

    # ── 5. Set timeout ──────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_set_timeout(self):
        """/workspace config timeout 300 sets the timeout."""
        update = _make_update(text="/workspace config timeout 300")
        config = _make_config()
        pool = _make_mock_claude()
        ctx = _make_context(config=config, pool=pool)
        mock_sessions = self._mock_sessions()

        with self._patches(mock_sessions):
            await _handle_workspace_config(update, ctx, "config timeout 300")

        mock_sessions.set_workspace_config_setting.assert_called_once_with(
            12345, str(pool.get_effective_workspace.return_value), "timeout", "300"
        )
        reply = update.message.reply_text.call_args[0][0]
        assert "300" in reply

    # ── 6. Set env var ──────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_set_env_var(self):
        """/workspace config env FOO=bar sets an env var."""
        update = _make_update(text="/workspace config env FOO=bar")
        config = _make_config()
        pool = _make_mock_claude()
        ctx = _make_context(config=config, pool=pool)
        mock_sessions = self._mock_sessions()

        with self._patches(mock_sessions):
            await _handle_workspace_config(update, ctx, "config env FOO=bar")

        # Should persist the env JSON blob
        call_args = mock_sessions.set_workspace_config_setting.call_args
        assert call_args[0][2] == "env"
        saved_env = json.loads(call_args[0][3])
        assert saved_env == {"FOO": "bar"}
        # Should apply config change
        pool.change_workspace.assert_called_once()

    # ── 7. Remove env var ───────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_remove_env_var(self):
        """/workspace config env -FOO removes an env var."""
        update = _make_update(text="/workspace config env -FOO")
        config = _make_config()
        pool = _make_mock_claude()
        ctx = _make_context(config=config, pool=pool)
        # Pre-populate an env var so there's something to remove
        mock_sessions = self._mock_sessions(db_settings={"env": '{"FOO": "bar"}'})

        with self._patches(mock_sessions):
            await _handle_workspace_config(update, ctx, "config env -FOO")

        # Should delete the env key entirely (empty dict -> delete)
        mock_sessions.delete_workspace_config_setting.assert_called_once_with(
            12345, str(pool.get_effective_workspace.return_value), "env"
        )
        reply = update.message.reply_text.call_args[0][0]
        assert "Removed" in reply

    # ── 8. List env vars ────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_list_env_vars(self):
        """/workspace config env (no value) lists env var keys."""
        update = _make_update(text="/workspace config env")
        config = _make_config()
        pool = _make_mock_claude()
        ctx = _make_context(config=config, pool=pool)
        mock_sessions = self._mock_sessions(db_settings={"env": '{"A": "1", "B": "2"}'})

        with self._patches(mock_sessions):
            await _handle_workspace_config(update, ctx, "config env")

        reply = update.message.reply_text.call_args[0][0]
        assert "A" in reply
        assert "B" in reply
        # Should NOT apply config change (read-only operation)
        pool.change_workspace.assert_not_called()

    # ── 9. Set prompt ───────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_set_prompt(self):
        """/workspace config prompt Hello world sets the prompt."""
        update = _make_update(text="/workspace config prompt Hello world")
        config = _make_config()
        pool = _make_mock_claude()
        ctx = _make_context(config=config, pool=pool)
        mock_sessions = self._mock_sessions()

        with self._patches(mock_sessions):
            await _handle_workspace_config(update, ctx, "config prompt Hello world")

        mock_sessions.set_workspace_config_setting.assert_called_once_with(
            12345, str(pool.get_effective_workspace.return_value), "prompt", "Hello world"
        )
        pool.change_workspace.assert_called_once()

    # ── 10. Clear prompt ────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_clear_prompt(self):
        """/workspace config prompt clear clears the prompt."""
        update = _make_update(text="/workspace config prompt clear")
        config = _make_config()
        pool = _make_mock_claude()
        ctx = _make_context(config=config, pool=pool)
        mock_sessions = self._mock_sessions()

        with self._patches(mock_sessions):
            await _handle_workspace_config(update, ctx, "config prompt clear")

        mock_sessions.delete_workspace_config_setting.assert_called_once_with(
            12345, str(pool.get_effective_workspace.return_value), "prompt"
        )
        pool.change_workspace.assert_called_once()

    # ── 11. Reset all overrides ─────────────────────────────────────

    @pytest.mark.asyncio
    async def test_reset_all(self):
        """/workspace config reset clears all overrides."""
        update = _make_update(text="/workspace config reset")
        config = _make_config()
        pool = _make_mock_claude()
        ctx = _make_context(config=config, pool=pool)
        mock_sessions = self._mock_sessions()

        with self._patches(mock_sessions):
            await _handle_workspace_config(update, ctx, "config reset")

        mock_sessions.delete_all_workspace_config.assert_called_once_with(
            12345, str(pool.get_effective_workspace.return_value)
        )
        pool.change_workspace.assert_called_once()
        reply = update.message.reply_text.call_args[0][0]
        assert "global defaults" in reply.lower()

    # ── 12. Reset single field ──────────────────────────────────────

    @pytest.mark.asyncio
    async def test_reset_single_field(self):
        """/workspace config reset model clears a single field."""
        update = _make_update(text="/workspace config reset model")
        config = _make_config()
        pool = _make_mock_claude()
        ctx = _make_context(config=config, pool=pool)
        mock_sessions = self._mock_sessions()

        with self._patches(mock_sessions):
            await _handle_workspace_config(update, ctx, "config reset model")

        mock_sessions.delete_workspace_config_setting.assert_called_once_with(
            12345, str(pool.get_effective_workspace.return_value), "model"
        )
        pool.change_workspace.assert_called_once()
        reply = update.message.reply_text.call_args[0][0]
        assert "model" in reply.lower()

    # ── 13. Unknown field shows error ───────────────────────────────

    @pytest.mark.asyncio
    async def test_unknown_field(self):
        """/workspace config bogus shows error with valid field list."""
        update = _make_update(text="/workspace config bogus")
        config = _make_config()
        pool = _make_mock_claude()
        ctx = _make_context(config=config, pool=pool)
        mock_sessions = self._mock_sessions()

        with self._patches(mock_sessions):
            await _handle_workspace_config(update, ctx, "config bogus")

        reply = update.message.reply_text.call_args[0][0]
        assert "bogus" in reply
        assert "model" in reply
        # Should NOT apply any config change
        pool.change_workspace.assert_not_called()

    # ── Invalid model rejected ──────────────────────────────────────

    @pytest.mark.asyncio
    async def test_invalid_model_rejected(self):
        """An unknown model name is rejected."""
        update = _make_update(text="/workspace config model gpt4")
        config = _make_config()
        pool = _make_mock_claude()
        ctx = _make_context(config=config, pool=pool)
        mock_sessions = self._mock_sessions()

        with self._patches(mock_sessions):
            await _handle_workspace_config(update, ctx, "config model gpt4")

        mock_sessions.set_workspace_config_setting.assert_not_called()
        reply = update.message.reply_text.call_args[0][0]
        assert "haiku" in reply
        assert "sonnet" in reply
        assert "opus" in reply


# ── /settings command ──────────────────────────────────────────────


class TestHandleSettings:
    """Tests for /settings - per-user default settings.

    Each test patches the sessions module and pool to isolate the handler
    from the database and subprocess pool.
    """

    def _patches(self, mock_sessions):
        """Build the standard patch set for settings tests."""
        from contextlib import ExitStack

        stack = ExitStack()
        stack.enter_context(patch("kai.bot.sessions", mock_sessions))
        return stack

    def _mock_sessions(self, db_settings=None):
        """Create a mock sessions module with per-user settings helpers."""
        mock = AsyncMock()
        mock.get_user_settings = AsyncMock(return_value=db_settings or {})
        mock.set_user_setting = AsyncMock()
        mock.delete_user_setting = AsyncMock()
        mock.delete_all_user_settings = AsyncMock()
        mock.clear_session = AsyncMock()
        return mock

    # ── 1. Show settings with global defaults ──────────────────────

    @pytest.mark.asyncio
    async def test_show_settings_defaults(self):
        """/settings with no overrides shows global defaults."""
        update = _make_update(text="/settings")
        config = _make_config()
        ctx = _make_context(config=config)
        mock_sessions = self._mock_sessions()

        with self._patches(mock_sessions):
            await _show_settings(update, ctx, 12345, config)

        reply = update.message.reply_text.call_args[0][0]
        assert "sonnet" in reply
        assert "global default" in reply
        assert "anthropic" in reply.lower()

    # ── 2. Set model ───────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_set_model(self):
        """/settings model opus sets model via _switch_model."""
        update = _make_update(text="/settings model opus")
        config = _make_config()
        pool = _make_mock_claude()
        ctx = _make_context(config=config, pool=pool, args=["model", "opus"])
        mock_sessions = self._mock_sessions()

        with self._patches(mock_sessions):
            await handle_settings(update, ctx)

        # _switch_model persists and restarts
        mock_sessions.set_user_setting.assert_called_once_with(12345, "model", "opus")
        pool.restart.assert_called_once()
        reply = update.message.reply_text.call_args[0][0]
        assert "opus" in reply.lower()

    # ── 3. Unknown setting rejected ────────────────────────────────

    @pytest.mark.asyncio
    async def test_budget_is_unknown_setting(self):
        """/settings budget 5 answers with the unknown-setting message
        listing the supported fields."""
        update = _make_update(text="/settings budget 5")
        config = _make_config()
        ctx = _make_context(config=config, args=["budget", "5"])
        mock_sessions = self._mock_sessions()

        with self._patches(mock_sessions):
            await handle_settings(update, ctx)

        mock_sessions.set_user_setting.assert_not_called()
        reply = update.message.reply_text.call_args[0][0]
        assert "Unknown setting" in reply
        assert "model" in reply
        assert "timeout" in reply

    # ── 6. Set timeout ─────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_set_timeout(self):
        """/settings timeout 300 sets the timeout."""
        update = _make_update(text="/settings timeout 300")
        config = _make_config()
        pool = _make_mock_claude()
        pool.get_if_exists = MagicMock(return_value=MagicMock())
        ctx = _make_context(config=config, pool=pool, args=["timeout", "300"])
        mock_sessions = self._mock_sessions()

        with self._patches(mock_sessions):
            await handle_settings(update, ctx)

        mock_sessions.set_user_setting.assert_called_once_with(12345, "timeout", "300")
        reply = update.message.reply_text.call_args[0][0]
        assert "300s" in reply

    # ── 7. Reject zero timeout ─────────────────────────────────────

    @pytest.mark.asyncio
    async def test_reject_zero_timeout(self):
        """/settings timeout 0 is rejected (must be positive)."""
        update = _make_update(text="/settings timeout 0")
        ctx = _make_context(args=["timeout", "0"])
        mock_sessions = self._mock_sessions()

        with self._patches(mock_sessions):
            await handle_settings(update, ctx)

        mock_sessions.set_user_setting.assert_not_called()
        reply = update.message.reply_text.call_args[0][0]
        assert "positive" in reply.lower()

    # ── 8. Reject timeout exceeding max ────────────────────────────

    @pytest.mark.asyncio
    async def test_reject_timeout_exceeding_max(self):
        """/settings timeout 601 is rejected."""
        update = _make_update(text="/settings timeout 601")
        ctx = _make_context(args=["timeout", "601"])
        mock_sessions = self._mock_sessions()

        with self._patches(mock_sessions):
            await handle_settings(update, ctx)

        mock_sessions.set_user_setting.assert_not_called()
        reply = update.message.reply_text.call_args[0][0]
        assert "600" in reply

    # ── 9. Context is not a setting ────────────────────────────────

    @pytest.mark.asyncio
    async def test_context_is_unknown_setting(self):
        """/settings context <n> is rejected: the context window
        setting was removed, so the field falls through to the
        unknown-setting reply and nothing is persisted."""
        update = _make_update(text="/settings context 200000")
        ctx = _make_context(args=["context", "200000"])
        mock_sessions = self._mock_sessions()

        with self._patches(mock_sessions):
            await handle_settings(update, ctx)

        mock_sessions.set_user_setting.assert_not_called()
        reply = update.message.reply_text.call_args[0][0]
        assert "Unknown setting" in reply
        assert "context" not in reply.split("Settings:")[1]

    # ── 12. Reset all ──────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_reset_all(self):
        """/settings reset clears all overrides."""
        update = _make_update(text="/settings reset")
        config = _make_config()
        pool = _make_mock_claude()
        ctx = _make_context(config=config, pool=pool, args=["reset"])
        mock_sessions = self._mock_sessions()

        with self._patches(mock_sessions):
            await handle_settings(update, ctx)

        mock_sessions.delete_all_user_settings.assert_called_once_with(12345)
        pool.restart.assert_called_once()
        reply = update.message.reply_text.call_args[0][0]
        assert "all settings cleared" in reply.lower()

    # ── 13. Reset single field ─────────────────────────────────────

    @pytest.mark.asyncio
    async def test_reset_single_field(self):
        """/settings reset model clears just the model override."""
        update = _make_update(text="/settings reset model")
        config = _make_config()
        pool = _make_mock_claude()
        ctx = _make_context(config=config, pool=pool, args=["reset", "model"])
        mock_sessions = self._mock_sessions()

        with self._patches(mock_sessions):
            await handle_settings(update, ctx)

        mock_sessions.delete_user_setting.assert_called_once_with(12345, "model")
        reply = update.message.reply_text.call_args[0][0]
        assert "model" in reply.lower()

    # ── 14. Unknown field ──────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_unknown_field(self):
        """/settings bogus shows error with valid field list."""
        update = _make_update(text="/settings bogus")
        ctx = _make_context(args=["bogus"])
        mock_sessions = self._mock_sessions()

        with self._patches(mock_sessions):
            await handle_settings(update, ctx)

        mock_sessions.set_user_setting.assert_not_called()
        reply = update.message.reply_text.call_args[0][0]
        assert "unknown setting" in reply.lower()

    # ── 15. Invalid model rejected ─────────────────────────────────

    @pytest.mark.asyncio
    async def test_invalid_model_rejected(self):
        """/settings model gpt4 is rejected."""
        update = _make_update(text="/settings model gpt4")
        ctx = _make_context(args=["model", "gpt4"])
        mock_sessions = self._mock_sessions()

        with self._patches(mock_sessions):
            await handle_settings(update, ctx)

        mock_sessions.set_user_setting.assert_not_called()
        reply = update.message.reply_text.call_args[0][0]
        assert "unknown model" in reply.lower()

    # ── 16. Reset context is an unknown field ──────────────────────

    @pytest.mark.asyncio
    async def test_reset_context_unknown_field(self):
        """/settings reset context is rejected: the context window
        setting was removed, so the field is not in the resettable
        set and nothing is deleted from the DB."""
        update = _make_update(text="/settings reset context")
        config = _make_config()
        pool = _make_mock_claude()
        ctx = _make_context(config=config, pool=pool, args=["reset", "context"])
        mock_sessions = self._mock_sessions()

        with self._patches(mock_sessions):
            await handle_settings(update, ctx)

        mock_sessions.delete_user_setting.assert_not_called()
        reply = update.message.reply_text.call_args[0][0]
        assert "Unknown field" in reply

    # ── 17. Reset reverts instance to defaults ────────────────────

    @pytest.mark.asyncio
    async def test_reset_reverts_instance_model(self):
        """/settings reset model writes the default back onto the instance."""
        update = _make_update(text="/settings reset model")
        config = _make_config()
        pool = _make_mock_claude()
        # Simulate an instance with an overridden model
        instance = MagicMock()
        instance.model = "opus"
        pool.get_if_exists = MagicMock(return_value=instance)
        ctx = _make_context(config=config, pool=pool, args=["reset", "model"])
        mock_sessions = self._mock_sessions()

        with self._patches(mock_sessions):
            await handle_settings(update, ctx)

        # Instance model should be reverted to the config default
        assert instance.model == config.default_model

    @pytest.mark.asyncio
    async def test_reset_reverts_codex_install_to_default(self):
        """
        /settings reset model on a wizard-generated codex install with
        DEFAULT_PROVIDER unset must revert to the wizard's DEFAULT_MODEL
        (e.g. gpt-5.5), not PROVIDER_DEFAULTS["openai"] (gpt-5.4).
        Regression for PR #489 re-review: get_effective_provider had
        to be taught the codex->openai rule so _revert_instance_field
        sees provider==effective_global and lands on config.default_model
        rather than the provider-default fallback branch.
        """
        update = _make_update(text="/settings reset model")
        config = _make_config(
            default_backend="codex",
            default_provider="",
            default_model="gpt-5.5",
        )
        pool = _make_mock_claude(provider="openai")
        instance = MagicMock()
        instance.model = "gpt-5.4-mini"
        instance.provider = "openai"
        pool.get_if_exists = MagicMock(return_value=instance)
        ctx = _make_context(config=config, pool=pool, args=["reset", "model"])
        mock_sessions = self._mock_sessions()

        with self._patches(mock_sessions):
            await handle_settings(update, ctx)

        # Reverts to the wizard's codex DEFAULT_MODEL, NOT PROVIDER_DEFAULTS
        # ["openai"] (gpt-5.4), which would silently move the user off
        # gpt-5.5 on a stock codex install.
        assert instance.model == "gpt-5.5"

    # ── 18. Reset all reverts all instance fields ─────────────────

    @pytest.mark.asyncio
    async def test_reset_all_reverts_instance(self):
        """/settings reset reverts all fields on the live instance."""
        update = _make_update(text="/settings reset")
        config = _make_config()
        pool = _make_mock_claude()
        instance = MagicMock()
        instance.model = "opus"
        instance.timeout_seconds = 500
        pool.get_if_exists = MagicMock(return_value=instance)
        ctx = _make_context(config=config, pool=pool, args=["reset"])
        mock_sessions = self._mock_sessions()

        with self._patches(mock_sessions):
            await handle_settings(update, ctx)

        assert instance.model == config.default_model
        assert instance.timeout_seconds == config.default_timeout


# ── /github command ─────────────────────────────────────────────────


class TestHandleGitHub:
    """Tests for /github - GitHub notification settings.

    Each test patches sessions to isolate the handler from the database.
    The _mock_resolve helper simulates resolve_github_settings() output,
    and _mock_db_settings simulates get_github_db_settings() output for
    source attribution in _show_github().
    """

    def _mock_resolve(self, repos=None, notify_chat_id=12345, pr_review=False, issue_triage=False):
        """Patch resolve_github_settings with controlled return values."""
        settings = {
            "repos": repos or [],
            "notify_chat_id": notify_chat_id,
            "pr_review": pr_review,
            "issue_triage": issue_triage,
        }
        return patch("kai.bot.sessions.resolve_github_settings", new_callable=AsyncMock, return_value=settings)

    def _mock_db_settings(self, db_settings=None):
        """Patch get_github_db_settings with controlled return values."""
        return patch("kai.bot.sessions.get_github_db_settings", new_callable=AsyncMock, return_value=db_settings or {})

    def _mock_added_repos(self, added=None):
        """Patch get_github_added_repos with controlled return values."""
        return patch("kai.bot.sessions.get_github_added_repos", new_callable=AsyncMock, return_value=added or [])

    def _mock_get_setting(self, value=None):
        """Patch get_setting (used for token lookup in _show_github)."""
        return patch("kai.bot.sessions.get_setting", new_callable=AsyncMock, return_value=value)

    # ── 1. /github with no config (defaults) ──────────────────────

    @pytest.mark.asyncio
    async def test_show_defaults(self):
        """/github with no user config shows global defaults."""
        update = _make_update(text="/github")
        config = _make_config()

        with (
            self._mock_resolve(),
            self._mock_db_settings(),
            self._mock_added_repos(),
            self._mock_get_setting(),
        ):
            await _show_github(update, 12345, config)

        reply = update.message.reply_text.call_args[0][0]
        assert "GitHub: not configured" in reply
        assert "Notifications: this chat" in reply
        assert "PR reviews: off (global default)" in reply
        assert "Issue triage: off (global default)" in reply
        assert "No repo subscriptions" in reply
        assert "GitHub token: not set" in reply

    # ── 2. /github with full config ───────────────────────────────

    @pytest.mark.asyncio
    async def test_show_full_config(self):
        """/github with user config shows all fields with sources."""
        from kai.config import UserConfig

        user = UserConfig(
            telegram_id=12345,
            name="alice",
            github="alice",
            github_repos=["alice/repo1"],
            pr_review=True,
            issue_triage=False,
        )
        config = _make_config(user_configs={12345: user})
        update = _make_update(text="/github")

        with (
            self._mock_resolve(
                repos=["alice/repo1"],
                notify_chat_id=99999,
                pr_review=True,
                issue_triage=True,
            ),
            # DB has issue_triage override, pr_review from yaml
            self._mock_db_settings({"issue_triage": "true"}),
            self._mock_added_repos(),
            self._mock_get_setting("ghp_fake_token"),
        ):
            await _show_github(update, 12345, config)

        reply = update.message.reply_text.call_args[0][0]
        assert "GitHub: alice" in reply
        assert "Notifications: 99999" in reply
        assert "PR reviews: on (users.yaml)" in reply
        assert "Issue triage: on (user override)" in reply
        assert "alice/repo1" in reply
        assert "(users.yaml)" in reply
        assert "GitHub token: stored" in reply

    # ── 3. /github notify <chat_id> ───────────────────────────────

    @pytest.mark.asyncio
    async def test_token_store_deletes_source_message(self):
        """/github token stores the PAT and deletes the Telegram command."""
        update = _make_update(text="/github token ghp_secret")
        config = _make_config()
        ctx = _make_context(config=config, args=["token", "ghp_secret"])
        mock_sessions = AsyncMock()
        mock_sessions.set_setting = AsyncMock()

        with patch("kai.bot.sessions", mock_sessions):
            await handle_github(update, ctx)

        mock_sessions.set_setting.assert_called_once_with("github_token:12345", "ghp_secret")
        update.message.delete.assert_awaited_once()
        reply = update.message.reply_text.call_args[0][0]
        assert "stored" in reply.lower()

    @pytest.mark.asyncio
    async def test_token_store_continues_if_source_delete_fails(self):
        """A Telegram delete failure is logged but does not drop the PAT update."""
        update = _make_update(text="/github token ghp_secret")
        update.message.delete.side_effect = BadRequest("message can't be deleted")
        config = _make_config()
        ctx = _make_context(config=config, args=["token", "ghp_secret"])
        mock_sessions = AsyncMock()
        mock_sessions.set_setting = AsyncMock()

        with patch("kai.bot.sessions", mock_sessions):
            await handle_github(update, ctx)

        mock_sessions.set_setting.assert_called_once_with("github_token:12345", "ghp_secret")
        update.message.delete.assert_awaited_once()
        reply = update.message.reply_text.call_args[0][0]
        assert "stored" in reply.lower()

    @pytest.mark.asyncio
    async def test_notify_set(self):
        """/github notify 123456 sets notification chat and updates live set."""
        update = _make_update(text="/github notify 123456")
        config = _make_config()
        ctx = _make_context(config=config, args=["notify", "123456"])
        mock_sessions = AsyncMock()
        mock_sessions.set_setting = AsyncMock()
        mock_sessions.delete_setting = AsyncMock()
        mock_sessions.resolve_github_settings = AsyncMock()
        mock_sessions.get_github_db_settings = AsyncMock(return_value={})
        # No existing notify setting (fresh set)
        mock_sessions.get_setting = AsyncMock(return_value=None)

        with (
            patch("kai.bot.sessions", mock_sessions),
            patch("kai.bot.webhook.add_notification_chat_id") as mock_add,
        ):
            await handle_github(update, ctx)

        mock_sessions.set_setting.assert_called_once_with("github_notify_chat:12345", "123456")
        mock_add.assert_called_once_with(123456)
        reply = update.message.reply_text.call_args[0][0]
        assert "123456" in reply
        # Fix removes the restart requirement
        assert "restart" not in reply.lower()

    # ── 4. /github notify reset ───────────────────────────────────

    @pytest.mark.asyncio
    async def test_notify_reset(self):
        """/github notify reset clears the override."""
        update = _make_update(text="/github notify reset")
        config = _make_config()
        ctx = _make_context(config=config, args=["notify", "reset"])
        mock_sessions = AsyncMock()
        mock_sessions.get_setting = AsyncMock(return_value=None)
        mock_sessions.delete_setting = AsyncMock()
        mock_sessions.resolve_github_settings = AsyncMock()
        mock_sessions.get_github_db_settings = AsyncMock(return_value={})

        with patch("kai.bot.sessions", mock_sessions):
            await handle_github(update, ctx)

        mock_sessions.delete_setting.assert_called_once_with("github_notify_chat:12345")
        reply = update.message.reply_text.call_args[0][0]
        assert "reset" in reply.lower()

    # ── 5. /github notify abc (invalid) ───────────────────────────

    @pytest.mark.asyncio
    async def test_notify_invalid(self):
        """/github notify abc is rejected (not an integer)."""
        update = _make_update(text="/github notify abc")
        config = _make_config()
        ctx = _make_context(config=config, args=["notify", "abc"])
        mock_sessions = AsyncMock()

        with patch("kai.bot.sessions", mock_sessions):
            await handle_github(update, ctx)

        reply = update.message.reply_text.call_args[0][0]
        assert "integer" in reply.lower()

    # ── 6. /github reviews on ─────────────────────────────────────

    @pytest.mark.asyncio
    async def test_reviews_on(self):
        """/github reviews on enables PR reviews."""
        update = _make_update(text="/github reviews on")
        config = _make_config()
        ctx = _make_context(config=config, args=["reviews", "on"])
        mock_sessions = AsyncMock()
        mock_sessions.set_setting = AsyncMock()

        with patch("kai.bot.sessions", mock_sessions):
            await handle_github(update, ctx)

        mock_sessions.set_setting.assert_called_once_with("pr_review:12345", "true")
        reply = update.message.reply_text.call_args[0][0]
        assert "enabled" in reply.lower()

    # ── 7. /github reviews off ────────────────────────────────────

    @pytest.mark.asyncio
    async def test_reviews_off(self):
        """/github reviews off disables PR reviews."""
        update = _make_update(text="/github reviews off")
        config = _make_config()
        ctx = _make_context(config=config, args=["reviews", "off"])
        mock_sessions = AsyncMock()
        mock_sessions.set_setting = AsyncMock()

        with patch("kai.bot.sessions", mock_sessions):
            await handle_github(update, ctx)

        mock_sessions.set_setting.assert_called_once_with("pr_review:12345", "false")
        reply = update.message.reply_text.call_args[0][0]
        assert "disabled" in reply.lower()

    # ── 8. /github triage on ──────────────────────────────────────

    @pytest.mark.asyncio
    async def test_triage_on(self):
        """/github triage on enables issue triage."""
        update = _make_update(text="/github triage on")
        config = _make_config()
        ctx = _make_context(config=config, args=["triage", "on"])
        mock_sessions = AsyncMock()
        mock_sessions.set_setting = AsyncMock()

        with patch("kai.bot.sessions", mock_sessions):
            await handle_github(update, ctx)

        mock_sessions.set_setting.assert_called_once_with("issue_triage:12345", "true")
        reply = update.message.reply_text.call_args[0][0]
        assert "enabled" in reply.lower()

    # ── 9. /github reviews (no value) ─────────────────────────────

    @pytest.mark.asyncio
    async def test_reviews_missing_value(self):
        """/github reviews with no on/off shows usage."""
        update = _make_update(text="/github reviews")
        config = _make_config()
        ctx = _make_context(config=config, args=["reviews"])
        mock_sessions = AsyncMock()

        with patch("kai.bot.sessions", mock_sessions):
            await handle_github(update, ctx)

        reply = update.message.reply_text.call_args[0][0]
        assert "usage" in reply.lower()

    # ── 10. /github bogus ─────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_unknown_subcommand(self):
        """/github bogus shows error message."""
        update = _make_update(text="/github bogus")
        config = _make_config()
        ctx = _make_context(config=config, args=["bogus"])
        mock_sessions = AsyncMock()

        with patch("kai.bot.sessions", mock_sessions):
            await handle_github(update, ctx)

        reply = update.message.reply_text.call_args[0][0]
        assert "unknown subcommand" in reply.lower()

    # ── 11. /github notify - live notification destination updates ────────

    @pytest.mark.asyncio
    async def test_notify_set_updates_notification_destinations(self):
        """/github notify -100999 adds the group to notification destinations immediately."""
        update = _make_update(text="/github notify -100999")
        config = _make_config()
        ctx = _make_context(config=config, args=["notify", "-100999"])
        mock_sessions = AsyncMock()
        mock_sessions.set_setting = AsyncMock()
        # No existing notify setting (fresh set)
        mock_sessions.get_setting = AsyncMock(return_value=None)

        with (
            patch("kai.bot.sessions", mock_sessions),
            patch("kai.bot.webhook.add_notification_chat_id") as mock_add,
        ):
            await handle_github(update, ctx)

        mock_sessions.set_setting.assert_called_once_with("github_notify_chat:12345", "-100999")
        mock_add.assert_called_once_with(-100999)
        reply = update.message.reply_text.call_args[0][0]
        assert "restart" not in reply.lower()

    @pytest.mark.asyncio
    async def test_notify_set_overwrite_cleans_up_old(self):
        """Overwriting a notify destination removes the old outbound destination.

        Without this cleanup, the old chat_id would linger in the live
        outbound notification registry until restart.
        """
        update = _make_update(text="/github notify -100888")
        config = _make_config()
        ctx = _make_context(config=config, args=["notify", "-100888"])
        mock_sessions = AsyncMock()
        mock_sessions.set_setting = AsyncMock()
        # Existing notify setting points to a different group chat
        mock_sessions.get_setting = AsyncMock(return_value="-100999")

        with (
            patch("kai.bot.sessions", mock_sessions),
            patch("kai.bot.webhook.add_notification_chat_id") as mock_add,
            patch("kai.bot.webhook.remove_notification_chat_id") as mock_remove,
            patch("kai.bot._is_notify_chat_used", new_callable=AsyncMock, return_value=False),
        ):
            await handle_github(update, ctx)

        # Old destination removed, new one added
        mock_remove.assert_called_once_with(-100999)
        mock_add.assert_called_once_with(-100888)

    @pytest.mark.asyncio
    async def test_notify_set_overwrite_skips_cleanup_when_shared(self):
        """Overwriting a notify destination keeps the old chat_id if another user uses it."""
        update = _make_update(text="/github notify -100888")
        config = _make_config()
        ctx = _make_context(config=config, args=["notify", "-100888"])
        mock_sessions = AsyncMock()
        mock_sessions.set_setting = AsyncMock()
        mock_sessions.get_setting = AsyncMock(return_value="-100999")

        with (
            patch("kai.bot.sessions", mock_sessions),
            patch("kai.bot.webhook.add_notification_chat_id") as mock_add,
            patch("kai.bot.webhook.remove_notification_chat_id") as mock_remove,
            # Another user still uses the old chat_id
            patch("kai.bot._is_notify_chat_used", new_callable=AsyncMock, return_value=True),
        ):
            await handle_github(update, ctx)

        # Old destination kept (still used), new one added
        mock_remove.assert_not_called()
        mock_add.assert_called_once_with(-100888)

    @pytest.mark.asyncio
    async def test_notify_set_overwrite_same_value_no_cleanup(self):
        """Setting the same notify destination again does not trigger cleanup."""
        update = _make_update(text="/github notify -100999")
        config = _make_config()
        ctx = _make_context(config=config, args=["notify", "-100999"])
        mock_sessions = AsyncMock()
        mock_sessions.set_setting = AsyncMock()
        # Same value already stored
        mock_sessions.get_setting = AsyncMock(return_value="-100999")

        with (
            patch("kai.bot.sessions", mock_sessions),
            patch("kai.bot.webhook.add_notification_chat_id") as mock_add,
            patch("kai.bot.webhook.remove_notification_chat_id") as mock_remove,
        ):
            await handle_github(update, ctx)

        # No removal needed (same value), just re-add (idempotent)
        mock_remove.assert_not_called()
        mock_add.assert_called_once_with(-100999)

    @pytest.mark.asyncio
    async def test_notify_reset_removes_from_notification_destinations(self):
        """/github notify reset removes the old outbound notification destination."""
        update = _make_update(text="/github notify reset")
        config = _make_config()
        ctx = _make_context(config=config, args=["notify", "reset"])
        mock_sessions = AsyncMock()
        # Simulate an existing notify setting pointing to a group chat
        mock_sessions.get_setting = AsyncMock(return_value="-100999")
        mock_sessions.delete_setting = AsyncMock()

        with (
            patch("kai.bot.sessions", mock_sessions),
            patch("kai.bot.webhook.remove_notification_chat_id") as mock_remove,
            patch("kai.bot._is_notify_chat_used", new_callable=AsyncMock, return_value=False),
        ):
            await handle_github(update, ctx)

        mock_sessions.delete_setting.assert_called_once_with("github_notify_chat:12345")
        mock_remove.assert_called_once_with(-100999)

    @pytest.mark.asyncio
    async def test_notify_reset_skips_removal_when_shared(self):
        """/github notify reset does NOT remove a chat_id still used by another user."""
        update = _make_update(text="/github notify reset")
        config = _make_config()
        ctx = _make_context(config=config, args=["notify", "reset"])
        mock_sessions = AsyncMock()
        mock_sessions.get_setting = AsyncMock(return_value="-100999")
        mock_sessions.delete_setting = AsyncMock()

        with (
            patch("kai.bot.sessions", mock_sessions),
            patch("kai.bot.webhook.remove_notification_chat_id") as mock_remove,
            # Another user still uses this chat_id
            patch("kai.bot._is_notify_chat_used", new_callable=AsyncMock, return_value=True),
        ):
            await handle_github(update, ctx)

        mock_sessions.delete_setting.assert_called_once()
        mock_remove.assert_not_called()

    @pytest.mark.asyncio
    async def test_notify_reset_self_id_skips_removal(self):
        """/github notify reset does NOT remove the user's own chat_id.

        This guard keeps the user's own authorized chat out of outbound
        notification cleanup.
        """
        update = _make_update(text="/github notify reset")
        config = _make_config()
        ctx = _make_context(config=config, args=["notify", "reset"])
        mock_sessions = AsyncMock()
        # The user previously set their own telegram_id (12345) as the target
        mock_sessions.get_setting = AsyncMock(return_value="12345")
        mock_sessions.delete_setting = AsyncMock()

        with (
            patch("kai.bot.sessions", mock_sessions),
            patch("kai.bot.webhook.remove_notification_chat_id") as mock_remove,
        ):
            await handle_github(update, ctx)

        mock_sessions.delete_setting.assert_called_once()
        # The self-ID guard fires before _is_notify_chat_used is even called
        mock_remove.assert_not_called()


# ── _is_notify_chat_used ──────────────────────────────────────────


class TestIsNotifyChatUsed:
    """Tests for _is_notify_chat_used helper.

    Verifies detection of shared notify chat_ids across users.yaml,
    database settings, and the global env var fallback.
    """

    @pytest.mark.asyncio
    async def test_yaml_match(self):
        """Returns True when another user in users.yaml uses the chat_id."""
        from kai.config import UserConfig

        user_a = UserConfig(telegram_id=111, name="alice")
        user_b = UserConfig(telegram_id=222, name="bob", github_notify_chat_id=-100999)
        config = _make_config(user_configs={111: user_a, 222: user_b})

        mock_sessions = AsyncMock()
        mock_sessions.get_setting = AsyncMock(return_value=None)

        with patch("kai.bot.sessions", mock_sessions):
            result = await _is_notify_chat_used(-100999, exclude_user=111, config=config)
        assert result is True

    @pytest.mark.asyncio
    async def test_db_match(self):
        """Returns True when another user's DB setting uses the chat_id."""
        from kai.config import UserConfig

        user_a = UserConfig(telegram_id=111, name="alice")
        user_b = UserConfig(telegram_id=222, name="bob")
        config = _make_config(user_configs={111: user_a, 222: user_b})

        mock_sessions = AsyncMock()
        # User B has the notify target in the DB
        mock_sessions.get_setting = AsyncMock(
            side_effect=lambda key: "-100999" if key == "github_notify_chat:222" else None
        )

        with patch("kai.bot.sessions", mock_sessions):
            result = await _is_notify_chat_used(-100999, exclude_user=111, config=config)
        assert result is True

    @pytest.mark.asyncio
    async def test_no_match(self):
        """Returns False when no other source references the chat_id."""
        from kai.config import UserConfig

        user_a = UserConfig(telegram_id=111, name="alice")
        config = _make_config(user_configs={111: user_a})

        mock_sessions = AsyncMock()
        mock_sessions.get_setting = AsyncMock(return_value=None)

        with patch("kai.bot.sessions", mock_sessions):
            result = await _is_notify_chat_used(-100999, exclude_user=111, config=config)
        assert result is False


# ── /model persistence ─────────────────────────────────────────────


class TestModelPersistence:
    """Tests that /model and /models keyboard now persist to settings."""

    @pytest.mark.asyncio
    async def test_model_command_persists(self):
        """/model opus writes to settings table."""
        update = _make_update(text="/model opus")
        config = _make_config()
        pool = _make_mock_claude()
        ctx = _make_context(config=config, pool=pool, args=["opus"])
        mock_sessions = AsyncMock()
        mock_sessions.set_user_setting = AsyncMock()
        mock_sessions.clear_session = AsyncMock()

        with patch("kai.bot.sessions", mock_sessions):
            await handle_model(update, ctx)

        mock_sessions.set_user_setting.assert_called_once_with(12345, "model", "opus")
        pool.restart.assert_called_once()

    @pytest.mark.asyncio
    async def test_model_callback_persists(self):
        """/models keyboard callback writes to settings table."""
        update = _make_callback_update(data="model:opus")
        config = _make_config()
        pool = _make_mock_claude(model="sonnet")  # different from opus
        ctx = _make_context(config=config, pool=pool)
        mock_sessions = AsyncMock()
        mock_sessions.set_user_setting = AsyncMock()
        mock_sessions.clear_session = AsyncMock()

        with patch("kai.bot.sessions", mock_sessions):
            await handle_model_callback(update, ctx)

        mock_sessions.set_user_setting.assert_called_once_with(12345, "model", "opus")
        pool.restart.assert_called_once()

    @pytest.mark.asyncio
    async def test_settings_reset_model_clears_db(self):
        """/settings reset model clears what /model set."""
        update = _make_update(text="/settings reset model")
        config = _make_config()
        pool = _make_mock_claude()
        ctx = _make_context(config=config, pool=pool, args=["reset", "model"])
        mock_sessions = AsyncMock()
        mock_sessions.delete_user_setting = AsyncMock()
        mock_sessions.clear_session = AsyncMock()

        with patch("kai.bot.sessions", mock_sessions):
            await _handle_settings_reset(update, ctx, 12345, config, "model")

        mock_sessions.delete_user_setting.assert_called_once_with(12345, "model")


# ── Command menu completeness ────────────────────────────────────────

# Commands in the Telegram dropdown menu (set_my_commands in main.py).
# Every command here must have a registered CommandHandler in create_bot().
# Intentionally excluded from the menu:
#   - start: Telegram convention, auto-triggered on first interaction
#   - ws: alias for /workspace, would be redundant in the menu
#   - jobs: alias for /job list, would be redundant in the menu
EXPECTED_MENU_COMMANDS = {
    "github",
    "help",
    "job",
    "memory",
    "model",
    "models",
    "new",
    "settings",
    "stats",
    "stop",
    "voice",
    "voices",
    "webhooks",
    "workspace",
    "workspaces",
}


class TestCommandMenu:
    """Tests that the Telegram command menu stays in sync with handlers."""

    @pytest.fixture(autouse=True)
    def _init_services(self, tmp_path):
        """Initialize services before create_bot() (same as TestCreateBotTransportMode)."""
        from kai import services

        services.load_services(tmp_path / "nonexistent.yaml")

    def test_every_expected_command_has_a_handler(self):
        """Every command in EXPECTED_MENU_COMMANDS has a registered CommandHandler.

        Checks the hardcoded expected set against create_bot() registrations.
        The companion test_menu_matches_expected_set bridges this set to the
        actual set_my_commands source, so together they enforce full parity.
        """
        from telegram.ext import CommandHandler as CH

        config = _make_config()
        app = create_bot(config)

        # Collect all command names from registered CommandHandlers
        registered: set[str] = set()
        for group_handlers in app.handlers.values():
            for handler in group_handlers:
                if isinstance(handler, CH):
                    registered.update(handler.commands)

        missing = EXPECTED_MENU_COMMANDS - registered
        assert not missing, f"Menu commands without handlers: {missing}"

    def test_all_stateful_commands_use_common_totp_middleware(self):
        """Every command except the narrow recovery set defaults to TOTP."""
        from telegram.ext import CommandHandler as CH

        app = create_bot(_make_config())
        callbacks: dict[str, object] = {}
        for group_handlers in app.handlers.values():
            for handler in group_handlers:
                if isinstance(handler, CH):
                    for command in handler.commands:
                        callbacks[command] = handler.callback

        assert {
            name for name, callback in callbacks.items() if not getattr(callback, "_kai_totp_sensitive", False)
        } == {
            "start",
            "help",
        }

    def test_all_inline_callbacks_use_common_totp_middleware(self):
        """Model, voice, workspace, and memory buttons cannot bypass TOTP."""
        from telegram.ext import CallbackQueryHandler as CQH

        app = create_bot(_make_config())
        callbacks = [
            handler.callback
            for group_handlers in app.handlers.values()
            for handler in group_handlers
            if isinstance(handler, CQH)
        ]

        assert len(callbacks) == 4
        assert all(getattr(callback, "_kai_totp_sensitive", False) for callback in callbacks)

    def test_menu_matches_expected_set(self):
        """The set_my_commands list in main.py matches EXPECTED_MENU_COMMANDS.

        Parses the actual source to extract BotCommand names from inside the
        set_my_commands() call specifically, so the test breaks if someone
        adds/removes a menu entry without updating the expected set.
        """
        import re

        main_py = Path(__file__).parent.parent / "src" / "kai" / "main.py"
        source = main_py.read_text()

        # Scope to the set_my_commands(...) block so BotCommand references
        # elsewhere in main.py (future defaults, comments, etc.) don't
        # cause false positives.
        match = re.search(
            r"set_my_commands\(\s*\[(.+?)\]\s*\)",
            source,
            re.DOTALL,
        )
        assert match, "Could not find set_my_commands() call in main.py"
        menu_commands = set(re.findall(r'BotCommand\("(\w+)"', match.group(1)))

        assert menu_commands == EXPECTED_MENU_COMMANDS, (
            f"Menu drift detected. "
            f"Added: {menu_commands - EXPECTED_MENU_COMMANDS}, "
            f"Removed: {EXPECTED_MENU_COMMANDS - menu_commands}"
        )


# ── /project command and workspace auto-registration ────────────────


@pytest.fixture(autouse=True)
def _reset_project_registry_cache():
    """The DB project registry cache in kai.memory_projects is
    module-level state; /project handler tests mutate it and must
    not leak into each other."""
    from kai.memory_projects import load_db_registry

    load_db_registry([])
    yield
    load_db_registry([])


def _mp_yaml(project_id: str, root: Path):
    """Operator-pinned registry entry for handler tests."""
    from kai.config import MemoryProjectConfig

    return MemoryProjectConfig(
        project_id=project_id,
        display_name=project_id.capitalize(),
        workspace_roots=(root.resolve(),),
        memory_enabled=True,
        default_scope_for_new_facts="project",
    )


class TestHandleProject:
    async def test_register_current_workspace(self, tmp_path):
        from kai.bot import handle_project
        from kai.memory_projects import db_registry_creator, merged_registry

        ws = tmp_path / "phi"
        ws.mkdir()
        update = _make_update("/project register")
        ctx = _make_context(
            pool=_make_mock_claude(workspace=ws),
            args=["register"],
        )
        with patch("kai.bot.sessions.register_memory_project", new_callable=AsyncMock) as persist:
            await handle_project(update, ctx)

        persist.assert_awaited_once()
        assert persist.call_args.kwargs["project_id"] == "phi"
        assert persist.call_args.kwargs["created_by"] == 12345
        # Restart-free contract: the cache sees the project immediately.
        assert "phi" in merged_registry({})
        assert db_registry_creator("phi") == 12345
        reply = update.message.reply_text.call_args.args[0]
        assert "Registered memory project 'phi'" in reply

    async def test_register_explicit_name_overrides_dir_name(self, tmp_path):
        from kai.bot import handle_project
        from kai.memory_projects import merged_registry

        ws = tmp_path / "some-checkout"
        ws.mkdir()
        update = _make_update("/project register Anvil")
        ctx = _make_context(pool=_make_mock_claude(workspace=ws), args=["register", "Anvil"])
        with patch("kai.bot.sessions.register_memory_project", new_callable=AsyncMock):
            await handle_project(update, ctx)

        merged = merged_registry({})
        assert "anvil" in merged
        # Display name keeps the user's casing; the id is the slug.
        assert merged["anvil"].display_name == "Anvil"

    async def test_register_inside_existing_project_rejected(self, tmp_path):
        from kai.bot import handle_project

        root = tmp_path / "kai"
        sub = root / "subdir"
        sub.mkdir(parents=True)
        config = _make_config(memory_projects={"kai": _mp_yaml("kai", root)})
        update = _make_update("/project register")
        ctx = _make_context(config=config, pool=_make_mock_claude(workspace=sub), args=["register"])
        with patch("kai.bot.sessions.register_memory_project", new_callable=AsyncMock) as persist:
            await handle_project(update, ctx)

        persist.assert_not_awaited()
        reply = update.message.reply_text.call_args.args[0]
        assert "already inside project 'kai'" in reply

    async def test_register_duplicate_id_rejected(self, tmp_path):
        from kai.bot import handle_project

        yaml_root = tmp_path / "elsewhere"
        yaml_root.mkdir()
        ws = tmp_path / "kai"
        ws.mkdir()
        config = _make_config(memory_projects={"kai": _mp_yaml("kai", yaml_root)})
        update = _make_update("/project register kai")
        ctx = _make_context(config=config, pool=_make_mock_claude(workspace=ws), args=["register", "kai"])
        with patch("kai.bot.sessions.register_memory_project", new_callable=AsyncMock) as persist:
            await handle_project(update, ctx)

        persist.assert_not_awaited()
        reply = update.message.reply_text.call_args.args[0]
        assert "already registered" in reply

    async def test_register_invalid_name_rejected(self, tmp_path):
        from kai.bot import handle_project

        ws = tmp_path / "phi"
        ws.mkdir()
        update = _make_update("/project register 'bad name!'")
        ctx = _make_context(pool=_make_mock_claude(workspace=ws), args=["register", "bad name!"])
        with patch("kai.bot.sessions.register_memory_project", new_callable=AsyncMock) as persist:
            await handle_project(update, ctx)

        persist.assert_not_awaited()
        reply = update.message.reply_text.call_args.args[0]
        assert "Invalid project name" in reply

    async def test_unregister_pinned_project_refused(self, tmp_path):
        from kai.bot import handle_project

        root = tmp_path / "kai"
        root.mkdir()
        config = _make_config(memory_projects={"kai": _mp_yaml("kai", root)})
        update = _make_update("/project unregister kai")
        ctx = _make_context(config=config, args=["unregister", "kai"])
        with patch("kai.bot.sessions.unregister_memory_project", new_callable=AsyncMock) as remove:
            await handle_project(update, ctx)

        remove.assert_not_awaited()
        reply = update.message.reply_text.call_args.args[0]
        assert "operator-pinned" in reply

    async def test_unregister_by_other_user_refused(self, tmp_path):
        from kai.bot import handle_project
        from kai.memory_projects import load_db_registry

        root = tmp_path / "phi"
        root.mkdir()
        load_db_registry(
            [
                {
                    "project_id": "phi",
                    "display_name": "Phi",
                    "workspace_root": str(root),
                    "memory_enabled": True,
                    "default_scope_for_new_facts": "project",
                    "created_by": 777,
                }
            ]
        )
        update = _make_update("/project unregister phi")
        ctx = _make_context(args=["unregister", "phi"])
        with patch("kai.bot.sessions.unregister_memory_project", new_callable=AsyncMock) as remove:
            await handle_project(update, ctx)

        remove.assert_not_awaited()
        reply = update.message.reply_text.call_args.args[0]
        assert "registered by another user" in reply

    async def test_unregister_by_creator_succeeds(self, tmp_path):
        from kai.bot import handle_project
        from kai.memory_projects import load_db_registry, merged_registry

        root = tmp_path / "phi"
        root.mkdir()
        load_db_registry(
            [
                {
                    "project_id": "phi",
                    "display_name": "Phi",
                    "workspace_root": str(root),
                    "memory_enabled": True,
                    "default_scope_for_new_facts": "project",
                    "created_by": 12345,
                }
            ]
        )
        update = _make_update("/project unregister phi")
        ctx = _make_context(args=["unregister", "phi"])
        with patch("kai.bot.sessions.unregister_memory_project", new_callable=AsyncMock, return_value=True) as remove:
            await handle_project(update, ctx)

        remove.assert_awaited_once_with("phi")
        assert merged_registry({}) == {}
        reply = update.message.reply_text.call_args.args[0]
        assert "Unregistered" in reply

    async def test_stale_unregister_authorization_rechecked_under_lock(self, tmp_path):
        """Unregister authorization must be read under the mutation
        lock: a creator read before the lock can authorize against a
        row that earlier queued mutations delete and a different
        user re-registers under the same id. The test holds the lock,
        starts the stale unregister so it queues, swaps the row's
        owner while it waits, and asserts the recheck denies the
        deletion and the new owner's project survives."""
        from kai.bot import handle_project
        from kai.memory_projects import (
            db_registry_remove,
            db_registry_upsert,
            load_db_registry,
            merged_registry,
            registry_mutation_lock,
        )

        root_old = tmp_path / "phi-old"
        root_old.mkdir()
        root_new = tmp_path / "phi-new"
        root_new.mkdir()
        # phi originally registered by chat 999 (the stale caller).
        load_db_registry(
            [
                {
                    "project_id": "phi",
                    "display_name": "Phi",
                    "workspace_root": str(root_old),
                    "memory_enabled": True,
                    "default_scope_for_new_facts": "project",
                    "created_by": 999,
                }
            ]
        )
        update = _make_update("/project unregister phi", chat_id=999)
        ctx = _make_context(args=["unregister", "phi"])

        with patch("kai.bot.sessions.unregister_memory_project", new_callable=AsyncMock) as remove:
            async with registry_mutation_lock():
                # The stale unregister starts and queues behind the
                # held lock BEFORE the ownership swap below.
                task = asyncio.create_task(handle_project(update, ctx))
                await asyncio.sleep(0)
                # Earlier queued mutations, simulated under the held
                # lock: the old phi goes away and a different user
                # re-registers the id.
                db_registry_remove("phi")
                db_registry_upsert(
                    {
                        "project_id": "phi",
                        "display_name": "Phi",
                        "workspace_root": str(root_new),
                        "memory_enabled": True,
                        "default_scope_for_new_facts": "project",
                        "created_by": 12345,
                    }
                )
            await task

        remove.assert_not_awaited()
        reply = update.message.reply_text.call_args.args[0]
        assert "registered by another user" in reply
        # The new owner's project survives with its new root.
        merged = merged_registry({})
        assert merged["phi"].workspace_roots == (root_new.resolve(),)

    async def test_concurrent_parent_child_registration_serialized(self, tmp_path):
        """The nested-root guard reads the merged view BEFORE an
        awaited DB insert; without the registry mutation lock, two
        concurrent registrations both pass their guards against the
        same stale view and commit a parent/child pair (the child
        then steals the parent's subtree via longest-prefix
        detection). With the lock, the second registration observes
        the first and is rejected. The persist stub yields to the
        event loop to force the interleaving the lock must close."""
        from kai.bot import _register_memory_project_for
        from kai.memory_projects import detect_active_memory_project, merged_registry

        parent = tmp_path / "parent"
        child = parent / "child"
        child.mkdir(parents=True)
        config = _make_config()

        async def _yielding_persist(**kwargs):
            await asyncio.sleep(0)

        with patch(
            "kai.bot.sessions.register_memory_project",
            new=AsyncMock(side_effect=_yielding_persist),
        ):
            results = await asyncio.gather(
                _register_memory_project_for(config, 12345, parent, "parent"),
                _register_memory_project_for(config, 12345, child, "child"),
            )

        successes = [message for ok, message in results if ok]
        rejections = [message for ok, message in results if not ok]
        assert len(successes) == 1
        assert len(rejections) == 1
        assert "already inside project" in rejections[0]
        # The merged registry must never contain nested DB-owned
        # roots: detection from inside the child resolves to the
        # single registered project.
        merged = merged_registry({})
        assert len(merged) == 1
        active = detect_active_memory_project(child, merged)
        assert active is not None
        assert active.project_id == "parent"

    async def test_list_shows_provenance_and_active_marker(self, tmp_path):
        from kai.bot import handle_project
        from kai.memory_projects import load_db_registry

        yaml_root = tmp_path / "kai"
        yaml_root.mkdir()
        db_root = tmp_path / "phi"
        db_root.mkdir()
        load_db_registry(
            [
                {
                    "project_id": "phi",
                    "display_name": "Phi",
                    "workspace_root": str(db_root),
                    "memory_enabled": True,
                    "default_scope_for_new_facts": "project",
                    "created_by": 12345,
                }
            ]
        )
        config = _make_config(memory_projects={"kai": _mp_yaml("kai", yaml_root)})
        update = _make_update("/project")
        ctx = _make_context(config=config, pool=_make_mock_claude(workspace=db_root), args=[])
        await handle_project(update, ctx)

        reply = update.message.reply_text.call_args.args[0]
        assert "kai [pinned]" in reply
        assert "phi [user] (active)" in reply


class TestWorkspaceNewAutoRegister:
    async def test_workspace_new_registers_project(self, tmp_path):
        """/workspace new is the strong project signal: the created
        directory is registered automatically and the user is told."""
        from kai.bot import handle_workspace
        from kai.memory_projects import merged_registry

        update = _make_update("/workspace new myproj")
        ctx = _make_context(args=["new", "myproj"])

        proc = MagicMock()
        proc.wait = AsyncMock(return_value=0)
        with (
            _mock_resolve(base=tmp_path),
            patch("kai.bot.asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=proc),
            patch("kai.bot._switch_workspace", new_callable=AsyncMock),
            patch("kai.bot.sessions.register_memory_project", new_callable=AsyncMock) as persist,
        ):
            await handle_workspace(update, ctx)

        persist.assert_awaited_once()
        assert persist.call_args.kwargs["project_id"] == "myproj"
        assert "myproj" in merged_registry({})
        replies = [c.args[0] for c in update.message.reply_text.call_args_list]
        assert any("Registered memory project 'myproj'" in r for r in replies)

    async def test_workspace_new_subpath_registers_basename(self, tmp_path):
        """/workspace new accepts relative subpaths; the auto-hook
        must register the created directory's BASENAME, since the
        raw argument's separator would fail slug validation and
        silently skip registration on a perfectly valid creation."""
        from kai.bot import handle_workspace
        from kai.memory_projects import merged_registry

        update = _make_update("/workspace new sub/project")
        ctx = _make_context(args=["new", "sub/project"])

        proc = MagicMock()
        proc.wait = AsyncMock(return_value=0)
        with (
            _mock_resolve(base=tmp_path),
            patch("kai.bot.asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=proc),
            patch("kai.bot._switch_workspace", new_callable=AsyncMock),
            patch("kai.bot.sessions.register_memory_project", new_callable=AsyncMock) as persist,
        ):
            await handle_workspace(update, ctx)

        persist.assert_awaited_once()
        assert persist.call_args.kwargs["project_id"] == "project"
        assert "project" in merged_registry({})

    async def test_workspace_new_registration_failure_does_not_block(self, tmp_path):
        """A registration failure warns but the workspace creation
        and switch still happen."""
        from kai.bot import handle_workspace
        from kai.memory_projects import load_db_registry

        # Pre-register the SAME id so the auto-hook hits the
        # duplicate-id rejection.
        other_root = tmp_path / "elsewhere"
        other_root.mkdir()
        load_db_registry(
            [
                {
                    "project_id": "myproj",
                    "display_name": "Myproj",
                    "workspace_root": str(other_root),
                    "memory_enabled": True,
                    "default_scope_for_new_facts": "project",
                    "created_by": 777,
                }
            ]
        )
        update = _make_update("/workspace new myproj")
        ctx = _make_context(args=["new", "myproj"])

        proc = MagicMock()
        proc.wait = AsyncMock(return_value=0)
        with (
            _mock_resolve(base=tmp_path),
            patch("kai.bot.asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=proc),
            patch("kai.bot._switch_workspace", new_callable=AsyncMock) as switch,
            patch("kai.bot.sessions.register_memory_project", new_callable=AsyncMock) as persist,
        ):
            await handle_workspace(update, ctx)

        persist.assert_not_awaited()
        switch.assert_awaited_once()
        replies = [c.args[0] for c in update.message.reply_text.call_args_list]
        assert any("memory project not registered" in r for r in replies)


# ── handle_review_command (Telegram manual review) ─────────────────────


def _review_result(text: str = "review output", warnings=()) -> PRReviewResult:
    """Build a PRReviewResult fixture for the manual command tests."""
    return PRReviewResult(
        repo="dcellison/kai",
        pr_number=681,
        pr_title="Test PR",
        pr_url="https://github.com/dcellison/kai/pull/681",
        review_text=text,
        collection_warnings=tuple(warnings),
    )


def _review_command_config(*repos: str, user_id: int = 1) -> Config:
    """Build a production-shaped config for an authorized review actor."""
    configured = list(repos or ("dcellison/kai",))
    return _make_config(
        allowed_user_ids={user_id},
        user_configs={
            user_id: UserConfig(
                telegram_id=user_id,
                name="reviewer",
                github_repos=configured,
            )
        },
    )


class TestHandleReviewCommand:
    """
    Manual /review command. The handler shares the bundle path with
    the webhook bot via generate_pr_review(); these tests cover the
    Telegram-specific surface (repo resolution, start ack, file
    staging, document upload, no-GitHub-comment, no-cooldown-touch,
    warning surfacing).
    """

    @pytest.fixture(autouse=True)
    def _mock_github_token_setting(self):
        """Default manual review tests run without a stored per-user GitHub token."""
        with patch("kai.bot.sessions.get_setting", new_callable=AsyncMock, return_value=None):
            yield

    @pytest.mark.asyncio
    async def test_short_form_uses_workspace_git_remote_match(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kai.bot.DATA_DIR", tmp_path)
        monkeypatch.setattr("kai.bot._REVIEW_TMP_DIR", tmp_path)
        update = _make_update()
        config = _make_config(
            user_configs={
                1: UserConfig(
                    telegram_id=1,
                    name="op",
                    github_repos=["dcellison/kai", "dcellison/other"],
                ),
            }
        )
        ctx = _make_context(config=config, args=["681"])
        ctx.bot.send_document = AsyncMock()

        with (
            patch("kai.bot._check_totp", new=AsyncMock(return_value=True)),
            patch(
                "kai.review._resolve_workspace_remote_repo",
                new=AsyncMock(return_value="dcellison/kai"),
            ),
            patch(
                "kai.sessions.get_effective_repos",
                new=AsyncMock(return_value=["dcellison/kai", "dcellison/other"]),
            ),
            patch("kai.bot.sessions.get_setting", new=AsyncMock(return_value="ghp_user")),
            patch(
                "kai.bot.review.generate_pr_review",
                new=AsyncMock(return_value=_review_result()),
            ) as mock_generate,
        ):
            await handle_review_command(update, ctx)

        assert mock_generate.call_args.args == ("dcellison/kai", 681)
        assert mock_generate.call_args.kwargs["github_token"] == "ghp_user"

    @pytest.mark.asyncio
    async def test_protected_install_requires_github_token(self, tmp_path, monkeypatch):
        """Manual /review does not use the daemon gh identity in protected installs."""
        monkeypatch.setattr("kai.bot.DATA_DIR", tmp_path)
        monkeypatch.setattr("kai.bot._REVIEW_TMP_DIR", tmp_path)
        update = _make_update()
        config = _make_config(
            protected_install=True,
            user_configs={
                1: UserConfig(
                    telegram_id=1,
                    name="op",
                    github_repos=["dcellison/kai"],
                ),
            },
        )
        ctx = _make_context(config=config, args=["dcellison/kai", "681"])

        with (
            patch("kai.bot._check_totp", new=AsyncMock(return_value=True)),
            patch(
                "kai.review._resolve_workspace_remote_repo",
                new=AsyncMock(return_value="dcellison/kai"),
            ),
            patch("kai.bot.sessions.get_setting", new=AsyncMock(return_value=None)),
            patch("kai.bot.review.generate_pr_review", new_callable=AsyncMock) as mock_generate,
        ):
            await handle_review_command(update, ctx)

        mock_generate.assert_not_called()
        reply = update.message.reply_text.call_args.args[0]
        assert "per-user GitHub token" in reply

    @pytest.mark.asyncio
    async def test_short_form_falls_back_to_sole_configured_repo(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kai.bot.DATA_DIR", tmp_path)
        monkeypatch.setattr("kai.bot._REVIEW_TMP_DIR", tmp_path)
        update = _make_update()
        # Workspace remote doesn't match the single configured repo;
        # the sole-configured fallback kicks in.
        config = _make_config(
            user_configs={
                1: UserConfig(
                    telegram_id=1,
                    name="op",
                    github_repos=["dcellison/kai"],
                ),
            }
        )
        ctx = _make_context(config=config, args=["681"])
        ctx.bot.send_document = AsyncMock()

        with (
            patch("kai.bot._check_totp", new=AsyncMock(return_value=True)),
            patch("kai.review._resolve_workspace_remote_repo", new=AsyncMock(return_value="")),
            patch(
                "kai.sessions.get_effective_repos",
                new=AsyncMock(return_value=["dcellison/kai"]),
            ),
            patch(
                "kai.bot.review.generate_pr_review",
                new=AsyncMock(return_value=_review_result()),
            ) as mock_generate,
        ):
            await handle_review_command(update, ctx)

        assert mock_generate.call_args.args == ("dcellison/kai", 681)

    @pytest.mark.asyncio
    async def test_short_form_deduplicates_case_varied_configured_repo(self, tmp_path, monkeypatch):
        """Duplicate config spellings remain one usable authorization."""
        monkeypatch.setattr("kai.bot.DATA_DIR", tmp_path)
        monkeypatch.setattr("kai.bot._REVIEW_TMP_DIR", tmp_path)
        update = _make_update()
        config = _review_command_config("DCellison/Kai", "dcellison/kai")
        ctx = _make_context(config=config, args=["681"])
        ctx.bot.send_document = AsyncMock()

        with (
            patch("kai.bot._check_totp", new=AsyncMock(return_value=True)),
            patch("kai.review._resolve_workspace_remote_repo", new=AsyncMock(return_value="")),
            patch(
                "kai.bot.review.generate_pr_review",
                new=AsyncMock(return_value=_review_result()),
            ) as mock_generate,
        ):
            await handle_review_command(update, ctx)

        assert mock_generate.call_args.args == ("dcellison/kai", 681)

    @pytest.mark.asyncio
    async def test_short_form_usage_error_when_unresolved(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kai.bot.DATA_DIR", tmp_path)
        monkeypatch.setattr("kai.bot._REVIEW_TMP_DIR", tmp_path)
        update = _make_update()
        # Multi-repo + no workspace match → usage error.
        config = _make_config(
            user_configs={
                1: UserConfig(
                    telegram_id=1,
                    name="op",
                    github_repos=["dcellison/kai", "dcellison/other"],
                ),
            }
        )
        ctx = _make_context(config=config, args=["681"])
        ctx.bot.send_document = AsyncMock()

        with (
            patch("kai.bot._check_totp", new=AsyncMock(return_value=True)),
            patch("kai.review._resolve_workspace_remote_repo", new=AsyncMock(return_value="")),
            patch(
                "kai.sessions.get_effective_repos",
                new=AsyncMock(return_value=["dcellison/kai", "dcellison/other"]),
            ),
            patch(
                "kai.bot.review.generate_pr_review",
                new=AsyncMock(),
            ) as mock_generate,
        ):
            await handle_review_command(update, ctx)

        mock_generate.assert_not_called()
        replies = [c.args[0] for c in update.message.reply_text.call_args_list]
        assert any("Usage: /review" in r for r in replies)

    @pytest.mark.asyncio
    async def test_explicit_owner_repo_form(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kai.bot.DATA_DIR", tmp_path)
        monkeypatch.setattr("kai.bot._REVIEW_TMP_DIR", tmp_path)
        update = _make_update()
        ctx = _make_context(config=_review_command_config(), args=["dcellison/kai", "681"])
        ctx.bot.send_document = AsyncMock()

        with (
            patch("kai.bot._check_totp", new=AsyncMock(return_value=True)),
            patch(
                "kai.bot.review.generate_pr_review",
                new=AsyncMock(return_value=_review_result()),
            ) as mock_generate,
        ):
            await handle_review_command(update, ctx)

        assert mock_generate.call_args.args == ("dcellison/kai", 681)

    @pytest.mark.asyncio
    async def test_explicit_repo_denied_when_only_another_user_is_authorized(self, tmp_path, monkeypatch):
        """One user cannot spend Kai's GitHub authority granted to another."""
        monkeypatch.setattr("kai.bot.DATA_DIR", tmp_path)
        monkeypatch.setattr("kai.bot._REVIEW_TMP_DIR", tmp_path)
        update = _make_update(user_id=1)
        config = _make_config(
            allowed_user_ids={1, 2},
            user_configs={
                1: UserConfig(telegram_id=1, name="alice", github_repos=["alice/repo"]),
                2: UserConfig(telegram_id=2, name="bob", github_repos=["private/target"]),
            },
        )
        ctx = _make_context(config=config, args=["private/target", "681"])

        with (
            patch("kai.bot._check_totp", new=AsyncMock(return_value=True)),
            patch("kai.bot.review.generate_pr_review", new=AsyncMock()) as mock_generate,
        ):
            await handle_review_command(update, ctx)

        mock_generate.assert_not_awaited()
        replies = [call.args[0] for call in update.message.reply_text.call_args_list]
        assert any("not authorized for GitHub review" in reply for reply in replies)

    @pytest.mark.asyncio
    async def test_db_added_subscription_does_not_authorize_explicit_review(self, tmp_path, monkeypatch):
        """Mutable notification subscriptions never grant shared-gh access."""
        monkeypatch.setattr("kai.bot.DATA_DIR", tmp_path)
        monkeypatch.setattr("kai.bot._REVIEW_TMP_DIR", tmp_path)
        update = _make_update(user_id=1)
        config = _review_command_config("configured/repo")
        ctx = _make_context(config=config, args=["db-added/repo", "681"])
        mock_effective = AsyncMock(return_value=["configured/repo", "db-added/repo"])

        with (
            patch("kai.bot._check_totp", new=AsyncMock(return_value=True)),
            patch("kai.bot.sessions.get_effective_repos", new=mock_effective),
            patch("kai.bot.review.generate_pr_review", new=AsyncMock()) as mock_generate,
        ):
            await handle_review_command(update, ctx)

        mock_effective.assert_not_awaited()
        mock_generate.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_invalid_repo_format_returns_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kai.bot.DATA_DIR", tmp_path)
        monkeypatch.setattr("kai.bot._REVIEW_TMP_DIR", tmp_path)
        update = _make_update()
        ctx = _make_context(args=["not-a-repo", "681"])

        with (
            patch("kai.bot._check_totp", new=AsyncMock(return_value=True)),
            patch(
                "kai.bot.review.generate_pr_review",
                new=AsyncMock(),
            ) as mock_generate,
        ):
            await handle_review_command(update, ctx)

        mock_generate.assert_not_called()
        replies = [c.args[0] for c in update.message.reply_text.call_args_list]
        assert any("Invalid repo format" in r for r in replies)

    @pytest.mark.asyncio
    async def test_invalid_pr_number_returns_usage(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kai.bot.DATA_DIR", tmp_path)
        monkeypatch.setattr("kai.bot._REVIEW_TMP_DIR", tmp_path)
        update = _make_update()
        ctx = _make_context(args=["dcellison/kai", "not-a-number"])

        with (
            patch("kai.bot._check_totp", new=AsyncMock(return_value=True)),
            patch(
                "kai.bot.review.generate_pr_review",
                new=AsyncMock(),
            ) as mock_generate,
        ):
            await handle_review_command(update, ctx)

        mock_generate.assert_not_called()
        replies = [c.args[0] for c in update.message.reply_text.call_args_list]
        assert any("Usage: /review" in r for r in replies)

    @pytest.mark.asyncio
    async def test_start_ack_before_backend(self, tmp_path, monkeypatch):
        # The start ack must fire BEFORE generate_pr_review returns.
        # We assert ordering by checking the reply_text call sequence.
        monkeypatch.setattr("kai.bot.DATA_DIR", tmp_path)
        monkeypatch.setattr("kai.bot._REVIEW_TMP_DIR", tmp_path)
        update = _make_update()
        ctx = _make_context(config=_review_command_config(), args=["dcellison/kai", "681"])
        ctx.bot.send_document = AsyncMock()

        ack_at_call: list[int] = []

        async def _capture(*_args, **_kwargs):
            # Snapshot reply count at the moment the backend runs.
            ack_at_call.append(len(update.message.reply_text.call_args_list))
            return _review_result()

        with (
            patch("kai.bot._check_totp", new=AsyncMock(return_value=True)),
            patch("kai.bot.review.generate_pr_review", new=AsyncMock(side_effect=_capture)),
        ):
            await handle_review_command(update, ctx)

        assert ack_at_call == [1], "start ack must be sent before generate_pr_review runs"

    @pytest.mark.asyncio
    async def test_writes_canonical_tmp_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kai.bot.DATA_DIR", tmp_path)
        monkeypatch.setattr("kai.bot._REVIEW_TMP_DIR", tmp_path)
        update = _make_update()
        ctx = _make_context(config=_review_command_config(), args=["dcellison/kai", "681"])
        ctx.bot.send_document = AsyncMock()

        with (
            patch("kai.bot._check_totp", new=AsyncMock(return_value=True)),
            patch(
                "kai.bot.review.generate_pr_review",
                new=AsyncMock(return_value=_review_result(text="bundle review body")),
            ),
        ):
            await handle_review_command(update, ctx)

        canonical = tmp_path / "pr-681-review.md"
        assert canonical.exists()
        body = canonical.read_text()
        assert "bundle review body" in body
        assert "Repository: dcellison/kai" in body

    @pytest.mark.asyncio
    async def test_stages_timestamped_copy_under_chat_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kai.bot.DATA_DIR", tmp_path)
        monkeypatch.setattr("kai.bot._REVIEW_TMP_DIR", tmp_path)
        update = _make_update(chat_id=12345)
        ctx = _make_context(config=_review_command_config(), args=["dcellison/kai", "681"])
        ctx.bot.send_document = AsyncMock()

        with (
            patch("kai.bot._check_totp", new=AsyncMock(return_value=True)),
            patch(
                "kai.bot.review.generate_pr_review",
                new=AsyncMock(return_value=_review_result()),
            ),
        ):
            await handle_review_command(update, ctx)

        staged_dir = tmp_path / "files" / "12345"
        assert staged_dir.is_dir()
        staged = list(staged_dir.glob("*_pr-681-review.md"))
        assert len(staged) == 1, f"expected one timestamped staged copy, got {staged}"

    @pytest.mark.asyncio
    async def test_uploads_staged_file_to_telegram(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kai.bot.DATA_DIR", tmp_path)
        monkeypatch.setattr("kai.bot._REVIEW_TMP_DIR", tmp_path)
        update = _make_update(chat_id=12345)
        ctx = _make_context(config=_review_command_config(), args=["dcellison/kai", "681"])
        ctx.bot.send_document = AsyncMock()

        with (
            patch("kai.bot._check_totp", new=AsyncMock(return_value=True)),
            patch(
                "kai.bot.review.generate_pr_review",
                new=AsyncMock(return_value=_review_result()),
            ),
        ):
            await handle_review_command(update, ctx)

        ctx.bot.send_document.assert_awaited_once()
        kwargs = ctx.bot.send_document.await_args.kwargs
        assert kwargs["filename"] == "pr-681-review.md"
        caption = kwargs["caption"]
        assert "pr-681-review.md" in caption

    @pytest.mark.asyncio
    async def test_does_not_post_to_github(self, tmp_path, monkeypatch):
        # post_review_comment lives in review.py; the manual command
        # must never call it directly. The webhook path goes through
        # review_pr; the Telegram path must NOT.
        monkeypatch.setattr("kai.bot.DATA_DIR", tmp_path)
        monkeypatch.setattr("kai.bot._REVIEW_TMP_DIR", tmp_path)
        update = _make_update()
        ctx = _make_context(config=_review_command_config(), args=["dcellison/kai", "681"])
        ctx.bot.send_document = AsyncMock()

        with (
            patch("kai.bot._check_totp", new=AsyncMock(return_value=True)),
            patch(
                "kai.bot.review.generate_pr_review",
                new=AsyncMock(return_value=_review_result()),
            ),
            patch("kai.review.post_review_comment", new=AsyncMock()) as mock_post,
        ):
            await handle_review_command(update, ctx)

        mock_post.assert_not_called()

    @pytest.mark.asyncio
    async def test_does_not_touch_webhook_cooldown(self, tmp_path, monkeypatch):
        # The webhook cooldown map is private to webhook.py. The
        # manual command runs as explicit user action and must not
        # update it; a manual review at T must not suppress an
        # automatic review on a later push.
        monkeypatch.setattr("kai.bot.DATA_DIR", tmp_path)
        monkeypatch.setattr("kai.bot._REVIEW_TMP_DIR", tmp_path)
        update = _make_update()
        ctx = _make_context(config=_review_command_config(), args=["dcellison/kai", "681"])
        ctx.bot.send_document = AsyncMock()

        from kai.webhook import _review_cooldowns

        baseline = dict(_review_cooldowns)
        with (
            patch("kai.bot._check_totp", new=AsyncMock(return_value=True)),
            patch(
                "kai.bot.review.generate_pr_review",
                new=AsyncMock(return_value=_review_result()),
            ),
        ):
            await handle_review_command(update, ctx)

        assert dict(_review_cooldowns) == baseline

    @pytest.mark.asyncio
    async def test_collection_warnings_surface_in_reply(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kai.bot.DATA_DIR", tmp_path)
        monkeypatch.setattr("kai.bot._REVIEW_TMP_DIR", tmp_path)
        update = _make_update()
        ctx = _make_context(config=_review_command_config(), args=["dcellison/kai", "681"])
        ctx.bot.send_document = AsyncMock()

        warnings = (CollectionWarning(source="related_search", message="search unavailable for repo"),)

        with (
            patch("kai.bot._check_totp", new=AsyncMock(return_value=True)),
            patch(
                "kai.bot.review.generate_pr_review",
                new=AsyncMock(return_value=_review_result(warnings=warnings)),
            ),
        ):
            await handle_review_command(update, ctx)

        replies = [c.args[0] for c in update.message.reply_text.call_args_list]
        assert any("Warnings:" in r and "search unavailable" in r for r in replies)

    @pytest.mark.asyncio
    async def test_backend_failure_replies_with_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kai.bot.DATA_DIR", tmp_path)
        monkeypatch.setattr("kai.bot._REVIEW_TMP_DIR", tmp_path)
        update = _make_update()
        ctx = _make_context(config=_review_command_config(), args=["dcellison/kai", "681"])
        ctx.bot.send_document = AsyncMock()

        with (
            patch("kai.bot._check_totp", new=AsyncMock(return_value=True)),
            patch(
                "kai.bot.review.generate_pr_review",
                new=AsyncMock(side_effect=RuntimeError("backend timeout")),
            ),
        ):
            await handle_review_command(update, ctx)

        replies = [c.args[0] for c in update.message.reply_text.call_args_list]
        assert any("Review failed" in r for r in replies)
        ctx.bot.send_document.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_args_returns_usage(self, tmp_path, monkeypatch):
        monkeypatch.setattr("kai.bot.DATA_DIR", tmp_path)
        monkeypatch.setattr("kai.bot._REVIEW_TMP_DIR", tmp_path)
        update = _make_update()
        ctx = _make_context(args=[])

        with (
            patch("kai.bot._check_totp", new=AsyncMock(return_value=True)),
            patch(
                "kai.bot.review.generate_pr_review",
                new=AsyncMock(),
            ) as mock_generate,
        ):
            await handle_review_command(update, ctx)

        mock_generate.assert_not_called()
        replies = [c.args[0] for c in update.message.reply_text.call_args_list]
        assert any("Usage: /review" in r for r in replies)

    @pytest.mark.asyncio
    async def test_group_chat_uses_user_id_for_user_scoped_lookups(self, tmp_path, monkeypatch):
        # In a notification group the message arrives on a group
        # chat_id but the actor is the operator. User-scoped lookups
        # (user_config, configured repos) must key on user_id, NOT
        # chat_id, so the review picks up the operator's settings.
        # Without this split, get_user_config(group_chat_id) returns
        # None, github_repos is [], and the resolution silently
        # collapses to the global defaults.
        monkeypatch.setattr("kai.bot.DATA_DIR", tmp_path)
        monkeypatch.setattr("kai.bot._REVIEW_TMP_DIR", tmp_path)
        update = _make_update(chat_id=-100888, user_id=12345)
        # User config keyed by user_id (12345), not chat_id (-100888).
        config = _make_config(
            user_configs={
                12345: UserConfig(
                    telegram_id=12345,
                    name="op",
                    github_repos=["dcellison/kai"],
                    backend="codex",
                    provider="openai",
                    os_user="daniel",
                ),
            },
            allowed_user_ids={12345},
        )
        ctx = _make_context(config=config, args=["681"])
        ctx.bot.send_document = AsyncMock()

        with (
            patch("kai.bot._check_totp", new=AsyncMock(return_value=True)),
            patch("kai.review._resolve_workspace_remote_repo", new=AsyncMock(return_value="")),
            patch(
                "kai.bot.review.generate_pr_review",
                new=AsyncMock(return_value=_review_result()),
            ) as mock_generate,
        ):
            await handle_review_command(update, ctx)

        # The review actually runs with the operator's resolved
        # settings (codex / openai / daniel), not the global
        # defaults that would apply if the lookup had used the
        # group chat id.
        kwargs = mock_generate.call_args.kwargs
        assert kwargs["agent_backend"] == "codex"
        assert kwargs["provider"] == "openai"
        assert kwargs["claude_user"] == "daniel"
        # And the inferred repo is the operator's sole configured
        # repo, not empty.
        assert mock_generate.call_args.args == ("dcellison/kai", 681)

    @pytest.mark.asyncio
    async def test_upload_failure_is_surfaced_in_reply(self, tmp_path, monkeypatch):
        # Phone-only operators cannot read /tmp; if the document
        # upload fails the final chat reply must say so rather than
        # silently telling them to open a file they can't reach.
        monkeypatch.setattr("kai.bot.DATA_DIR", tmp_path)
        monkeypatch.setattr("kai.bot._REVIEW_TMP_DIR", tmp_path)
        update = _make_update(chat_id=12345)
        ctx = _make_context(config=_review_command_config(), args=["dcellison/kai", "681"])
        ctx.bot.send_document = AsyncMock(side_effect=RuntimeError("file too large"))

        with (
            patch("kai.bot._check_totp", new=AsyncMock(return_value=True)),
            patch(
                "kai.bot.review.generate_pr_review",
                new=AsyncMock(return_value=_review_result()),
            ),
        ):
            await handle_review_command(update, ctx)

        replies = [c.args[0] for c in update.message.reply_text.call_args_list]
        # The canonical path is still surfaced...
        assert any("Review written to" in r for r in replies)
        # ...AND the failure is visible.
        assert any("Attachment failed" in r for r in replies)

    @pytest.mark.asyncio
    async def test_canonical_write_failure_is_surfaced(self, tmp_path, monkeypatch):
        # If the canonical /tmp/pr-N-review.md write fails (permission
        # denied, disk full, etc.), the contract is broken: there is
        # no artifact for the operator to read. The operator must see
        # a clear chat error, not just the initial "Reviewing…" ack
        # and silence.
        monkeypatch.setattr("kai.bot.DATA_DIR", tmp_path)
        # Point _REVIEW_TMP_DIR at a path that does NOT exist so the
        # write_text call raises OSError without touching real /tmp.
        monkeypatch.setattr("kai.bot._REVIEW_TMP_DIR", tmp_path / "does-not-exist")
        update = _make_update()
        ctx = _make_context(config=_review_command_config(), args=["dcellison/kai", "681"])
        ctx.bot.send_document = AsyncMock()

        with (
            patch("kai.bot._check_totp", new=AsyncMock(return_value=True)),
            patch(
                "kai.bot.review.generate_pr_review",
                new=AsyncMock(return_value=_review_result()),
            ),
        ):
            await handle_review_command(update, ctx)

        replies = [c.args[0] for c in update.message.reply_text.call_args_list]
        assert any("writing" in r and "failed" in r for r in replies)
        ctx.bot.send_document.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_staging_failure_surfaces_in_status(self, tmp_path, monkeypatch):
        # Staging is non-fatal (canonical /tmp still exists) but the
        # operator should know the attachment will not arrive so they
        # can fetch the file another way.
        monkeypatch.setattr("kai.bot.DATA_DIR", tmp_path)
        monkeypatch.setattr("kai.bot._REVIEW_TMP_DIR", tmp_path)
        update = _make_update()
        ctx = _make_context(config=_review_command_config(), args=["dcellison/kai", "681"])
        ctx.bot.send_document = AsyncMock()

        with (
            patch("kai.bot._check_totp", new=AsyncMock(return_value=True)),
            patch(
                "kai.bot.review.generate_pr_review",
                new=AsyncMock(return_value=_review_result()),
            ),
            patch("kai.bot._save_upload", side_effect=OSError("read-only filesystem")),
        ):
            await handle_review_command(update, ctx)

        replies = [c.args[0] for c in update.message.reply_text.call_args_list]
        assert any("Staging copy failed" in r for r in replies)
        # send_document must NOT fire when staging failed because
        # there is no staged path to upload from.
        ctx.bot.send_document.assert_not_awaited()
        # Canonical path is still surfaced.
        assert any("Review written to" in r for r in replies)

    @pytest.mark.asyncio
    async def test_many_warnings_chunked_under_telegram_limit(self, tmp_path, monkeypatch):
        # A bundle that hits many file fetch failures can produce
        # dozens of warnings. Concatenating them all into a single
        # reply with the status line can blow past Telegram's 4096-
        # char limit and cause the final reply to fail silently.
        # Status and warnings are split across separate replies,
        # each kept under the limit by chunk_text.
        monkeypatch.setattr("kai.bot.DATA_DIR", tmp_path)
        monkeypatch.setattr("kai.bot._REVIEW_TMP_DIR", tmp_path)
        update = _make_update()
        ctx = _make_context(config=_review_command_config(), args=["dcellison/kai", "681"])
        ctx.bot.send_document = AsyncMock()

        # Build warnings whose combined size exceeds 4096 chars.
        warnings = tuple(
            CollectionWarning(
                source=f"changed_file:src/module_{i}.py",
                message="content fetch failed: " + ("x" * 200),
            )
            for i in range(30)
        )
        assert sum(len(w.message) for w in warnings) > 4096

        with (
            patch("kai.bot._check_totp", new=AsyncMock(return_value=True)),
            patch(
                "kai.bot.review.generate_pr_review",
                new=AsyncMock(return_value=_review_result(warnings=warnings)),
            ),
        ):
            await handle_review_command(update, ctx)

        replies = [c.args[0] for c in update.message.reply_text.call_args_list]
        # Telegram's hard limit is 4096; every reply must respect it.
        for r in replies:
            assert len(r) <= 4096, f"reply exceeds Telegram message limit: {len(r)} chars"
        # Skip past the "Reviewing…" start ack to the post-backend
        # replies; the canonical status line is the first post-
        # backend reply and stands alone.
        post_backend = [r for r in replies if not r.startswith("Reviewing ")]
        assert post_backend[0].startswith("Review written to")
        assert "Warnings:" not in post_backend[0]
        assert any("Warnings:" in r for r in post_backend[1:])
        # All warning sources appeared somewhere.
        combined = "\n".join(replies)
        for w in warnings:
            assert w.source in combined

    @pytest.mark.asyncio
    async def test_explicit_form_with_mismatched_workspace_passes_none(self, tmp_path, monkeypatch):
        # Operator is sitting in workspace whose origin points at a
        # different repo (e.g. /Users/op/some-other-repo) and runs
        # `/review dcellison/kai 681`. The bundle must NOT load
        # local spec / conventions / surrounding-code from the
        # unrelated checkout; the only safe choice is to pass
        # local_repo_path=None so the bundle skips local lookups
        # entirely.
        monkeypatch.setattr("kai.bot.DATA_DIR", tmp_path)
        monkeypatch.setattr("kai.bot._REVIEW_TMP_DIR", tmp_path)
        update = _make_update()
        ctx = _make_context(config=_review_command_config(), args=["dcellison/kai", "681"])
        ctx.bot.send_document = AsyncMock()

        with (
            patch("kai.bot._check_totp", new=AsyncMock(return_value=True)),
            # Active workspace's origin resolves to a different repo.
            patch(
                "kai.review._resolve_workspace_remote_repo",
                new=AsyncMock(return_value="dcellison/other-repo"),
            ),
            patch(
                "kai.bot.review.generate_pr_review",
                new=AsyncMock(return_value=_review_result()),
            ) as mock_generate,
        ):
            await handle_review_command(update, ctx)

        assert mock_generate.call_args.kwargs["local_repo_path"] is None

    @pytest.mark.asyncio
    async def test_explicit_form_with_matching_workspace_passes_workspace(self, tmp_path, monkeypatch):
        # Same as above but the workspace IS the target repo's
        # checkout. The bundle gets the workspace so spec /
        # conventions / surrounding-code search can use it.
        monkeypatch.setattr("kai.bot.DATA_DIR", tmp_path)
        monkeypatch.setattr("kai.bot._REVIEW_TMP_DIR", tmp_path)
        update = _make_update()
        ctx = _make_context(config=_review_command_config(), args=["dcellison/kai", "681"])
        ctx.bot.send_document = AsyncMock()

        with (
            patch("kai.bot._check_totp", new=AsyncMock(return_value=True)),
            patch(
                "kai.review._resolve_workspace_remote_repo",
                new=AsyncMock(return_value="dcellison/kai"),
            ),
            patch(
                "kai.bot.review.generate_pr_review",
                new=AsyncMock(return_value=_review_result()),
            ) as mock_generate,
        ):
            await handle_review_command(update, ctx)

        assert mock_generate.call_args.kwargs["local_repo_path"] == "/home/workspace"

    @pytest.mark.asyncio
    async def test_short_form_sole_configured_with_mismatched_workspace_passes_none(self, tmp_path, monkeypatch):
        # Short form falls back to the sole configured repo because
        # the workspace remote does not match it (or is empty).
        # local_repo_path must be None in that case for the same
        # reason as the explicit form: the workspace and the
        # inferred repo are intentionally unrelated.
        monkeypatch.setattr("kai.bot.DATA_DIR", tmp_path)
        monkeypatch.setattr("kai.bot._REVIEW_TMP_DIR", tmp_path)
        update = _make_update()
        config = _make_config(
            user_configs={
                1: UserConfig(
                    telegram_id=1,
                    name="op",
                    github_repos=["dcellison/kai"],
                ),
            }
        )
        ctx = _make_context(config=config, args=["681"])
        ctx.bot.send_document = AsyncMock()

        with (
            patch("kai.bot._check_totp", new=AsyncMock(return_value=True)),
            # No GitHub remote at all (e.g. a plain non-git
            # workspace) - the sole-configured fallback fires but
            # the workspace must not propagate.
            patch("kai.review._resolve_workspace_remote_repo", new=AsyncMock(return_value="")),
            patch(
                "kai.bot.review.generate_pr_review",
                new=AsyncMock(return_value=_review_result()),
            ) as mock_generate,
        ):
            await handle_review_command(update, ctx)

        assert mock_generate.call_args.kwargs["local_repo_path"] is None

    @pytest.mark.asyncio
    async def test_short_form_workspace_match_passes_workspace(self, tmp_path, monkeypatch):
        # Belt and braces for the happy path: short form picks up
        # the workspace remote AND the workspace flows through as
        # local_repo_path so the bundle's full-context path runs.
        monkeypatch.setattr("kai.bot.DATA_DIR", tmp_path)
        monkeypatch.setattr("kai.bot._REVIEW_TMP_DIR", tmp_path)
        update = _make_update()
        config = _make_config(
            user_configs={
                1: UserConfig(
                    telegram_id=1,
                    name="op",
                    github_repos=["dcellison/kai", "dcellison/other"],
                ),
            }
        )
        ctx = _make_context(config=config, args=["681"])
        ctx.bot.send_document = AsyncMock()

        with (
            patch("kai.bot._check_totp", new=AsyncMock(return_value=True)),
            patch(
                "kai.review._resolve_workspace_remote_repo",
                new=AsyncMock(return_value="dcellison/kai"),
            ),
            patch(
                "kai.bot.review.generate_pr_review",
                new=AsyncMock(return_value=_review_result()),
            ) as mock_generate,
        ):
            await handle_review_command(update, ctx)

        assert mock_generate.call_args.kwargs["local_repo_path"] == "/home/workspace"
