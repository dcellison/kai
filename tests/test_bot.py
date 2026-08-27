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
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
from telegram.error import BadRequest

from kai import sessions
from kai.backend import AgentResponse, resolve_home_workspace
from kai.bot import (
    KaiTelegramApplication,
    _do_switch_workspace,
    _edit_message_safe,
    _handle_settings_reset,
    _handle_workspace_allow,
    _handle_workspace_allowed,
    _handle_workspace_config,
    _handle_workspace_deny,
    _is_authorized,
    _models_keyboard,
    _reply_safe,
    _require_auth,
    _resolve_workspace_path,
    _save_upload,
    _short_workspace_name,
    _show_github,
    _show_settings,
    _switch_workspace,
    _truncate_for_telegram,
    _voices_keyboard,
    _workspace_config_suffix,
    _workspaces_keyboard,
    create_bot,
    handle_backend,
    handle_backend_callback,
    handle_backends,
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
from kai.config import (
    PROVIDER_MODELS,
    Config,
    ModelRole,
    UserConfig,
    get_default_model_for_backend,
    get_user_backend_and_provider,
    resolve_user_model,
)
from kai.review import CollectionWarning, PRReviewResult
from kai.transcribe import TranscriptionError
from kai.tts import DEFAULT_VOICE, VOICES, TTSError
from kai.workshop.artifacts import (
    StagedArtifact,
    canonical_artifact_media_type,
)
from kai.workshop.conversation_commands import ConversationCommandDisposition
from kai.workshop.conversation_runs import WorkshopConversationRunService
from kai.workshop.domain import AgentId, ChannelId, MessageId, PrincipalId, RunId
from kai.workshop.execution_coordinator import (
    CanonicalCancellationDisposition,
    CanonicalExecutionDisposition,
)
from kai.workshop.inbound import InboundMessage
from kai.workshop.runtime_pool import WorkshopRuntimePool
from kai.workshop.settings_workspaces import (
    EffectiveValue,
    WorkshopSettingsWorkspaceConflict,
    WorkspaceConfigSnapshot,
)
from kai.workshop.storage_namespaces import (
    WorkshopPrincipalStorageNamespace,
    WorkshopPrincipalStorageRegistry,
)
from kai.workspace_utils import is_workspace_allowed
from tests.workshop_profiles import profile_id, profile_registry

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
        pool.get_backend_provider = MagicMock(return_value=("opencode", ""))

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
        pool.get_backend_provider = MagicMock(return_value=("codex", ""))

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
        principal_id = PrincipalId("prn_" + "1" * 32)
        result = _save_upload(b"x", "report.pdf", principal_id=principal_id)
        assert result.parent == tmp_path / "files" / principal_id
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
            result = _save_upload(
                b"x",
                "report.pdf",
                principal_id=PrincipalId("prn_" + "1" * 32),
                reader_user="alice",
            )

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
            _save_upload(
                b"x",
                "report.pdf",
                principal_id=PrincipalId("prn_" + "1" * 32),
                reader_user="alice",
            )

        assert list((tmp_path / "files" / ("prn_" + "1" * 32)).glob("*")) == []

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


def _bot_core_services(config):
    """Build unstarted typed core dependencies for Telegram factory tests."""
    from kai import services
    from kai.pool import SubprocessPool

    profiles = profile_registry(1)
    pool = SubprocessPool(
        config=config,
        services_info=services.get_available_services(),
        runtime_profiles=profiles,
    )
    runtime_pool = WorkshopRuntimePool(pool, profiles)
    principal_storage = WorkshopPrincipalStorageRegistry(
        (
            WorkshopPrincipalStorageNamespace(
                PrincipalId("prn_" + "1" * 32),
                profile_id(1),
                1,
            ),
        )
    )
    return SimpleNamespace(
        subprocess_pool=pool,
        runtime_pool=runtime_pool,
        conversation_runs=WorkshopConversationRunService(
            runtime_pool,
            sessions.resolve_workshop_conversation_run,
        ),
        private_text_execution=MagicMock(),
        principal_storage=principal_storage,
    )


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
        app = create_bot(config, use_webhook=True, core_services=_bot_core_services(config))
        assert app.updater is None

    def test_polling_mode_keeps_updater(self):
        """In polling mode, the Updater is present for start_polling()."""
        config = _make_config()
        app = create_bot(config, use_webhook=False, core_services=_bot_core_services(config))
        assert app.updater is not None

    def test_rejects_construction_without_telegram_token(self):
        config = _make_config(
            telegram_bot_token=None,
            enabled_adapters=frozenset({"workshop"}),
        )

        with pytest.raises(RuntimeError, match="cannot start without TELEGRAM_BOT_TOKEN"):
            create_bot(config, core_services=_bot_core_services(config))

    def test_installs_typed_core_services_without_shadow_recorders(self):
        config = _make_config()
        app = create_bot(config, core_services=_bot_core_services(config))

        assert isinstance(app, KaiTelegramApplication)
        assert isinstance(app.core_services.runtime_pool, WorkshopRuntimePool)
        assert isinstance(app.core_services.conversation_runs, WorkshopConversationRunService)
        for removed_key in (
            "pool",
            "workshop_runtime_pool",
            "workshop_conversation_run_service",
            "workshop_private_text_execution",
            "workshop_principal_storage",
            "workshop_inbound_recorder",
            "workshop_artifact_recorder",
            "workshop_outbound_recorder",
            "workshop_delivery_recorder",
            "workshop_streaming_preview_recorder",
            "workshop_streaming_finalizer",
        ):
            assert removed_key not in app.bot_data

    def test_does_not_publish_untyped_run_lifecycle_alias(self):
        config = _make_config()
        app = create_bot(config, core_services=_bot_core_services(config))

        # The application host owns the typed canonical services; Telegram
        # must not publish an alternate lifecycle through mutable bot_data.
        assert "workshop_run_lifecycle" not in app.bot_data


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
    pool.get_runtime_profile = MagicMock(return_value=None)
    pool.get_backend_provider = MagicMock(return_value=("claude", provider))
    pool.get_os_user = MagicMock(return_value=None)
    pool.get_role_model = MagicMock(return_value=model)
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


class _TestArtifactService:
    async def stage_upload(
        self,
        *,
        principal_id,
        filename,
        claimed_media_type,
        chunks,
        source_transport,
        source_unique_id,
        occurred_at,
        kind,
        **_unused,
    ):
        content = b"".join([chunk async for chunk in chunks])
        from kai import bot as bot_module

        path = bot_module._save_upload(
            content,
            filename,
            principal_id=principal_id,
        )
        return StagedArtifact(
            kind=kind,
            media_type=canonical_artifact_media_type(filename, claimed_media_type),
            storage_path=path,
            source_transport=source_transport,
            source_unique_id=source_unique_id,
            occurred_at=occurred_at,
            original_filename=(
                Path(filename.replace("\\", "/")).name if source_transport != "telegram" or kind == "document" else None
            ),
            created_for_attempt=True,
        )


def _make_context(config=None, claude=None, pool=None, args=None, user_data=None, job_queue=None):
    """Create a mock PTB context with bot_data, args, and user_data."""
    ctx = MagicMock()
    # Accept either pool (Phase 3) or claude (legacy test compat) as the
    # mock subprocess manager. Pool is preferred for new tests.
    created_pool = pool is None and claude is None
    mock_pool = pool or claude or _make_mock_claude()
    resolved_config = config or _make_config()
    if created_pool:
        mock_pool.get_backend_provider.side_effect = lambda runtime_config_id: get_user_backend_and_provider(
            resolved_config.get_user_config(runtime_config_id),
            resolved_config,
        )
        mock_pool.get_os_user.side_effect = lambda runtime_config_id: (
            user.os_user if (user := resolved_config.get_user_config(runtime_config_id)) is not None else None
        )
        mock_pool.get_role_model.side_effect = lambda runtime_config_id, role: resolve_user_model(
            role,
            resolved_config.get_user_config(runtime_config_id),
            resolved_config,
            backend=mock_pool.get_backend_provider(runtime_config_id)[0],
            provider=mock_pool.get_backend_provider(runtime_config_id)[1],
        )
    if pool is None:
        mock_pool.get_home_workspace = MagicMock(
            side_effect=lambda runtime_config_id: resolve_home_workspace(
                runtime_config_id,
                resolved_config,
            )
        )
        mock_pool.get_static_workspace_policy = MagicMock(
            side_effect=lambda runtime_config_id: (
                (
                    resolved_config.get_user_config(runtime_config_id).workspace_base,
                    tuple(resolved_config.get_user_config(runtime_config_id).allowed_workspaces),
                    False,
                )
                if resolved_config.get_user_config(runtime_config_id) is not None
                else (None, (), False)
            )
        )

        async def resolve_access(runtime_config_id):
            return await sessions.resolve_workspace_access(runtime_config_id, resolved_config)

        mock_pool.resolve_workspace_access = AsyncMock(side_effect=resolve_access)
    runtime_config_ids = {
        runtime_config_id
        for runtime_config_id in (
            set(resolved_config.allowed_user_ids) | set(resolved_config.user_configs) | {1, 12345}
        )
        if isinstance(runtime_config_id, int) and runtime_config_id > 0
    }
    principal_storage = WorkshopPrincipalStorageRegistry(
        tuple(
            WorkshopPrincipalStorageNamespace(
                PrincipalId(f"prn_{runtime_config_id:032x}"),
                profile_id(runtime_config_id),
                runtime_config_id,
            )
            for runtime_config_id in sorted(runtime_config_ids)
        )
    )
    ctx.bot_data = {
        "config": resolved_config,
    }
    application = MagicMock(spec=KaiTelegramApplication)
    internal_api_contexts = MagicMock()
    internal_api_contexts.for_runtime_profile.side_effect = lambda runtime_profile_id: SimpleNamespace(
        principal_id=principal_storage.for_runtime_profile(runtime_profile_id).principal_id,
        channel_id=ChannelId(f"chn_{int(str(runtime_profile_id).partition('_')[2], 16):032x}"),
        agent_id=AgentId("agt_" + "a" * 32),
        runtime_profile_id=runtime_profile_id,
    )
    application.core_services = SimpleNamespace(
        subprocess_pool=mock_pool,
        runtime_profiles=profile_registry(*sorted(runtime_config_ids)),
        private_text_execution=None,
        conversation_runs=None,
        principal_storage=principal_storage,
        artifacts=_TestArtifactService(),
        internal_api_contexts=internal_api_contexts,
        scheduler=SimpleNamespace(
            list_jobs=AsyncMock(return_value=[]),
            get_job=AsyncMock(return_value=None),
            delete_job=AsyncMock(return_value=False),
        ),
    )
    ctx.application = application
    ctx.args = args or []
    ctx.user_data = user_data if user_data is not None else {}
    ctx.bot.send_chat_action = AsyncMock()
    ctx.bot.get_file = AsyncMock()
    ctx.bot.send_voice = AsyncMock()
    if job_queue is not None:
        ctx.application.job_queue = job_queue
    return ctx


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

        # /github notify accepts <number|reset>, not [on|off]. The
        # [on|off] shape belongs to /github reviews and /github triage;
        # confusing them sent operators down the wrong path when
        # routing notifications.
        assert "/github notify [number|reset]" in reply
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

    @pytest.mark.asyncio
    async def test_private_text_stop_uses_canonical_cancellation_without_pool_kill(self):
        pool = _make_mock_claude()
        update = _make_update(chat_id=1, user_id=1)
        ctx = _make_context(claude=pool, config=_make_config(allowed_user_ids={1}))
        execution = MagicMock()
        execution.request_transport_cancellation = AsyncMock(return_value=CanonicalCancellationDisposition.REQUESTED)
        ctx.application.core_services.private_text_execution = execution

        await handle_stop(update, ctx)

        execution.request_transport_cancellation.assert_awaited_once_with(
            transport="telegram",
            sender_subject="1",
            channel_subject="1",
        )
        pool.force_kill.assert_not_called()
        assert update.message.reply_text.await_args.args[0] == "Stopping..."


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
            "created_at": "2026-01-01 12:00:00",
            "last_used_at": "2026-01-02 15:30:45",
        }
        with patch("kai.bot.sessions.get_stats", new_callable=AsyncMock, return_value=stats):
            await handle_stats(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "abcd1234" in reply
        assert "sonnet" in reply
        # Stored values are UTC; the reply must show them converted to the
        # host zone with a zone label. Expected values are computed with an
        # independent conversion so the assertion holds in any host zone,
        # including across the EST/EDT boundary.
        expected_started = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        expected_last_used = datetime(2026, 1, 2, 15, 30, 45, tzinfo=UTC).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        assert f"Started: {expected_started}" in reply
        assert f"Last used: {expected_last_used}" in reply


# ── handle_jobs ──────────────────────────────────────────────────────


class TestHandleJob:
    """Tests for the unified /job command and its subcommands."""

    # ── /job (no args) and /job list ────────────────────────────────

    @pytest.mark.asyncio
    async def test_no_args_lists_jobs(self):
        """/job with no args lists all jobs."""
        update = _make_update()
        ctx = _make_context()
        await handle_job(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "No active" in reply

    @pytest.mark.asyncio
    async def test_list_subcommand(self):
        """/job list is equivalent to /job with no args."""
        update = _make_update()
        ctx = _make_context(args=["list"])
        await handle_job(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "No active" in reply

    @pytest.mark.asyncio
    async def test_group_command_uses_authenticated_human_not_telegram_chat(self):
        update = _make_update(chat_id=-100123, user_id=1)
        ctx = _make_context()

        await handle_job(update, ctx)

        ctx.application.core_services.internal_api_contexts.for_runtime_profile.assert_called_once_with(profile_id(1))

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
        ctx.application.core_services.scheduler.list_jobs.return_value = jobs
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
        ctx.application.core_services.scheduler.list_jobs.return_value = jobs
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
        ctx.application.core_services.scheduler.list_jobs.return_value = jobs
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
            "name": "Weather report",
            "job_type": "claude",
            "prompt": "What is the weather today?",
            "schedule_type": "daily",
            "schedule_data": json.dumps({"times": ["08:00"]}),
            "auto_remove": False,
            "notify_on_check": False,
        }
        ctx.application.core_services.scheduler.get_job.return_value = job
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
        await handle_job(update, ctx)
        reply = update.message.reply_text.call_args[0][0]
        assert "not found" in reply.lower()

    @pytest.mark.asyncio
    async def test_info_wrong_owner(self):
        """/job info <id> returns not found for a job owned by another chat."""
        update = _make_update(chat_id=12345)
        ctx = _make_context(args=["info", "4"])
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
        """Deletes from DB and removes the core scheduler task."""
        update = _make_update()
        ctx = _make_context(args=["cancel", "5"])
        ctx.application.core_services.scheduler.delete_job.return_value = True
        await handle_job(update, ctx)
        ctx.application.core_services.scheduler.delete_job.assert_awaited_once()
        reply = update.message.reply_text.call_args[0][0]
        assert "cancelled" in reply.lower()

    @pytest.mark.asyncio
    async def test_cancel_not_found(self):
        update = _make_update()
        ctx = _make_context(args=["cancel", "99"])
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
    async def test_protected_model_switch_uses_canonical_settings_service(self):
        from kai.bot import _switch_model

        ctx = _make_context()
        authority = SimpleNamespace(runtime_profile_id=profile_id(12345))
        service = MagicMock()
        service.authority_for_principal_profile.return_value = authority
        service.set_model = AsyncMock()
        ctx.application.core_services.settings_workspaces = service

        await _switch_model(ctx, 12345, "gpt-5.6-terra")

        service.set_model.assert_awaited_once_with(
            authority,
            "gpt-5.6-terra",
        )

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
            patch("kai.bot.sessions.delete_active_workspace", new_callable=AsyncMock),
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
            patch("kai.bot.sessions.delete_active_workspace", new_callable=AsyncMock),
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
            patch("kai.bot.sessions.set_active_workspace", new_callable=AsyncMock),
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
            patch("kai.bot.sessions.set_active_workspace", new_callable=AsyncMock),
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
            patch("kai.bot.sessions.set_active_workspace", new_callable=AsyncMock),
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
            patch("kai.bot.sessions.set_active_workspace", new_callable=AsyncMock),
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
            patch("kai.bot.sessions.set_active_workspace", new_callable=AsyncMock),
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
            patch("kai.bot.sessions.delete_active_workspace", new_callable=AsyncMock),
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
            patch("kai.bot.sessions.set_active_workspace", new_callable=AsyncMock),
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
            patch("kai.bot.sessions.set_active_workspace", new_callable=AsyncMock),
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
            patch("kai.bot.sessions.set_active_workspace", new_callable=AsyncMock),
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
    async def test_records_authenticated_message_after_totp_gate(self):
        update = _make_update(text="canonical input", chat_id=1, user_id=1)
        ctx = _make_context(config=_make_config(allowed_user_ids={1}))
        inbound_id = MessageId("msg_00000000000000000000000000000001")
        run_id = RunId("run_00000000000000000000000000000001")
        execution = MagicMock()
        execution.accept = AsyncMock()
        execution.accept.return_value.message.event.envelope.aggregate_id = inbound_id
        execution.accept.return_value.run.run_id = run_id
        execution.accept.return_value.disposition = ConversationCommandDisposition.NEWLY_ACCEPTED
        ctx.application.core_services.private_text_execution = execution

        with (
            patch("kai.bot.is_totp_configured", return_value=False),
            patch("kai.bot._handle_workshop_private_text", new_callable=AsyncMock) as response,
        ):
            await handle_message(update, ctx)

        execution.accept.assert_awaited_once_with(
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
        response.assert_awaited_once_with(
            update,
            ctx,
            chat_id=1,
            run_id=run_id,
            inbound_message_id=inbound_id,
            prompt="canonical input",
            voice_mode="off",
        )

    @pytest.mark.asyncio
    async def test_private_text_voice_mode_still_uses_canonical_execution(self):
        update = _make_update(text="speak this", chat_id=1, user_id=1)
        ctx = _make_context(config=_make_config(allowed_user_ids={1}, tts_enabled=True))
        execution, _, _ = _configure_canonical_media_execution(ctx)

        with (
            patch("kai.bot.is_totp_configured", return_value=False),
            patch("kai.bot.sessions.get_setting", new_callable=AsyncMock, return_value="only"),
            patch("kai.bot._handle_workshop_private_text", new_callable=AsyncMock) as canonical_execute,
        ):
            await handle_message(update, ctx)

        execution.accept.assert_awaited_once()
        assert canonical_execute.await_args.kwargs["voice_mode"] == "only"

    @pytest.mark.asyncio
    async def test_private_text_without_durable_service_fails_closed(self):
        update = _make_update(text="do not dispatch", chat_id=1, user_id=1)
        pool = _make_mock_claude()
        ctx = _make_context(claude=pool, config=_make_config(allowed_user_ids={1}))

        with patch("kai.bot.is_totp_configured", return_value=False):
            await handle_message(update, ctx)

        pool.send.assert_not_called()
        assert "durable execution service is unavailable" in update.message.reply_text.await_args.args[0]

    @pytest.mark.asyncio
    async def test_private_text_terminal_replay_uses_canonical_execution_only(self):
        update = _make_update(text="same durable update", chat_id=1, user_id=1)
        ctx = _make_context(config=_make_config(allowed_user_ids={1}))
        execution, _, _ = _configure_canonical_media_execution(ctx)
        execution.accept.return_value.disposition = ConversationCommandDisposition.TERMINAL_REPLAY

        with (
            patch("kai.bot.is_totp_configured", return_value=False),
            patch("kai.bot._handle_workshop_private_text", new_callable=AsyncMock) as execute,
        ):
            await handle_message(update, ctx)

        execution.accept.assert_awaited_once()
        execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_totp_denial_does_not_accept_message(self):
        update = _make_update(chat_id=1, user_id=1)
        ctx = _make_context(config=_make_config(allowed_user_ids={1}))
        execution, _, _ = _configure_canonical_media_execution(ctx)

        with patch("kai.bot._check_totp_text", new_callable=AsyncMock, return_value=False):
            await handle_message(update, ctx)

        execution.accept.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unauthorized_message_is_not_accepted(self):
        update = _make_update(chat_id=99, user_id=99)
        ctx = _make_context(config=_make_config(allowed_user_ids={1}))
        execution, _, _ = _configure_canonical_media_execution(ctx)

        await handle_message(update, ctx)

        execution.accept.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_group_text_is_ignored_outside_canonical_agent_addressing(self):
        update = _make_update(chat_id=-100, user_id=1)
        ctx = _make_context(config=_make_config(allowed_user_ids={1}))
        execution, _, _ = _configure_canonical_media_execution(ctx)

        with patch("kai.bot._check_totp_text", new_callable=AsyncMock) as totp:
            await handle_message(update, ctx)

        totp.assert_not_awaited()
        execution.accept.assert_not_awaited()
        update.message.reply_text.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_message(self):
        update = _make_update(chat_id=1, user_id=1)
        update.message.text = None
        ctx = _make_context(config=_make_config(allowed_user_ids={1}))
        execution, _, _ = _configure_canonical_media_execution(ctx)

        await handle_message(update, ctx)

        execution.accept.assert_not_awaited()


# ── handle_photo ─────────────────────────────────────────────────────


def _configure_canonical_media_execution(ctx):
    inbound_id = MessageId("msg_00000000000000000000000000000001")
    run_id = RunId("run_00000000000000000000000000000001")
    execution = MagicMock()
    execution.accept = AsyncMock()
    execution.accept.return_value.message.event.envelope.aggregate_id = inbound_id
    execution.accept.return_value.run.run_id = run_id
    execution.accept.return_value.disposition = ConversationCommandDisposition.NEWLY_ACCEPTED
    ctx.application.core_services.private_text_execution = execution
    return execution, inbound_id, run_id


class TestHandlePhoto:
    @pytest.mark.asyncio
    async def test_accepts_photo_as_one_canonical_artifact_run(self, tmp_path):
        update = _make_update(chat_id=1, user_id=1)
        photo = MagicMock(file_id="download-capability", file_unique_id="stable-photo-id")
        update.message.photo = [MagicMock(), photo]
        update.message.caption = "Inspect this detail"
        downloaded = MagicMock()
        downloaded.download_as_bytearray = AsyncMock(return_value=bytearray(b"image-data"))
        ctx = _make_context(config=_make_config(allowed_user_ids={1}))
        ctx.bot.get_file = AsyncMock(return_value=downloaded)
        execution, inbound_id, run_id = _configure_canonical_media_execution(ctx)

        with (
            patch("kai.bot.DATA_DIR", tmp_path),
            patch("kai.bot._handle_workshop_private_text", new_callable=AsyncMock) as execute,
        ):
            await handle_photo(update, ctx)

        ctx.bot.get_file.assert_awaited_once_with("download-capability")
        accepted_message = execution.accept.await_args.args[0]
        artifact = execution.accept.await_args.kwargs["artifact"]
        assert accepted_message == InboundMessage(
            transport="telegram",
            update_id="9001",
            message_id="42",
            sender_subject="1",
            channel_subject="1",
            body="Inspect this detail",
            occurred_at=datetime(2026, 8, 11, 12, 0, tzinfo=UTC),
        )
        assert artifact.kind == "photo"
        assert artifact.media_type == "image/jpeg"
        assert artifact.source_unique_id == "stable-photo-id"
        assert artifact.storage_path.read_bytes() == b"image-data"
        execute.assert_awaited_once_with(
            update,
            ctx,
            chat_id=1,
            run_id=run_id,
            inbound_message_id=inbound_id,
            prompt="Inspect this detail",
        )

    @pytest.mark.asyncio
    async def test_failed_photo_acceptance_discards_new_staging(self, tmp_path):
        update = _make_update(chat_id=1, user_id=1)
        update.message.photo = [MagicMock(file_id="photo", file_unique_id="stable-photo-id")]
        downloaded = MagicMock()
        downloaded.download_as_bytearray = AsyncMock(return_value=bytearray(b"image-data"))
        ctx = _make_context(config=_make_config(allowed_user_ids={1}))
        ctx.bot.get_file = AsyncMock(return_value=downloaded)
        execution = MagicMock()
        execution.accept = AsyncMock(side_effect=RuntimeError("canonical unavailable"))
        ctx.application.core_services.private_text_execution = execution

        with (
            patch("kai.bot.DATA_DIR", tmp_path),
            patch("kai.bot._handle_workshop_private_text", new_callable=AsyncMock) as execute,
        ):
            await handle_photo(update, ctx)

        execute.assert_not_awaited()
        assert not [path for path in (tmp_path / "files").rglob("*") if path.is_file()]
        assert "could not safely accept" in update.message.reply_text.await_args.args[0].lower()

    @pytest.mark.asyncio
    async def test_group_photo_is_rejected_before_download(self):
        update = _make_update(chat_id=-100999, user_id=1)
        update.message.photo = [MagicMock(file_id="photo", file_unique_id="stable-photo-id")]
        ctx = _make_context(config=_make_config(allowed_user_ids={1}))
        ctx.bot.get_file = AsyncMock()

        with patch("kai.bot._check_totp", new_callable=AsyncMock) as totp:
            await handle_photo(update, ctx)

        totp.assert_not_awaited()
        ctx.bot.get_file.assert_not_awaited()
        assert "direct chat" in update.message.reply_text.await_args.args[0]


class TestHandleDocument:
    @staticmethod
    def _setup_doc(update, file_name, mime_type, content):
        update.message.document = MagicMock(
            file_id="document-download-capability",
            file_unique_id="stable-document-id",
            file_name=file_name,
            mime_type=mime_type,
        )
        downloaded = MagicMock()
        downloaded.download_as_bytearray = AsyncMock(return_value=bytearray(content))
        return downloaded

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("file_name", "mime_type", "content", "expected_kind", "expected_body"),
        (
            ("logo.png", "image/png", b"png-data", "document", "What's in this image (logo.png)?"),
            ("script.py", "text/x-python", b"print('hi')", "document", "[file: script.py]"),
            ("archive.zip", "application/zip", b"PK...", "document", "[file: archive.zip]"),
            ("data.txt", "text/plain", b"\xff\xfe", "document", "[file: data.txt]"),
        ),
    )
    async def test_accepts_documents_through_shared_artifact_prompt_policy(
        self,
        tmp_path,
        file_name,
        mime_type,
        content,
        expected_kind,
        expected_body,
    ):
        update = _make_update(chat_id=1, user_id=1)
        downloaded = self._setup_doc(update, file_name, mime_type, content)
        ctx = _make_context(config=_make_config(allowed_user_ids={1}))
        ctx.bot.get_file = AsyncMock(return_value=downloaded)
        execution, inbound_id, run_id = _configure_canonical_media_execution(ctx)

        with (
            patch("kai.bot.DATA_DIR", tmp_path),
            patch("kai.bot._handle_workshop_private_text", new_callable=AsyncMock) as execute,
        ):
            await handle_document(update, ctx)

        accepted_message = execution.accept.await_args.args[0]
        artifact = execution.accept.await_args.kwargs["artifact"]
        assert accepted_message.body == expected_body
        assert artifact.kind == expected_kind
        assert artifact.source_unique_id == "stable-document-id"
        assert artifact.original_filename == file_name
        assert artifact.storage_path.read_bytes() == content
        execute.assert_awaited_once_with(
            update,
            ctx,
            chat_id=1,
            run_id=run_id,
            inbound_message_id=inbound_id,
            prompt=expected_body,
        )

    @pytest.mark.asyncio
    async def test_document_caption_is_canonical_message_body(self, tmp_path):
        update = _make_update(chat_id=1, user_id=1)
        update.message.caption = "Summarize this"
        downloaded = self._setup_doc(update, "report.txt", "text/plain", b"report")
        ctx = _make_context(config=_make_config(allowed_user_ids={1}))
        ctx.bot.get_file = AsyncMock(return_value=downloaded)
        execution, _, _ = _configure_canonical_media_execution(ctx)

        with (
            patch("kai.bot.DATA_DIR", tmp_path),
            patch("kai.bot._handle_workshop_private_text", new_callable=AsyncMock),
        ):
            await handle_document(update, ctx)

        assert execution.accept.await_args.args[0].body == "Summarize this"

    @pytest.mark.asyncio
    async def test_group_document_is_rejected_before_download(self):
        update = _make_update(chat_id=-100999, user_id=1)
        self._setup_doc(update, "report.txt", "text/plain", b"report")
        ctx = _make_context(config=_make_config(allowed_user_ids={1}))
        ctx.bot.get_file = AsyncMock()

        await handle_document(update, ctx)

        ctx.bot.get_file.assert_not_awaited()
        assert "direct chat" in update.message.reply_text.await_args.args[0]


class TestHandleVoice:
    @staticmethod
    def _setup_voice(update):
        update.message.voice = MagicMock(
            file_id="voice-download-capability",
            file_unique_id="stable-voice-id",
            duration=5,
        )
        downloaded = MagicMock()
        downloaded.download_as_bytearray = AsyncMock(return_value=bytearray(b"audio-data"))
        return downloaded

    @pytest.mark.asyncio
    async def test_accepts_transcribed_voice_as_one_canonical_artifact_run(self, tmp_path):
        model_file = tmp_path / "model.bin"
        model_file.touch()
        update = _make_update(chat_id=1, user_id=1)
        downloaded = self._setup_voice(update)
        ctx = _make_context(
            config=_make_config(
                allowed_user_ids={1},
                voice_enabled=True,
                whisper_model_path=model_file,
            )
        )
        ctx.bot.get_file = AsyncMock(return_value=downloaded)
        execution, inbound_id, run_id = _configure_canonical_media_execution(ctx)

        with (
            patch("shutil.which", return_value="/usr/bin/tool"),
            patch("kai.bot.DATA_DIR", tmp_path),
            patch("kai.bot.transcribe_voice", new_callable=AsyncMock, return_value="Hello there"),
            patch("kai.bot._handle_workshop_private_text", new_callable=AsyncMock) as execute,
        ):
            await handle_voice(update, ctx)

        accepted_message = execution.accept.await_args.args[0]
        artifact = execution.accept.await_args.kwargs["artifact"]
        assert accepted_message.body == "Hello there"
        assert artifact.kind == "voice"
        assert artifact.media_type == "audio/ogg"
        assert artifact.source_unique_id == "stable-voice-id"
        assert artifact.storage_path.read_bytes() == b"audio-data"
        assert update.message.reply_text.await_args_list[0].args[0] == "_Heard:_ Hello there"
        execute.assert_awaited_once_with(
            update,
            ctx,
            chat_id=1,
            run_id=run_id,
            inbound_message_id=inbound_id,
            prompt="Hello there",
        )

    @pytest.mark.asyncio
    async def test_transcription_error_does_not_accept_run(self, tmp_path):
        model_file = tmp_path / "model.bin"
        model_file.touch()
        update = _make_update(chat_id=1, user_id=1)
        downloaded = self._setup_voice(update)
        ctx = _make_context(
            config=_make_config(
                allowed_user_ids={1},
                voice_enabled=True,
                whisper_model_path=model_file,
            )
        )
        ctx.bot.get_file = AsyncMock(return_value=downloaded)
        execution, _, _ = _configure_canonical_media_execution(ctx)

        with (
            patch("shutil.which", return_value="/usr/bin/tool"),
            patch("kai.bot.transcribe_voice", new_callable=AsyncMock, side_effect=TranscriptionError("bad audio")),
        ):
            await handle_voice(update, ctx)

        execution.accept.assert_not_awaited()
        assert "Transcription failed" in update.message.reply_text.await_args.args[0]

    @pytest.mark.asyncio
    async def test_empty_transcription_does_not_accept_run(self, tmp_path):
        model_file = tmp_path / "model.bin"
        model_file.touch()
        update = _make_update(chat_id=1, user_id=1)
        downloaded = self._setup_voice(update)
        ctx = _make_context(
            config=_make_config(
                allowed_user_ids={1},
                voice_enabled=True,
                whisper_model_path=model_file,
            )
        )
        ctx.bot.get_file = AsyncMock(return_value=downloaded)
        execution, _, _ = _configure_canonical_media_execution(ctx)

        with (
            patch("shutil.which", return_value="/usr/bin/tool"),
            patch("kai.bot.transcribe_voice", new_callable=AsyncMock, return_value=""),
        ):
            await handle_voice(update, ctx)

        execution.accept.assert_not_awaited()
        assert "make out" in update.message.reply_text.await_args.args[0]

    @pytest.mark.asyncio
    async def test_voice_not_enabled(self):
        update = _make_update(chat_id=1, user_id=1)
        update.message.voice = MagicMock()
        ctx = _make_context(config=_make_config(allowed_user_ids={1}, voice_enabled=False))

        await handle_voice(update, ctx)

        assert "not enabled" in update.message.reply_text.await_args.args[0].lower()

    @pytest.mark.asyncio
    async def test_missing_dependencies(self, tmp_path):
        update = _make_update(chat_id=1, user_id=1)
        update.message.voice = MagicMock()
        ctx = _make_context(
            config=_make_config(
                allowed_user_ids={1},
                voice_enabled=True,
                whisper_model_path=tmp_path / "missing-model",
            )
        )

        with patch("shutil.which", return_value=None):
            await handle_voice(update, ctx)

        reply = update.message.reply_text.await_args.args[0]
        assert "ffmpeg" in reply
        assert "whisper-cpp" in reply
        assert "whisper model" in reply

    @pytest.mark.asyncio
    async def test_group_voice_is_rejected_before_download(self):
        update = _make_update(chat_id=-100999, user_id=1)
        update.message.voice = MagicMock()
        ctx = _make_context(config=_make_config(allowed_user_ids={1}, voice_enabled=True))
        ctx.bot.get_file = AsyncMock()

        await handle_voice(update, ctx)

        ctx.bot.get_file.assert_not_awaited()
        assert "direct chat" in update.message.reply_text.await_args.args[0]


class TestCanonicalTelegramVoiceRendering:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("voice_mode", "request_delivery"),
        (("on", True), ("only", False)),
    )
    async def test_tts_is_transport_rendering_of_canonical_result(
        self,
        voice_mode,
        request_delivery,
    ):
        from kai.bot import _handle_workshop_private_text

        update = _make_update(chat_id=1, user_id=1)
        ctx = _make_context(config=_make_config(allowed_user_ids={1}, tts_enabled=True))
        execution = MagicMock()
        outcomes = []

        async def execute(_run_id, *, stream_observer, success_transformer):
            del stream_observer
            outcomes.append(await success_transformer(AgentResponse(success=True, text="Canonical voice answer")))
            return SimpleNamespace(
                disposition=CanonicalExecutionDisposition.COMPLETED,
                terminal=SimpleNamespace(body="Canonical voice answer"),
                session_id=None,
                selection=None,
                workspace=None,
            )

        execution.execute = AsyncMock(side_effect=execute)
        ctx.application.core_services.private_text_execution = execution

        with (
            patch("kai.bot.sessions.get_setting", new_callable=AsyncMock, return_value=DEFAULT_VOICE),
            patch("kai.bot.synthesize_speech", new_callable=AsyncMock, return_value=b"voice"),
        ):
            await _handle_workshop_private_text(
                update,
                ctx,
                chat_id=1,
                run_id=RunId("run_00000000000000000000000000000001"),
                inbound_message_id=MessageId("msg_00000000000000000000000000000001"),
                prompt="Speak",
                voice_mode=voice_mode,
            )

        assert outcomes[0].body == "Canonical voice answer"
        assert outcomes[0].request_delivery is request_delivery
        ctx.bot.send_voice.assert_awaited_once_with(chat_id=1, voice=b"voice")

    @pytest.mark.asyncio
    async def test_voice_only_synthesis_failure_requests_canonical_text_delivery(self):
        from kai.bot import _handle_workshop_private_text

        update = _make_update(chat_id=1, user_id=1)
        ctx = _make_context(config=_make_config(allowed_user_ids={1}, tts_enabled=True))
        execution = MagicMock()
        outcomes = []

        async def execute(_run_id, *, stream_observer, success_transformer):
            del stream_observer
            outcomes.append(await success_transformer(AgentResponse(success=True, text="Canonical text fallback")))
            return SimpleNamespace(
                disposition=CanonicalExecutionDisposition.COMPLETED,
                terminal=SimpleNamespace(body="Canonical text fallback"),
                session_id=None,
                selection=None,
                workspace=None,
            )

        execution.execute = AsyncMock(side_effect=execute)
        ctx.application.core_services.private_text_execution = execution

        with (
            patch("kai.bot.sessions.get_setting", new_callable=AsyncMock, return_value=DEFAULT_VOICE),
            patch("kai.bot.synthesize_speech", new_callable=AsyncMock, side_effect=TTSError("unavailable")),
            patch("kai.bot._send_response", new_callable=AsyncMock) as direct_fallback,
        ):
            await _handle_workshop_private_text(
                update,
                ctx,
                chat_id=1,
                run_id=RunId("run_00000000000000000000000000000001"),
                inbound_message_id=MessageId("msg_00000000000000000000000000000001"),
                prompt="Speak",
                voice_mode="only",
            )

        assert outcomes[0].request_delivery is True
        ctx.bot.send_voice.assert_not_awaited()
        direct_fallback.assert_not_awaited()


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

    @pytest.mark.asyncio
    async def test_protected_runtime_uses_core_workspace_config_service(self):
        update = _make_update(text="/workspace config model opus")
        ctx = _make_context()
        authority = SimpleNamespace(runtime_profile_id=profile_id(12345))
        service = MagicMock()
        service.authority_for_principal_profile.return_value = authority
        service.set_workspace_config = AsyncMock(
            return_value=WorkspaceConfigSnapshot(
                workspace="/srv/kai",
                model=EffectiveValue("opus", "workspace override", "sonnet"),
                timeout_seconds=EffectiveValue(120, "runtime policy", 120),
                environment_keys=(),
                prompt=None,
                has_prompt=False,
                prompt_source=None,
                override_fields=("model",),
                revision="sws_test",
                capabilities=(),
            )
        )
        ctx.application.core_services.settings_workspaces = service

        await _handle_workspace_config(
            update,
            ctx,
            "config model opus",
        )

        service.set_workspace_config.assert_awaited_once_with(
            authority,
            field="model",
            value="opus",
        )
        assert "opus" in update.message.reply_text.call_args[0][0]

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

    @pytest.mark.asyncio
    async def test_protected_backend_switch_uses_canonical_settings_service(self):
        update = _make_update(text="/settings backend codex:openai")
        ctx = _make_context(args=["backend", "codex:openai"])
        authority = SimpleNamespace(runtime_profile_id=profile_id(12345))
        service = MagicMock()
        service.authority_for_principal_profile.return_value = authority
        service.set_backend = AsyncMock(return_value=SimpleNamespace(backend_option_id="codex:openai"))
        ctx.application.core_services.settings_workspaces = service

        await handle_settings(update, ctx)

        service.set_backend.assert_awaited_once_with(authority, "codex:openai")
        reply = update.message.reply_text.call_args[0][0]
        assert "Backend switched to codex:openai" in reply
        assert "other users were not restarted" in reply

    @pytest.mark.asyncio
    async def test_backend_alias_lists_canonical_options(self):
        update = _make_update(text="/backend")
        ctx = _make_context(args=[])
        authority = SimpleNamespace(runtime_profile_id=profile_id(12345))
        service = MagicMock()
        service.authority_for_principal_profile.return_value = authority
        service.inspect = AsyncMock(
            return_value=SimpleNamespace(
                backend_option_id="claude:anthropic",
                backend_options=(
                    SimpleNamespace(
                        option_id="claude:anthropic",
                        backend="claude",
                        provider="anthropic",
                    ),
                    SimpleNamespace(
                        option_id="opencode:deepseek",
                        backend="opencode",
                        provider="deepseek",
                    ),
                ),
            )
        )
        ctx.application.core_services.settings_workspaces = service

        await handle_backend(update, ctx)

        service.inspect.assert_awaited_once_with(authority)
        reply = update.message.reply_text.call_args[0][0]
        assert "Current backend: claude:anthropic" in reply
        assert "opencode:deepseek" in reply
        assert "Usage: /backend <backend:provider>" in reply

    @pytest.mark.asyncio
    async def test_backend_alias_switches_through_canonical_service(self):
        update = _make_update(text="/backend opencode:deepseek")
        ctx = _make_context(args=["opencode:deepseek"])
        authority = SimpleNamespace(runtime_profile_id=profile_id(12345))
        service = MagicMock()
        service.authority_for_principal_profile.return_value = authority
        service.set_backend = AsyncMock(return_value=SimpleNamespace(backend_option_id="opencode:deepseek"))
        ctx.application.core_services.settings_workspaces = service

        await handle_backend(update, ctx)

        service.set_backend.assert_awaited_once_with(authority, "opencode:deepseek")
        assert "Backend switched to opencode:deepseek" in update.message.reply_text.call_args[0][0]

    @pytest.mark.asyncio
    async def test_backends_command_shows_authorized_options_as_buttons(self):
        update = _make_update(text="/backends")
        ctx = _make_context()
        authority = SimpleNamespace(runtime_profile_id=profile_id(12345))
        snapshot = SimpleNamespace(
            backend_option_id="claude:anthropic",
            backend_options=(
                SimpleNamespace(
                    option_id="claude:anthropic",
                    backend="claude",
                    provider="anthropic",
                ),
                SimpleNamespace(
                    option_id="opencode:deepseek",
                    backend="opencode",
                    provider="deepseek",
                ),
            ),
        )
        service = MagicMock()
        service.authority_for_principal_profile.return_value = authority
        service.inspect = AsyncMock(return_value=snapshot)
        ctx.application.core_services.settings_workspaces = service

        await handle_backends(update, ctx)

        markup = update.message.reply_text.call_args.kwargs["reply_markup"]
        buttons = [row[0] for row in markup.inline_keyboard]
        assert [button.callback_data for button in buttons] == [
            "backend:claude:anthropic",
            "backend:opencode:deepseek",
        ]
        assert buttons[0].text.endswith("\U0001f7e2")

    @pytest.mark.asyncio
    async def test_backend_callback_switches_through_canonical_service(self):
        update = _make_callback_update(data="backend:opencode:deepseek")
        ctx = _make_context()
        authority = SimpleNamespace(runtime_profile_id=profile_id(12345))
        service = MagicMock()
        service.authority_for_principal_profile.return_value = authority
        service.inspect = AsyncMock(return_value=SimpleNamespace(backend_option_id="claude:anthropic"))
        service.set_backend = AsyncMock(return_value=SimpleNamespace(backend_option_id="opencode:deepseek"))
        ctx.application.core_services.settings_workspaces = service

        await handle_backend_callback(update, ctx)

        service.set_backend.assert_awaited_once_with(authority, "opencode:deepseek")
        update.callback_query.answer.assert_awaited_once_with()
        assert "Switched to opencode:deepseek" in update.callback_query.edit_message_text.call_args[0][0]

    @pytest.mark.asyncio
    async def test_backend_switch_conflict_is_reported_without_success(self):
        update = _make_update(text="/settings backend codex")
        ctx = _make_context(args=["backend", "codex"])
        authority = SimpleNamespace(runtime_profile_id=profile_id(12345))
        service = MagicMock()
        service.authority_for_principal_profile.return_value = authority
        service.set_backend = AsyncMock(side_effect=WorkshopSettingsWorkspaceConflict("active run"))
        ctx.application.core_services.settings_workspaces = service

        await handle_settings(update, ctx)

        assert update.message.reply_text.call_args[0][0] == "active run"

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

    @pytest.mark.asyncio
    async def test_show_settings_ignores_invalid_protected_model_override(self):
        """The display matches the model that protected execution will use."""
        update = _make_update(text="/settings")
        config = _make_config(protected_install=True)
        pool = _make_mock_claude()
        pool.get_runtime_profile.return_value = profile_registry(12345).profile_for_legacy_runtime_key(12345)
        pool.get_backend_provider.return_value = ("codex", "openai")
        ctx = _make_context(config=config, pool=pool)
        mock_sessions = self._mock_sessions({"model": "opus"})

        with self._patches(mock_sessions):
            await _show_settings(update, ctx, 12345, config)

        reply = update.message.reply_text.call_args[0][0]
        assert "gpt-5.6-sol (runtime policy)" in reply
        assert "opus (user override)" not in reply
        assert "openai" in reply.lower()

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

    @pytest.mark.asyncio
    async def test_protected_timeout_uses_canonical_policy_above_compatibility_cap(self):
        update = _make_update(text="/settings timeout 1800")
        ctx = _make_context(args=["timeout", "1800"])
        authority = SimpleNamespace(runtime_profile_id=profile_id(12345))
        service = MagicMock()
        service.authority_for_principal_profile.return_value = authority
        service.set_timeout = AsyncMock()
        ctx.application.core_services.settings_workspaces = service

        await handle_settings(update, ctx)

        service.set_timeout.assert_awaited_once_with(authority, 1800)
        assert "1800s" in update.message.reply_text.call_args[0][0]

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

    # ── 3. /github notify <number> ───────────────────────────────

    @pytest.mark.asyncio
    async def test_notify_lists_only_canonical_display_choices(self):
        update = _make_update(text="/github notify")
        ctx = _make_context(config=_make_config(), args=["notify"])
        service = MagicMock()
        service.authority_for_principal_profile.return_value = object()
        service.inspect = AsyncMock(
            return_value=SimpleNamespace(
                revision="revision-1",
                preferences=(
                    SimpleNamespace(
                        integration_class="github",
                        destination_name="Notifications",
                        source="protected policy",
                    ),
                ),
                destinations=(
                    SimpleNamespace(
                        choice_id="ndst_home",
                        display_name="Home",
                        kind="direct",
                        supported_classes=("github",),
                    ),
                    SimpleNamespace(
                        choice_id="ndst_notifications",
                        display_name="Notifications",
                        kind="notification",
                        supported_classes=("github",),
                    ),
                ),
            )
        )
        ctx.application.core_services.notification_preferences = service

        await handle_github(update, ctx)

        reply = update.message.reply_text.call_args[0][0]
        assert "1. Home (direct)" in reply
        assert "2. Notifications (notification)" in reply
        assert "ndst_" not in reply
        assert "telegram" not in reply.lower()

    @pytest.mark.asyncio
    async def test_token_store_deletes_source_message(self):
        """/github token stores the PAT and deletes the Telegram command."""
        update = _make_update(text="/github token ghp_secret")
        config = _make_config()
        ctx = _make_context(config=config, args=["token", "ghp_secret"])
        mock_sessions = AsyncMock()
        mock_sessions.set_github_toggle = AsyncMock()

        with patch("kai.bot.sessions", mock_sessions):
            await handle_github(update, ctx)

        mock_sessions.set_github_token.assert_called_once_with(12345, "ghp_secret")
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
        mock_sessions.set_github_token = AsyncMock()

        with patch("kai.bot.sessions", mock_sessions):
            await handle_github(update, ctx)

        mock_sessions.set_github_token.assert_called_once_with(12345, "ghp_secret")
        update.message.delete.assert_awaited_once()
        reply = update.message.reply_text.call_args[0][0]
        assert "stored" in reply.lower()

    @pytest.mark.asyncio
    async def test_notify_set(self):
        """/github notify 2 selects a server-authorized canonical destination."""
        update = _make_update(text="/github notify 2")
        config = _make_config()
        ctx = _make_context(config=config, args=["notify", "2"])
        mock_sessions = AsyncMock()
        authority = object()
        service = MagicMock()
        service.authority_for_principal_profile.return_value = authority
        snapshot = SimpleNamespace(
            revision="revision-1",
            preferences=(
                SimpleNamespace(
                    integration_class="github",
                    destination_name="Home",
                    source="protected policy",
                ),
            ),
            destinations=(
                SimpleNamespace(
                    choice_id="ndst_home",
                    display_name="Home",
                    kind="direct",
                    supported_classes=("github",),
                ),
                SimpleNamespace(
                    choice_id="ndst_notifications",
                    display_name="Notifications",
                    kind="notification",
                    supported_classes=("github",),
                ),
            ),
        )
        selected = SimpleNamespace(
            preferences=(
                SimpleNamespace(
                    integration_class="github",
                    destination_name="Notifications",
                ),
            ),
        )
        service.inspect = AsyncMock(return_value=snapshot)
        service.select = AsyncMock(return_value=selected)
        ctx.application.core_services.notification_preferences = service

        with patch("kai.bot.sessions", mock_sessions):
            await handle_github(update, ctx)

        service.select.assert_awaited_once_with(
            authority,
            "github",
            "ndst_notifications",
            expected_revision="revision-1",
        )
        reply = update.message.reply_text.call_args[0][0]
        assert "Notifications" in reply

    # ── 4. /github notify reset ───────────────────────────────────

    @pytest.mark.asyncio
    async def test_notify_reset(self):
        """/github notify reset restores protected canonical policy."""
        update = _make_update(text="/github notify reset")
        config = _make_config()
        ctx = _make_context(config=config, args=["notify", "reset"])
        mock_sessions = AsyncMock()
        authority = object()
        service = MagicMock()
        service.authority_for_principal_profile.return_value = authority
        service.inspect = AsyncMock(
            return_value=SimpleNamespace(
                revision="revision-1",
                preferences=(
                    SimpleNamespace(
                        integration_class="github",
                        destination_name="Notifications",
                        source="personal override",
                    ),
                ),
                destinations=(),
            )
        )
        service.reset = AsyncMock(
            return_value=SimpleNamespace(
                preferences=(
                    SimpleNamespace(
                        integration_class="github",
                        destination_name="Home",
                        source="protected policy",
                    ),
                ),
            )
        )
        ctx.application.core_services.notification_preferences = service

        with patch("kai.bot.sessions", mock_sessions):
            await handle_github(update, ctx)

        service.reset.assert_awaited_once_with(
            authority,
            "github",
            expected_revision="revision-1",
        )
        reply = update.message.reply_text.call_args[0][0]
        assert "reset to Home" in reply

    # ── 5. /github notify abc (invalid) ───────────────────────────

    @pytest.mark.asyncio
    async def test_notify_invalid(self):
        """/github notify abc is rejected because it is not a displayed number."""
        update = _make_update(text="/github notify abc")
        config = _make_config()
        ctx = _make_context(config=config, args=["notify", "abc"])
        mock_sessions = AsyncMock()
        service = MagicMock()
        service.authority_for_principal_profile.return_value = object()
        service.inspect = AsyncMock(
            return_value=SimpleNamespace(
                revision="revision-1",
                preferences=(
                    SimpleNamespace(
                        integration_class="github",
                        destination_name="Home",
                        source="protected policy",
                    ),
                ),
                destinations=(),
            )
        )
        ctx.application.core_services.notification_preferences = service

        with patch("kai.bot.sessions", mock_sessions):
            await handle_github(update, ctx)

        reply = update.message.reply_text.call_args[0][0]
        assert "number" in reply.lower()

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

        mock_sessions.set_github_toggle.assert_called_once_with(12345, "pr_review", True)
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
        mock_sessions.set_github_toggle = AsyncMock()

        with patch("kai.bot.sessions", mock_sessions):
            await handle_github(update, ctx)

        mock_sessions.set_github_toggle.assert_called_once_with(12345, "pr_review", False)
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
        mock_sessions.set_github_toggle = AsyncMock()

        with patch("kai.bot.sessions", mock_sessions):
            await handle_github(update, ctx)

        mock_sessions.set_github_toggle.assert_called_once_with(12345, "issue_triage", True)
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
    "backend",
    "backends",
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
        app = create_bot(config, core_services=_bot_core_services(config))

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

        config = _make_config()
        app = create_bot(config, core_services=_bot_core_services(config))
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

        config = _make_config()
        app = create_bot(config, core_services=_bot_core_services(config))
        callbacks = [
            handler.callback
            for group_handlers in app.handlers.values()
            for handler in group_handlers
            if isinstance(handler, CQH)
        ]

        assert len(callbacks) == 5
        assert all(getattr(callback, "_kai_totp_sensitive", False) for callback in callbacks)

    def test_menu_matches_expected_set(self):
        """The adapter-owned Telegram command menu matches the contract."""
        from kai.telegram_adapter import _TELEGRAM_COMMANDS

        menu_commands = {command.command for command in _TELEGRAM_COMMANDS}

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
    async def test_stages_timestamped_copy_under_principal_files(self, tmp_path, monkeypatch):
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

        namespace = ctx.application.core_services.principal_storage.for_runtime_config_id(1)
        staged_dir = namespace.files_directory(tmp_path)
        assert staged_dir.is_dir()
        staged = list(staged_dir.glob("*_pr-681-review.md"))
        assert len(staged) == 1, f"expected one timestamped staged copy, got {staged}"

    @pytest.mark.asyncio
    async def test_group_review_staging_belongs_to_human_actor(
        self,
        tmp_path,
        monkeypatch,
    ):
        """A notification-group chat ID never becomes a storage owner."""
        monkeypatch.setattr("kai.bot.DATA_DIR", tmp_path)
        monkeypatch.setattr("kai.bot._REVIEW_TMP_DIR", tmp_path)
        update = _make_update(chat_id=-100999, user_id=1)
        ctx = _make_context(
            config=_review_command_config(),
            args=["dcellison/kai", "681"],
        )
        ctx.bot.send_document = AsyncMock()

        with (
            patch("kai.bot._check_totp", new=AsyncMock(return_value=True)),
            patch(
                "kai.bot.review.generate_pr_review",
                new=AsyncMock(return_value=_review_result()),
            ),
        ):
            await handle_review_command(update, ctx)

        namespace = ctx.application.core_services.principal_storage.for_runtime_config_id(1)
        staged = list(namespace.files_directory(tmp_path).glob("*_pr-681-review.md"))
        assert len(staged) == 1
        assert not (tmp_path / "files" / "-100999").exists()
        ctx.bot.send_document.assert_awaited_once()
        assert ctx.bot.send_document.await_args.args[0] == -100999

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
    async def test_protected_review_uses_pool_execution_policy(self, tmp_path, monkeypatch):
        """Compatibility fields cannot select the protected review subprocess."""
        monkeypatch.setattr("kai.bot.DATA_DIR", tmp_path)
        monkeypatch.setattr("kai.bot._REVIEW_TMP_DIR", tmp_path)
        update = _make_update(user_id=1)
        config = _make_config(
            protected_install=True,
            allowed_user_ids={1},
            user_configs={
                1: UserConfig(
                    telegram_id=1,
                    name="operator",
                    github_repos=["dcellison/kai"],
                    backend="claude",
                    provider="anthropic",
                    os_user="compatibility-user",
                    models={"pr_review": "opus"},
                )
            },
        )
        pool = _make_mock_claude()
        pool.get_os_user.return_value = "policy-user"
        pool.get_backend_provider.return_value = ("codex", "openai")
        pool.get_role_model.return_value = "gpt-5.5"
        ctx = _make_context(
            config=config,
            pool=pool,
            args=["dcellison/kai", "681"],
        )
        ctx.bot.send_document = AsyncMock()

        with (
            patch("kai.bot._check_totp", new=AsyncMock(return_value=True)),
            patch("kai.bot.sessions.get_setting", new=AsyncMock(return_value="ghp_user")),
            patch(
                "kai.bot.review.generate_pr_review",
                new=AsyncMock(return_value=_review_result()),
            ) as mock_generate,
        ):
            await handle_review_command(update, ctx)

        kwargs = mock_generate.call_args.kwargs
        assert kwargs["claude_user"] == "policy-user"
        assert kwargs["agent_backend"] == "codex"
        assert kwargs["provider"] == "openai"
        assert kwargs["model_override"] == "gpt-5.5"
        pool.get_os_user.assert_called_once_with(1)
        pool.get_backend_provider.assert_called_once_with(1)
        pool.get_role_model.assert_called_once_with(1, ModelRole.PR_REVIEW)

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
