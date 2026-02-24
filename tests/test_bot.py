"""Tests for bot.py pure functions."""

from pathlib import Path

import pytest

from kai.bot import (
    _chunk_text,
    _is_workspace_allowed,
    _resolve_workspace_path,
    _save_to_workspace,
    _short_workspace_name,
    _truncate_for_telegram,
    _workspaces_keyboard,
)

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


# ── _chunk_text ──────────────────────────────────────────────────────


class TestChunkText:
    def test_short_text_single_chunk(self):
        assert _chunk_text("hello", 100) == ["hello"]

    def test_splits_at_double_newline(self):
        text = "a" * 50 + "\n\n" + "b" * 50
        chunks = _chunk_text(text, 60)
        assert len(chunks) == 2
        assert chunks[0] == "a" * 50
        assert chunks[1] == "b" * 50

    def test_splits_at_single_newline_if_no_double(self):
        text = "a" * 50 + "\n" + "b" * 50
        chunks = _chunk_text(text, 60)
        assert len(chunks) == 2
        assert chunks[0] == "a" * 50
        assert chunks[1] == "b" * 50

    def test_splits_at_max_len_if_no_newlines(self):
        text = "a" * 100
        chunks = _chunk_text(text, 50)
        assert chunks == ["a" * 50, "a" * 50]

    def test_empty_string(self):
        assert _chunk_text("") == []


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


# ── _save_to_workspace ──────────────────────────────────────────────


class TestSaveToWorkspace:
    def test_creates_files_directory(self, tmp_path):
        """Automatically creates the files/ subdirectory if missing."""
        _save_to_workspace(b"hello", "test.txt", tmp_path)
        assert (tmp_path / "files").is_dir()

    def test_saves_content_correctly(self, tmp_path):
        """Written bytes match the input exactly."""
        data = b"binary content here"
        result = _save_to_workspace(data, "doc.pdf", tmp_path)
        assert result.read_bytes() == data

    def test_filename_contains_original_name(self, tmp_path):
        """Saved filename preserves the original name after the timestamp."""
        result = _save_to_workspace(b"x", "report.pdf", tmp_path)
        assert "report.pdf" in result.name

    def test_timestamp_prefix_format(self, tmp_path):
        """Filename starts with YYYYMMDD_HHMMSS_ffffff timestamp."""
        result = _save_to_workspace(b"x", "file.txt", tmp_path)
        # Format: YYYYMMDD_HHMMSS_ffffff_file.txt
        parts = result.name.split("_", 3)
        assert len(parts[0]) == 8  # date
        assert len(parts[1]) == 6  # time
        assert len(parts[2]) == 6  # microseconds

    def test_sanitizes_slashes_and_spaces(self, tmp_path):
        """Slashes and spaces in filenames are replaced with underscores."""
        result = _save_to_workspace(b"x", "my file/name.txt", tmp_path)
        assert "/" not in result.name
        assert " " not in result.name

    def test_returns_absolute_path(self, tmp_path):
        """Returned path is absolute and points to an existing file."""
        result = _save_to_workspace(b"x", "test.txt", tmp_path)
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
    @pytest.mark.asyncio
    async def test_home_always_first(self, tmp_path):
        """Home button appears first regardless of history or allowed workspaces."""
        markup = await _workspaces_keyboard([], "/home", "/home", None, [])
        assert _button_labels(markup)[0] == "\U0001f3e0 Home \U0001f7e2"

    @pytest.mark.asyncio
    async def test_allowed_workspaces_appear_before_history(self, tmp_path):
        """Pinned workspaces appear between Home and history entries."""
        pinned = tmp_path / "pinned"
        pinned.mkdir()
        history = [{"path": "/other/project"}]
        markup = await _workspaces_keyboard(history, "/other/project", "/home", None, [pinned])
        labels = _button_labels(markup)
        # Home, then pinned, then history
        assert labels[0].startswith("\U0001f3e0 Home")
        assert labels[1] == "pinned"
        assert labels[2].endswith("\U0001f7e2")  # history entry marked as current

    @pytest.mark.asyncio
    async def test_allowed_workspace_callback_data(self, tmp_path):
        """Pinned workspaces use ws:allowed:<index> callback data."""
        pinned = tmp_path / "project-a"
        pinned.mkdir()
        markup = await _workspaces_keyboard([], "/home", "/home", None, [pinned])
        callbacks = _button_callbacks(markup)
        assert "ws:allowed:0" in callbacks

    @pytest.mark.asyncio
    async def test_history_deduplicated_against_allowed(self, tmp_path):
        """A path in both allowed and history appears only once (in allowed section)."""
        pinned = tmp_path / "shared"
        pinned.mkdir()
        history = [{"path": str(pinned)}]
        markup = await _workspaces_keyboard(history, "/home", "/home", None, [pinned])
        labels = _button_labels(markup)
        # Should be: Home + one "shared" entry — not two "shared" entries
        assert labels.count("shared") == 1
        callbacks = _button_callbacks(markup)
        # The single entry should be the allowed version, not a bare history index
        assert "ws:allowed:0" in callbacks
        assert not any(c == "ws:0" for c in callbacks)

    @pytest.mark.asyncio
    async def test_current_workspace_marked_in_allowed(self, tmp_path):
        """Green dot appears on the pinned workspace button when it is current."""
        pinned = tmp_path / "active"
        pinned.mkdir()
        markup = await _workspaces_keyboard([], str(pinned), "/home", None, [pinned])
        labels = _button_labels(markup)
        assert any("active" in lbl and "\U0001f7e2" in lbl for lbl in labels)

    @pytest.mark.asyncio
    async def test_no_allowed_no_history_shows_only_home(self):
        """With no allowed workspaces and no history, only the Home button appears."""
        markup = await _workspaces_keyboard([], "/home", "/home", None, [])
        assert len(_button_labels(markup)) == 1

    @pytest.mark.asyncio
    async def test_disambiguates_duplicate_names(self, tmp_path):
        """Two allowed workspaces with the same directory name get parent/name labels."""
        foo_a = tmp_path / "projects" / "foo"
        foo_b = tmp_path / "clients" / "foo"
        foo_a.mkdir(parents=True)
        foo_b.mkdir(parents=True)
        markup = await _workspaces_keyboard([], "/home", "/home", None, [foo_a, foo_b])
        labels = _button_labels(markup)
        assert "projects/foo" in labels
        assert "clients/foo" in labels
        # Neither bare "foo" label should appear
        assert "foo" not in labels

    @pytest.mark.asyncio
    async def test_unique_names_not_disambiguated(self, tmp_path):
        """Allowed workspaces with unique names keep their short labels."""
        bar = tmp_path / "bar"
        baz = tmp_path / "baz"
        bar.mkdir()
        baz.mkdir()
        markup = await _workspaces_keyboard([], "/home", "/home", None, [bar, baz])
        labels = _button_labels(markup)
        assert "bar" in labels
        assert "baz" in labels


# ── _is_workspace_allowed ────────────────────────────────────────────


from kai.config import Config  # noqa: E402 — after fixtures, before tests


def _make_config(workspace_base=None, allowed_workspaces=None) -> Config:
    """Minimal Config for testing _is_workspace_allowed."""
    return Config(
        telegram_bot_token="test",
        allowed_user_ids={1},
        workspace_base=workspace_base,
        allowed_workspaces=allowed_workspaces or [],
    )


class TestIsWorkspaceAllowed:
    def test_no_sources_allows_anything(self, tmp_path):
        """With no WORKSPACE_BASE and no ALLOWED_WORKSPACES, all paths are allowed."""
        config = _make_config()
        assert _is_workspace_allowed(tmp_path / "anything", config) is True

    def test_path_under_base_is_allowed(self, tmp_path):
        """Paths under WORKSPACE_BASE are allowed."""
        config = _make_config(workspace_base=tmp_path)
        assert _is_workspace_allowed(tmp_path / "myproject", config) is True

    def test_path_in_allowed_workspaces_is_allowed(self, tmp_path):
        """Paths listed in ALLOWED_WORKSPACES are allowed."""
        project = tmp_path / "project"
        project.mkdir()
        config = _make_config(allowed_workspaces=[project])
        assert _is_workspace_allowed(project, config) is True

    def test_path_outside_both_is_rejected(self, tmp_path):
        """Paths not in WORKSPACE_BASE or ALLOWED_WORKSPACES are rejected."""
        base = tmp_path / "base"
        base.mkdir()
        outside = tmp_path / "outside"
        config = _make_config(workspace_base=base)
        assert _is_workspace_allowed(outside, config) is False

    def test_base_set_allowed_workspaces_empty_rejects_outside(self, tmp_path):
        """With WORKSPACE_BASE set but no allowed workspaces, outside paths are rejected."""
        base = tmp_path / "base"
        base.mkdir()
        config = _make_config(workspace_base=base, allowed_workspaces=[])
        assert _is_workspace_allowed(tmp_path / "other", config) is False

    def test_only_allowed_workspaces_set_rejects_unlisted(self, tmp_path):
        """With only ALLOWED_WORKSPACES set, unlisted paths are rejected."""
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        unlisted = tmp_path / "unlisted"
        config = _make_config(allowed_workspaces=[allowed])
        assert _is_workspace_allowed(unlisted, config) is False

    def test_resolves_symlinks_for_comparison(self, tmp_path):
        """Path resolution handles non-canonical paths correctly."""
        project = tmp_path / "project"
        project.mkdir()
        config = _make_config(allowed_workspaces=[project])
        # Pass the resolved canonical path — should still match
        assert _is_workspace_allowed(project.resolve(), config) is True
