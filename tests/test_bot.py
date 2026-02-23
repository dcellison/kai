"""Tests for bot.py pure functions."""

from pathlib import Path

from kai.bot import (
    _chunk_text,
    _resolve_workspace_path,
    _save_to_workspace,
    _short_workspace_name,
    _truncate_for_telegram,
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
