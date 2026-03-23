"""Tests for telegram_utils.py shared Telegram messaging utilities."""

from kai.telegram_utils import chunk_text


class TestChunkText:
    def test_short_text_single_chunk(self):
        assert chunk_text("hello", 100) == ["hello"]

    def test_splits_at_double_newline(self):
        text = "a" * 50 + "\n\n" + "b" * 50
        chunks = chunk_text(text, 60)
        assert len(chunks) == 2
        assert chunks[0] == "a" * 50
        assert chunks[1] == "b" * 50

    def test_splits_at_single_newline_if_no_double(self):
        text = "a" * 50 + "\n" + "b" * 50
        chunks = chunk_text(text, 60)
        assert len(chunks) == 2
        assert chunks[0] == "a" * 50
        assert chunks[1] == "b" * 50

    def test_splits_at_max_len_if_no_newlines(self):
        text = "a" * 100
        chunks = chunk_text(text, 50)
        assert chunks == ["a" * 50, "a" * 50]

    def test_empty_string(self):
        assert chunk_text("") == []

    def test_exactly_4096_no_split(self):
        """A string of exactly 4096 characters returns a single chunk."""
        text = "x" * 4096
        chunks = chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_leading_double_newline_no_empty_chunk(self):
        """Text starting with \\n\\n doesn't produce an empty first chunk."""
        text = "\n\n" + "a" * 5000
        chunks = chunk_text(text)
        assert all(c and len(c) <= 4096 for c in chunks)
        # Content preserved (leading newlines stripped, nothing lost)
        assert "".join(chunks) == "a" * 5000

    def test_4097_triggers_split(self):
        """A string of 4097 characters with a newline is split into two chunks."""
        text = "a" * 2000 + "\n" + "b" * 2096
        chunks = chunk_text(text)
        assert len(chunks) == 2
        assert chunks[0] == "a" * 2000
        assert chunks[1] == "b" * 2096
