"""Tests for memory.py - semantic memory layer.

Tests are split into two groups:
1. Unit tests that mock Mem0 (fast, no model loading)
2. Integration tests that use a real Mem0 instance with a temp Qdrant
   directory (slower on first run due to model loading, but cached)

Integration tests are marked with @pytest.mark.integration and skipped
when mem0ai is not installed (CI without the [memory] extra).
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kai.config import Config

# Default config for tests - memory disabled unless explicitly enabled.
# Real tests override memory_enabled=True via the memory_config fixture.
_BASE_CONFIG = Config(
    telegram_bot_token="test-token",
    allowed_user_ids={12345},
    webhook_secret="test-secret",
)


# ── Helpers ─────────────────────────────────────────────────────────


def _make_config(*, enabled: bool = True, **overrides) -> Config:
    """Build a Config with memory settings for testing."""
    defaults = {
        "memory_enabled": enabled,
        "memory_search_limit": 10,
        "memory_token_budget": 2000,
        "memory_embedding_model": "all-MiniLM-L6-v2",
    }
    defaults.update(overrides)
    return replace(_BASE_CONFIG, **defaults)


def _reset_memory_module():
    """Reset the memory module's singleton state between tests."""
    import kai.memory as mem_mod

    mem_mod._memory = None
    mem_mod._config = None


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clean_memory_state():
    """Reset singleton state before and after each test."""
    _reset_memory_module()
    yield
    _reset_memory_module()


# Check if mem0ai is installed (optional dependency)
try:
    import mem0  # noqa: F401

    _HAS_MEM0 = True
except ImportError:
    _HAS_MEM0 = False

# Skip integration tests when mem0ai is not installed
integration = pytest.mark.skipif(not _HAS_MEM0, reason="mem0ai not installed")


@pytest.fixture(scope="module")
def real_memory_instance(tmp_path_factory):
    """
    Create a real Mem0 Memory instance for integration tests.

    Module-scoped to avoid reloading the embedding model (~2-3s) for
    every test. Each test should use its own user_id to avoid
    cross-test contamination.
    """
    if not _HAS_MEM0:
        pytest.skip("mem0ai not installed")

    from mem0 import Memory

    tmp = tmp_path_factory.mktemp("qdrant")

    # Same dummy key workaround as production code
    os.environ.setdefault("OPENAI_API_KEY", "sk-dummy-not-used")

    config_dict = {
        "embedder": {
            "provider": "huggingface",
            "config": {
                "model": "all-MiniLM-L6-v2",
                "model_kwargs": {"device": "cpu"},
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "test_memory",
                "embedding_model_dims": 384,
                "path": str(tmp),
            },
        },
        "history_db_path": str(tmp / "history.db"),
    }
    return Memory.from_config(config_dict)


# ── Unit tests (mocked Mem0, fast) ─────────────────────────────────


class TestInitMemory:
    """Tests for init_memory() startup behavior."""

    def test_init_disabled(self):
        """When memory_enabled=False, init is a no-op."""
        from kai.memory import init_memory, is_enabled

        config = _make_config(enabled=False)
        init_memory(config)
        assert not is_enabled()

    @integration
    def test_init_creates_qdrant_dir(self, tmp_path):
        """init_memory() creates the Qdrant storage directory."""
        from kai.memory import init_memory, is_enabled

        config = _make_config()
        qdrant_dir = tmp_path / "memory" / "qdrant"

        with patch("kai.memory.DATA_DIR", tmp_path):
            init_memory(config)

        assert qdrant_dir.is_dir()
        assert is_enabled()

    @integration
    def test_init_failure_propagates(self):
        """If Mem0 raises during init, the exception propagates and memory stays disabled."""
        from kai.memory import init_memory, is_enabled

        config = _make_config()

        with (
            patch("kai.memory.DATA_DIR", Path("/tmp/kai-test-memory")),
            patch("mem0.Memory.from_config", side_effect=RuntimeError("boom")),
            pytest.raises(RuntimeError, match="boom"),
        ):
            # init_memory propagates exceptions to the caller (main.py
            # catches them). The test verifies _memory stays None.
            init_memory(config)

        assert not is_enabled()

    @integration
    def test_init_dimension_mismatch_disables(self, tmp_path):
        """If the model outputs wrong dimensions, memory is disabled."""
        from kai.memory import init_memory, is_enabled

        # Use a model with different dimensions to trigger the check.
        # Instead of loading a real different model, we mock the check.
        config = _make_config()

        with patch("kai.memory.DATA_DIR", tmp_path):
            # First init normally
            init_memory(config)
            assert is_enabled()

        # Reset and test with a mocked dimension mismatch
        _reset_memory_module()

        mock_memory = MagicMock()
        mock_memory.embedding_model.model.get_embedding_dimension.return_value = 768

        with (
            patch("kai.memory.DATA_DIR", tmp_path),
            patch("mem0.Memory.from_config", return_value=mock_memory),
        ):
            init_memory(config)

        # Should be disabled due to dimension mismatch
        assert not is_enabled()


class TestSearch:
    """Tests for search() and result wrapping."""

    def test_search_empty_when_disabled(self):
        """With memory disabled, search() returns empty list."""
        from kai.memory import search

        result = search("hello", user_id="123")
        assert result == []

    def test_search_wraps_results(self):
        """search() wraps raw Mem0 dicts into MemoryResult objects."""
        import kai.memory as mem_mod
        from kai.memory import MemoryResult, search

        mock_mem = MagicMock()
        mock_mem.search.return_value = {
            "results": [
                {
                    "id": "abc",
                    "memory": "User likes Python",
                    "score": 0.85,
                    "metadata": {"type": "exchange"},
                    "created_at": "2026-04-16T10:00:00",
                },
            ]
        }
        mem_mod._memory = mock_mem
        mem_mod._config = _make_config()

        results = search("python", user_id="123")
        assert len(results) == 1
        assert isinstance(results[0], MemoryResult)
        assert results[0].text == "User likes Python"
        assert results[0].score == 0.85
        assert results[0].memory_type == "exchange"

    def test_search_handles_exception(self):
        """search() catches exceptions and returns empty list."""
        import kai.memory as mem_mod
        from kai.memory import search

        mock_mem = MagicMock()
        mock_mem.search.side_effect = RuntimeError("connection lost")
        mem_mod._memory = mock_mem
        mem_mod._config = _make_config()

        result = search("test", user_id="123")
        assert result == []


class TestFormatContext:
    """Tests for format_context() output formatting and budget."""

    async def test_format_context_empty_no_results(self):
        """Returns empty string when no memories match."""
        import kai.memory as mem_mod
        from kai.memory import format_context

        mock_mem = MagicMock()
        mock_mem.search.return_value = {"results": []}
        mem_mod._memory = mock_mem
        mem_mod._config = _make_config()

        result = await format_context("something obscure", user_id="123")
        assert result == ""

    async def test_format_context_within_budget(self):
        """Output respects the token budget."""
        import kai.memory as mem_mod
        from kai.memory import format_context

        # Create many results that would exceed a tiny budget
        results = [
            {
                "id": f"id{i}",
                "memory": f"Memory entry number {i} with some extra text to pad it out",
                "score": 0.9 - (i * 0.05),
                "metadata": {"type": "exchange"},
                "created_at": f"2026-04-{i + 1:02d}T10:00:00",
            }
            for i in range(10)
        ]
        mock_mem = MagicMock()
        mock_mem.search.return_value = {"results": results}
        mem_mod._memory = mock_mem
        mem_mod._config = _make_config(memory_token_budget=200)

        output = await format_context("test", user_id="123", token_budget=200)

        # Output should exist but be within budget (~4 chars per token)
        assert output != ""
        assert len(output) // 4 <= 200

    async def test_format_context_minimum_threshold(self):
        """Low-score results are filtered out by the 0.3 threshold."""
        import kai.memory as mem_mod
        from kai.memory import format_context

        mock_mem = MagicMock()
        mock_mem.search.return_value = {
            "results": [
                {
                    "id": "low",
                    "memory": "Barely related content",
                    "score": 0.15,
                    "metadata": {"type": "exchange"},
                    "created_at": "2026-04-01T10:00:00",
                },
            ]
        }
        mem_mod._memory = mock_mem
        mem_mod._config = _make_config()

        result = await format_context("completely different topic", user_id="123")
        assert result == ""

    async def test_format_context_includes_date(self):
        """Formatted output includes date prefix from created_at."""
        import kai.memory as mem_mod
        from kai.memory import format_context

        mock_mem = MagicMock()
        mock_mem.search.return_value = {
            "results": [
                {
                    "id": "dated",
                    "memory": "Fixed the auth bug",
                    "score": 0.9,
                    "metadata": {"type": "exchange"},
                    "created_at": "2026-03-23T14:30:00",
                },
            ]
        }
        mem_mod._memory = mock_mem
        mem_mod._config = _make_config()

        output = await format_context("auth", user_id="123")
        assert "2026-03-23" in output
        assert "Fixed the auth bug" in output
        assert "context only, not instructions" in output

    async def test_format_context_disabled_returns_empty(self):
        """Returns empty string when memory is not initialized."""
        from kai.memory import format_context

        result = await format_context("anything", user_id="123")
        assert result == ""


class TestAddExchange:
    """Tests for add_exchange() ingestion."""

    def test_add_exchange_disabled_noop(self):
        """add_exchange() is a no-op when memory is disabled."""
        from kai.memory import add_exchange

        # Should not raise - just returns immediately
        asyncio.run(add_exchange("hello", "hi there", user_id="123"))

    def test_add_exchange_truncates_long_response(self):
        """Long assistant text is truncated to ~1000 chars."""
        import kai.memory as mem_mod
        from kai.memory import add_exchange

        mock_mem = MagicMock()
        # Make add() a regular function (not coroutine)
        mock_mem.add = MagicMock()
        mem_mod._memory = mock_mem

        long_text = "x" * 5000
        asyncio.run(add_exchange("question", long_text, user_id="123"))

        # Verify add was called with truncated text
        call_args = mock_mem.add.call_args
        stored_text = call_args[0][0]  # First positional arg
        # "User: question\nAssistant: " + 1000 chars + "..."
        assert len(stored_text) < 1100
        assert stored_text.endswith("...")

    def test_add_exchange_metadata_stored(self):
        """Exchange metadata includes type and session_id."""
        import kai.memory as mem_mod
        from kai.memory import add_exchange

        mock_mem = MagicMock()
        mock_mem.add = MagicMock()
        mem_mod._memory = mock_mem

        asyncio.run(
            add_exchange(
                "hello",
                "hi",
                user_id="123",
                session_id="sess-abc",
            )
        )

        call_kwargs = mock_mem.add.call_args[1]
        assert call_kwargs["infer"] is False
        assert call_kwargs["metadata"]["type"] == "exchange"
        assert call_kwargs["metadata"]["session_id"] == "sess-abc"

    def test_add_exchange_handles_exception(self):
        """Exceptions in add_exchange are caught, not propagated."""
        import kai.memory as mem_mod
        from kai.memory import add_exchange

        mock_mem = MagicMock()
        mock_mem.add.side_effect = RuntimeError("disk full")
        mem_mod._memory = mock_mem

        # Should not raise
        asyncio.run(add_exchange("hello", "hi", user_id="123"))


class TestGetAll:
    """Tests for get_all() retrieval."""

    def test_get_all_disabled_returns_empty(self):
        """get_all() returns empty list when disabled."""
        from kai.memory import get_all

        assert get_all(user_id="123") == []

    def test_get_all_wraps_results(self):
        """get_all() wraps raw results into MemoryResult objects."""
        import kai.memory as mem_mod
        from kai.memory import get_all

        mock_mem = MagicMock()
        mock_mem.get_all.return_value = {
            "results": [
                {
                    "id": "a",
                    "memory": "fact one",
                    "score": 0.0,
                    "metadata": {"type": "exchange"},
                    "created_at": "2026-04-01T00:00:00",
                },
                {
                    "id": "b",
                    "memory": "fact two",
                    "score": 0.0,
                    "metadata": {"type": "fact"},
                    "created_at": "2026-04-02T00:00:00",
                },
            ]
        }
        mem_mod._memory = mock_mem

        results = get_all(user_id="123")
        assert len(results) == 2


class TestDeleteAll:
    """Tests for delete_all()."""

    def test_delete_all_disabled_noop(self):
        """delete_all() is a no-op when disabled."""
        from kai.memory import delete_all

        # Should not raise
        delete_all(user_id="123")

    def test_delete_all_calls_mem0(self):
        """delete_all() delegates to Mem0's delete_all."""
        import kai.memory as mem_mod
        from kai.memory import delete_all

        mock_mem = MagicMock()
        mem_mod._memory = mock_mem

        delete_all(user_id="user-abc")
        mock_mem.delete_all.assert_called_once_with(user_id="user-abc")


class TestGetStats:
    """Tests for get_stats()."""

    def test_get_stats_counts_by_type(self):
        """get_stats() returns correct counts grouped by type."""
        import kai.memory as mem_mod
        from kai.memory import get_stats

        mock_mem = MagicMock()
        mock_mem.get_all.return_value = {
            "results": [
                {"id": "1", "memory": "a", "score": 0, "metadata": {"type": "exchange"}, "created_at": ""},
                {"id": "2", "memory": "b", "score": 0, "metadata": {"type": "exchange"}, "created_at": ""},
                {"id": "3", "memory": "c", "score": 0, "metadata": {"type": "fact"}, "created_at": ""},
            ]
        }
        mem_mod._memory = mock_mem

        stats = get_stats(user_id="123")
        assert stats.total_count == 3
        assert stats.by_type == {"exchange": 2, "fact": 1}

    def test_get_stats_disabled_returns_zeroed(self):
        """get_stats() returns zeroed stats when disabled."""
        from kai.memory import get_stats

        stats = get_stats(user_id="123")
        assert stats.total_count == 0
        assert stats.by_type == {}


# ── Integration tests (real Mem0 + Qdrant, slower) ──────────────────


@integration
class TestMemoryIntegration:
    """End-to-end tests with a real Mem0 instance and Qdrant storage."""

    def test_add_and_search(self, real_memory_instance):
        """Add an exchange, then search for it by semantic similarity."""
        import kai.memory as mem_mod

        mem_mod._memory = real_memory_instance
        mem_mod._config = _make_config()

        user_id = "integration-add-search"

        # Clean slate
        real_memory_instance.delete_all(user_id=user_id)

        # Add an exchange
        asyncio.run(
            mem_mod.add_exchange(
                "How do I set up the webhook server?",
                "Run make run to start the aiohttp server on port 8080.",
                user_id=user_id,
            )
        )

        # Search for it
        results = mem_mod.search("webhook server setup", user_id=user_id)
        assert len(results) >= 1
        assert any("webhook" in r.text.lower() for r in results)
        assert results[0].score > 0

    def test_search_relevance_ranking(self, real_memory_instance):
        """Most relevant result should rank first."""
        import kai.memory as mem_mod

        mem_mod._memory = real_memory_instance
        mem_mod._config = _make_config()

        user_id = "integration-ranking"
        real_memory_instance.delete_all(user_id=user_id)

        # Add three exchanges on different topics
        exchanges = [
            ("What is the weather today?", "It is sunny and 72F in Toronto."),
            ("How do I deploy to production?", "Run sudo make install to deploy to /opt/kai/."),
            ("What is for dinner?", "How about pasta with garlic bread?"),
        ]
        for user_text, assistant_text in exchanges:
            asyncio.run(mem_mod.add_exchange(user_text, assistant_text, user_id=user_id))

        # Search for deployment - should rank the deploy exchange first
        results = mem_mod.search("production deployment process", user_id=user_id)
        assert len(results) >= 1
        # The top result should be about deployment
        assert "deploy" in results[0].text.lower() or "production" in results[0].text.lower()

    def test_multi_user_isolation(self, real_memory_instance):
        """Memories for user A should not appear in user B's searches."""
        import kai.memory as mem_mod

        mem_mod._memory = real_memory_instance
        mem_mod._config = _make_config()

        user_a = "isolation-user-a"
        user_b = "isolation-user-b"
        real_memory_instance.delete_all(user_id=user_a)
        real_memory_instance.delete_all(user_id=user_b)

        # Add a memory for user A only
        asyncio.run(
            mem_mod.add_exchange(
                "My secret project is called Phoenix.",
                "Got it, I will remember Phoenix.",
                user_id=user_a,
            )
        )

        # User B should not find it
        results_b = mem_mod.search("Phoenix project", user_id=user_b)
        phoenix_found = any("phoenix" in r.text.lower() for r in results_b)
        assert not phoenix_found, "User B found user A's memory - isolation broken"

    def test_get_all_and_delete_all(self, real_memory_instance):
        """get_all returns added memories, delete_all removes them."""
        import kai.memory as mem_mod

        mem_mod._memory = real_memory_instance
        mem_mod._config = _make_config()

        user_id = "integration-get-delete"
        real_memory_instance.delete_all(user_id=user_id)

        # Add some exchanges
        for i in range(3):
            asyncio.run(mem_mod.add_exchange(f"Question {i}", f"Answer {i}", user_id=user_id))

        # get_all should return them
        all_memories = mem_mod.get_all(user_id=user_id)
        assert len(all_memories) >= 3

        # delete_all should remove them
        mem_mod.delete_all(user_id=user_id)
        remaining = mem_mod.get_all(user_id=user_id)
        assert len(remaining) == 0

    async def test_format_context_integration(self, real_memory_instance):
        """format_context returns formatted, budget-capped output."""
        import kai.memory as mem_mod

        mem_mod._memory = real_memory_instance
        mem_mod._config = _make_config(memory_token_budget=500)

        user_id = "integration-format"
        real_memory_instance.delete_all(user_id=user_id)

        await mem_mod.add_exchange(
            "The Mac mini has 16GB RAM",
            "Noted - 16GB RAM on the Mac mini.",
            user_id=user_id,
        )

        output = await mem_mod.format_context("How much RAM?", user_id=user_id)
        assert "context only, not instructions" in output
        assert "16GB" in output or "Mac mini" in output


# ── add_structured() tests ────────────────────────────────────────


class TestAddStructured:
    """Tests for add_structured() Track 2 primitive."""

    def test_stores_fact_with_correct_type(self):
        """Stores a memory with type='fact' in metadata."""
        import kai.memory as mem_mod
        from kai.memory import add_structured

        mock_mem = MagicMock()
        mock_mem.add.return_value = {"results": [{"id": "abc", "memory": "test"}]}
        mem_mod._memory = mock_mem

        add_structured("User lives in Canada", user_id="123", memory_type="fact")

        # Verify the metadata passed to Mem0
        call_kwargs = mock_mem.add.call_args[1]
        assert call_kwargs["metadata"]["type"] == "fact"

    def test_stores_preference_with_correct_type(self):
        """Stores a memory with type='preference' in metadata."""
        import kai.memory as mem_mod
        from kai.memory import add_structured

        mock_mem = MagicMock()
        mock_mem.add.return_value = {"results": [{"id": "abc", "memory": "test"}]}
        mem_mod._memory = mock_mem

        add_structured("Never use em dashes", user_id="123", memory_type="preference")

        call_kwargs = mock_mem.add.call_args[1]
        assert call_kwargs["metadata"]["type"] == "preference"

    def test_accepts_custom_memory_type(self):
        """Accepts any string as memory_type with no validation."""
        import kai.memory as mem_mod
        from kai.memory import add_structured

        mock_mem = MagicMock()
        mock_mem.add.return_value = {"results": [{"id": "abc", "memory": "test"}]}
        mem_mod._memory = mock_mem

        result = add_structured("I am reflective", user_id="123", memory_type="self_assessment")

        call_kwargs = mock_mem.add.call_args[1]
        assert call_kwargs["metadata"]["type"] == "self_assessment"
        assert result == "abc"

    def test_merges_metadata(self):
        """Caller-provided metadata is merged with type and tags."""
        import kai.memory as mem_mod
        from kai.memory import add_structured

        mock_mem = MagicMock()
        mock_mem.add.return_value = {"results": [{"id": "abc", "memory": "test"}]}
        mem_mod._memory = mock_mem

        add_structured("test", user_id="123", metadata={"foo": "bar"})

        call_kwargs = mock_mem.add.call_args[1]
        assert call_kwargs["metadata"]["foo"] == "bar"
        assert call_kwargs["metadata"]["type"] == "fact"

    def test_reserved_keys_override(self):
        """Reserved keys (type, tags) override caller-provided metadata."""
        import kai.memory as mem_mod
        from kai.memory import add_structured

        mock_mem = MagicMock()
        mock_mem.add.return_value = {"results": [{"id": "abc", "memory": "test"}]}
        mem_mod._memory = mock_mem

        add_structured(
            "test",
            user_id="123",
            memory_type="preference",
            metadata={"type": "spoof"},
        )

        call_kwargs = mock_mem.add.call_args[1]
        assert call_kwargs["metadata"]["type"] == "preference"

    def test_stores_tags(self):
        """Tags are stored in metadata['tags']."""
        import kai.memory as mem_mod
        from kai.memory import add_structured

        mock_mem = MagicMock()
        mock_mem.add.return_value = {"results": [{"id": "abc", "memory": "test"}]}
        mem_mod._memory = mock_mem

        add_structured("test", user_id="123", tags=["a", "b"])

        call_kwargs = mock_mem.add.call_args[1]
        assert call_kwargs["metadata"]["tags"] == ["a", "b"]

    def test_empty_content_returns_none(self):
        """Empty or whitespace-only content returns None without calling Mem0."""
        import kai.memory as mem_mod
        from kai.memory import add_structured

        mock_mem = MagicMock()
        mem_mod._memory = mock_mem

        assert add_structured("", user_id="123") is None
        assert add_structured("   ", user_id="123") is None
        mock_mem.add.assert_not_called()

    def test_disabled_returns_none(self):
        """Returns None when memory is not initialized."""
        from kai.memory import add_structured

        assert add_structured("test", user_id="123") is None

    def test_returns_id_string(self):
        """Returns the Mem0 memory ID as a string on success."""
        import kai.memory as mem_mod
        from kai.memory import add_structured

        mock_mem = MagicMock()
        mock_mem.add.return_value = {"results": [{"id": "mem-uuid-123", "memory": "test"}]}
        mem_mod._memory = mock_mem

        result = add_structured("test", user_id="123")
        assert result == "mem-uuid-123"
        assert isinstance(result, str)

    def test_mem0_failure_returns_none_and_logs(self, caplog):
        """Mem0 add() exceptions are caught, logged, and return None."""
        import kai.memory as mem_mod
        from kai.memory import add_structured

        mock_mem = MagicMock()
        mock_mem.add.side_effect = RuntimeError("disk full")
        mem_mod._memory = mock_mem

        with caplog.at_level("WARNING", logger="kai.memory"):
            result = add_structured("test", user_id="123")

        assert result is None
        assert "add_structured failed" in caplog.text


# ── _parse_topic_file() tests ─────────────────────────────────────


class TestParseTopicFile:
    """Tests for _parse_topic_file() markdown parser."""

    def test_bullets_become_candidates(self, tmp_path):
        """Each bullet line becomes one memory candidate."""
        from kai.memory import _parse_topic_file

        f = tmp_path / "test.md"
        f.write_text("# Heading\n\n- First item\n- Second item\n- Third item\n")

        result = _parse_topic_file(f)
        assert len(result) == 3
        assert result[0]["content"] == "First item"
        assert result[1]["content"] == "Second item"

    def test_headings_stored_as_context(self, tmp_path):
        """Heading text is stored in the 'heading' key, not as content."""
        from kai.memory import _parse_topic_file

        f = tmp_path / "test.md"
        f.write_text("# Main\n\n## Communication\n\n- Be concise\n")

        result = _parse_topic_file(f)
        assert len(result) == 1
        assert result[0]["content"] == "Be concise"
        assert result[0]["heading"] == "Communication"

    def test_paragraphs_become_candidates(self, tmp_path):
        """Non-bullet, non-heading text is joined into paragraph candidates."""
        from kai.memory import _parse_topic_file

        f = tmp_path / "test.md"
        f.write_text("# Notes\n\nFirst line of paragraph.\nSecond line of paragraph.\n\nAnother paragraph.\n")

        result = _parse_topic_file(f)
        assert len(result) == 2
        assert result[0]["content"] == "First line of paragraph. Second line of paragraph."
        assert result[1]["content"] == "Another paragraph."

    def test_code_blocks_skipped(self, tmp_path):
        """Content inside fenced code blocks is not seeded."""
        from kai.memory import _parse_topic_file

        f = tmp_path / "test.md"
        f.write_text(
            "# Reference\n\n- Real memory\n\n```\n- Not a memory\nAlso not a memory\n```\n\n- Another real one\n"
        )

        result = _parse_topic_file(f)
        contents = [r["content"] for r in result]
        assert "Real memory" in contents
        assert "Another real one" in contents
        assert "Not a memory" not in contents
        assert "Also not a memory" not in contents

    def test_empty_file_returns_empty(self, tmp_path):
        """An empty file produces no candidates."""
        from kai.memory import _parse_topic_file

        f = tmp_path / "test.md"
        f.write_text("")

        assert _parse_topic_file(f) == []

    def test_heading_only_file_returns_empty(self, tmp_path):
        """A file with only headings and no content produces no candidates."""
        from kai.memory import _parse_topic_file

        f = tmp_path / "test.md"
        f.write_text("# Title\n\n## Section\n\n### Subsection\n")

        assert _parse_topic_file(f) == []

    def test_bare_hash_not_treated_as_heading(self, tmp_path):
        """Lines like #311, #hashtag, or ##cross-ref are paragraph text."""
        from kai.memory import _parse_topic_file

        f = tmp_path / "test.md"
        # All ATX levels require a space: #, ##, ###, etc.
        f.write_text("# Real Heading\n\n#311 is an issue reference\n#hashtag\n##nospace\n")

        result = _parse_topic_file(f)
        # All three bare-hash lines should be joined into one paragraph
        assert len(result) == 1
        assert "#311 is an issue reference" in result[0]["content"]
        assert "#hashtag" in result[0]["content"]
        assert "##nospace" in result[0]["content"]
        # The real heading (with space) should be context, not content
        assert result[0]["heading"] == "Real Heading"


# ── _classify_source_file() tests ─────────────────────────────────


class TestClassifySourceFile:
    """Tests for _classify_source_file() file-to-type mapping."""

    def test_known_files(self):
        """Known files map to their expected types."""
        from kai.memory import _classify_source_file

        assert _classify_source_file("preferences.md") == "preference"
        assert _classify_source_file("hard-lessons.md") == "preference"
        assert _classify_source_file("user.md") == "fact"
        assert _classify_source_file("projects.md") == "fact"
        assert _classify_source_file("notes.md") == "fact"
        assert _classify_source_file("planned-features.md") == "fact"

    def test_skip_files(self):
        """MEMORY.md and api-reference.md return None (skip)."""
        from kai.memory import _classify_source_file

        assert _classify_source_file("MEMORY.md") is None
        assert _classify_source_file("api-reference.md") is None

    def test_unknown_files_return_none(self):
        """Unknown files default to None (skip), not 'fact'."""
        from kai.memory import _classify_source_file

        assert _classify_source_file("random.md") is None
        assert _classify_source_file("todo.md") is None


# ── _is_duplicate() tests ────────────────────────────────────────


class TestIsDuplicate:
    """Tests for _is_duplicate() dedup helper."""

    def test_search_exception_returns_false(self):
        """When search() raises, _is_duplicate returns False (insert, don't skip)."""
        import kai.memory as mem_mod
        from kai.memory import _is_duplicate

        # Set _config so search() doesn't short-circuit on the None guard
        mem_mod._config = _make_config()
        # Mock _memory.search to raise inside search()
        mock_mem = MagicMock()
        mock_mem.search.side_effect = RuntimeError("qdrant connection refused")
        mem_mod._memory = mock_mem

        # Should return False (not a duplicate), not raise
        result = _is_duplicate("some content", user_id="123")
        assert result is False


# ── seed_from_memory_md() tests ───────────────────────────────────


class TestSeedFromMemoryMd:
    """Tests for seed_from_memory_md() one-time migration."""

    def test_parses_preferences_file_as_preference(self, tmp_path):
        """Preferences file bullets are seeded with type='preference'."""
        import kai.memory as mem_mod
        from kai.memory import seed_from_memory_md

        mock_mem = MagicMock()
        mock_mem.add.return_value = {"results": [{"id": "abc", "memory": "test"}]}
        mock_mem.search.return_value = {"results": []}
        mem_mod._memory = mock_mem

        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        (memory_dir / "preferences.md").write_text("# Preferences\n\n- Item A\n- Item B\n- Item C\n")

        counts = seed_from_memory_md(user_ids=["123"], memory_dir=memory_dir)

        assert counts["123"]["seeded"] == 3
        # Verify all calls used memory_type="preference" via metadata
        for call in mock_mem.add.call_args_list:
            assert call[1]["metadata"]["type"] == "preference"

    def test_parses_user_file_as_fact(self, tmp_path):
        """User file bullets are seeded with type='fact'."""
        import kai.memory as mem_mod
        from kai.memory import seed_from_memory_md

        mock_mem = MagicMock()
        mock_mem.add.return_value = {"results": [{"id": "abc", "memory": "test"}]}
        mock_mem.search.return_value = {"results": []}
        mem_mod._memory = mock_mem

        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        (memory_dir / "user.md").write_text("# User\n\n- Location: Canada\n- Timezone: EST\n")

        counts = seed_from_memory_md(user_ids=["123"], memory_dir=memory_dir)

        assert counts["123"]["seeded"] == 2
        for call in mock_mem.add.call_args_list:
            assert call[1]["metadata"]["type"] == "fact"

    def test_skips_api_reference_file(self, tmp_path):
        """api-reference.md is not seeded even when present."""
        import kai.memory as mem_mod
        from kai.memory import seed_from_memory_md

        mock_mem = MagicMock()
        mock_mem.add.return_value = {"results": [{"id": "abc", "memory": "test"}]}
        mock_mem.search.return_value = {"results": []}
        mem_mod._memory = mock_mem

        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        (memory_dir / "api-reference.md").write_text("# API\n\n- Endpoint A\n")
        (memory_dir / "user.md").write_text("# User\n\n- Location: Canada\n")

        counts = seed_from_memory_md(user_ids=["123"], memory_dir=memory_dir)

        # Only user.md should be seeded, not api-reference.md
        assert counts["123"]["seeded"] == 1
        for call in mock_mem.add.call_args_list:
            assert call[1]["metadata"]["source_file"] != "api-reference.md"

    def test_skips_memory_md_index(self, tmp_path):
        """MEMORY.md index file is not seeded."""
        import kai.memory as mem_mod
        from kai.memory import seed_from_memory_md

        mock_mem = MagicMock()
        mock_mem.add.return_value = {"results": [{"id": "abc", "memory": "test"}]}
        mock_mem.search.return_value = {"results": []}
        mem_mod._memory = mock_mem

        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        (memory_dir / "MEMORY.md").write_text("# Memory\n\n- [User](user.md)\n")
        (memory_dir / "user.md").write_text("# User\n\n- Location: Canada\n")

        counts = seed_from_memory_md(user_ids=["123"], memory_dir=memory_dir)

        assert counts["123"]["seeded"] == 1

    def test_skips_unknown_files(self, tmp_path):
        """Files not in the classification mapping are ignored."""
        import kai.memory as mem_mod
        from kai.memory import seed_from_memory_md

        mock_mem = MagicMock()
        mock_mem.add.return_value = {"results": [{"id": "abc", "memory": "test"}]}
        mock_mem.search.return_value = {"results": []}
        mem_mod._memory = mock_mem

        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        (memory_dir / "random.md").write_text("# Random\n\n- Should be ignored\n")

        counts = seed_from_memory_md(user_ids=["123"], memory_dir=memory_dir)

        assert counts["123"]["seeded"] == 0
        mock_mem.add.assert_not_called()

    def test_is_idempotent_on_rerun(self, tmp_path):
        """Second run skips all entries via dedup (skipped == first run's seeded)."""
        import kai.memory as mem_mod
        from kai.memory import seed_from_memory_md

        # Track stored memories to simulate search returning them on second run
        stored: list[dict] = []
        call_count = 0

        def mock_add(content, **kwargs):
            nonlocal call_count
            call_count += 1
            mem_id = f"id-{call_count}"
            stored.append({"id": mem_id, "memory": content, "score": 0.95, "metadata": kwargs.get("metadata", {})})
            return {"results": [{"id": mem_id, "memory": content}]}

        def mock_search(query, **kwargs):
            # Return the best match from stored memories (simulate high score for exact match)
            for s in stored:
                if s["memory"] == query:
                    return {
                        "results": [{"id": s["id"], "memory": s["memory"], "score": 0.95, "metadata": s["metadata"]}]
                    }
            return {"results": []}

        mock_mem = MagicMock()
        mock_mem.add.side_effect = mock_add
        mock_mem.search.side_effect = mock_search
        mem_mod._memory = mock_mem
        # search() requires _config to be set (returns [] otherwise)
        mem_mod._config = _make_config()

        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        (memory_dir / "user.md").write_text("# User\n\n- Fact A\n- Fact B\n")

        # First run: seeds everything
        counts1 = seed_from_memory_md(user_ids=["123"], memory_dir=memory_dir)
        assert counts1["123"]["seeded"] == 2

        # Second run: everything should be skipped via dedup
        counts2 = seed_from_memory_md(user_ids=["123"], memory_dir=memory_dir)
        assert counts2["123"]["skipped"] == 2
        assert counts2["123"]["seeded"] == 0

    def test_multi_user_isolation(self, tmp_path):
        """Each user_id gets their own copy of the seeded content."""
        import kai.memory as mem_mod
        from kai.memory import seed_from_memory_md

        mock_mem = MagicMock()
        mock_mem.add.return_value = {"results": [{"id": "abc", "memory": "test"}]}
        mock_mem.search.return_value = {"results": []}
        mem_mod._memory = mock_mem

        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        (memory_dir / "user.md").write_text("# User\n\n- Fact A\n")

        counts = seed_from_memory_md(user_ids=["111", "222"], memory_dir=memory_dir)

        assert counts["111"]["seeded"] == 1
        assert counts["222"]["seeded"] == 1
        # Two calls total - one per user
        assert mock_mem.add.call_count == 2

    def test_partial_failure_counts_failures(self, tmp_path):
        """File read errors are counted as failures; other files still seed."""
        import kai.memory as mem_mod
        from kai.memory import seed_from_memory_md

        mock_mem = MagicMock()
        mock_mem.add.return_value = {"results": [{"id": "abc", "memory": "test"}]}
        mock_mem.search.return_value = {"results": []}
        mem_mod._memory = mock_mem

        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        (memory_dir / "user.md").write_text("# User\n\n- Fact A\n")

        # Create a notes.md that will fail to read by making it a directory
        # (reading a directory raises OSError/IsADirectoryError)
        (memory_dir / "notes.md").mkdir()

        counts = seed_from_memory_md(user_ids=["123"], memory_dir=memory_dir)

        assert counts["123"]["seeded"] == 1  # user.md succeeded
        assert counts["123"]["failed"] == 1  # notes.md failed

    def test_unicode_decode_error_counts_as_failure(self, tmp_path):
        """Non-UTF-8 files are caught and counted as failures, not crashes."""
        import kai.memory as mem_mod
        from kai.memory import seed_from_memory_md

        mock_mem = MagicMock()
        mock_mem.add.return_value = {"results": [{"id": "abc", "memory": "test"}]}
        mock_mem.search.return_value = {"results": []}
        mem_mod._memory = mock_mem

        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        (memory_dir / "user.md").write_text("# User\n\n- Fact A\n")
        # Write raw bytes that are not valid UTF-8
        (memory_dir / "notes.md").write_bytes(b"\xff\xfe# Notes\n\n- Broken\n")

        counts = seed_from_memory_md(user_ids=["123"], memory_dir=memory_dir)

        assert counts["123"]["seeded"] == 1  # user.md succeeded
        assert counts["123"]["failed"] == 1  # notes.md failed (UnicodeDecodeError)

    def test_preserves_heading_context(self, tmp_path):
        """Headings are stored as metadata['heading'] on subsequent bullets."""
        import kai.memory as mem_mod
        from kai.memory import seed_from_memory_md

        mock_mem = MagicMock()
        mock_mem.add.return_value = {"results": [{"id": "abc", "memory": "test"}]}
        mock_mem.search.return_value = {"results": []}
        mem_mod._memory = mock_mem

        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        (memory_dir / "preferences.md").write_text("# Preferences\n\n## Communication\n\n- Be concise\n")

        seed_from_memory_md(user_ids=["123"], memory_dir=memory_dir)

        call_kwargs = mock_mem.add.call_args[1]
        assert call_kwargs["metadata"]["heading"] == "Communication"

    def test_stores_source_file_metadata(self, tmp_path):
        """Every seeded memory has metadata['source_file'] set."""
        import kai.memory as mem_mod
        from kai.memory import seed_from_memory_md

        mock_mem = MagicMock()
        mock_mem.add.return_value = {"results": [{"id": "abc", "memory": "test"}]}
        mock_mem.search.return_value = {"results": []}
        mem_mod._memory = mock_mem

        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        (memory_dir / "user.md").write_text("# User\n\n- Fact A\n")

        seed_from_memory_md(user_ids=["123"], memory_dir=memory_dir)

        call_kwargs = mock_mem.add.call_args[1]
        assert call_kwargs["metadata"]["source_file"] == "user.md"

    def test_stores_source_migration_tag(self, tmp_path):
        """Every seeded memory has metadata['source'] == 'memory_md_migration'."""
        import kai.memory as mem_mod
        from kai.memory import seed_from_memory_md

        mock_mem = MagicMock()
        mock_mem.add.return_value = {"results": [{"id": "abc", "memory": "test"}]}
        mock_mem.search.return_value = {"results": []}
        mem_mod._memory = mock_mem

        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        (memory_dir / "user.md").write_text("# User\n\n- Fact A\n")

        seed_from_memory_md(user_ids=["123"], memory_dir=memory_dir)

        call_kwargs = mock_mem.add.call_args[1]
        assert call_kwargs["metadata"]["source"] == "memory_md_migration"

    def test_stores_tag_from_file_stem(self, tmp_path):
        """Tags contain the file stem (e.g. 'preferences' for preferences.md)."""
        import kai.memory as mem_mod
        from kai.memory import seed_from_memory_md

        mock_mem = MagicMock()
        mock_mem.add.return_value = {"results": [{"id": "abc", "memory": "test"}]}
        mock_mem.search.return_value = {"results": []}
        mem_mod._memory = mock_mem

        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        (memory_dir / "preferences.md").write_text("# Preferences\n\n- Item A\n")

        seed_from_memory_md(user_ids=["123"], memory_dir=memory_dir)

        call_kwargs = mock_mem.add.call_args[1]
        assert call_kwargs["metadata"]["tags"] == ["preferences"]

    def test_disabled_returns_zero_counts(self):
        """With memory disabled, returns all-zero counts without exceptions."""
        from kai.memory import seed_from_memory_md

        counts = seed_from_memory_md(user_ids=["123", "456"])

        assert counts["123"] == {"seeded": 0, "skipped": 0, "failed": 0}
        assert counts["456"] == {"seeded": 0, "skipped": 0, "failed": 0}

    def test_missing_directory_returns_zero_counts(self, tmp_path):
        """Non-existent memory directory returns zero counts, not an error."""
        import kai.memory as mem_mod
        from kai.memory import seed_from_memory_md

        mock_mem = MagicMock()
        mem_mod._memory = mock_mem

        # Point to a directory that does not exist
        missing_dir = tmp_path / "does_not_exist" / "memory"

        counts = seed_from_memory_md(user_ids=["123"], memory_dir=missing_dir)

        assert counts["123"] == {"seeded": 0, "skipped": 0, "failed": 0}
        # No Mem0 calls should be made
        mock_mem.add.assert_not_called()

    def test_code_blocks_not_stored(self, tmp_path):
        """Content inside fenced code blocks is not seeded."""
        import kai.memory as mem_mod
        from kai.memory import seed_from_memory_md

        mock_mem = MagicMock()
        mock_mem.add.return_value = {"results": [{"id": "abc", "memory": "test"}]}
        mock_mem.search.return_value = {"results": []}
        mem_mod._memory = mock_mem

        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        (memory_dir / "notes.md").write_text(
            "# Notes\n\n- Real fact\n\n```\n- Not a fact\nAlso not a fact\n```\n\n- Another real fact\n"
        )

        counts = seed_from_memory_md(user_ids=["123"], memory_dir=memory_dir)

        assert counts["123"]["seeded"] == 2
        # Verify the stored content
        stored_texts = [call[0][0] for call in mock_mem.add.call_args_list]
        assert "Real fact" in stored_texts
        assert "Another real fact" in stored_texts
        assert "Not a fact" not in stored_texts

    def test_paragraphs_stored_when_not_bullets(self, tmp_path):
        """Non-bullet prose paragraphs are seeded as single memories."""
        import kai.memory as mem_mod
        from kai.memory import seed_from_memory_md

        mock_mem = MagicMock()
        mock_mem.add.return_value = {"results": [{"id": "abc", "memory": "test"}]}
        mock_mem.search.return_value = {"results": []}
        mem_mod._memory = mock_mem

        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        (memory_dir / "hard-lessons.md").write_text(
            "# Hard Lessons\n\n## Never do X\n\nFirst line of lesson.\nSecond line of lesson.\n\n## Also bad\n\nAnother paragraph here.\n"
        )

        counts = seed_from_memory_md(user_ids=["123"], memory_dir=memory_dir)

        assert counts["123"]["seeded"] == 2
        stored_texts = [call[0][0] for call in mock_mem.add.call_args_list]
        assert "First line of lesson. Second line of lesson." in stored_texts
        assert "Another paragraph here." in stored_texts
