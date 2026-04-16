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
