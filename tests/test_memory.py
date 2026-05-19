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
import json
import logging
import os
import re
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
    """Build a Config with memory settings for testing.

    The memory_search_floor default mirrors the production default (0.3,
    matching the prior hard-coded `_MIN_RELEVANCE_THRESHOLD` constant).
    Tests that exercise the floor explicitly should pass a different
    `memory_search_floor=...` override; everything else inherits 0.3 so
    the existing threshold tests continue to assert the same behavior.
    """
    defaults = {
        "memory_enabled": enabled,
        "memory_search_limit": 10,
        "memory_token_budget": 2000,
        "memory_embedding_model": "all-MiniLM-L6-v2",
        "memory_search_floor": 0.3,
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

    def test_init_disabled_emits_config_log_with_null_fields(self, caplog):
        """Even when memory is disabled, init_memory emits exactly
        one structured `memory.config` log so an operator scanning
        the log post-restart can confirm the configured state. All
        non-`enabled` fields elide to null."""
        import json as json_module

        from kai.memory import init_memory

        config = _make_config(enabled=False)
        with caplog.at_level(logging.INFO, logger="kai.memory"):
            init_memory(config)
        records = [r for r in caplog.records if "memory.config" in r.getMessage()]
        assert len(records) == 1
        payload = json_module.loads(records[0].getMessage().split("memory.config ", 1)[1])
        assert payload == {
            "enabled": False,
            "extraction_enabled": False,
            "reasoner_backend": None,
            "extraction_model": None,
            "episode_model": None,
            "extraction_binary": None,
        }

    @integration
    def test_init_retrieval_only_emits_null_extraction_binary(self, caplog, tmp_path, monkeypatch):
        """Retrieval-only memory (MEMORY_ENABLED=true with extraction
        disabled) MUST NOT call the resolver: the resolver call is
        gated on `memory_extraction_enabled`, so an install without a
        claude or codex binary on PATH still initializes
        successfully. The resolver mock fires loudly if called."""
        import json as json_module

        from kai.memory import init_memory

        called = []

        def fake_resolve(backend: str) -> str:
            called.append(backend)
            raise RuntimeError("resolver should not run on retrieval-only init")

        monkeypatch.setattr("kai.oneshot_binary.resolve_oneshot_binary", fake_resolve)
        # Memory enabled but extraction disabled = retrieval-only.
        config = _make_config(
            memory_extraction_enabled=False,
            memory_reasoner_backend="claude",
            memory_extraction_model="claude-haiku-4-5",
            memory_episode_model="",
        )
        with (
            caplog.at_level(logging.INFO, logger="kai.memory"),
            patch("kai.memory.DATA_DIR", tmp_path),
            patch("mem0.Memory") as mock_mem,
        ):
            # Set the embedding model dim to the expected 384 so init
            # proceeds past the dim-check guard.
            mock_mem.from_config.return_value.embedding_model.model.get_embedding_dimension.return_value = 384
            init_memory(config)
        records = [r for r in caplog.records if "memory.config" in r.getMessage()]
        assert len(records) == 1
        payload = json_module.loads(records[0].getMessage().split("memory.config ", 1)[1])
        assert payload["enabled"] is True
        assert payload["extraction_enabled"] is False
        assert payload["extraction_binary"] is None
        # Critical contract: resolver MUST NOT have been called.
        assert called == [], f"resolver was invoked on retrieval-only init: {called}"

    @integration
    def test_init_extraction_enabled_resolves_binary_in_log(self, caplog, tmp_path, monkeypatch):
        """When extraction is enabled, init_memory calls the resolver
        and writes the resolved path into the `extraction_binary`
        field of the structured log."""
        import json as json_module

        from kai.memory import init_memory

        monkeypatch.setattr(
            "kai.oneshot_binary.resolve_oneshot_binary",
            lambda backend: f"/fake/{backend}-binary",
        )
        config = _make_config(
            memory_extraction_enabled=True,
            memory_reasoner_backend="codex",
            memory_extraction_model="gpt-5.4-mini",
            memory_episode_model="",
        )
        with (
            caplog.at_level(logging.INFO, logger="kai.memory"),
            patch("kai.memory.DATA_DIR", tmp_path),
            patch("mem0.Memory") as mock_mem,
        ):
            mock_mem.from_config.return_value.embedding_model.model.get_embedding_dimension.return_value = 384
            init_memory(config)
        records = [r for r in caplog.records if "memory.config" in r.getMessage()]
        assert len(records) == 1
        payload = json_module.loads(records[0].getMessage().split("memory.config ", 1)[1])
        assert payload["enabled"] is True
        assert payload["extraction_enabled"] is True
        assert payload["reasoner_backend"] == "codex"
        assert payload["extraction_binary"] == "/fake/codex-binary"

    @integration
    def test_mem0_telemetry_path_is_isolated(self):
        """Mem0's hardcoded telemetry Qdrant path must live under the
        test-only MEM0_DIR set by conftest, never under $HOME/.mem0.

        Regression guard for issue #357. Mem0 opens a second Qdrant
        local-mode client at $MEM0_DIR/migrations_qdrant for its own
        telemetry/migration tracking, with the path resolved at module
        import time (mem0/memory/setup.py:8). If MEM0_DIR is not set
        before any `import mem0` runs, that path freezes to the
        production default $HOME/.mem0 and collides with the running
        production service's portalocker lock - making the entire
        TestMemoryIntegration suite unrunnable on a dev machine.

        If a future refactor removes the conftest-level env override,
        this test fails fast on CI and on dev machines alike, before
        the heavier integration tests hit the RuntimeError.
        """
        mem0_dir_env = os.environ.get("MEM0_DIR", "")
        assert mem0_dir_env, "MEM0_DIR must be set by conftest.py"

        # Must not be the production default - catches a regression
        # where someone replaces tempfile.mkdtemp with a literal
        # path that happens to land on the user's home dir.
        assert Path(mem0_dir_env).resolve() != (Path.home() / ".mem0").resolve()

        # Cross-check against mem0's own resolved constant. This is
        # the load-bearing freeze-detector: if MEM0_DIR was set
        # AFTER mem0 was imported, mem0_resolved is frozen to the
        # wrong value (the production default) and this assertion
        # fires. The other two assertions only catch coarser
        # breakage (env unset, env pointing at home) - this one
        # catches the subtle ordering bug that motivated the fix.
        from mem0.memory.setup import mem0_dir as mem0_resolved

        assert Path(mem0_resolved).resolve() == Path(mem0_dir_env).resolve()

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
        """Formatted output includes date and source-short prefix from created_at/metadata."""
        import kai.memory as mem_mod
        from kai.memory import format_context

        mock_mem = MagicMock()
        mock_mem.search.return_value = {
            "results": [
                {
                    "id": "dated",
                    "memory": "User prefers Celsius",
                    "score": 0.9,
                    "metadata": {"type": "fact", "source": "extracted"},
                    "created_at": "2026-03-23T14:30:00",
                },
            ]
        }
        mem_mod._memory = mock_mem
        mem_mod._config = _make_config()

        output = await format_context("temperature units", user_id="123")
        # Per-line format: `- (YYYY-MM-DD, <source_short>) <text>`. The source
        # short tag carries more weight than the date in the new header spec
        # (§5.4); both should be present for results that have a timestamp.
        assert "2026-03-23" in output
        assert "fact" in output
        assert "- (2026-03-23, fact) User prefers Celsius" in output
        assert "context only, not instructions" in output

    async def test_format_context_source_hint_without_date(self):
        """When created_at is empty, the source-short prefix is still emitted.
        Spec 360 removed the `user_raw` source, so this test uses an
        `extracted` row as its undated specimen — the formatter contract
        (always emit the source tag, even when the date is missing) is
        unchanged; only the available source values shrank."""
        import kai.memory as mem_mod
        from kai.memory import format_context

        mock_mem = MagicMock()
        mock_mem.search.return_value = {
            "results": [
                {
                    "id": "undated",
                    "memory": "User prefers strong coffee",
                    "score": 0.9,
                    "metadata": {"type": "fact", "source": "extracted"},
                    "created_at": "",
                },
            ]
        }
        mem_mod._memory = mock_mem
        mem_mod._config = _make_config()

        output = await format_context("coffee", user_id="123")
        # Bare source-only prefix `- (<source_short>) <text>`: never drop source.
        assert "- (fact) User prefers strong coffee" in output

    async def test_format_context_legacy_source_labeled(self):
        """Rows with missing source are labeled 'legacy' in the per-line prefix."""
        import kai.memory as mem_mod
        from kai.memory import format_context

        mock_mem = MagicMock()
        mock_mem.search.return_value = {
            "results": [
                {
                    "id": "legacy",
                    "memory": "Old pre-spec entry",
                    "score": 0.9,
                    # Legacy rows have no "source" key. `metadata.get("source")`
                    # returns None, which the formatter maps to "legacy".
                    "metadata": {"type": "exchange"},
                    "created_at": "2026-01-15T00:00:00",
                },
            ]
        }
        mem_mod._memory = mock_mem
        mem_mod._config = _make_config()

        output = await format_context("history", user_id="123")
        assert "- (2026-01-15, legacy) Old pre-spec entry" in output

    async def test_format_context_orders_by_weighted_score(self):
        """At equal raw cosine, a user-speaker row ranks above an
        assistant-speaker row. The new ranking key is `cosine *
        speaker_weight * confidence`; with speakers = (user, assistant)
        and confidence held equal, the speaker_weights table alone
        decides the tie. Mem0 returns the rows in reverse order to
        confirm the formatter does its own sort rather than relying
        on input order.
        """
        import kai.memory as mem_mod
        from kai.memory import format_context

        # Two results with IDENTICAL raw score and confidence; only
        # speaker differs. Both confidences are 0.9 so a difference
        # in confidence cannot account for any reordering.
        mock_mem = MagicMock()
        mock_mem.search.return_value = {
            "results": [
                {
                    "id": "a",
                    "memory": "Assistant inferred a pattern",
                    "score": 0.8,
                    "metadata": {
                        "type": "fact",
                        "source": "extracted",
                        "speaker": "assistant",
                        "confidence": 0.9,
                    },
                    "created_at": "2026-01-01T00:00:00",
                },
                {
                    "id": "c",
                    "memory": "User prefers vim for editing",
                    "score": 0.8,
                    "metadata": {
                        "type": "fact",
                        "source": "extracted",
                        "speaker": "user",
                        "confidence": 0.9,
                    },
                    "created_at": "2026-03-01T00:00:00",
                },
            ]
        }
        mem_mod._memory = mock_mem
        mem_mod._config = _make_config()

        output = await format_context("editor", user_id="123")
        lines = output.splitlines()[1:]  # skip header

        # User-speaker outranks assistant-speaker at equal cosine and
        # equal confidence. With speaker weights at 1.0 vs 0.7 the
        # adjusted scores are 0.72 vs 0.504, ordering: user, assistant.
        assert "User prefers vim for editing" in lines[0]
        assert "Assistant inferred a pattern" in lines[1]

    async def test_format_context_floor_applies_to_raw_cosine(self):
        """A sub-threshold row stays filtered even though weighting
        could in principle re-rank rows past the floor. The new
        weights are demote-only (every value <= 1.0), so weighting
        can only lower a row's adjusted score. The floor check runs
        on raw cosine, BEFORE the speaker multiplier, which means a
        sub-threshold row never reaches the walk regardless of its
        speaker class.
        """
        import kai.memory as mem_mod
        from kai.memory import format_context

        # raw_score 0.25 < memory_search_floor (0.3). Even an
        # otherwise-promotable user-speaker row (speaker_weight 1.0)
        # cannot rescue it because the filter runs on raw cosine.
        mock_mem = MagicMock()
        mock_mem.search.return_value = {
            "results": [
                {
                    "id": "sub",
                    "memory": "Borderline user-claimed fact",
                    "score": 0.25,
                    "metadata": {
                        "type": "fact",
                        "source": "extracted",
                        "speaker": "user",
                        "confidence": 1.0,
                    },
                    "created_at": "2026-04-01T00:00:00",
                },
            ]
        }
        mem_mod._memory = mock_mem
        mem_mod._config = _make_config()

        output = await format_context("anything", user_id="123")
        assert output == ""

    async def test_format_context_log_payload_carries_speaker_and_confidence(self, caplog):
        """The per-hit log payload exposes `speaker` and `confidence`
        siblings of `source`. For a legacy row missing both fields
        from metadata, the logged values must be the defaulted
        constants returned by `_read_time_speaker`, NOT a missing-
        field marker - so a log analyst reading the line can
        reconstruct the demote multiplier (`speaker_weight *
        confidence`) without re-fetching the row.

        This is the test that catches a regression where the per-hit
        builder reads `r.metadata["speaker"]` directly (which would
        be None for legacy rows) instead of going through the
        helper. Without the helper indirection, every legacy row
        would log speaker=null and the eval harness would lose its
        ability to attribute ranking decisions.
        """
        import kai.memory as mem_mod
        from kai.memory import (
            _LEGACY_CONFIDENCE,
            _LEGACY_SPEAKER,
            format_context,
        )

        # One legacy row (no speaker/confidence in metadata, source ==
        # "extracted"): the read-time helper falls into branch 4 and
        # supplies the legacy defaults. The assertion below pins
        # against the bound module constants so a swap-the-constants
        # follow-up does not break this test.
        mock_mem = MagicMock()
        mock_mem.search.return_value = {
            "results": [
                {
                    "id": "legacy",
                    "memory": "Some legacy claim",
                    "score": 0.8,
                    "metadata": {"type": "fact", "source": "extracted"},
                    "created_at": "2026-01-01T00:00:00",
                },
            ]
        }
        mem_mod._memory = mock_mem
        mem_mod._config = _make_config()

        with caplog.at_level(logging.INFO, logger="kai.memory"):
            await format_context("legacy fact", user_id="42")

        payload = _parse_recall_log(caplog)
        assert len(payload["hits"]) == 1
        hit = payload["hits"][0]
        # Speaker and confidence carry the defaulted values.
        assert hit["speaker"] == _LEGACY_SPEAKER
        assert hit["confidence"] == _LEGACY_CONFIDENCE
        # Source still passes through as recorded on the row, NOT
        # collapsed into the speaker default. The two fields describe
        # different axes (write path vs whose claim) and must both
        # remain readable.
        assert hit["source"] == "extracted"

    async def test_format_context_disabled_returns_empty(self):
        """Returns empty string when memory is not initialized."""
        from kai.memory import format_context

        result = await format_context("anything", user_id="123")
        assert result == ""

    async def test_format_context_floor_from_config(self):
        """The relevance floor is read from `config.memory_search_floor`,
        not from a module-level constant. Spec 310 §7.5 requires the same
        knob to govern both this path and the `/memory search` UI; this
        test pins the read so a future refactor cannot reintroduce a
        hard-coded constant without breaking it.

        A row scoring 0.4 passes when the floor is 0.3 (default behavior)
        but is filtered out when the floor is raised to 0.5 - verifying
        the value is sourced from config at call time."""
        import kai.memory as mem_mod
        from kai.memory import format_context

        # Single result whose score sits between the two test floors.
        mock_mem = MagicMock()
        mock_mem.search.return_value = {
            "results": [
                {
                    "id": "mid",
                    "memory": "Mid-confidence fact",
                    "score": 0.4,
                    "metadata": {"type": "fact", "source": "extracted"},
                    "created_at": "2026-04-01T00:00:00",
                },
            ]
        }
        mem_mod._memory = mock_mem

        # Floor at default 0.3: row passes.
        mem_mod._config = _make_config(memory_search_floor=0.3)
        out_low = await format_context("anything", user_id="123")
        assert "Mid-confidence fact" in out_low

        # Floor raised to 0.5: same row now filtered.
        mem_mod._config = _make_config(memory_search_floor=0.5)
        out_high = await format_context("anything", user_id="123")
        assert out_high == ""

    async def test_format_context_extracted_only(self):
        """Spec 360 invariant: post-Track-1 retrieval renders only extracted
        facts. There must be no `User said:` prefix anywhere in the output
        and every per-line tag must be `(fact)` (the `_SOURCE_SHORT` value
        for `extracted`). This test seeds the formatter with five
        extracted-source rows and confirms the output is homogeneous —
        the original incident (#360) was triggered by retrieval blocks
        densely populated with `User said:` quote-shaped lines that the
        agent could not distinguish from the real current message; this
        regression test pins the new shape so the failure mode cannot
        creep back via a re-introduced `user_raw` row type."""
        import kai.memory as mem_mod
        from kai.memory import format_context

        # Five rows, all source="extracted", varying scores so the formatter
        # has real ranking work to do (not just a single-row degenerate case).
        mock_mem = MagicMock()
        mock_mem.search.return_value = {
            "results": [
                {
                    "id": f"e{i}",
                    "memory": memory_text,
                    "score": score,
                    "metadata": {"type": "fact", "source": "extracted"},
                    "created_at": "2026-04-01T00:00:00",
                }
                for i, (memory_text, score) in enumerate(
                    [
                        ("User prefers vim for editing", 0.9),
                        ("User uses macOS on a Mac mini", 0.85),
                        ("User is in the Eastern timezone", 0.8),
                        ("User runs Python 3.12", 0.75),
                        ("User dislikes em dashes in writing", 0.7),
                    ]
                )
            ]
        }
        mem_mod._memory = mock_mem
        mem_mod._config = _make_config()

        output = await format_context("editor preferences", user_id="123")

        # No legacy `User said:` shape anywhere — the spec 360 incident
        # was that those quote-shaped lines mimicked real user input.
        assert "User said:" not in output

        # Every per-line tag must be `(fact)`. Walk the body lines (skip
        # the header) and look for the source short tag in each. The
        # regex matches the per-line prefix shape exactly — `(fact)` with
        # an optional `YYYY-MM-DD, ` date prefix — rather than searching
        # for the bare substring `"fact)"`, which a memory text could
        # theoretically contain (e.g. a stored fact about "a known fact)").
        # The seeded data here is controlled, but the precise form keeps
        # the test honest if future seed text drifts.
        tag_re = re.compile(r"\((?:\d{4}-\d{2}-\d{2}, )?fact\)")
        body_lines = [ln for ln in output.splitlines()[1:] if ln.strip()]
        assert body_lines, "expected non-empty body"
        for line in body_lines:
            assert tag_re.search(line), f"non-(fact) tag on line: {line!r}"
            assert "(user)" not in line


# ── Speaker weight function ────────────────────────────────────────


class TestReadTimeSpeaker:
    """Tests for `_read_time_speaker`'s independent-field resolution.

    The two metadata fields (`speaker`, `confidence`) are filled in
    independently: an explicit value on either side is preserved, and
    only the missing side falls back to a source-appropriate default.
    Three angles need pinning:

    1. Both fields present: explicit values pass through unchanged.
    2. Speaker present, confidence missing: speaker stays, confidence
       comes from the source default. Defends against a future write
       path or a partial Mem0 round-trip from silently dropping the
       explicit speaker into the legacy bucket.
    3. Confidence present, speaker missing: confidence stays, speaker
       comes from the source default. Symmetric protection.
    """

    def test_both_fields_present_passes_through(self):
        from kai.memory import _read_time_speaker

        # Source set to "extracted" so the source-based defaults
        # would otherwise apply; both fields explicit so they win.
        meta = {"source": "extracted", "speaker": "user", "confidence": 0.85}
        assert _read_time_speaker(meta) == ("user", 0.85)

    def test_speaker_present_confidence_missing_keeps_speaker(self):
        from kai.memory import _LEGACY_CONFIDENCE, _read_time_speaker

        # Speaker explicit, confidence absent. The legacy default
        # confidence (0.5) fills in; the speaker is NOT dragged down
        # to _LEGACY_SPEAKER. Pin against the bound constant so a
        # swap-the-constants follow-up flows through.
        meta = {"source": "extracted", "speaker": "user"}
        assert _read_time_speaker(meta) == ("user", _LEGACY_CONFIDENCE)

    def test_confidence_present_speaker_missing_keeps_confidence(self):
        from kai.memory import _LEGACY_SPEAKER, _read_time_speaker

        # Symmetric counterpart: confidence explicit, speaker absent.
        # The legacy default speaker fills in; the confidence is
        # preserved. A pre-spec extracted row carrying confidence
        # but no speaker hits this branch in production.
        meta = {"source": "extracted", "confidence": 0.92}
        assert _read_time_speaker(meta) == (_LEGACY_SPEAKER, 0.92)

    def test_speaker_present_episode_source_uses_episode_confidence(self):
        from kai.memory import _read_time_speaker

        # Episode-source row carries an explicit non-canonical speaker
        # (extractor of a future write path). The speaker is preserved;
        # the missing confidence picks up the episode default of 1.0.
        meta = {"source": "episode", "speaker": "user"}
        assert _read_time_speaker(meta) == ("user", 1.0)

    def test_speaker_present_migration_source_uses_migration_confidence(self):
        from kai.memory import _MIGRATION_CONFIDENCE, _read_time_speaker

        # Migration-source row with an explicit speaker that does NOT
        # match the canonical migration speaker. Speaker is preserved;
        # confidence picks up the migration default.
        meta = {"source": "migration", "speaker": "assistant"}
        assert _read_time_speaker(meta) == ("assistant", _MIGRATION_CONFIDENCE)


class TestSpeakerWeight:
    """Tests for `_speaker_weight`, the read-time multiplier the
    retrieval sort uses in place of the older `_source_weight`. The
    helper composes two factors: a speaker_weights table lookup and
    the row's confidence value, both surfaced by `_read_time_speaker`.
    """

    def test_speaker_weight_combines_factors(self):
        """speaker_weight * confidence is the contract; pin the
        multiplication so a refactor that swaps order or drops a
        factor fails immediately. Use a row carrying speaker and
        confidence in metadata so the `_read_time_speaker` path is
        the explicit-fields branch, not a defaulted branch.
        """
        from kai.memory import _SPEAKER_WEIGHTS, MemoryResult, _speaker_weight

        row = MemoryResult(
            id="x",
            text="User prefers dark mode",
            score=0.5,
            memory_type="fact",
            metadata={
                "type": "fact",
                "source": "extracted",
                "speaker": "user",
                "confidence": 0.8,
            },
            created_at="2026-04-01T00:00:00",
        )

        # Expected: speaker_weights["user"] * confidence
        # Held against the table rather than literal 1.0 so the test
        # stays honest if the calibration sweep retunes "user".
        assert _speaker_weight(row) == _SPEAKER_WEIGHTS["user"] * 0.8

    def test_speaker_weight_unknown_speaker(self):
        """A row whose speaker value is not in the speaker_weights
        table falls back to _UNKNOWN_SPEAKER_WEIGHT (aliased to the
        assistant weight). Confidence is unaffected by the unknown
        path; the multiplier is `_UNKNOWN_SPEAKER_WEIGHT * confidence`.

        Pins the unknown-class fallback against the named alias rather
        than a literal so a future change to the alias target (e.g.,
        if the design shifts unknown back toward the legacy floor)
        flows through automatically.
        """
        from kai.memory import _UNKNOWN_SPEAKER_WEIGHT, MemoryResult, _speaker_weight

        row = MemoryResult(
            id="x",
            text="Some claim of unknown origin",
            score=0.5,
            memory_type="fact",
            metadata={
                "type": "fact",
                "source": "extracted",
                "speaker": "mystery_class",
                "confidence": 0.7,
            },
            created_at="2026-04-01T00:00:00",
        )

        assert _speaker_weight(row) == _UNKNOWN_SPEAKER_WEIGHT * 0.7

    def test_speaker_weight_demote_only(self):
        """For every (in-enum speaker, in-range confidence) pair, the
        combined multiplier stays in [0.0, 1.0]. This is the load-
        bearing invariant for the floor check: as long as the weight
        is <= 1.0, raw cosine remains an upper bound on adjusted
        score, and a sub-threshold row (raw_score < floor) cannot be
        rescued by a high speaker weight.

        Iterates the production table directly so a calibration sweep
        that bumps a value above 1.0 in a future tune fails this test
        loudly rather than silently breaking the floor invariant.
        """
        from kai.memory import _SPEAKER_WEIGHTS, MemoryResult, _speaker_weight

        # Confidence floor (0.5) and ceiling (1.0) come from the fact
        # schema's [0.5, 1.0] range. Ceilings above 1.0 are not valid
        # production values and would themselves be a separate bug.
        confidences = (0.5, 0.7, 0.85, 1.0)
        for speaker in _SPEAKER_WEIGHTS:
            for confidence in confidences:
                row = MemoryResult(
                    id=f"{speaker}-{confidence}",
                    text="placeholder",
                    score=0.5,
                    memory_type="fact",
                    metadata={
                        "type": "fact",
                        "source": "extracted",
                        "speaker": speaker,
                        "confidence": confidence,
                    },
                    created_at="2026-04-01T00:00:00",
                )
                w = _speaker_weight(row)
                assert 0.0 <= w <= 1.0, f"speaker={speaker} confidence={confidence} weight={w}"


# ── memory.recall logging ───────────────────────────────────────────


# Every uniform-shape field that must appear on every memory.recall log
# line, regardless of which return site emitted it. `reason` is
# deliberately excluded; it is the one non-uniform field, present only
# on short-circuit lines and absent on success.
_RECALL_UNIFORM_FIELDS = {
    "user_id",
    "query_len",
    "query",
    "fetch_limit",
    "hits_raw",
    "hits_after_floor",
    "floor",
    "latency_ms",
    "returned_empty",
    "lines_used",
    "budget_tokens",
    "hits",
}


def _parse_recall_log(caplog) -> dict[str, object]:
    """
    Find the single memory.recall record in caplog and return its
    parsed JSON payload.

    Asserts exactly one record so tests catch any future regression
    that double-emits or fails to emit. The "memory.recall " prefix
    is stripped before json.loads.

    Reads each record via `getMessage()` rather than the `.message`
    attribute. `LogRecord.message` is set as a side effect of
    `Formatter.format()`; pytest caplog populates it in practice but
    `getMessage()` is the documented stable API and renders the
    formatted message directly from the args, so it works whether or
    not a formatter has run on the record yet.
    """
    recall_records = [r for r in caplog.records if r.getMessage().startswith("memory.recall ")]
    assert len(recall_records) == 1, (
        f"expected exactly one memory.recall log, got {len(recall_records)}: {[r.getMessage() for r in recall_records]}"
    )
    blob = recall_records[0].getMessage()[len("memory.recall ") :]
    return json.loads(blob)


class TestRecallLogging:
    """
    Tests for the memory.recall structured log emit in format_context.

    The log line is the contract surface for the retrieval eval harness;
    schema regressions here would silently break downstream precision and
    recall scoring, so each shape and field invariant is asserted
    explicitly rather than spot-checked.
    """

    async def test_memory_recall_log_success_has_all_fields(self, caplog):
        """Success path emits one memory.recall line with every uniform field
        and no reason key."""
        import kai.memory as mem_mod
        from kai.memory import format_context

        # Two results, both above the default 0.3 floor, with raw scores
        # arranged so ONLY the speaker weighting can flip them: the
        # legacy/no-source row has the higher raw score (0.95), the
        # user-claimed row the lower (0.90). Adjusted scores are
        # 0.95 * 0.7 * 0.5 = 0.3325 (legacy default = assistant/0.5)
        # and 0.90 * 1.0 * 1.0 = 0.90 (user, full confidence), so the
        # user-claimed entry must come first in post-sort order. If
        # `_speaker_weight` were disabled or returned 1.0 for every
        # row, the assertion below on `payload["hits"][0]["id"]` would
        # fail because raw ordering would put the legacy row first.
        mock_mem = MagicMock()
        mock_mem.search.return_value = {
            "results": [
                {
                    "id": "a",
                    "memory": "User prefers Celsius",
                    "score": 0.90,
                    "metadata": {
                        "type": "fact",
                        "source": "extracted",
                        "speaker": "user",
                        "confidence": 1.0,
                    },
                    "created_at": "2026-04-01T10:00:00",
                },
                {
                    "id": "b",
                    "memory": "User said something old",
                    "score": 0.95,
                    # No source / speaker / confidence keys: simulates a
                    # legacy row from before this spec landed. The
                    # _read_time_speaker helper supplies the documented
                    # default (assistant, 0.5) at the ranking step, which
                    # is what makes adj = 0.3325 below the user-claimed
                    # row's 0.90. The per-hit log payload still records
                    # source="" and speaker="assistant" / confidence=0.5
                    # so a log analyst can reconstruct the multiplier.
                    "metadata": {"type": "exchange"},
                    "created_at": "2026-01-01T00:00:00",
                },
            ]
        }
        mem_mod._memory = mock_mem
        mem_mod._config = _make_config(memory_token_budget=2000)

        with caplog.at_level(logging.INFO, logger="kai.memory"):
            output = await format_context("temperature units", user_id="42")

        assert output != ""

        payload = _parse_recall_log(caplog)

        # Uniform-shape fields must all be present on success.
        missing = _RECALL_UNIFORM_FIELDS - set(payload.keys())
        assert not missing, f"missing uniform fields on success: {missing}"

        # `reason` is the one non-uniform field; success paths must omit it.
        assert "reason" not in payload, "success log must not carry a reason"

        # Spot-check field types and values.
        assert payload["user_id"] == "42"
        assert payload["query"] == "temperature units"
        assert payload["query_len"] == len("temperature units")
        assert payload["returned_empty"] is False
        assert payload["hits_raw"] == 2
        assert payload["hits_after_floor"] == 2
        assert payload["lines_used"] == 2
        assert payload["floor"] == 0.3
        assert payload["budget_tokens"] == 2000
        assert payload["fetch_limit"] >= 10  # at least the configured search limit
        assert payload["latency_ms"] >= 0  # wall-time, can be 0 for mocked instant search
        assert isinstance(payload["hits"], list) and len(payload["hits"]) == 2

        # Per-hit shape: id, source, speaker, confidence, score, adj,
        # snippet. `speaker` and `confidence` ride alongside `source`
        # so a log analyst can reconstruct the demote multiplier
        # without re-fetching the row; both are derived through
        # _read_time_speaker so legacy rows log defaulted constants
        # rather than missing-field markers. `id` exists so a
        # downstream consumer (the retrieval eval harness) can match
        # a probe's expected_fact_id against the actual hit.
        for hit in payload["hits"]:
            assert set(hit.keys()) == {
                "id",
                "source",
                "speaker",
                "confidence",
                "score",
                "adj",
                "snippet",
            }
            assert isinstance(hit["id"], str)
            assert isinstance(hit["score"], float)
            assert isinstance(hit["adj"], float)
            assert isinstance(hit["snippet"], str)
            assert isinstance(hit["speaker"], str)
            assert isinstance(hit["confidence"], (int, float))

        # Per-hit `id` passes through from MemoryResult.id. The mocked
        # search returned rows with ids "a" (extracted) and "b" (legacy);
        # both must appear in the payload's hits array. Asserting on the
        # set rather than ordered list keeps this independent of the
        # adjusted-score sort assertion below.
        assert {hit["id"] for hit in payload["hits"]} == {"a", "b"}

        # Post-sort order: the user-claimed row (speaker=user,
        # confidence=1.0, weight 1.0) outranks the legacy row
        # (defaulted to assistant/0.5, weight 0.7) by adjusted score,
        # so the user-claimed hit must come first in the hits array
        # even though Mem0 returned the legacy row with the higher raw
        # score (0.95 vs 0.90). This is the assertion that would fail
        # if `_speaker_weight` ever stopped applying its multiplier;
        # the mock is built so raw ordering and adjusted
        # ordering disagree.
        assert payload["hits"][0]["source"] == "extracted"
        assert payload["hits"][1]["source"] == ""

    @pytest.mark.parametrize(
        "reason,setup",
        [
            # disabled: _memory is None or _config is None. Easiest to
            # leave both at their reset defaults via the autouse fixture.
            ("disabled", "disabled"),
            # empty_query: passes a whitespace-only string after init.
            ("empty_query", "empty_query"),
            # no_results: search returns []. Exercises the post-search
            # short-circuit before the floor filter.
            ("no_results", "no_results"),
            # all_below_floor: search returns one result whose raw score
            # sits beneath the configured floor.
            ("all_below_floor", "all_below_floor"),
            # budget_exhausted: results pass the floor but no formatted
            # line fits the configured budget.
            ("budget_exhausted", "budget_exhausted"),
        ],
    )
    async def test_memory_recall_log_uniform_shape_on_short_circuit(self, caplog, reason, setup):
        """Each short-circuit return emits exactly one memory.recall line
        with the uniform schema, returned_empty=True, and the expected
        reason value. Parametrized so a regression that collapses two
        paths to the same reason value fails loudly."""
        import kai.memory as mem_mod
        from kai.memory import format_context

        # Setup per case. Each branch leaves the module state in the
        # exact shape needed to trigger one and only one short-circuit.
        if setup == "disabled":
            # Default state from the autouse fixture: _memory and
            # _config are both None.
            query = "anything"
        elif setup == "empty_query":
            mem_mod._memory = MagicMock()
            mem_mod._config = _make_config()
            query = "   "  # whitespace-only, hits the strip() guard
        elif setup == "no_results":
            mock = MagicMock()
            mock.search.return_value = {"results": []}
            mem_mod._memory = mock
            mem_mod._config = _make_config()
            query = "anything"
        elif setup == "all_below_floor":
            mock = MagicMock()
            mock.search.return_value = {
                "results": [
                    {
                        "id": "low",
                        "memory": "barely related",
                        "score": 0.1,
                        "metadata": {"type": "exchange"},
                        "created_at": "2026-04-01T00:00:00",
                    }
                ]
            }
            mem_mod._memory = mock
            # Floor at 0.3 (default in _make_config); the 0.1 score is below it.
            mem_mod._config = _make_config()
            query = "anything"
        elif setup == "budget_exhausted":
            mock = MagicMock()
            mock.search.return_value = {
                "results": [
                    {
                        "id": "any",
                        # The header alone (~70 chars / ~17 tokens via
                        # _estimate_tokens) already exceeds budget=10
                        # below, so the for-loop's first iteration
                        # `if used_tokens + line_tokens > budget: break`
                        # fires regardless of memory text. Any non-empty
                        # text triggers the same path; the content
                        # itself is not what's load-bearing here, the
                        # header's own token cost is.
                        "memory": "any text suffices",
                        "score": 0.9,
                        "metadata": {"type": "fact", "source": "extracted"},
                        "created_at": "2026-04-01T00:00:00",
                    }
                ]
            }
            mem_mod._memory = mock
            # Budget set below the header's own token cost so no line
            # can ever be appended; len(lines) <= 1 then short-circuits.
            mem_mod._config = _make_config(memory_token_budget=10)
            query = "anything"
        else:
            raise AssertionError(f"unhandled setup: {setup}")

        with caplog.at_level(logging.INFO, logger="kai.memory"):
            output = await format_context(query, user_id="99")

        assert output == "", f"short-circuit case {reason} unexpectedly returned non-empty"

        payload = _parse_recall_log(caplog)

        # Uniform-shape fields all present on every short-circuit line.
        missing = _RECALL_UNIFORM_FIELDS - set(payload.keys())
        assert not missing, f"missing uniform fields on {reason}: {missing}"

        # Always-real fields: user_id and query_len use real values
        # (no sentineling), even when other fields are sentineled.
        assert payload["user_id"] == "99"
        assert payload["query_len"] == len(query)
        assert payload["returned_empty"] is True
        assert payload["reason"] == reason

        # Floor is read from _config BEFORE the search call (not at the
        # filter site), so post-search short-circuits carry the real
        # threshold. disabled and empty_query short-circuit earlier and
        # keep the 0.0 sentinel from _base_recall_payload. Locks in the
        # eval-harness contract that "operator can recover the floor in
        # effect for any no_results / all_below_floor / budget_exhausted
        # log line" without re-running search.
        if reason in ("no_results", "all_below_floor", "budget_exhausted"):
            assert payload["floor"] == 0.3, (
                f"post-search short-circuit {reason} must carry real floor; got {payload['floor']}"
            )
        else:
            assert payload["floor"] == 0.0, (
                f"pre-search short-circuit {reason} should keep 0.0 sentinel; got {payload['floor']}"
            )

    async def test_memory_recall_log_snippet_and_query_truncated_and_sanitized(self, caplog):
        """Both the query field and per-hit snippets honor the 80-char
        truncation cap and rewrite \\n and \\r into single spaces. The
        eval harness treats snippets as fingerprints and parses log
        lines as single-line JSON; either escape leaking through
        breaks both contracts at once."""
        import kai.memory as mem_mod
        from kai.memory import format_context

        # Long memory text that contains both \n and \r. The text is
        # well over 80 chars so truncation is exercised.
        long_text = "Line one of stored memory\nLine two with newline\rAnd a CR plus more padding text " * 5
        mock_mem = MagicMock()
        mock_mem.search.return_value = {
            "results": [
                {
                    "id": "long",
                    "memory": long_text,
                    "score": 0.9,
                    "metadata": {"type": "fact", "source": "extracted"},
                    "created_at": "2026-04-01T00:00:00",
                }
            ]
        }
        mem_mod._memory = mock_mem
        mem_mod._config = _make_config(memory_token_budget=2000)

        # Query also contains both escapes and exceeds 80 chars so the
        # query-field path is exercised on the same call.
        long_query = "what do I prefer for editing\nplus a newline\rand carriage return then a long tail" * 3

        with caplog.at_level(logging.INFO, logger="kai.memory"):
            await format_context(long_query, user_id="7")

        payload = _parse_recall_log(caplog)

        # Query field must be capped and free of newlines / CRs.
        assert len(payload["query"]) <= 80
        assert "\n" not in payload["query"]
        assert "\r" not in payload["query"]

        # Every snippet must be capped and free of newlines / CRs.
        assert payload["hits"], "expected at least one hit"
        for hit in payload["hits"]:
            assert len(hit["snippet"]) <= 80
            assert "\n" not in hit["snippet"]
            assert "\r" not in hit["snippet"]


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


class TestDeleteBySource:
    """Tests for delete_by_source() scoped delete primitive (spec §6.2)."""

    def test_delete_by_source_disabled_returns_zero(self):
        """delete_by_source returns 0 when memory is disabled; never raises."""
        from kai.memory import delete_by_source

        assert asyncio.run(delete_by_source("user-a", "extracted")) == 0

    def test_delete_by_source_deletes_only_matching_source(self):
        """Only rows whose metadata source equals `source` are deleted."""
        import kai.memory as mem_mod
        from kai.memory import delete_by_source

        mock_mem = MagicMock()
        # Three rows: two extracted, one user_raw. Expect two deletes and
        # zero touches on the user_raw row's id.
        mock_mem.get_all.return_value = {
            "results": [
                {"id": "e1", "metadata": {"source": "extracted"}},
                {"id": "u1", "metadata": {"source": "user_raw"}},
                {"id": "e2", "metadata": {"source": "extracted"}},
            ]
        }
        mem_mod._memory = mock_mem

        count = asyncio.run(delete_by_source("user-a", "extracted"))

        assert count == 2
        deleted_ids = {call.kwargs["memory_id"] for call in mock_mem.delete.call_args_list}
        assert deleted_ids == {"e1", "e2"}
        # user_raw row must not have been deleted.
        assert "u1" not in deleted_ids

    def test_delete_by_source_empty_string_matches_legacy_and_empty(self):
        """source='' matches rows with missing metadata, absent source key, and empty-string source."""
        import kai.memory as mem_mod
        from kai.memory import delete_by_source

        mock_mem = MagicMock()
        mock_mem.get_all.return_value = {
            "results": [
                # Row 1: metadata key absent entirely (pre-spec rows).
                {"id": "legacy-no-meta"},
                # Row 2: metadata dict present but "source" key missing.
                {"id": "legacy-no-src", "metadata": {"type": "exchange"}},
                # Row 3: explicit empty-string source.
                {"id": "empty-src", "metadata": {"source": ""}},
                # Row 4: a real source value - should NOT be deleted.
                {"id": "keep-me", "metadata": {"source": "extracted"}},
            ]
        }
        mem_mod._memory = mock_mem

        count = asyncio.run(delete_by_source("user-a", ""))

        assert count == 3
        deleted_ids = {call.kwargs["memory_id"] for call in mock_mem.delete.call_args_list}
        assert deleted_ids == {"legacy-no-meta", "legacy-no-src", "empty-src"}
        assert "keep-me" not in deleted_ids

    def test_delete_by_source_tolerates_valueerror(self):
        """ValueError from Mem0.delete is swallowed mid-loop; remaining matches still delete."""
        import kai.memory as mem_mod
        from kai.memory import delete_by_source

        mock_mem = MagicMock()
        mock_mem.get_all.return_value = {
            "results": [
                {"id": "gone", "metadata": {"source": "extracted"}},
                {"id": "still-there", "metadata": {"source": "extracted"}},
            ]
        }
        # First delete raises ValueError (simulating an already-gone id under
        # concurrent cleanup); second delete succeeds. Count reflects only
        # successful deletes.
        mock_mem.delete.side_effect = [ValueError("not found"), None]
        mem_mod._memory = mock_mem

        count = asyncio.run(delete_by_source("user-a", "extracted"))

        assert count == 1
        assert mock_mem.delete.call_count == 2

    def test_delete_by_source_drains_multiple_pages(self, monkeypatch):
        """The loop keeps calling get_all until a short page or non-match page terminates it."""
        import kai.memory as mem_mod
        from kai.memory import delete_by_source

        # Shrink page size instead of synthesizing 20k rows. The spec calls
        # out this monkeypatch approach explicitly (§13.1).
        monkeypatch.setattr(mem_mod, "_DELETE_PAGE_SIZE", 4)

        mock_mem = MagicMock()
        full_page = [{"id": f"p1-{i}", "metadata": {"source": "extracted"}} for i in range(4)]
        partial_page = [
            {"id": "p2-0", "metadata": {"source": "extracted"}},
            {"id": "p2-1", "metadata": {"source": "extracted"}},
        ]
        # Round 1: full page of matches -> drain + loop continues.
        # Round 2: partial page (len < page size) -> terminates after delete.
        mock_mem.get_all.side_effect = [
            {"results": full_page},
            {"results": partial_page},
        ]
        mem_mod._memory = mock_mem

        count = asyncio.run(delete_by_source("user-a", "extracted"))

        assert count == 6
        assert mock_mem.get_all.call_count == 2

    def test_delete_by_source_live_lock_guard_on_full_non_matching_page(self, monkeypatch):
        """A full page with zero matches terminates the loop (live-lock guard)."""
        import kai.memory as mem_mod
        from kai.memory import delete_by_source

        monkeypatch.setattr(mem_mod, "_DELETE_PAGE_SIZE", 4)

        mock_mem = MagicMock()
        # Full page (len == _DELETE_PAGE_SIZE) but zero rows match. Without
        # the live-lock guard, the loop would spin on the same page forever
        # because Mem0's get_all is cursor-less.
        mock_mem.get_all.return_value = {
            "results": [{"id": f"u-{i}", "metadata": {"source": "user_raw"}} for i in range(4)]
        }
        mem_mod._memory = mock_mem

        count = asyncio.run(delete_by_source("user-a", "extracted"))

        assert count == 0
        # Exactly one call; the guard fires immediately on the non-matching page.
        assert mock_mem.get_all.call_count == 1

    def test_delete_by_source_tail_miss_emits_explicit_warning(self, monkeypatch, caplog):
        """PR #333 review finding #3. The live-lock termination path is
        also a correctness-loss path: if matching rows exist PAST the
        first _DELETE_PAGE_SIZE non-matching rows in Mem0's row order,
        they are not deleted. Operators need to see an explicit log
        line for this case; the generic page-drain warning is not
        enough to distinguish 'completed normally' from 'terminated
        early, possible tail miss'."""
        import logging

        import kai.memory as mem_mod
        from kai.memory import delete_by_source

        monkeypatch.setattr(mem_mod, "_DELETE_PAGE_SIZE", 4)

        mock_mem = MagicMock()
        mock_mem.get_all.return_value = {
            "results": [{"id": f"u-{i}", "metadata": {"source": "user_raw"}} for i in range(4)]
        }
        mem_mod._memory = mock_mem

        with caplog.at_level(logging.WARNING, logger="kai.memory"):
            asyncio.run(delete_by_source("user-a", "extracted"))

        # The warning must name the tail-miss specifically; a generic
        # drain message would not help an operator diagnose an
        # incomplete delete after the fact.
        joined = "\n".join(r.message for r in caplog.records)
        assert "Possible incomplete delete" in joined
        assert "user-a" in joined
        assert "'extracted'" in joined


class TestCountBySource:
    """Tests for count_by_source() (issue #406, Phase 4 of #396).

    Read-side companion to delete_by_source. Sync (no asyncio plumbing)
    because the migration script's idempotency guard runs on the main
    sync thread before any async rollback work.
    """

    def test_count_by_source_disabled_returns_zero(self):
        """count_by_source returns 0 when memory is disabled; never raises."""
        from kai.memory import count_by_source

        assert count_by_source("user-a", "migration") == 0

    def test_count_by_source_returns_count(self, monkeypatch):
        """Counts only rows whose metadata source matches; ignores others.

        Uses monkeypatch (not direct attribute assignment) so an
        assertion failure does not leak the mock into subsequent tests.
        """
        from kai.memory import count_by_source

        mock_mem = MagicMock()
        # Mixed-source rows: three migration, one extracted, one legacy.
        # Expected count for "migration" is 3; extracted/legacy are not
        # counted under that filter.
        mock_mem.get_all.return_value = {
            "results": [
                {"id": "m1", "metadata": {"source": "migration"}},
                {"id": "e1", "metadata": {"source": "extracted"}},
                {"id": "m2", "metadata": {"source": "migration"}},
                {"id": "leg", "metadata": {}},
                {"id": "m3", "metadata": {"source": "migration"}},
            ]
        }
        monkeypatch.setattr("kai.memory._memory", mock_mem)

        assert count_by_source("user-a", "migration") == 3

    def test_count_by_source_handles_empty_user(self, monkeypatch):
        """Empty store returns 0 cleanly (no exceptions)."""
        from kai.memory import count_by_source

        mock_mem = MagicMock()
        mock_mem.get_all.return_value = {"results": []}
        monkeypatch.setattr("kai.memory._memory", mock_mem)

        assert count_by_source("user-a", "migration") == 0

    def test_count_by_source_does_not_loop_on_full_page(self, monkeypatch, caplog):
        """Regression: count_by_source must call get_all exactly once.

        An earlier shape (mirroring delete_by_source's paged loop)
        would re-fetch when the page came back full with matches,
        because Mem0's get_all has no offset and read-only operations
        don't shrink the store. The loop would either hang forever or
        silently double-count matching rows. Pin the single-fetch
        contract so a future refactor cannot reintroduce the loop.
        """
        import logging

        import kai.memory as mem_mod
        from kai.memory import count_by_source

        # Build a full page (_DELETE_PAGE_SIZE rows) of matches. If
        # the function loops, the second iteration would re-add the
        # same _DELETE_PAGE_SIZE matches and the assertion would
        # observe call_count > 1 OR a doubled count (whichever the
        # broken loop hit first).
        full_page = [{"id": f"m{i}", "metadata": {"source": "migration"}} for i in range(mem_mod._DELETE_PAGE_SIZE)]
        mock_mem = MagicMock()
        mock_mem.get_all.return_value = {"results": full_page}
        monkeypatch.setattr("kai.memory._memory", mock_mem)

        with caplog.at_level(logging.WARNING, logger="kai.memory"):
            count = count_by_source("user-a", "migration")

        assert count == mem_mod._DELETE_PAGE_SIZE
        assert mock_mem.get_all.call_count == 1
        # When the cap fires, the function logs a warning so the
        # operator knows the count is a lower bound, not silent.
        # The wording must spell out total vs matched so an operator
        # whose match count is 0 but whose total exceeds the cap does
        # not misread the warning as alarming.
        joined = "\n".join(r.message for r in caplog.records)
        assert "page cap" in joined
        assert "lower bound" in joined


class TestMigrationSourceMetadata:
    """Tests for the migration source tag's rendering label in
    _SOURCE_SHORT. The retrieval-side weighting that previously lived
    in _SOURCE_WEIGHTS now goes through _speaker_weight, which uses
    speaker rather than source; migration rows pick up speaker="user"
    and confidence=0.9 via the read-time helper. The "Speaker" axis
    tests live with the speaker-weight tests; this class keeps only
    the per-line label assertion that is genuinely about source.
    """

    def test_migration_renders_as_fact_prefix_in_format_context(self):
        """Migration rows render with the same line-prefix label as
        extracted rows. Spec §D3: source tag is for dedup/rollback,
        not for prompt-side labeling.
        """
        from kai.memory import _SOURCE_SHORT

        assert _SOURCE_SHORT["migration"] == _SOURCE_SHORT["extracted"]

    def test_build_migration_metadata_sets_required_fields(self):
        """The migration writer and the tests both drive
        `build_migration_metadata` for the metadata bundle. Pin the
        full dict shape: source / speaker / confidence / section /
        subsection. The speaker and confidence values come from the
        migration constants so a swap-the-constants follow-up flows
        through automatically.
        """
        from kai.memory import (
            _MIGRATION_CONFIDENCE,
            _MIGRATION_SPEAKER,
            build_migration_metadata,
        )

        # Standard h3 chunk shape: section + subsection both populated.
        meta = build_migration_metadata(section="Architecture", subsection="Memory")
        assert meta == {
            "source": "migration",
            "speaker": _MIGRATION_SPEAKER,
            "confidence": _MIGRATION_CONFIDENCE,
            "section": "Architecture",
            "subsection": "Memory",
        }

        # H2-chunk shape: subsection is the empty string, NOT
        # missing. Centralizing this here means the Qdrant rows
        # have a uniform schema regardless of chunk depth, so a
        # future tag-renderer or section browser does not have to
        # branch on key presence.
        meta_h2 = build_migration_metadata(section="Conventions", subsection="")
        assert meta_h2["subsection"] == ""
        assert meta_h2["section"] == "Conventions"


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


# ── Spec 310 §7.2: get_all limit + new helpers + extended stats ─────


class TestGetAllLimit:
    """The new `limit` parameter on `get_all`. Default behavior is
    preserved for legacy callers (top_k=1000); /memory paths pass
    `limit=None` to bypass the cap so users with >1000 facts see a
    true total instead of a silent flatten."""

    def test_default_limit_is_1000(self):
        """No-arg call still requests top_k=1000 so existing callers
        (delete_by_source pagination, ad-hoc admin) keep their bounded
        memory footprint."""
        import kai.memory as mem_mod
        from kai.memory import get_all

        mock_mem = MagicMock()
        mock_mem.get_all.return_value = {"results": []}
        mem_mod._memory = mock_mem

        get_all(user_id="123")
        mock_mem.get_all.assert_called_once()
        kwargs = mock_mem.get_all.call_args.kwargs
        assert kwargs["top_k"] == 1000

    def test_explicit_limit_passed_through(self):
        import kai.memory as mem_mod
        from kai.memory import get_all

        mock_mem = MagicMock()
        mock_mem.get_all.return_value = {"results": []}
        mem_mod._memory = mock_mem

        get_all(user_id="123", limit=42)
        assert mock_mem.get_all.call_args.kwargs["top_k"] == 42

    def test_none_limit_uses_high_ceiling(self):
        """`limit=None` requests a top_k far above any realistic
        per-user count. Spec 310 §7.2.1 names 100000."""
        import kai.memory as mem_mod
        from kai.memory import get_all

        mock_mem = MagicMock()
        mock_mem.get_all.return_value = {"results": []}
        mem_mod._memory = mock_mem

        get_all(user_id="123", limit=None)
        # The exact ceiling is an internal choice; require only that it
        # is "well above realistic per-user counts" which the spec
        # quantifies as 100000.
        assert mock_mem.get_all.call_args.kwargs["top_k"] >= 100_000


class TestGetByTag:
    """Data-layer helper for tag-keyed lookups across user-visible sources."""

    def test_disabled_returns_empty(self):
        from kai.memory import get_by_tag

        assert get_by_tag(user_id="123", tag="preference") == []

    def test_filters_by_tag_and_source(self):
        """Only rows that are BOTH source==extracted AND have the tag
        are returned. user_raw rows that happen to carry a tag
        (defensive against future writers) are excluded."""
        import kai.memory as mem_mod
        from kai.memory import get_by_tag

        mock_mem = MagicMock()
        mock_mem.get_all.return_value = {
            "results": [
                # Match: extracted + tag.
                {
                    "id": "1",
                    "memory": "Prefers Celsius",
                    "score": 0.0,
                    "metadata": {
                        "type": "fact",
                        "source": "extracted",
                        "tags": ["preference"],
                    },
                    "created_at": "2026-04-01T00:00:00",
                    "updated_at": "2026-04-01T00:00:00",
                },
                # Excluded: extracted but different tag.
                {
                    "id": "2",
                    "memory": "Lives in Boston",
                    "score": 0.0,
                    "metadata": {
                        "type": "fact",
                        "source": "extracted",
                        "tags": ["location"],
                    },
                    "created_at": "2026-04-02T00:00:00",
                    "updated_at": "2026-04-02T00:00:00",
                },
                # Excluded: user_raw source even with a matching tag.
                # Defends against future writers; the UI is documented
                # as extracted-only.
                {
                    "id": "3",
                    "memory": "Likes vim",
                    "score": 0.0,
                    "metadata": {
                        "type": "exchange",
                        "source": "user_raw",
                        "tags": ["preference"],
                    },
                    "created_at": "2026-04-03T00:00:00",
                    "updated_at": "2026-04-03T00:00:00",
                },
                # Excluded: legacy row missing source.
                {
                    "id": "4",
                    "memory": "Old entry",
                    "score": 0.0,
                    "metadata": {"type": "exchange", "tags": ["preference"]},
                    "created_at": "2026-01-01T00:00:00",
                    "updated_at": "2026-01-01T00:00:00",
                },
            ]
        }
        mem_mod._memory = mock_mem

        results = get_by_tag(user_id="123", tag="preference")
        assert [r.id for r in results] == ["1"]

    def test_handles_multiple_tags_per_row(self):
        """A fact tagged [preference, constraint] matches BOTH
        get_by_tag('preference') AND get_by_tag('constraint'). Tags
        are independent in metadata, so a row can be reached through
        any of its tags."""
        import kai.memory as mem_mod
        from kai.memory import get_by_tag

        mock_mem = MagicMock()
        mock_mem.get_all.return_value = {
            "results": [
                {
                    "id": "1",
                    "memory": "No em dashes",
                    "score": 0.0,
                    "metadata": {
                        "type": "fact",
                        "source": "extracted",
                        "tags": ["preference", "constraint"],
                    },
                    "created_at": "2026-04-01T00:00:00",
                    "updated_at": "2026-04-01T00:00:00",
                },
            ]
        }
        mem_mod._memory = mock_mem

        assert len(get_by_tag(user_id="123", tag="preference")) == 1
        assert len(get_by_tag(user_id="123", tag="constraint")) == 1

    def test_sorted_by_updated_at_desc(self):
        """Newest-updated row comes first - so a re-extracted fact
        bubbles to the top of the tag list (spec §6.2)."""
        import kai.memory as mem_mod
        from kai.memory import get_by_tag

        mock_mem = MagicMock()
        mock_mem.get_all.return_value = {
            "results": [
                {
                    "id": "old",
                    "memory": "Older fact",
                    "score": 0.0,
                    "metadata": {
                        "type": "fact",
                        "source": "extracted",
                        "tags": ["preference"],
                    },
                    "created_at": "2026-01-01T00:00:00",
                    "updated_at": "2026-01-01T00:00:00",
                },
                {
                    "id": "new",
                    "memory": "Newer fact",
                    "score": 0.0,
                    "metadata": {
                        "type": "fact",
                        "source": "extracted",
                        "tags": ["preference"],
                    },
                    "created_at": "2026-04-01T00:00:00",
                    "updated_at": "2026-04-15T00:00:00",
                },
            ]
        }
        mem_mod._memory = mock_mem

        results = get_by_tag(user_id="123", tag="preference")
        assert [r.id for r in results] == ["new", "old"]

    def test_handles_missing_tags_metadata(self):
        """A row without a tags list (defensive) simply does not
        match any tag. Should not raise."""
        import kai.memory as mem_mod
        from kai.memory import get_by_tag

        mock_mem = MagicMock()
        mock_mem.get_all.return_value = {
            "results": [
                {
                    "id": "no-tags",
                    "memory": "Some fact",
                    "score": 0.0,
                    "metadata": {"type": "fact", "source": "extracted"},
                    "created_at": "2026-04-01T00:00:00",
                    "updated_at": "2026-04-01T00:00:00",
                },
            ]
        }
        mem_mod._memory = mock_mem

        assert get_by_tag(user_id="123", tag="preference") == []

    def test_passes_limit_none_to_get_all(self):
        """get_by_tag must call get_all with limit=None so the tag
        listing is not silently truncated for users with >1000 facts."""
        import kai.memory as mem_mod
        from kai.memory import get_by_tag

        mock_mem = MagicMock()
        mock_mem.get_all.return_value = {"results": []}
        mem_mod._memory = mock_mem

        get_by_tag(user_id="123", tag="preference")
        # top_k must be the high ceiling, not the legacy 1000.
        assert mock_mem.get_all.call_args.kwargs["top_k"] >= 100_000


class TestGetById:
    """Spec 310 §7.2 helper: ownership + source-scoped single fetch.

    `get_by_id` is the single source of truth for "is this fact
    addressable by /memory under this user?" - delete_by_id calls
    it, and the fact-view / forget-fact-confirm screens call it
    directly. The four not-found cases (missing, wrong user,
    non-extracted, fetch error) all collapse to None; the UI treats
    them identically."""

    def test_disabled_returns_none(self):
        from kai.memory import get_by_id

        assert get_by_id(user_id="123", memory_id="abc") is None

    def test_returns_none_when_row_missing(self):
        import kai.memory as mem_mod
        from kai.memory import get_by_id

        mock_mem = MagicMock()
        mock_mem.get.return_value = None
        mem_mod._memory = mock_mem

        assert get_by_id(user_id="123", memory_id="missing") is None

    def test_returns_none_on_user_mismatch(self):
        """A memory_id that resolves to another user's row must not
        leak through this fetch. Same blast radius rationale as
        delete_by_id: cross-user data exposure on a multi-user install."""
        import kai.memory as mem_mod
        from kai.memory import get_by_id

        mock_mem = MagicMock()
        mock_mem.get.return_value = {
            "id": "abc",
            "memory": "secret",
            "user_id": "other-user",
            "metadata": {"source": "extracted"},
        }
        mem_mod._memory = mock_mem

        assert get_by_id(user_id="123", memory_id="abc") is None

    def test_returns_none_on_non_extracted_source(self):
        """Track 1 / legacy rows are intentionally invisible to
        /memory UI surfaces - they belong to memory_admin.py."""
        import kai.memory as mem_mod
        from kai.memory import get_by_id

        mock_mem = MagicMock()
        mock_mem.get.return_value = {
            "id": "abc",
            "memory": "raw exchange",
            "user_id": "123",
            "metadata": {"source": "user_raw"},
        }
        mem_mod._memory = mock_mem

        assert get_by_id(user_id="123", memory_id="abc") is None

    def test_returns_none_on_legacy_missing_source(self):
        """Legacy rows with no source key are also invisible."""
        import kai.memory as mem_mod
        from kai.memory import get_by_id

        mock_mem = MagicMock()
        mock_mem.get.return_value = {
            "id": "abc",
            "memory": "old",
            "user_id": "123",
            "metadata": {"type": "exchange"},
        }
        mem_mod._memory = mock_mem

        assert get_by_id(user_id="123", memory_id="abc") is None

    def test_returns_none_on_fetch_exception(self):
        """An unexpected Mem0 failure during get must not raise to
        the caller - the UI cannot do anything useful with a stack
        trace, and "no such fact" is a survivable degradation."""
        import kai.memory as mem_mod
        from kai.memory import get_by_id

        mock_mem = MagicMock()
        mock_mem.get.side_effect = RuntimeError("vector store down")
        mem_mod._memory = mock_mem

        assert get_by_id(user_id="123", memory_id="abc") is None

    def test_happy_path_wraps_result(self):
        import kai.memory as mem_mod
        from kai.memory import get_by_id

        mock_mem = MagicMock()
        mock_mem.get.return_value = {
            "id": "abc",
            "memory": "user prefers tea",
            "user_id": "123",
            "metadata": {"source": "extracted", "type": "preference"},
            "created_at": "2026-04-17T10:00:00Z",
            "updated_at": "2026-04-17T11:00:00Z",
        }
        mem_mod._memory = mock_mem

        result = get_by_id(user_id="123", memory_id="abc")
        assert result is not None
        assert result.id == "abc"
        assert result.text == "user prefers tea"
        assert result.memory_type == "preference"
        assert result.metadata.get("source") == "extracted"
        assert result.created_at == "2026-04-17T10:00:00Z"
        assert result.updated_at == "2026-04-17T11:00:00Z"


class TestDeleteById:
    """Spec 310 §7.2 helper: ownership + source-checked single delete.

    Now delegates ownership/source verification to get_by_id, so the
    test suite below covers the delete-specific behavior (the actual
    delete call, ValueError swallowing). The verify rules themselves
    are exercised in TestGetById."""

    def test_disabled_returns_false(self):
        from kai.memory import delete_by_id

        assert delete_by_id(user_id="123", memory_id="abc") is False

    def test_returns_false_when_row_missing(self):
        """Mem0's get returns None for an unknown id; delete_by_id
        must propagate that as False without calling delete."""
        import kai.memory as mem_mod
        from kai.memory import delete_by_id

        mock_mem = MagicMock()
        mock_mem.get.return_value = None
        mem_mod._memory = mock_mem

        assert delete_by_id(user_id="123", memory_id="missing") is False
        mock_mem.delete.assert_not_called()

    def test_returns_false_on_user_mismatch(self):
        """A memory_id that resolves to a different user's row must
        not be deleted - cross-user data deletion is the worst-case
        consequence of malformed callback data on a multi-user install."""
        import kai.memory as mem_mod
        from kai.memory import delete_by_id

        mock_mem = MagicMock()
        mock_mem.get.return_value = {
            "id": "abc",
            "user_id": "other-user",
            "metadata": {"source": "extracted"},
        }
        mem_mod._memory = mock_mem

        assert delete_by_id(user_id="123", memory_id="abc") is False
        mock_mem.delete.assert_not_called()

    def test_returns_false_on_non_extracted_source(self):
        """The /memory UI is documented to manage extracted memories
        only. Track 1 (user_raw) and legacy rows are owned by the
        admin CLI; refuse to delete them through this path."""
        import kai.memory as mem_mod
        from kai.memory import delete_by_id

        mock_mem = MagicMock()
        mock_mem.get.return_value = {
            "id": "abc",
            "user_id": "123",
            "metadata": {"source": "user_raw"},
        }
        mem_mod._memory = mock_mem

        assert delete_by_id(user_id="123", memory_id="abc") is False
        mock_mem.delete.assert_not_called()

    def test_returns_false_on_legacy_missing_source(self):
        """A legacy row whose metadata lacks a source key entirely
        must NOT be deletable through /memory. Same rationale as
        non-extracted: legacy is admin-CLI territory."""
        import kai.memory as mem_mod
        from kai.memory import delete_by_id

        mock_mem = MagicMock()
        mock_mem.get.return_value = {
            "id": "abc",
            "user_id": "123",
            "metadata": {"type": "exchange"},
        }
        mem_mod._memory = mock_mem

        assert delete_by_id(user_id="123", memory_id="abc") is False
        mock_mem.delete.assert_not_called()

    def test_happy_path_deletes_and_returns_true(self):
        import kai.memory as mem_mod
        from kai.memory import delete_by_id

        mock_mem = MagicMock()
        mock_mem.get.return_value = {
            "id": "abc",
            "user_id": "123",
            "metadata": {"source": "extracted"},
        }
        mem_mod._memory = mock_mem

        assert delete_by_id(user_id="123", memory_id="abc") is True
        mock_mem.delete.assert_called_once_with(memory_id="abc")

    def test_value_error_swallowed_as_false(self):
        """Mem0 raises ValueError when the row vanishes between get and
        delete. Treated as "nothing to do" (False) rather than an error
        - the user ends up where they wanted (row gone) and the UI is
        not blocked rendering a confusing failure."""
        import kai.memory as mem_mod
        from kai.memory import delete_by_id

        mock_mem = MagicMock()
        mock_mem.get.return_value = {
            "id": "abc",
            "user_id": "123",
            "metadata": {"source": "extracted"},
        }
        mock_mem.delete.side_effect = ValueError("Memory with id abc not found")
        mem_mod._memory = mock_mem

        assert delete_by_id(user_id="123", memory_id="abc") is False

    def test_handles_missing_metadata_dict(self):
        """Mem0 omits the metadata key entirely when the row has no
        extra payload (verified at mem0/memory/main.py). delete_by_id
        must not raise KeyError on that shape."""
        import kai.memory as mem_mod
        from kai.memory import delete_by_id

        mock_mem = MagicMock()
        mock_mem.get.return_value = {"id": "abc", "user_id": "123"}
        mem_mod._memory = mock_mem

        # No metadata -> source check fails, refuse to delete.
        assert delete_by_id(user_id="123", memory_id="abc") is False
        mock_mem.delete.assert_not_called()


class TestUpdateMetadataWrapper:
    """Issue #418, Sub D of #388: thin wrapper around Mem0's
    `update` that pins ownership + source and forwards the call.

    The wrapper's docstring documents (a) the metadata-overwrite
    contract (Mem0 REPLACES wholesale; callers must read-merge-write
    for partial updates) and (b) the source-scope contract (admits any
    USER_VISIBLE_SOURCES row). These tests pin both contracts so a
    future "helpful" auto-merge refactor cannot silently change the
    semantics that callers depend on."""

    def test_disabled_returns_false(self):
        from kai.memory import update_metadata

        assert update_metadata(user_id="123", memory_id="abc", data="x", metadata={"tags": []}) is False

    def test_returns_false_when_row_missing(self):
        import kai.memory as mem_mod
        from kai.memory import update_metadata

        mock_mem = MagicMock()
        mock_mem.get.return_value = None
        mem_mod._memory = mock_mem

        result = update_metadata(
            user_id="123",
            memory_id="missing",
            data="x",
            metadata={"tags": ["preference"]},
        )
        assert result is False
        mock_mem.update.assert_not_called()

    def test_returns_false_on_user_mismatch(self):
        """Cross-user metadata writes are the worst-case consequence
        of a malformed memory_id leak; the gate must refuse."""
        import kai.memory as mem_mod
        from kai.memory import update_metadata

        mock_mem = MagicMock()
        mock_mem.get.return_value = {
            "id": "abc",
            "user_id": "other-user",
            "metadata": {"source": "extracted"},
        }
        mem_mod._memory = mock_mem

        result = update_metadata(
            user_id="123",
            memory_id="abc",
            data="x",
            metadata={"tags": ["preference"]},
        )
        assert result is False
        mock_mem.update.assert_not_called()

    def test_returns_false_on_source_outside_user_visible(self):
        """Legacy ""-source rows and any future non-USER_VISIBLE
        sources are admin-CLI territory; refuse to write to them
        through this wrapper."""
        import kai.memory as mem_mod
        from kai.memory import update_metadata

        mock_mem = MagicMock()
        mock_mem.get.return_value = {
            "id": "abc",
            "user_id": "123",
            "metadata": {"source": ""},  # legacy
        }
        mem_mod._memory = mock_mem

        result = update_metadata(
            user_id="123",
            memory_id="abc",
            data="x",
            metadata={"tags": ["preference"]},
        )
        assert result is False
        mock_mem.update.assert_not_called()

    def test_mem0_exception_returns_false(self):
        """A Mem0 raise during update is caught, logged, and surfaces
        as False. Callers can treat False as "nothing to do" without
        an exception bubble."""
        import kai.memory as mem_mod
        from kai.memory import update_metadata

        mock_mem = MagicMock()
        mock_mem.get.return_value = {
            "id": "abc",
            "user_id": "123",
            "metadata": {"source": "extracted"},
        }
        mock_mem.update.side_effect = RuntimeError("vector store down")
        mem_mod._memory = mock_mem

        result = update_metadata(
            user_id="123",
            memory_id="abc",
            data="x",
            metadata={"tags": ["preference"]},
        )
        assert result is False

    def test_happy_path_calls_mem0_update_and_returns_true(self):
        import kai.memory as mem_mod
        from kai.memory import update_metadata

        mock_mem = MagicMock()
        mock_mem.get.return_value = {
            "id": "abc",
            "user_id": "123",
            "metadata": {"source": "extracted"},
        }
        mem_mod._memory = mock_mem

        result = update_metadata(
            user_id="123",
            memory_id="abc",
            data="fact text",
            metadata={"source": "extracted", "tags": ["preference"]},
        )
        assert result is True
        # The wrapper passes the caller-provided metadata dict through
        # to Mem0 unchanged; no auto-merge happens at this layer.
        mock_mem.update.assert_called_once_with(
            memory_id="abc",
            data="fact text",
            metadata={"source": "extracted", "tags": ["preference"]},
        )

    def test_sparse_metadata_passes_through_unchanged(self):
        """Pin the wrapper's destructive behavior: a sparse caller-
        provided metadata dict (e.g. {"tags": [...]}) MUST be passed
        through to Mem0 unmodified. The wrapper does NOT auto-merge
        with the existing row; callers wanting to preserve other
        fields are responsible for read-merge-write at the call site.

        A future "helpful" refactor that auto-merges in the wrapper
        would change the contract that the dedup script's
        apply_rewrites depends on (the script reads, merges, then
        passes the merged dict). Surfacing the pin here means the
        refactor cannot land silently."""
        import kai.memory as mem_mod
        from kai.memory import update_metadata

        mock_mem = MagicMock()
        mock_mem.get.return_value = {
            "id": "abc",
            "user_id": "123",
            "metadata": {
                "source": "extracted",
                "confidence": 0.9,
                "tags": ["original"],
            },
        }
        mem_mod._memory = mock_mem

        # Caller passes a sparse dict; the wrapper forwards it AS-IS.
        result = update_metadata(
            user_id="123",
            memory_id="abc",
            data="x",
            metadata={"tags": ["new"]},  # missing source, confidence
        )
        assert result is True
        # Verify Mem0 received the sparse dict literally; no merge with
        # existing fields happened in the wrapper.
        call_args = mock_mem.update.call_args
        passed_metadata = call_args.kwargs["metadata"]
        assert passed_metadata == {"tags": ["new"]}
        assert "source" not in passed_metadata
        assert "confidence" not in passed_metadata


class TestGetStatsExtended:
    """Spec 310 §6.6 / §7.2: extended MemoryStats fields.

    The legacy `total_count` / `by_type` fields are tested separately
    in TestGetStats; this class focuses on the new extracted-only
    aggregates."""

    def _stats_with(self, rows):
        """Helper: install a mock returning `rows` and call get_stats."""
        import kai.memory as mem_mod
        from kai.memory import get_stats

        mock_mem = MagicMock()
        mock_mem.get_all.return_value = {"results": rows}
        mem_mod._memory = mock_mem
        return get_stats(user_id="123")

    def test_extracted_count_excludes_other_sources(self):
        """extracted_count counts only source==extracted rows; total_count
        still counts everything for backward compat."""
        stats = self._stats_with(
            [
                {
                    "id": "1",
                    "memory": "f",
                    "score": 0.0,
                    "metadata": {"type": "fact", "source": "extracted", "confidence": 0.9},
                    "created_at": "",
                },
                {
                    "id": "2",
                    "memory": "u",
                    "score": 0.0,
                    "metadata": {"type": "exchange", "source": "user_raw"},
                    "created_at": "",
                },
                {
                    "id": "3",
                    "memory": "l",
                    "score": 0.0,
                    "metadata": {"type": "exchange"},
                    "created_at": "",
                },
            ]
        )
        assert stats.total_count == 3
        assert stats.extracted_count == 1

    def test_by_tag_aggregates_extracted_only(self):
        """A user_raw row carrying a tag does not contribute to by_tag,
        even if a future writer starts emitting tags on Track 1 rows."""
        stats = self._stats_with(
            [
                {
                    "id": "1",
                    "memory": "a",
                    "score": 0.0,
                    "metadata": {
                        "type": "fact",
                        "source": "extracted",
                        "tags": ["preference", "constraint"],
                        "confidence": 0.9,
                    },
                    "created_at": "",
                },
                {
                    "id": "2",
                    "memory": "b",
                    "score": 0.0,
                    "metadata": {
                        "type": "fact",
                        "source": "extracted",
                        "tags": ["preference"],
                        "confidence": 0.8,
                    },
                    "created_at": "",
                },
                {
                    "id": "3",
                    "memory": "c",
                    "score": 0.0,
                    "metadata": {
                        "type": "exchange",
                        "source": "user_raw",
                        "tags": ["preference"],
                    },
                    "created_at": "",
                },
            ]
        )
        assert stats.by_tag == {"preference": 2, "constraint": 1}

    def test_confidence_min_median_max_odd_count(self):
        """Three values [0.6, 0.8, 0.9]: min=0.6, median=0.8, max=0.9."""
        stats = self._stats_with(
            [
                {
                    "id": str(i),
                    "memory": "x",
                    "score": 0.0,
                    "metadata": {"source": "extracted", "tags": ["fact"], "confidence": c},
                    "created_at": "",
                }
                for i, c in enumerate([0.9, 0.6, 0.8])
            ]
        )
        assert stats.confidence_min == 0.6
        assert stats.confidence_median == 0.8
        assert stats.confidence_max == 0.9

    def test_confidence_median_even_count_picks_lower(self):
        """Spec §6.6: median with even count picks the LOWER of the two
        middle values (statistics.median_low semantics) rather than
        averaging them - averaging would synthesize a value no fact
        actually had."""
        # Four values [0.5, 0.7, 0.8, 1.0]; mean of middle two would
        # be 0.75, but median_low picks 0.7.
        stats = self._stats_with(
            [
                {
                    "id": str(i),
                    "memory": "x",
                    "score": 0.0,
                    "metadata": {"source": "extracted", "tags": ["fact"], "confidence": c},
                    "created_at": "",
                }
                for i, c in enumerate([1.0, 0.5, 0.8, 0.7])
            ]
        )
        assert stats.confidence_median == 0.7

    def test_confidence_below_thresholds(self):
        """Counts at the 0.7 and 0.6 cutoffs. Strictly less-than per spec
        (the boundary value itself does not count toward "below")."""
        stats = self._stats_with(
            [
                {
                    "id": str(i),
                    "memory": "x",
                    "score": 0.0,
                    "metadata": {"source": "extracted", "tags": ["fact"], "confidence": c},
                    "created_at": "",
                }
                # 0.55 below both; 0.65 below 0.7 only; 0.7 below neither;
                # 0.85 below neither.
                for i, c in enumerate([0.55, 0.65, 0.7, 0.85])
            ]
        )
        assert stats.confidence_below_0_7 == 2
        assert stats.confidence_below_0_6 == 1

    def test_confirmation_quote_count(self):
        stats = self._stats_with(
            [
                {
                    "id": "1",
                    "memory": "x",
                    "score": 0.0,
                    "metadata": {
                        "source": "extracted",
                        "tags": ["confirmed_action"],
                        "confidence": 0.9,
                        "confirmation_quote": "yes please go ahead and do that",
                    },
                    "created_at": "",
                },
                {
                    "id": "2",
                    "memory": "y",
                    "score": 0.0,
                    "metadata": {"source": "extracted", "tags": ["fact"], "confidence": 0.9},
                    "created_at": "",
                },
            ]
        )
        assert stats.confirmation_quote_count == 1

    def test_by_prompt_version_counts(self):
        stats = self._stats_with(
            [
                {
                    "id": "1",
                    "memory": "x",
                    "score": 0.0,
                    "metadata": {
                        "source": "extracted",
                        "tags": ["fact"],
                        "confidence": 0.9,
                        "prompt_version": "v3",
                    },
                    "created_at": "",
                },
                {
                    "id": "2",
                    "memory": "y",
                    "score": 0.0,
                    "metadata": {
                        "source": "extracted",
                        "tags": ["fact"],
                        "confidence": 0.9,
                        "prompt_version": "v3",
                    },
                    "created_at": "",
                },
                {
                    "id": "3",
                    "memory": "z",
                    "score": 0.0,
                    "metadata": {
                        "source": "extracted",
                        "tags": ["fact"],
                        "confidence": 0.9,
                        "prompt_version": "v2",
                    },
                    "created_at": "",
                },
            ]
        )
        assert stats.by_prompt_version == {"v3": 2, "v2": 1}

    def test_by_prompt_version_normalizes_int_to_str(self):
        # The aggregation must cast prompt_version to str at the
        # read boundary so that rows whose metadata stores the
        # version as an int (older revisions of the extraction code
        # wrote ints) bucket together with rows that store it as a
        # str. Without the cast, the dict ends up with mixed-type
        # keys, the dict[str, int] annotation becomes a lie, and
        # downstream renderers crash on len(int).
        stats = self._stats_with(
            [
                {
                    "id": "1",
                    "memory": "x",
                    "score": 0.0,
                    "metadata": {
                        "source": "extracted",
                        "tags": ["fact"],
                        "confidence": 0.9,
                        # Int-typed row - the shape produced by
                        # older extraction code that wrote int 1.
                        "prompt_version": 1,
                    },
                    "created_at": "",
                },
                {
                    "id": "2",
                    "memory": "y",
                    "score": 0.0,
                    "metadata": {
                        "source": "extracted",
                        "tags": ["fact"],
                        "confidence": 0.9,
                        # Str-typed row - what current writes produce.
                        "prompt_version": "1",
                    },
                    "created_at": "",
                },
            ]
        )
        # Single str-keyed bucket. Without the cast the dict has
        # mixed keys ({1: 1, "1": 1}) and this equality fails.
        assert stats.by_prompt_version == {"1": 2}

    def test_by_prompt_version_collapses_null_with_missing(self):
        # A metadata dict that explicitly stores prompt_version=None
        # must bucket together with rows that have no prompt_version
        # key at all - both indicate "version not stamped" and
        # surface the same way in the stats view. Without the
        # `... or ""` guard the cast turns None into the literal
        # string "None", producing a phantom bucket that looks like
        # a real version label.
        stats = self._stats_with(
            [
                {
                    "id": "1",
                    "memory": "x",
                    "score": 0.0,
                    "metadata": {
                        "source": "extracted",
                        "tags": ["fact"],
                        "confidence": 0.9,
                        # Explicit None - key present, value null.
                        "prompt_version": None,
                    },
                    "created_at": "",
                },
                {
                    "id": "2",
                    "memory": "y",
                    "score": 0.0,
                    "metadata": {
                        "source": "extracted",
                        "tags": ["fact"],
                        "confidence": 0.9,
                        # Key absent entirely - no prompt_version.
                    },
                    "created_at": "",
                },
            ]
        )
        # Both rows bucket under the empty-string sentinel; no
        # spurious "None" bucket appears.
        assert stats.by_prompt_version == {"": 2}
        assert "None" not in stats.by_prompt_version

    def test_empty_extracted_set_yields_none_confidence(self):
        """No extracted rows -> min/median/max are None (NOT 0.0) so
        the UI can render "n/a" rather than a misleading score."""
        stats = self._stats_with(
            [
                {
                    "id": "1",
                    "memory": "u",
                    "score": 0.0,
                    "metadata": {"type": "exchange", "source": "user_raw"},
                    "created_at": "",
                },
            ]
        )
        assert stats.extracted_count == 0
        assert stats.confidence_min is None
        assert stats.confidence_median is None
        assert stats.confidence_max is None
        assert stats.confidence_below_0_7 == 0
        assert stats.confidence_below_0_6 == 0
        assert stats.confirmation_quote_count == 0
        assert stats.by_tag == {}
        assert stats.by_prompt_version == {}

    def test_disabled_returns_zeroed_extended_fields(self):
        """When memory is disabled the new fields collapse to their
        sentinel defaults, just like total_count and by_type."""
        from kai.memory import get_stats

        stats = get_stats(user_id="123")
        assert stats.extracted_count == 0
        assert stats.confidence_min is None
        assert stats.by_tag == {}
        assert stats.by_prompt_version == {}


class TestUserVisibleSources:
    """Issue #407: episode and migration rows joined extracted as
    addressable sources from the /memory UI.

    The source-admit gate now lives in
    `memory.USER_VISIBLE_SOURCES = {"extracted", "episode", "migration"}`,
    used by `get_by_id` and `get_by_tag`. `delete_by_id` inherits via
    its delegation to `get_by_id`. Legacy ""-source rows stay hidden;
    they are managed via `delete_by_source` / memory_admin.py."""

    # ── get_by_id ──────────────────────────────────────────────────

    def test_get_by_id_returns_extracted_row(self):
        """Extracted rows still resolve through get_by_id (regression
        pin against the source-admit-set expansion). Mirrors the
        existing `test_happy_path_wraps_result` but sized to the
        post-#407 contract: an extracted row is one of three admitted
        sources, not the only one."""
        import kai.memory as mem_mod
        from kai.memory import get_by_id

        mock_mem = MagicMock()
        mock_mem.get.return_value = {
            "id": "abc",
            "memory": "user prefers tea",
            "user_id": "123",
            "metadata": {"source": "extracted", "type": "preference"},
            "created_at": "2026-04-29T10:00:00Z",
        }
        mem_mod._memory = mock_mem

        result = get_by_id(user_id="123", memory_id="abc")
        assert result is not None
        assert result.id == "abc"
        assert result.metadata.get("source") == "extracted"

    def test_get_by_id_returns_episode_row(self):
        """Episode rows (#385/#387) are now addressable from /memory
        so the fact-view can render Sophia-style fields. Pre-#407
        the same input returned None and the screen rendered "This
        memory no longer exists." for a row the operator could see
        in retrieval-time prompt injection."""
        import kai.memory as mem_mod
        from kai.memory import get_by_id

        mock_mem = MagicMock()
        mock_mem.get.return_value = {
            "id": "ep-1",
            "memory": "Set up cron",
            "user_id": "123",
            "metadata": {"source": "episode", "outcome_quality": "good"},
            "created_at": "2026-04-28T15:00:00Z",
        }
        mem_mod._memory = mock_mem

        result = get_by_id(user_id="123", memory_id="ep-1")
        assert result is not None
        assert result.id == "ep-1"
        assert result.metadata.get("source") == "episode"

    def test_get_by_id_returns_migration_row(self):
        """Migration rows (#406/#408) are also addressable so the
        fact-view's `Imported` shape can render. Same regression
        rationale as the episode case."""
        import kai.memory as mem_mod
        from kai.memory import get_by_id

        mock_mem = MagicMock()
        mock_mem.get.return_value = {
            "id": "mig-1",
            "memory": "### /backend\nRole-based assignment",
            "user_id": "123",
            "metadata": {"source": "migration", "tags": ["migration"]},
            "created_at": "2026-04-29T16:30:00Z",
        }
        mem_mod._memory = mock_mem

        result = get_by_id(user_id="123", memory_id="mig-1")
        assert result is not None
        assert result.id == "mig-1"
        assert result.metadata.get("source") == "migration"

    def test_get_by_id_rejects_legacy_source(self):
        """Legacy ""-source rows stay invisible to /memory after #407.
        The admit list is intentionally enumerated rather than
        "everything except legacy"; a value not in the set returns
        None just like the pre-#407 non-extracted branch did."""
        import kai.memory as mem_mod
        from kai.memory import get_by_id

        mock_mem = MagicMock()
        mock_mem.get.return_value = {
            "id": "legacy-1",
            "memory": "ancient row",
            "user_id": "123",
            "metadata": {"source": ""},
        }
        mem_mod._memory = mock_mem

        assert get_by_id(user_id="123", memory_id="legacy-1") is None

    # ── get_by_tag ─────────────────────────────────────────────────

    def test_get_by_tag_includes_all_user_visible_sources(self):
        """`get_by_tag` returns rows of every user-visible source
        that carries the queried tag in metadata. Cross-source
        overlap on a single tag value is rare in production (episode
        rows usually carry Sophia-style tags and migration rows
        usually carry H3-slug tags, with little overlap), but the
        contract is regression-pinned here for the rare case."""
        import kai.memory as mem_mod
        from kai.memory import get_by_tag

        mock_mem = MagicMock()
        mock_mem.get_all.return_value = {
            "results": [
                {
                    "id": "1",
                    "memory": "Prefers Celsius",
                    "score": 0.0,
                    "metadata": {"source": "extracted", "tags": ["preference"]},
                    "created_at": "2026-04-01T00:00:00",
                    "updated_at": "2026-04-01T00:00:00",
                },
                {
                    "id": "2",
                    "memory": "Configured CI to run on PR",
                    "score": 0.0,
                    "metadata": {"source": "episode", "tags": ["preference"]},
                    "created_at": "2026-04-02T00:00:00",
                    "updated_at": "2026-04-02T00:00:00",
                },
                {
                    "id": "3",
                    "memory": "### preferences\nNo em dashes",
                    "score": 0.0,
                    "metadata": {"source": "migration", "tags": ["preference"]},
                    "created_at": "2026-04-03T00:00:00",
                    "updated_at": "2026-04-03T00:00:00",
                },
            ]
        }
        mem_mod._memory = mock_mem

        results = get_by_tag(user_id="123", tag="preference")
        # Sort order is updated_at desc; assert membership rather
        # than ordering so this test stays focused on the source
        # admit-list expansion.
        assert {r.id for r in results} == {"1", "2", "3"}

    def test_get_by_tag_excludes_legacy_source(self):
        """A legacy ""-source row that happens to carry an enum tag
        in its metadata is still excluded - the admit list is the
        gate, not the tag list."""
        import kai.memory as mem_mod
        from kai.memory import get_by_tag

        mock_mem = MagicMock()
        mock_mem.get_all.return_value = {
            "results": [
                {
                    "id": "legacy",
                    "memory": "Old preference row",
                    "score": 0.0,
                    "metadata": {"source": "", "tags": ["preference"]},
                    "created_at": "2026-01-01T00:00:00",
                    "updated_at": "2026-01-01T00:00:00",
                },
            ]
        }
        mem_mod._memory = mock_mem

        assert get_by_tag(user_id="123", tag="preference") == []

    # ── delete_by_id (via get_by_id delegation) ────────────────────

    def test_delete_by_id_allows_episode_via_delegation(self):
        """delete_by_id has no source filter of its own; it delegates
        ownership and source admit to get_by_id. Verifies the chain
        works end-to-end for episode rows after #407."""
        import kai.memory as mem_mod
        from kai.memory import delete_by_id

        mock_mem = MagicMock()
        mock_mem.get.return_value = {
            "id": "ep-1",
            "user_id": "123",
            "metadata": {"source": "episode"},
        }
        mem_mod._memory = mock_mem

        assert delete_by_id(user_id="123", memory_id="ep-1") is True
        mock_mem.delete.assert_called_once_with(memory_id="ep-1")

    def test_delete_by_id_allows_migration_via_delegation(self):
        """Same delegation contract as the episode case - migration
        rows also reach the actual Mem0 delete call now."""
        import kai.memory as mem_mod
        from kai.memory import delete_by_id

        mock_mem = MagicMock()
        mock_mem.get.return_value = {
            "id": "mig-1",
            "user_id": "123",
            "metadata": {"source": "migration"},
        }
        mem_mod._memory = mock_mem

        assert delete_by_id(user_id="123", memory_id="mig-1") is True
        mock_mem.delete.assert_called_once_with(memory_id="mig-1")

    def test_delete_by_id_rejects_legacy_via_delegation(self):
        """Legacy ""-source rows still cannot be deleted through
        /memory - the inherited admit gate refuses them. They remain
        managed via `delete_by_source` / memory_admin.py."""
        import kai.memory as mem_mod
        from kai.memory import delete_by_id

        mock_mem = MagicMock()
        mock_mem.get.return_value = {
            "id": "legacy-1",
            "user_id": "123",
            "metadata": {"source": ""},
        }
        mem_mod._memory = mock_mem

        assert delete_by_id(user_id="123", memory_id="legacy-1") is False
        mock_mem.delete.assert_not_called()

    # ── get_stats per-source counts ───────────────────────────────

    def test_get_stats_episode_count(self):
        """`MemoryStats.episode_count` is computed over rows with
        metadata.source == "episode". Counts are independent of
        extracted_count and confidence aggregates."""
        import kai.memory as mem_mod
        from kai.memory import get_stats

        mock_mem = MagicMock()
        mock_mem.get_all.return_value = {
            "results": [
                {
                    "id": "1",
                    "memory": "ep one",
                    "score": 0.0,
                    "metadata": {"type": "fact", "source": "episode"},
                    "created_at": "",
                },
                {
                    "id": "2",
                    "memory": "ep two",
                    "score": 0.0,
                    "metadata": {"type": "fact", "source": "episode"},
                    "created_at": "",
                },
                {
                    "id": "3",
                    "memory": "extracted one",
                    "score": 0.0,
                    "metadata": {"type": "fact", "source": "extracted"},
                    "created_at": "",
                },
            ]
        }
        mem_mod._memory = mock_mem

        stats = get_stats(user_id="123")
        assert stats.episode_count == 2
        # extracted_count is independent of episode_count.
        assert stats.extracted_count == 1
        assert stats.migration_count == 0

    def test_get_stats_migration_count(self):
        """`MemoryStats.migration_count` is computed over rows with
        metadata.source == "migration". Same independence contract
        as episode_count."""
        import kai.memory as mem_mod
        from kai.memory import get_stats

        mock_mem = MagicMock()
        mock_mem.get_all.return_value = {
            "results": [
                {
                    "id": "1",
                    "memory": "mig one",
                    "score": 0.0,
                    "metadata": {"type": "fact", "source": "migration"},
                    "created_at": "",
                },
                {
                    "id": "2",
                    "memory": "mig two",
                    "score": 0.0,
                    "metadata": {"type": "fact", "source": "migration"},
                    "created_at": "",
                },
                {
                    "id": "3",
                    "memory": "mig three",
                    "score": 0.0,
                    "metadata": {"type": "fact", "source": "migration"},
                    "created_at": "",
                },
            ]
        }
        mem_mod._memory = mock_mem

        stats = get_stats(user_id="123")
        assert stats.migration_count == 3
        assert stats.extracted_count == 0
        assert stats.episode_count == 0

    def test_get_stats_confidence_stays_extracted_only(self):
        """Confidence aggregates compute over extracted-source rows
        only, regardless of whether non-extracted rows happen to
        carry the field. Episode rows do not carry confidence in
        production today; this test pins the principle (rather than
        the current data shape) so a future schema where episodes
        gain a confidence field cannot quietly contaminate the
        confidence histogram."""
        import kai.memory as mem_mod
        from kai.memory import get_stats

        mock_mem = MagicMock()
        mock_mem.get_all.return_value = {
            "results": [
                # Extracted with high confidence - the only row that
                # should contribute to the confidence aggregates.
                {
                    "id": "1",
                    "memory": "fact",
                    "score": 0.0,
                    "metadata": {
                        "type": "fact",
                        "source": "extracted",
                        "confidence": 0.9,
                        "tags": ["fact"],
                    },
                    "created_at": "",
                },
                # Synthetic episode row carrying a low confidence
                # value. Must NOT pull min/median/max down.
                {
                    "id": "2",
                    "memory": "ep",
                    "score": 0.0,
                    "metadata": {
                        "type": "fact",
                        "source": "episode",
                        "confidence": 0.1,
                    },
                    "created_at": "",
                },
            ]
        }
        mem_mod._memory = mock_mem

        stats = get_stats(user_id="123")
        # Only the extracted row's 0.9 contributes; the episode's 0.1
        # is excluded so min == max == 0.9.
        assert stats.confidence_min == 0.9
        assert stats.confidence_max == 0.9
        assert stats.confidence_median == 0.9


class TestGetAllEpisodes:
    """Issue #410: single-source enumeration helper backing the
    /memory dashboard's episode-list browser. Sources outside the
    literal "episode" string are excluded - this is intentionally
    narrower than `USER_VISIBLE_SOURCES` because the function's
    purpose is single-source enumeration, not multi-source admission."""

    def test_get_all_episodes_returns_only_episode_source(self):
        """Mixed-source store: only the episode rows come back. The
        `USER_VISIBLE_SOURCES` admit list is broader (it accepts
        extracted and migration too) but does not apply here."""
        import kai.memory as mem_mod
        from kai.memory import get_all_episodes

        mock_mem = MagicMock()
        mock_mem.get_all.return_value = {
            "results": [
                {
                    "id": "ext",
                    "memory": "extracted fact",
                    "score": 0.0,
                    "metadata": {"source": "extracted", "type": "fact"},
                    "created_at": "2026-04-01T00:00:00",
                    "updated_at": "2026-04-01T00:00:00",
                },
                {
                    "id": "ep1",
                    "memory": "episode one",
                    "score": 0.0,
                    "metadata": {"source": "episode", "outcome_quality": "success"},
                    "created_at": "2026-04-02T00:00:00",
                    "updated_at": "2026-04-02T00:00:00",
                },
                {
                    "id": "mig",
                    "memory": "migration row",
                    "score": 0.0,
                    "metadata": {"source": "migration", "tags": ["migration"]},
                    "created_at": "2026-04-03T00:00:00",
                    "updated_at": "2026-04-03T00:00:00",
                },
                {
                    "id": "legacy",
                    "memory": "legacy row",
                    "score": 0.0,
                    "metadata": {"source": ""},
                    "created_at": "2026-01-01T00:00:00",
                    "updated_at": "2026-01-01T00:00:00",
                },
            ]
        }
        mem_mod._memory = mock_mem

        results = get_all_episodes(user_id="123")
        assert [r.id for r in results] == ["ep1"]

    def test_get_all_episodes_sorts_newest_first(self):
        """Two episode rows with different `updated_at`; assert
        descending order. Mirrors `get_by_tag`'s sort contract so
        the same conventions hold across both list surfaces."""
        import kai.memory as mem_mod
        from kai.memory import get_all_episodes

        mock_mem = MagicMock()
        mock_mem.get_all.return_value = {
            "results": [
                {
                    "id": "old",
                    "memory": "older episode",
                    "score": 0.0,
                    "metadata": {"source": "episode"},
                    "created_at": "2026-01-01T00:00:00",
                    "updated_at": "2026-01-01T00:00:00",
                },
                {
                    "id": "new",
                    "memory": "newer episode",
                    "score": 0.0,
                    "metadata": {"source": "episode"},
                    "created_at": "2026-04-01T00:00:00",
                    "updated_at": "2026-04-15T00:00:00",
                },
            ]
        }
        mem_mod._memory = mock_mem

        results = get_all_episodes(user_id="123")
        assert [r.id for r in results] == ["new", "old"]

    def test_get_all_episodes_empty_when_no_episodes(self):
        """Extracted-only store: empty list, not a None or an exception."""
        import kai.memory as mem_mod
        from kai.memory import get_all_episodes

        mock_mem = MagicMock()
        mock_mem.get_all.return_value = {
            "results": [
                {
                    "id": "ext",
                    "memory": "fact",
                    "score": 0.0,
                    "metadata": {"source": "extracted"},
                    "created_at": "",
                    "updated_at": "",
                },
            ]
        }
        mem_mod._memory = mock_mem

        assert get_all_episodes(user_id="123") == []

    def test_get_all_episodes_disabled_returns_empty(self):
        """With memory disabled (`_memory is None`), helper short-
        circuits to empty list. No exception, no get_all call."""
        from kai.memory import get_all_episodes

        # The autouse `_clean_memory_state` fixture leaves _memory at
        # None unless the test sets it explicitly.
        assert get_all_episodes(user_id="123") == []

    def test_get_all_episodes_passes_limit_none(self):
        """get_all is called with the high ceiling (top_k >= 100_000)
        so users with thousands of episodes get a complete listing.
        Parallel to test_passes_limit_none_to_get_all for get_by_tag."""
        import kai.memory as mem_mod
        from kai.memory import get_all_episodes

        mock_mem = MagicMock()
        mock_mem.get_all.return_value = {"results": []}
        mem_mod._memory = mock_mem

        get_all_episodes(user_id="123")
        assert mock_mem.get_all.call_args.kwargs["top_k"] >= 100_000


class TestGetAllFacts:
    """Multi-source enumeration helper backing the /memory dashboard's
    facts-list browser. Filter is `metadata.source in {"extracted",
    "migration"}` -- narrower than `USER_VISIBLE_SOURCES` (which also
    admits episode) and broader than `get_all_episodes` (which is
    scoped to a single source). Episodes are intentionally excluded
    because they have their own list view."""

    def test_get_all_facts_returns_only_extracted_and_migration_sources(self):
        """Mixed-source store: only the extracted and migration rows
        come back. Episode and legacy ""-source rows are filtered
        out at the data layer rather than in the UI, so the UI cache
        can't accidentally surface a non-fact-bucket row."""
        import kai.memory as mem_mod
        from kai.memory import get_all_facts

        mock_mem = MagicMock()
        mock_mem.get_all.return_value = {
            "results": [
                {
                    "id": "ext",
                    "memory": "extracted fact",
                    "score": 0.0,
                    "metadata": {"source": "extracted", "type": "fact"},
                    "created_at": "2026-04-01T00:00:00",
                    "updated_at": "2026-04-01T00:00:00",
                },
                {
                    "id": "ep1",
                    "memory": "episode one",
                    "score": 0.0,
                    "metadata": {"source": "episode", "outcome_quality": "success"},
                    "created_at": "2026-04-02T00:00:00",
                    "updated_at": "2026-04-02T00:00:00",
                },
                {
                    "id": "mig",
                    "memory": "migration row",
                    "score": 0.0,
                    "metadata": {"source": "migration", "tags": ["migration"]},
                    "created_at": "2026-04-03T00:00:00",
                    "updated_at": "2026-04-03T00:00:00",
                },
                {
                    "id": "legacy",
                    "memory": "legacy row",
                    "score": 0.0,
                    "metadata": {"source": ""},
                    "created_at": "2026-01-01T00:00:00",
                    "updated_at": "2026-01-01T00:00:00",
                },
            ]
        }
        mem_mod._memory = mock_mem

        results = get_all_facts(user_id="123")
        # Sort order is recency-desc on updated_at, so migration
        # (2026-04-03) precedes extracted (2026-04-01). Episode and
        # legacy rows are excluded.
        assert [r.id for r in results] == ["mig", "ext"]

    def test_get_all_facts_sorts_newest_first(self):
        """Two rows with different `updated_at` (one extracted, one
        migration) so the sort spans the full fact bucket. Mirrors
        `get_all_episodes`'s sort contract."""
        import kai.memory as mem_mod
        from kai.memory import get_all_facts

        mock_mem = MagicMock()
        mock_mem.get_all.return_value = {
            "results": [
                {
                    "id": "old",
                    "memory": "older extracted",
                    "score": 0.0,
                    "metadata": {"source": "extracted"},
                    "created_at": "2026-01-01T00:00:00",
                    "updated_at": "2026-01-01T00:00:00",
                },
                {
                    "id": "new",
                    "memory": "newer migration",
                    "score": 0.0,
                    "metadata": {"source": "migration"},
                    "created_at": "2026-04-01T00:00:00",
                    "updated_at": "2026-04-15T00:00:00",
                },
            ]
        }
        mem_mod._memory = mock_mem

        results = get_all_facts(user_id="123")
        assert [r.id for r in results] == ["new", "old"]

    def test_get_all_facts_empty_when_no_matching_sources(self):
        """Episode-only store: empty list, not None or an exception.
        The episode rows are visible to `get_all_episodes` but not to
        this helper."""
        import kai.memory as mem_mod
        from kai.memory import get_all_facts

        mock_mem = MagicMock()
        mock_mem.get_all.return_value = {
            "results": [
                {
                    "id": "ep",
                    "memory": "episode",
                    "score": 0.0,
                    "metadata": {"source": "episode"},
                    "created_at": "",
                    "updated_at": "",
                },
            ]
        }
        mem_mod._memory = mock_mem

        assert get_all_facts(user_id="123") == []

    def test_get_all_facts_disabled_returns_empty(self):
        """With memory disabled (`_memory is None`), helper short-
        circuits to empty list. No exception, no get_all call."""
        from kai.memory import get_all_facts

        # The autouse `_clean_memory_state` fixture leaves _memory at
        # None unless the test sets it explicitly.
        assert get_all_facts(user_id="123") == []

    def test_get_all_facts_passes_limit_none(self):
        """get_all is called with the high ceiling (top_k >= 100_000)
        so users with thousands of fact-bucket rows get a complete
        listing. Parallel to test_get_all_episodes_passes_limit_none."""
        import kai.memory as mem_mod
        from kai.memory import get_all_facts

        mock_mem = MagicMock()
        mock_mem.get_all.return_value = {"results": []}
        mem_mod._memory = mock_mem

        get_all_facts(user_id="123")
        assert mock_mem.get_all.call_args.kwargs["top_k"] >= 100_000


# ── Integration tests (real Mem0 + Qdrant, slower) ──────────────────


@integration
class TestMemoryIntegration:
    """End-to-end tests with a real Mem0 instance and Qdrant storage."""

    def test_add_and_search(self, real_memory_instance):
        """Add a structured fact, then search for it by semantic similarity.

        Spec 360 deleted Track 1 (`add_user_utterance`). The only ingestion
        primitive now is `add_structured`, which is what production calls
        from `memory_extraction.extract_and_store` after Haiku produces a
        fact. The test mirrors that path: pre-extracted text, sync call,
        explicit `memory_type`."""
        import kai.memory as mem_mod

        mem_mod._memory = real_memory_instance
        mem_mod._config = _make_config()

        user_id = "integration-add-search"

        # Clean slate
        real_memory_instance.delete_all(user_id=user_id)

        mem_mod.add_structured(
            "User asked how to set up the webhook server",
            user_id=user_id,
            memory_type="fact",
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

        # Add three pre-extracted facts on different topics. Track 2 is
        # the only ingestion path now, so each one stands on its own as a
        # standalone semantic unit (no user/assistant pairing).
        facts = [
            "User asked about today's weather",
            "User wants to deploy to production",
            "User is wondering what is for dinner",
        ]
        for fact in facts:
            mem_mod.add_structured(fact, user_id=user_id, memory_type="fact")

        # Search for deployment - should rank the deploy fact first
        results = mem_mod.search("production deployment process", user_id=user_id)
        assert len(results) >= 1
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

        mem_mod.add_structured(
            "User's secret project is called Phoenix",
            user_id=user_a,
            memory_type="fact",
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

        for i in range(3):
            mem_mod.add_structured(
                f"User raised question number {i}",
                user_id=user_id,
                memory_type="fact",
            )

        # get_all should return them
        all_memories = mem_mod.get_all(user_id=user_id)
        assert len(all_memories) >= 3

        # delete_all should remove them
        mem_mod.delete_all(user_id=user_id)
        remaining = mem_mod.get_all(user_id=user_id)
        assert len(remaining) == 0

    async def test_format_context_integration(self, real_memory_instance):
        """format_context returns formatted, budget-capped output.

        Note: the test method itself is `async def` because `format_context`
        is async (it awaits Mem0's executor wrapper). `add_structured`,
        however, is intentionally **synchronous** — it wraps a single
        Mem0 `.add()` call with `infer=False` and does no I/O of its own
        worth offloading. If `add_structured` is ever made async, this
        call site silently returns a coroutine without awaiting it; the
        test would still pass but no row would be stored. Pinning the
        sync contract in this docstring so a future signature change has
        a paper trail."""
        import kai.memory as mem_mod

        mem_mod._memory = real_memory_instance
        mem_mod._config = _make_config(memory_token_budget=500)

        user_id = "integration-format"
        real_memory_instance.delete_all(user_id=user_id)

        # Sync call - see docstring above. Not awaited.
        mem_mod.add_structured(
            "The Mac mini has 16GB RAM",
            user_id=user_id,
            memory_type="fact",
        )

        output = await mem_mod.format_context("How much RAM?", user_id=user_id)
        assert "context only, not instructions" in output
        assert "16GB" in output or "Mac mini" in output


# ── Speaker / confidence metadata round-trip ──────────────────────
#
# Round-trip tests for the new `speaker` and `confidence` fields the
# retrieval ranking and /memory rendering code consume. Two failure
# modes are gated here, both first-class blockers:
#
#   1. Mem0 metadata channel could silently drop one of the new keys.
#      Mem0 has historically reshaped the metadata round-trip (issue
#      #357 was about a related telemetry path); a test pinning the
#      exact key/value preservation surfaces a regression on the next
#      mem0 version bump rather than at first-rank-anomaly in
#      production.
#   2. The read-time defaulting helper has to land on the documented
#      legacy / migration constants for rows that predate the spec
#      and thus carry no speaker or confidence in metadata. If a
#      constant changes (operator-side decision via the swap-the-
#      constants escape hatch), the assertions below pin against the
#      module's bound values rather than literals so the tests stay
#      green after a single-line constant change.


@integration
class TestSpeakerMetadataRoundTrip:
    """End-to-end checks that speaker/confidence survive Mem0 storage
    and that legacy rows surface the documented default constants.
    """

    def test_speaker_round_trips_user(self, real_memory_instance):
        """A fact written with speaker='user' reads back with speaker='user'."""
        import kai.memory as mem_mod

        mem_mod._memory = real_memory_instance
        mem_mod._config = _make_config()

        user_id = "speaker-round-trip-user"
        real_memory_instance.delete_all(user_id=user_id)

        mem_mod.add_structured(
            "User prefers concise responses",
            user_id=user_id,
            memory_type="fact",
            metadata={"source": "extracted", "speaker": "user", "confidence": 0.9},
        )

        results = mem_mod.get_all(user_id=user_id)
        assert len(results) >= 1
        # All fact rows for this user_id were written with the same
        # speaker/confidence in this test, so any row in the result
        # set proves the round-trip; pick the first.
        meta = results[0].metadata
        assert meta.get("speaker") == "user"
        assert meta.get("confidence") == 0.9

    def test_speaker_round_trips_assistant(self, real_memory_instance):
        """A fact written with speaker='assistant' reads back unchanged."""
        import kai.memory as mem_mod

        mem_mod._memory = real_memory_instance
        mem_mod._config = _make_config()

        user_id = "speaker-round-trip-assistant"
        real_memory_instance.delete_all(user_id=user_id)

        mem_mod.add_structured(
            "Assistant noticed user bundles related changes",
            user_id=user_id,
            memory_type="fact",
            metadata={"source": "extracted", "speaker": "assistant", "confidence": 0.7},
        )

        results = mem_mod.get_all(user_id=user_id)
        assert len(results) >= 1
        meta = results[0].metadata
        assert meta.get("speaker") == "assistant"
        assert meta.get("confidence") == 0.7

    def test_speaker_round_trips_episode_summary(self, real_memory_instance):
        """An episode written with speaker='episode_summary' reads back unchanged."""
        import kai.memory as mem_mod

        mem_mod._memory = real_memory_instance
        mem_mod._config = _make_config()

        user_id = "speaker-round-trip-episode"
        real_memory_instance.delete_all(user_id=user_id)

        mem_mod.add_structured(
            "User shipped a new feature on Monday",
            user_id=user_id,
            memory_type="episode",
            metadata={
                "source": "episode",
                "speaker": "episode_summary",
                "confidence": 1.0,
            },
        )

        results = mem_mod.get_all(user_id=user_id)
        assert len(results) >= 1
        meta = results[0].metadata
        assert meta.get("speaker") == "episode_summary"
        assert meta.get("confidence") == 1.0

    def test_extracted_legacy_row_uses_legacy_constants(self, real_memory_instance):
        """A pre-spec extracted row with no speaker/confidence in metadata
        surfaces the documented legacy default through _read_time_speaker.
        """
        import kai.memory as mem_mod
        from kai.memory import _LEGACY_CONFIDENCE, _LEGACY_SPEAKER, _read_time_speaker

        mem_mod._memory = real_memory_instance
        mem_mod._config = _make_config()

        user_id = "speaker-legacy-extracted"
        real_memory_instance.delete_all(user_id=user_id)

        # No speaker, no confidence - simulates a row written by the
        # pre-spec extractor before the metadata channel carried these.
        mem_mod.add_structured(
            "User asked about deployment process",
            user_id=user_id,
            memory_type="fact",
            metadata={"source": "extracted"},
        )

        results = mem_mod.get_all(user_id=user_id)
        assert len(results) >= 1
        # The helper falls into branch 4 (extracted-or-empty source).
        # Pin against the bound constants rather than literals so the
        # test stays green after a swap-the-constants follow-up.
        assert _read_time_speaker(results[0].metadata) == (
            _LEGACY_SPEAKER,
            _LEGACY_CONFIDENCE,
        )

    def test_migration_legacy_row_uses_migration_constants(self, real_memory_instance):
        """A pre-spec migration row with no speaker/confidence surfaces
        the migration default through _read_time_speaker.
        """
        import kai.memory as mem_mod
        from kai.memory import (
            _MIGRATION_CONFIDENCE,
            _MIGRATION_SPEAKER,
            _read_time_speaker,
        )

        mem_mod._memory = real_memory_instance
        mem_mod._config = _make_config()

        user_id = "speaker-legacy-migration"
        real_memory_instance.delete_all(user_id=user_id)

        # Migration rows written before the build_migration_metadata
        # helper landed only carry source/section/subsection. The
        # read-time helper supplies speaker/confidence via the
        # migration constants.
        mem_mod.add_structured(
            "Kai's memory layer uses Mem0",
            user_id=user_id,
            memory_type="fact",
            metadata={"source": "migration", "section": "Architecture", "subsection": ""},
        )

        results = mem_mod.get_all(user_id=user_id)
        assert len(results) >= 1
        assert _read_time_speaker(results[0].metadata) == (
            _MIGRATION_SPEAKER,
            _MIGRATION_CONFIDENCE,
        )

    def test_episode_default_confidence_is_one(self, real_memory_instance):
        """A pre-spec episode row with no confidence in metadata surfaces
        the constant 1.0 via _read_time_speaker's episode branch.
        """
        import kai.memory as mem_mod
        from kai.memory import _read_time_speaker

        mem_mod._memory = real_memory_instance
        mem_mod._config = _make_config()

        user_id = "speaker-legacy-episode"
        real_memory_instance.delete_all(user_id=user_id)

        # Older episodes were written without speaker/confidence in
        # metadata; the helper's source=="episode" branch supplies
        # both ("episode_summary", 1.0). Pin both elements of the
        # tuple here (not just the confidence) because the speaker
        # default is the same ranking-relevant signal.
        mem_mod.add_structured(
            "User completed a refactor on Tuesday",
            user_id=user_id,
            memory_type="episode",
            metadata={"source": "episode"},
        )

        results = mem_mod.get_all(user_id=user_id)
        assert len(results) >= 1
        assert _read_time_speaker(results[0].metadata) == ("episode_summary", 1.0)


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

    def test_bare_dict_return_shape_yields_id(self):
        """Some Mem0 versions return the memory dict directly instead of
        wrapping it in `{"results": [...]}`. Round 7 review surfaced the
        upstream concern: _store_facts relies on add_structured returning
        truthy on success to count facts, so both shapes must resolve to
        a non-None id. This test covers the bare-dict fallback branch."""
        import kai.memory as mem_mod
        from kai.memory import add_structured

        mock_mem = MagicMock()
        mock_mem.add.return_value = {"id": "mem-uuid-bare", "memory": "test"}
        mem_mod._memory = mock_mem

        result = add_structured("test", user_id="123")
        assert result == "mem-uuid-bare"

    def test_unexpected_return_shape_yields_none(self):
        """Defensive: if a future Mem0 version returns something the
        unwrap logic does not recognize (a list, a bare string, a
        non-dict scalar), add_structured returns None rather than
        crashing. _store_facts treats None as "not actually stored" and
        simply under-counts in the log - never raises. Round 7 review
        noted that silent miscounting would be the failure mode; this
        test pins the current defensive behavior."""
        import kai.memory as mem_mod
        from kai.memory import add_structured

        mock_mem = MagicMock()
        mem_mod._memory = mock_mem

        for unexpected in (None, [], "raw-string-id", 42):
            mock_mem.add.return_value = unexpected
            assert add_structured("test", user_id="123") is None

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
