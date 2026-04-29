"""
Tests for scripts/migrate-memory-md-to-qdrant.py (issue #406, Phase 4 of #396).

The migration script lives at scripts/migrate-memory-md-to-qdrant.py
because that is the project convention for one-shot operator scripts.
The hyphenated filename is not a legal Python module name, so this
test loads it via importlib.util at module-import time and exposes
the helpers as `mig.<helper>` for the test functions below.

Tests cover:
- Chunking unit tests (1-7): the _chunk_memory_md and slugify helpers
  in isolation, no Qdrant dependency.
- Integration tests (8-16): the main() entry point with stubbed
  memory module functions to verify the dry-run / forward / rollback
  / guard / exit-code paths.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Load the migration script as a module so its helpers are importable
# despite the hyphenated filename. spec_from_file_location is the
# stdlib idiom for loading code from an arbitrary path; the loaded
# module behaves like any other import target.
#
# CRITICAL: register in sys.modules BEFORE exec_module(). The script
# uses @dataclass, whose decorator looks up cls.__module__ in
# sys.modules at class-creation time. Without the pre-registration,
# the loaded module is not yet visible there and dataclass crashes
# with AttributeError on a NoneType __dict__ lookup.
_SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "migrate-memory-md-to-qdrant.py"
_spec = importlib.util.spec_from_file_location("migrate_memory_md", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
mig = importlib.util.module_from_spec(_spec)
sys.modules["migrate_memory_md"] = mig
_spec.loader.exec_module(mig)


# ── Chunking unit tests ────────────────────────────────────────────


class TestChunking:
    """Tests for `_chunk_memory_md` and `slugify`."""

    def test_h2_only_file_produces_one_chunk_per_section(self):
        """Three H2s with substantive bodies, no H3s: three chunks at level 2."""
        text = (
            "# Memory\n\n"
            "## Section A\n\n"
            "Body of A. Has enough text to survive the empty-body filter.\n\n"
            "## Section B\n\n"
            "Body of B with similar substance.\n\n"
            "## Section C\n\n"
            "Body of C, also substantive.\n"
        )
        chunks = mig._chunk_memory_md(text, rollup_tokens=50)
        h2_chunks = [c for c in chunks if c.level == 2]
        assert len(h2_chunks) == 3
        assert [c.title for c in h2_chunks] == ["Section A", "Section B", "Section C"]
        for chunk in h2_chunks:
            assert chunk.parent_h2 is None
            assert chunk.tag_slug == ""

    def test_h3_below_threshold_rolls_up_to_h2(self):
        """A short H3 gets folded into its parent H2's chunk text."""
        text = "## Parent\n\nParent body.\n\n### Tiny\nShort.\n"
        # rollup_tokens=50; "Short." is ~2 tokens, well under threshold.
        chunks = mig._chunk_memory_md(text, rollup_tokens=50)
        # Only the H2 chunk survives; H3 rolled up.
        assert len(chunks) == 1
        parent = chunks[0]
        assert parent.level == 2
        assert parent.title == "Parent"
        # Spec D1 separator: \n\n### <title>\n<body>
        assert "### Tiny" in parent.body
        assert "Short." in parent.body

    def test_h3_above_threshold_stands_alone(self):
        """A long H3 becomes its own chunk with parent_h2 set."""
        long_body = "word " * 100  # ~500 chars, ~125 tokens, well above 50.
        text = (
            f"## Parent\n\nParent body that is also reasonably long so it survives filtering.\n\n### Big\n{long_body}\n"
        )
        chunks = mig._chunk_memory_md(text, rollup_tokens=50)
        assert len(chunks) == 2
        parent, child = chunks
        assert parent.level == 2
        assert parent.title == "Parent"
        assert "### Big" not in parent.body  # Did NOT roll up.
        assert child.level == 3
        assert child.title == "Big"
        assert child.parent_h2 == "Parent"
        assert child.tag_slug == "big"

    def test_mixed_h3s_within_one_h2(self):
        """Some H3s under one H2 roll up, others stand alone."""
        long_body = "word " * 100
        text = f"## Parent\n\nParent body.\n\n### Short1\nTiny.\n\n### Long1\n{long_body}\n\n### Short2\nAlso tiny.\n"
        chunks = mig._chunk_memory_md(text, rollup_tokens=50)
        # Expected: H2 (with Short1 + Short2 rolled up), Long1 standalone.
        assert len(chunks) == 2
        parent, long_chunk = chunks
        assert parent.level == 2
        assert "### Short1" in parent.body
        assert "### Short2" in parent.body
        assert "### Long1" not in parent.body
        assert long_chunk.level == 3
        assert long_chunk.title == "Long1"

    def test_h1_preamble_becomes_root_chunk(self):
        """The paragraph between `# Memory` and the first H2 becomes a level-1 chunk."""
        text = (
            "# Memory\n\nThis is the file preamble explaining what MEMORY.md is.\n\n## First Section\n\nSection body.\n"
        )
        chunks = mig._chunk_memory_md(text, rollup_tokens=50)
        assert len(chunks) == 2
        preamble = chunks[0]
        assert preamble.level == 1
        assert "preamble" in preamble.body
        assert preamble.title == "Memory"

    def test_empty_section_does_not_emit_chunk(self):
        """A header with no body and no children is dropped."""
        text = (
            "# Memory\n\n"
            "Preamble.\n\n"
            "## Empty Section\n\n"  # No body, no children.
            "## Section With Body\n\n"
            "Has substance.\n"
        )
        chunks = mig._chunk_memory_md(text, rollup_tokens=50)
        # Preamble + Section With Body = 2 chunks; Empty Section dropped.
        titles = [c.title for c in chunks]
        assert "Empty Section" not in titles
        assert "Section With Body" in titles

    @pytest.mark.parametrize(
        "title,expected",
        [
            ("Goose backend (shipped Apr 7, 2026)", "goose-backend-shipped-apr-7-2026"),
            ("Mem0 v2.0.0 gotchas (verified by testing)", "mem0-v2-0-0-gotchas-verified-by-testing"),
            ("/help text bug: /github notify", "help-text-bug-github-notify"),
            (
                "`/backend` slash command for runtime backend switching",
                "backend-slash-command-for-runtime-backend-switching",
            ),
            ("/model and /settings model are identical", "model-and-settings-model-are-identical"),
        ],
    )
    def test_h3_slug_generated_correctly(self, title, expected):
        """The slugify algorithm matches the spec §D6 worked examples."""
        assert mig.slugify(title) == expected


# ── Integration tests via main() ───────────────────────────────────


@pytest.fixture
def fake_memory_md(tmp_path):
    """A minimal MEMORY.md fixture with both rollup and standalone H3s."""
    long_body = "word " * 100
    content = (
        "# Memory\n\n"
        "Operator notes file.\n\n"
        "## Projects\n\n"
        "Projects body.\n\n"
        "### Phi\nShort entry.\n\n"
        f"### Kai\n{long_body}\n"
    )
    path = tmp_path / "MEMORY.md"
    path.write_text(content)
    return path


@pytest.fixture
def stub_memory_module(monkeypatch):
    """Stub the kai.memory functions the script calls.

    Yields a dict with handles to each stub so tests can set
    return values and inspect call args.
    """
    stubs = {
        "init_memory": MagicMock(),
        "search": MagicMock(return_value=[]),
        "add_structured": MagicMock(return_value="fake-mem-id"),
        "count_by_source": MagicMock(return_value=0),
        "delete_by_source": MagicMock(),
    }

    # async fn for delete_by_source: needs to be awaitable so the
    # script's `asyncio.run(delete_by_source(...))` works without
    # raising "object MagicMock can't be used in 'await' expression".
    async def _async_delete(*_args, **_kwargs):
        stubs["delete_by_source"](*_args, **_kwargs)
        return 7  # fake delete count

    # Patch on the loaded mig module's imports. The script does
    # lazy imports inside main()/_do_*, so patching kai.memory is
    # enough for the import-time resolution to pick up our stubs.
    monkeypatch.setattr("kai.memory.init_memory", stubs["init_memory"])
    monkeypatch.setattr("kai.memory.search", stubs["search"])
    monkeypatch.setattr("kai.memory.add_structured", stubs["add_structured"])
    monkeypatch.setattr("kai.memory.count_by_source", stubs["count_by_source"])
    monkeypatch.setattr("kai.memory.delete_by_source", _async_delete)
    return stubs


@pytest.fixture
def enabled_config(monkeypatch):
    """Stub load_config to return a config with memory_enabled=True."""
    fake_config = MagicMock()
    fake_config.memory_enabled = True
    monkeypatch.setattr("kai.config.load_config", lambda: fake_config)
    return fake_config


def test_dry_run_makes_no_writes(fake_memory_md, stub_memory_module, enabled_config, capsys):
    """Test 8: --dry-run prints plan + summary, calls no writes."""
    # Arrange: a low-similarity result for every chunk so all would ADD.
    low_result = MagicMock()
    low_result.score = 0.10
    stub_memory_module["search"].return_value = [low_result]

    # Act.
    rc = mig.main(
        [
            "--user-id",
            "123",
            "--memory-md",
            str(fake_memory_md),
            "--dry-run",
        ]
    )

    # Assert.
    assert rc == mig.EXIT_OK
    stub_memory_module["add_structured"].assert_not_called()
    captured = capsys.readouterr()
    # Summary block printed by default under --dry-run.
    assert "Similarity score distribution" in captured.out
    assert "Skipped at threshold 0.85" in captured.out


def test_below_threshold_chunk_writes(fake_memory_md, stub_memory_module, enabled_config):
    """Test 9: low-similarity results produce add_structured calls with migration tags."""
    low_result = MagicMock()
    low_result.score = 0.10
    stub_memory_module["search"].return_value = [low_result]

    rc = mig.main(["--user-id", "123", "--memory-md", str(fake_memory_md)])

    assert rc == mig.EXIT_OK
    assert stub_memory_module["add_structured"].called
    # Inspect the first call: tags include "migration", metadata source is "migration".
    call_kwargs = stub_memory_module["add_structured"].call_args_list[0].kwargs
    assert "migration" in call_kwargs["tags"]
    assert call_kwargs["metadata"]["source"] == "migration"


def test_above_threshold_chunk_skips(fake_memory_md, stub_memory_module, enabled_config):
    """Test 10: high-similarity results skip add_structured."""
    high_result = MagicMock()
    high_result.score = 0.99  # well above default threshold 0.85
    stub_memory_module["search"].return_value = [high_result]

    rc = mig.main(["--user-id", "123", "--memory-md", str(fake_memory_md)])

    assert rc == mig.EXIT_OK
    stub_memory_module["add_structured"].assert_not_called()


def test_idempotency_guard_blocks_second_run(fake_memory_md, stub_memory_module, enabled_config, capsys):
    """Test 11: existing migration rows + no --force = exit code 2."""
    stub_memory_module["count_by_source"].return_value = 5  # pretend 5 prior rows.

    rc = mig.main(["--user-id", "123", "--memory-md", str(fake_memory_md)])

    assert rc == mig.EXIT_GUARD
    captured = capsys.readouterr()
    assert "--force" in captured.err
    assert "--rollback" in captured.err
    stub_memory_module["add_structured"].assert_not_called()


def test_force_bypasses_idempotency_guard(fake_memory_md, stub_memory_module, enabled_config, capsys):
    """Test 12: --force lets the migration proceed despite existing rows."""
    stub_memory_module["count_by_source"].return_value = 5
    low_result = MagicMock()
    low_result.score = 0.10
    stub_memory_module["search"].return_value = [low_result]

    rc = mig.main(["--user-id", "123", "--memory-md", str(fake_memory_md), "--force"])

    assert rc == mig.EXIT_OK
    assert stub_memory_module["add_structured"].called
    captured = capsys.readouterr()
    assert "WARNING" in captured.err  # warning about layering


def test_rollback_calls_delete_by_source(stub_memory_module, enabled_config):
    """Test 13: --rollback --yes deletes via delete_by_source, no other writes.

    Sets count_by_source to 7 (non-zero) so the rollback proceeds past
    the pre-check; the zero-row short-circuit is exercised separately
    by test_rollback_short_circuits_when_no_migration_rows.
    """
    stub_memory_module["count_by_source"].return_value = 7
    rc = mig.main(["--user-id", "123", "--rollback", "--yes"])

    assert rc == mig.EXIT_OK
    stub_memory_module["delete_by_source"].assert_called_once_with("123", "migration")
    stub_memory_module["add_structured"].assert_not_called()


def test_rollback_short_circuits_when_no_migration_rows(stub_memory_module, enabled_config, capsys):
    """Pre-check: --rollback with zero existing rows exits cleanly without
    calling delete_by_source. Holds for both --yes and interactive paths
    so the operator never sees a confusing "Deleted 0 migration entries"
    line for what is meant as a no-op verification.
    """
    stub_memory_module["count_by_source"].return_value = 0
    rc = mig.main(["--user-id", "123", "--rollback", "--yes"])

    assert rc == mig.EXIT_OK
    stub_memory_module["delete_by_source"].assert_not_called()
    captured = capsys.readouterr()
    assert "No migration entries to delete" in captured.out


def test_rollback_dry_run_calls_count_not_delete(stub_memory_module, enabled_config, capsys):
    """Test 14: --rollback --dry-run previews via count_by_source."""
    stub_memory_module["count_by_source"].return_value = 12

    rc = mig.main(["--user-id", "123", "--rollback", "--dry-run"])

    assert rc == mig.EXIT_OK
    stub_memory_module["count_by_source"].assert_called_with("123", "migration")
    stub_memory_module["delete_by_source"].assert_not_called()
    captured = capsys.readouterr()
    assert "would delete 12" in captured.out


def test_memory_disabled_exits_cleanly(monkeypatch, fake_memory_md, capsys):
    """Test 15: memory_enabled=False = exit code 3 with clear error."""
    fake_config = MagicMock()
    fake_config.memory_enabled = False
    monkeypatch.setattr("kai.config.load_config", lambda: fake_config)

    rc = mig.main(["--user-id", "123", "--memory-md", str(fake_memory_md)])

    assert rc == mig.EXIT_MEMORY_DISABLED
    captured = capsys.readouterr()
    assert "memory subsystem is disabled" in captured.err.lower()


def test_memory_md_missing_exits_cleanly(stub_memory_module, enabled_config, tmp_path, capsys):
    """Test 16: nonexistent MEMORY.md path = exit code 4 with clear error."""
    nonexistent = tmp_path / "no-such-file.md"

    rc = mig.main(["--user-id", "123", "--memory-md", str(nonexistent)])

    assert rc == mig.EXIT_FILE_MISSING
    captured = capsys.readouterr()
    assert "MEMORY.md not found" in captured.err
