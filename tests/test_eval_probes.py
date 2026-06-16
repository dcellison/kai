"""Tests for the shared probe primitives (kai.eval._probes).

Covers the small public surface that the behavioral evaluator and
the collision-probe generator both depend on:

1. TestProbeLoading - JSONL parsing with the documented
   `#`-comment extension; error paths for malformed JSON,
   missing required keys, wrong types; and the probe_set_hash
   invariance under reordering of the probe file.
2. TestDriftDetection - `detect_drift` buckets a probe whose
   expected fact does not resolve via `memory.get_by_id` as
   drift rather than as a retrieval miss; the returned tag
   mapping is built during drift detection so the per-tag rollup
   later reuses it instead of re-fetching.
3. TestComputeRank - `compute_rank` returns 1-indexed position
   or None; empty input is None, not a raise.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from kai.eval._probes import (
    Probe,
    compute_rank,
    detect_drift,
    load_probes,
    probe_set_hash,
)

# ── Probe file parsing ─────────────────────────────────────────────


class TestProbeLoading:
    """JSONL parsing with the documented #-comment extension."""

    def test_loads_well_formed_lines_and_skips_comments(self, tmp_path: Path):
        # The probe file demonstrates every supported feature: leading
        # comment, blank line, indented comment, trailing newline. The
        # assertion below also locks in line-order preservation, which
        # is what lets per-probe output rows align back to source.
        text = (
            "# overall annotation\n"
            "\n"
            '{"question": "where do I live?", "expected_fact_id": "fact-1"}\n'
            "    # indented annotation between probes\n"
            '{"question": "what is my timezone?", "expected_fact_id": "fact-2", '
            '"source_turn_ts": "2026-04-01T10:00:00", "notes": "from session 12"}\n'
        )
        path = tmp_path / "probes.jsonl"
        path.write_text(text, encoding="utf-8")
        probes = load_probes(path)
        assert len(probes) == 2
        # Order preserved: probe 1 is the live-where question.
        assert probes[0].expected_fact_id == "fact-1"
        assert probes[1].expected_fact_id == "fact-2"
        assert probes[1].source_turn_ts == "2026-04-01T10:00:00"
        assert probes[1].notes == "from session 12"

    def test_rejects_malformed_json(self, tmp_path: Path):
        path = tmp_path / "probes.jsonl"
        path.write_text('{"question": "bad json"\n', encoding="utf-8")
        with pytest.raises(ValueError, match="malformed JSON"):
            load_probes(path)

    def test_rejects_missing_required_keys(self, tmp_path: Path):
        # `expected_fact_id` absent: must fail at load time so a
        # large probe set's bad row surfaces before any retrieval cost.
        path = tmp_path / "probes.jsonl"
        path.write_text('{"question": "no expected id here"}\n', encoding="utf-8")
        with pytest.raises(ValueError, match="expected_fact_id"):
            load_probes(path)

    def test_rejects_wrong_type(self, tmp_path: Path):
        # `expected_fact_id` is an int (spreadsheet export landmine):
        # without this guard the eventual hit comparison would silently
        # never match and report 0% precision.
        path = tmp_path / "probes.jsonl"
        path.write_text('{"question": "q", "expected_fact_id": 42}\n', encoding="utf-8")
        with pytest.raises(ValueError, match="not a non-empty string"):
            load_probes(path)

    def test_probe_set_hash_invariant_under_reorder(self, tmp_path: Path):
        # Two files with the same probes in different order must
        # produce the same hash; otherwise a baseline file's hash
        # would invalidate after a benign reorder.
        a = [
            Probe(question="q1", expected_fact_id="id1"),
            Probe(question="q2", expected_fact_id="id2"),
        ]
        b = list(reversed(a))
        assert probe_set_hash(a) == probe_set_hash(b)
        # Different content -> different hash, defending the
        # baseline-comparison contract.
        c = [Probe(question="q1", expected_fact_id="id1")]
        assert probe_set_hash(a) != probe_set_hash(c)


# ── Drift detection ────────────────────────────────────────────────


class TestDriftDetection:
    """A probe with a get_by_id None must bucket as drift, not a miss."""

    def test_none_from_get_by_id_buckets_as_drift(self):
        # Two probes: alive-fact resolves to a real MemoryResult,
        # dead-fact returns None (deleted, ownership mismatch, source
        # mismatch, or never-existed - all collapse to drift).
        from kai.memory import MemoryResult

        alive = MemoryResult(
            id="alive-fact",
            text="...",
            score=0.0,
            memory_type="fact",
            metadata={"source": "extracted", "tags": ["preferences"]},
            created_at="2026-04-01T00:00:00",
        )

        def fake_get_by_id(*, user_id: str, memory_id: str):
            return alive if memory_id == "alive-fact" else None

        with patch("kai.memory.get_by_id", side_effect=fake_get_by_id):
            scored, drifted, tags_by_id = detect_drift(
                [
                    Probe(question="q1", expected_fact_id="alive-fact"),
                    Probe(question="q2", expected_fact_id="dead-fact"),
                ],
                user_id="42",
            )

        assert [p.expected_fact_id for p in scored] == ["alive-fact"]
        assert [p.expected_fact_id for p in drifted] == ["dead-fact"]
        # Tag mapping captured during drift detection so the per-tag
        # rollup later reuses this lookup instead of refetching.
        assert tags_by_id == {"alive-fact": ("preferences",)}


# ── compute_rank ───────────────────────────────────────────────────


class TestComputeRank:
    """1-indexed position of expected_fact_id, or None."""

    def test_returns_one_indexed_or_none(self):
        # First hit -> rank 1 (1-indexed). No match -> None.
        hits = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        assert compute_rank(hits, "a") == 1
        assert compute_rank(hits, "c") == 3
        assert compute_rank(hits, "z") is None
        # Empty hits cleanly return None rather than raising.
        assert compute_rank([], "a") is None
