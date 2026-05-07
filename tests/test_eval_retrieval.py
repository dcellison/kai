"""Tests for the Layer 1 retrieval eval harness (kai.eval.retrieval).

Six test classes covering the harness's load-bearing seams:

1. TestMetricComputation - precision/recall/MRR math, including the
   drift-denominator contract (drift probes excluded from BOTH
   numerator and denominator).
2. TestProbeLoading - JSONL parsing with the #-comment extension,
   error paths for malformed JSON / missing keys / wrong types, and
   probe_set_hash invariance under reorder.
3. TestSweepRestoration - run_sweep restores all three pieces of
   mutated module state when a probe raises mid-sweep.
4. TestLogParserRoundTrip - capture a real memory.recall record via
   format_context (with Mem0 mocked) and assert the per-hit `id`
   field round-trips through the parser into compute_rank.
5. TestDriftDetection - get_by_id returning None buckets a probe as
   drift; pareto_frontier dominates correctly on (precision@5, p50).
6. TestEvaluateEndToEnd - drift detection + scoring composed through
   evaluate(); confirms metrics use N_scored, not N_probes.

Math tests are pure functions (no Mem0, no logging fixtures), so the
file is fast to run independently of the heavyweight memory fixtures.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kai.config import Config
from kai.eval import retrieval
from kai.eval.retrieval import (
    ConfigOverride,
    Probe,
    ProbeResult,
    aggregate_metrics,
    compute_rank,
    detect_drift,
    evaluate,
    load_probes,
    pareto_frontier,
    probe_set_hash,
    run_sweep,
    score_probes,
)

# Shared test config: memory enabled, modest budget, production floor.
# Used in tests that swap the live `_config` for a stable known shape
# so sweep-restoration assertions can compare the saved snapshot
# against the post-sweep state.
_BASE_CONFIG = Config(
    telegram_bot_token="test-token",
    allowed_user_ids={12345},
    webhook_secret="test-secret",
    memory_enabled=True,
    memory_search_limit=10,
    memory_token_budget=2000,
    memory_search_floor=0.3,
)


def _reset_memory_module() -> None:
    """Restore memory module state between tests.

    Mirrors the helper in tests/test_memory.py: harness tests mutate
    `_memory`, `_config`, `_SPEAKER_WEIGHTS`, and `_SEARCH_OVERFETCH`,
    and a leak between tests would silently couple them.
    """
    import kai.memory as mem_mod

    mem_mod._memory = None
    mem_mod._config = None
    # `_SPEAKER_WEIGHTS` is the production default snapshot; reset
    # the dict contents so a sweep test mutating any speaker entry
    # does not leak into the next test. Must mirror the production
    # default in src/kai/memory.py exactly, otherwise tests reading
    # the dict after this fixture has run will see a stale snapshot
    # missing entries.
    mem_mod._SPEAKER_WEIGHTS.clear()
    mem_mod._SPEAKER_WEIGHTS.update({"user": 1.0, "assistant": 0.7, "episode_summary": 0.85})
    # `_UNKNOWN_SPEAKER_WEIGHT` is aliased to the assistant entry
    # at module-load time (a float copy, not a live dict reference).
    # If a sweep test mutates `assistant` and a subsequent test
    # reads `_UNKNOWN_SPEAKER_WEIGHT`, the alias would carry the
    # mutated value into the next test. Re-bind explicitly to keep
    # the documented invariant ("unknown rows ride on the assistant
    # weight") holding across tests.
    mem_mod._UNKNOWN_SPEAKER_WEIGHT = mem_mod._SPEAKER_WEIGHTS["assistant"]
    mem_mod._SEARCH_OVERFETCH = 20


@pytest.fixture(autouse=True)
def _clean_memory_state():
    _reset_memory_module()
    yield
    _reset_memory_module()


# ── Test helpers ────────────────────────────────────────────────────


def _make_probe_result(
    *,
    expected_id: str,
    rank: int | None,
    lines_used: int,
    latency_ms: int = 50,
    tags: tuple[str, ...] = (),
) -> ProbeResult:
    """Construct a ProbeResult without going through format_context.

    The harness's scoring math is deliberately decoupled from the
    log-parsing path, so test #1 builds these directly. `in_prompt`
    is computed the same way the harness computes it (rank within
    lines_used) so we don't drift between test and production logic.
    """
    in_prompt = rank is not None and rank <= lines_used
    return ProbeResult(
        probe=Probe(question="q", expected_fact_id=expected_id),
        rank=rank,
        in_prompt=in_prompt,
        lines_used=lines_used,
        latency_ms=latency_ms,
        tags=tags,
    )


# ── Test 1: Metric computation incl. drift denominator ─────────────


class TestMetricComputation:
    """The math layer of the harness, pure-function tested.

    The combination of scored-hit, scored-miss, and drift probes is
    the load-bearing case: drift probes must be excluded from BOTH
    numerator and denominator (N_scored = N_probes - N_drift). A
    naive implementation that counted drift as zero-rank misses
    would silently understate retrieval quality on every probe set.
    """

    def test_precision_and_recall_at_k_single_hit(self):
        # One probe, fact at rank 2. Precision@1 = 0/1 = 0; @3 = 1/1.
        results = [_make_probe_result(expected_id="x", rank=2, lines_used=3)]
        out = score_probes(results)
        assert out["precision_at_k"][1] == 0.0
        assert out["precision_at_k"][3] == 1.0
        assert out["precision_at_k"][5] == 1.0
        # In single-answer mode (one expected_fact_id per probe), the
        # numerator/denominator are the same for precision and recall,
        # so the two dicts must match. Reported separately so a future
        # multi-answer probe format can diverge without reshaping.
        assert out["recall_at_k"] == out["precision_at_k"]

    def test_mrr_treats_misses_as_zero(self):
        # Two probes: one at rank 2 (1/2 = 0.5), one missed (0).
        # MRR = (0.5 + 0) / 2 = 0.25.
        results = [
            _make_probe_result(expected_id="a", rank=2, lines_used=5),
            _make_probe_result(expected_id="b", rank=None, lines_used=5),
        ]
        out = score_probes(results)
        assert out["mrr"] == pytest.approx(0.25)

    def test_in_prompt_uses_per_probe_lines_used(self):
        # Probe A: rank 3, lines_used 2 -> not in prompt (truncated).
        # Probe B: rank 1, lines_used 5 -> in prompt.
        # fraction_in_prompt = 1/2 = 0.5.
        results = [
            _make_probe_result(expected_id="a", rank=3, lines_used=2),
            _make_probe_result(expected_id="b", rank=1, lines_used=5),
        ]
        out = score_probes(results)
        assert out["fraction_in_prompt"] == pytest.approx(0.5)
        assert results[0].in_prompt is False
        assert results[1].in_prompt is True

    def test_drift_excluded_from_denominator(self):
        # Three original probes: one scored hit, one scored miss, one
        # drifted. The harness builds ProbeResult only for the two
        # scored probes; n_drift=1 is passed through. With a hit and
        # a miss in the scored bucket, precision@5 must be 1/2 = 0.5,
        # NOT 1/3 (which would be the wrong denominator).
        scored_results = [
            _make_probe_result(expected_id="hit", rank=1, lines_used=3),
            _make_probe_result(expected_id="miss", rank=None, lines_used=3),
        ]
        m = aggregate_metrics(scored_results, n_probes=3, n_drift=1)
        assert m.n_probes == 3
        assert m.n_scored == 2
        assert m.n_drift == 1
        # The whole point of the drift bucket: 1/2 not 1/3.
        assert m.precision_at_k[5] == pytest.approx(0.5)
        assert m.recall_at_k[5] == pytest.approx(0.5)
        # MRR also computed against N_scored: 1.0 (the hit) / 2 = 0.5.
        assert m.mrr == pytest.approx(0.5)

    def test_per_tag_buckets_share_multitag_probes(self):
        # A probe whose fact carries two tags appears in both buckets;
        # this matches the operator's mental model of "show me retrieval
        # quality for every probe whose fact touches tag X" rather than
        # partitioning probes across tags. A multi-tag fact is a multi-
        # tag probe by construction.
        results = [
            _make_probe_result(expected_id="a", rank=1, lines_used=3, tags=("preferences", "tech")),
            _make_probe_result(expected_id="b", rank=None, lines_used=3, tags=("tech",)),
        ]
        m = aggregate_metrics(results, n_probes=2, n_drift=0)
        assert "preferences" in m.by_tag
        assert "tech" in m.by_tag
        # preferences sees only probe a (a hit at rank 1) -> p@5 = 1.0.
        assert m.by_tag["preferences"]["precision_at_k"][5] == pytest.approx(1.0)
        # tech sees both probes (one hit, one miss) -> p@5 = 0.5.
        assert m.by_tag["tech"]["precision_at_k"][5] == pytest.approx(0.5)

    def test_compute_rank_returns_one_indexed_or_none(self):
        # First hit -> rank 1 (1-indexed). No match -> None.
        hits = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        assert compute_rank(hits, "a") == 1
        assert compute_rank(hits, "c") == 3
        assert compute_rank(hits, "z") is None
        # Empty hits cleanly return None rather than raising.
        assert compute_rank([], "a") is None


# ── Test 2: Probe file parsing ─────────────────────────────────────


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


# ── Test 3: Sweep restoration after exception ──────────────────────


class TestSweepRestoration:
    """run_sweep must restore module state even if a probe raises mid-sweep."""

    def test_module_state_restored_after_midsweep_exception(self):
        import kai.memory as mem_mod

        # Install known-good baseline state. Capture explicit copies
        # so post-sweep assertions can compare against frozen values
        # that won't accidentally alias the dict the harness mutated.
        mem_mod._config = _BASE_CONFIG
        original_floor = mem_mod._config.memory_search_floor
        original_weights = dict(mem_mod._SPEAKER_WEIGHTS)
        original_overfetch = mem_mod._SEARCH_OVERFETCH

        # The sweep loop now calls `_score_against_store` per grid
        # point (drift is hoisted to a single call at sweep entry, so
        # we also stub `detect_drift` to a no-op tuple). The stub is
        # patched to raise on the second call, mid-sweep. The first
        # call completes normally; the second raises and the exception
        # propagates out. The try/finally inside run_sweep must still
        # restore the original module state.
        call_count = {"n": 0}

        async def faulty_score(scored, drifted, tags_by_id, user_id):
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise RuntimeError("simulated mid-sweep failure")
            from kai.eval.retrieval import Metrics

            return Metrics(
                n_probes=1,
                n_scored=1,
                n_drift=0,
                precision_at_k={1: 0.0, 3: 0.0, 5: 0.0},
                recall_at_k={1: 0.0, 3: 0.0, 5: 0.0},
                mrr=0.0,
                fraction_in_prompt=0.0,
                latency_p50_ms=0.0,
                latency_p95_ms=0.0,
            )

        grid = [
            ConfigOverride(
                floor=0.20,
                user_weight=1.0,
                assistant_weight=0.6,
                episode_summary_weight=0.85,
                overfetch=10,
            ),
            ConfigOverride(
                floor=0.40,
                user_weight=0.85,
                assistant_weight=0.7,
                episode_summary_weight=0.7,
                overfetch=30,
            ),
        ]
        with (
            patch(
                "kai.eval.retrieval.detect_drift",
                return_value=([Probe(question="q", expected_fact_id="id")], [], {"id": ()}),
            ),
            patch("kai.eval.retrieval._score_against_store", side_effect=faulty_score),
            pytest.raises(RuntimeError, match="simulated mid-sweep failure"),
        ):
            asyncio.run(
                run_sweep(
                    [Probe(question="q", expected_fact_id="id")],
                    user_id="42",
                    grid=grid,
                )
            )

        # The whole point of test 3: state restored despite the abort.
        assert mem_mod._config.memory_search_floor == original_floor
        assert original_weights == mem_mod._SPEAKER_WEIGHTS
        assert original_overfetch == mem_mod._SEARCH_OVERFETCH


# ── Test 4: Log parser end-to-end (round-trip on `id` field) ──────


class TestLogParserRoundTrip:
    """Capture a real memory.recall record, parse it, assert id passthrough.

    The harness's ranking logic depends on each per-hit dict in the
    recall payload carrying `id` matching MemoryResult.id. This test
    wires the harness's `_RecallLogCapture` to a real format_context
    call (with a mocked Mem0 search) and confirms the id flows through
    end-to-end via the structured log line, not via any side-channel.
    """

    def test_capture_and_parse_id_field(self):
        import kai.memory as mem_mod
        from kai.memory import format_context

        # Mock Mem0 to return two rows with known IDs. The harness's
        # log capture handler reads format_context's emit; the parsed
        # payload must carry both IDs so a downstream `compute_rank`
        # against expected_fact_id="row-b" returns 2 (not None).
        mock_mem = MagicMock()
        mock_mem.search.return_value = {
            "results": [
                {
                    "id": "row-a",
                    "memory": "Fact A text",
                    "score": 0.9,
                    "metadata": {"type": "fact", "source": "extracted"},
                    "created_at": "2026-04-01T10:00:00",
                },
                {
                    "id": "row-b",
                    "memory": "Fact B text",
                    "score": 0.8,
                    "metadata": {"type": "fact", "source": "extracted"},
                    "created_at": "2026-04-02T10:00:00",
                },
            ]
        }
        mem_mod._memory = mock_mem
        mem_mod._config = _BASE_CONFIG

        # Attach the harness's capture handler exactly the way the
        # production code path does. _attach_capture also forces the
        # kai.memory logger level to INFO if it was higher (test runs
        # default to WARNING via pytest config), so the recall record
        # is guaranteed to reach the handler.
        capture = retrieval._attach_capture()
        try:
            asyncio.run(format_context("any query", user_id="42"))
            payloads = capture.drain()
        finally:
            retrieval._detach_capture(capture)

        assert len(payloads) == 1
        payload = payloads[0]
        ids = [h["id"] for h in payload["hits"]]
        # IDs round-tripped via JSON; compute_rank must locate row-b
        # at position 2 to score the prerequisite-dependent contract.
        assert ids == ["row-a", "row-b"]
        assert compute_rank(payload["hits"], "row-b") == 2
        assert compute_rank(payload["hits"], "missing-id") is None


# ── Test 5: Drift detection ────────────────────────────────────────


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

    def test_pareto_frontier_dominates_correctly(self):
        # Three configs:
        #   A: precision 0.8, latency 50  - dominates B (worse on both).
        #   B: precision 0.7, latency 100 - dominated.
        #   C: precision 0.9, latency 200 - non-dominated (better quality, worse latency).
        # Frontier should be {A, C}, sorted by precision desc -> [C, A].
        def _m(p5: float, lat: float):
            from kai.eval.retrieval import Metrics

            return Metrics(
                n_probes=1,
                n_scored=1,
                n_drift=0,
                precision_at_k={1: p5, 3: p5, 5: p5},
                recall_at_k={1: p5, 3: p5, 5: p5},
                mrr=p5,
                fraction_in_prompt=p5,
                latency_p50_ms=lat,
                latency_p95_ms=lat,
            )

        cfg_a = ConfigOverride(
            floor=0.2,
            user_weight=1.0,
            assistant_weight=0.5,
            episode_summary_weight=0.85,
            overfetch=10,
        )
        cfg_b = ConfigOverride(
            floor=0.3,
            user_weight=1.0,
            assistant_weight=0.7,
            episode_summary_weight=0.85,
            overfetch=20,
        )
        cfg_c = ConfigOverride(
            floor=0.4,
            user_weight=0.85,
            assistant_weight=0.8,
            episode_summary_weight=1.0,
            overfetch=30,
        )
        sweep = [(cfg_a, _m(0.8, 50)), (cfg_b, _m(0.7, 100)), (cfg_c, _m(0.9, 200))]
        front = pareto_frontier(sweep)
        front_cfgs = [c for c, _ in front]
        assert cfg_a in front_cfgs
        assert cfg_c in front_cfgs
        assert cfg_b not in front_cfgs  # dominated by A on both axes
        # Sort order: precision desc -> C (0.9) before A (0.8).
        assert front[0][0] is cfg_c
        assert front[1][0] is cfg_a


# ── Bonus: drift end-to-end through evaluate() ─────────────────────


class TestEvaluateEndToEnd:
    """evaluate() integrates drift detection + scoring; exercise the seam."""

    def test_evaluate_excludes_drift_from_metrics(self):
        # Mock the memory module: get_by_id resolves only "alive"
        # probes (so the dead probe falls into the drift bucket
        # before format_context is ever called), and format_context
        # emits a stub recall log naming alive-fact at rank 1. The
        # combined behavior must produce metrics that count the alive
        # probe but not the drifted one.
        import kai.memory as mem_mod
        from kai.memory import MemoryResult

        mem_mod._config = _BASE_CONFIG

        alive = MemoryResult(
            id="alive-fact",
            text="...",
            score=0.0,
            memory_type="fact",
            metadata={"source": "extracted", "tags": []},
            created_at="2026-04-01T00:00:00",
        )

        async def fake_format_context(query: str, *, user_id: str, token_budget=None):
            # Emit one memory.recall line per call so the harness's
            # capture handler sees exactly one record (the contract
            # the harness aborts loudly on if violated).
            payload = {
                "user_id": user_id,
                "query": query,
                "hits": [{"id": "alive-fact", "source": "extracted", "score": 0.9, "adj": 1.08, "snippet": "..."}],
                "lines_used": 1,
                "latency_ms": 7,
            }
            logging.getLogger("kai.memory").info("memory.recall %s", json.dumps(payload))
            return ""

        def fake_get_by_id(*, user_id: str, memory_id: str):
            return alive if memory_id == "alive-fact" else None

        with (
            patch("kai.memory.get_by_id", side_effect=fake_get_by_id),
            patch("kai.memory.format_context", side_effect=fake_format_context),
        ):
            metrics = asyncio.run(
                evaluate(
                    [
                        Probe(question="q1", expected_fact_id="alive-fact"),
                        Probe(question="q2", expected_fact_id="dead-fact"),
                    ],
                    user_id="42",
                )
            )

        assert metrics.n_probes == 2
        assert metrics.n_scored == 1
        assert metrics.n_drift == 1
        # The single scored probe was a hit at rank 1 -> p@5 = 1.0,
        # MRR = 1.0; both computed against N_scored=1, not N_probes=2.
        assert metrics.precision_at_k[5] == pytest.approx(1.0)
        assert metrics.mrr == pytest.approx(1.0)
        assert metrics.fraction_in_prompt == pytest.approx(1.0)


class TestApplyOverride:
    """`_apply_override` and `_restore_overrides` are the two helpers
    `run_sweep` calls per iteration to mutate / restore module state.
    Direct tests of the helpers stand in for the higher-level
    `run_sweep` test by exercising the same state-touch pattern in
    isolation, without spinning up an evaluate loop. Together they
    pin the schema-required restore-roundtrip property the sweep
    `finally` block depends on.
    """

    def test_apply_override_writes_three_weights(self):
        # `_apply_override` must mutate all three speaker entries in
        # place. The pre-mutation snapshot returned by the helper
        # carries the prior values so a `_restore_overrides` call
        # rolls them back exactly. Pin both sides of the contract:
        # post-apply state matches the override, snapshot matches
        # pre-apply state.
        import kai.memory as mem_mod
        from kai.eval.retrieval import _apply_override

        # Install a real Config object so the helper's
        # `dataclasses.replace(snap.config, ...)` call has something
        # to clone from. Autouse fixture leaves _config=None.
        mem_mod._config = _BASE_CONFIG

        # Known starting state (the autouse fixture pre-fills these
        # to production values; assert against those values rather
        # than against literals so a default-tune retro flows through).
        pre_user = mem_mod._SPEAKER_WEIGHTS["user"]
        pre_assistant = mem_mod._SPEAKER_WEIGHTS["assistant"]
        pre_episode = mem_mod._SPEAKER_WEIGHTS["episode_summary"]

        override = ConfigOverride(
            floor=0.20,
            user_weight=0.85,
            assistant_weight=0.6,
            episode_summary_weight=1.0,
            overfetch=15,
        )
        snap = _apply_override(override)

        # Post-apply: live dict matches override.
        assert mem_mod._SPEAKER_WEIGHTS["user"] == 0.85
        assert mem_mod._SPEAKER_WEIGHTS["assistant"] == 0.6
        assert mem_mod._SPEAKER_WEIGHTS["episode_summary"] == 1.0

        # Snapshot: pre-apply values captured for the restore path.
        assert snap.speaker_weights["user"] == pre_user
        assert snap.speaker_weights["assistant"] == pre_assistant
        assert snap.speaker_weights["episode_summary"] == pre_episode

    def test_apply_override_writes_overfetch_and_floor(self):
        # `_apply_override` mutates `_SEARCH_OVERFETCH` and rebuilds
        # `_config` with the override's floor; the snapshot carries
        # the pre-mutation values for both. Pin the restore path's
        # data here so a regression that captures stale values fails
        # this test rather than at sweep finally-time.
        import kai.memory as mem_mod
        from kai.eval.retrieval import _apply_override

        # Install a known config so the floor swap is observable.
        mem_mod._config = _BASE_CONFIG
        pre_overfetch = mem_mod._SEARCH_OVERFETCH

        override = ConfigOverride(
            floor=0.42,
            user_weight=1.0,
            assistant_weight=0.7,
            episode_summary_weight=0.85,
            overfetch=11,
        )
        snap = _apply_override(override)

        assert mem_mod._SEARCH_OVERFETCH == 11
        assert mem_mod._config is not None
        assert mem_mod._config.memory_search_floor == 0.42

        # Snapshot captures the pre-apply values.
        assert snap.overfetch == pre_overfetch
        assert snap.config is _BASE_CONFIG

    def test_restore_overrides_undoes_apply(self):
        # The full round-trip: apply, mutate observable state, restore
        # from the snapshot, assert the live module is back at its
        # pre-apply shape. This is the contract `run_sweep`'s
        # `finally` block depends on; testing it in isolation lets
        # a regression in the helper trip here rather than during a
        # mid-sweep abort path.
        import kai.memory as mem_mod
        from kai.eval.retrieval import _apply_override, _restore_overrides

        mem_mod._config = _BASE_CONFIG
        pre_speaker_weights = dict(mem_mod._SPEAKER_WEIGHTS)
        pre_overfetch = mem_mod._SEARCH_OVERFETCH
        pre_config = mem_mod._config

        override = ConfigOverride(
            floor=0.99,
            user_weight=0.1,
            assistant_weight=0.2,
            episode_summary_weight=0.3,
            overfetch=99,
        )
        snap = _apply_override(override)
        # Sanity check: we are NOT testing the no-op identity path -
        # the override must have actually mutated state for the
        # restore assertion below to mean anything.
        assert pre_speaker_weights != mem_mod._SPEAKER_WEIGHTS
        assert pre_overfetch != mem_mod._SEARCH_OVERFETCH

        _restore_overrides(snap)

        assert pre_speaker_weights == mem_mod._SPEAKER_WEIGHTS
        assert pre_overfetch == mem_mod._SEARCH_OVERFETCH
        assert mem_mod._config is pre_config
