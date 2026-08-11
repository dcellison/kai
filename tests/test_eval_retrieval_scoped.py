"""Tests for the scoped retrieval evaluator (kai.eval.retrieval_scoped).

Covers the spec test plan:

1. TestProbeLoading - schema v2 loader: v1-shape compatibility,
   per-line errors, polarity gate, element validation of
   expected_excluded_fact_ids.
2. TestDriftDetection - per-polarity drift: a probe can drift on one
   polarity and be clean on the other; reported separately.
3. TestPositiveScoringSplit - candidate_rank against the raw helper's
   adjusted-score order, prompt_position against the renderer's
   prompt order. The pinned divergence case (`candidate_rank=1` but
   `prompt_position=6`) is the load-bearing assertion.
4. TestNegativeScoring - excluded_in_prompt vs excluded_in_candidates
   for the three position cases (absent, in candidates only, in
   prompt slice).
5. TestAggregateMetrics - drift dropped from positive and negative
   denominators; by_scoped_reason and by_active_project populate
   from the rendered payload.
6. TestWorkspaceHandling - registered-root workspace routes through
   the project; unregistered or null routes global-only.
7. TestSweepRestoration - state restored on a mid-sweep raise.
8. TestCLIFilters - --non-project-only and --projects filtering
   shapes; mutually exclusive enforcement.
9. TestProjectRegistryBootstrap - the harness loads the merged
   registry via `kai.memory_projects.load_project_registry` BEFORE
   scoring; an empty registry degrades to non-project.
10. TestOutputFormat - schema version, per-probe details array
    present in single-config, omitted by default in sweep mode,
    re-included under --include-details.
11. TestStdoutSummary - distribution lines emitted on every run.
12. TestRelocation - kai.memory_projects.load_project_registry is
    the same callable kai.memory_reclassify re-exports.
13. TestFailClosedObservation - a scoped_error payload produces a
    full per-probe row with all-None ranks and does NOT abort.

The harness's pipeline calls are mocked at module boundaries
(retrieve_scoped_memories, format_scoped_context_with_recall_payload,
get_by_id) so the math layer is exercised without Mem0 or Qdrant.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

import kai.memory_projects as mp_mod
from kai.config import Config, MemoryProjectConfig
from kai.eval import retrieval_scoped
from kai.eval.retrieval_scoped import (
    ConfigOverride,
    ScopedProbe,
    aggregate_metrics,
    detect_drift,
    evaluate,
    load_probes,
    probe_set_hash,
    run_sweep,
    score_negative,
    score_positive,
)

# ── Shared fixtures ────────────────────────────────────────────────


# Shared test config: memory enabled, modest budget, production floor.
# Sweep tests swap `_config` for this to assert restoration; single-
# config tests use it through `evaluate` so the harness reads a known
# floor when forming the override snapshot.
_BASE_CONFIG = Config(
    telegram_bot_token="test-token",
    allowed_user_ids={12345},
    memory_enabled=True,
    memory_search_limit=10,
    memory_token_budget=2000,
    memory_search_floor=0.3,
)


def _reset_memory_module() -> None:
    """Restore memory module state between tests.

    Sweep tests mutate `_config`, `_SPEAKER_WEIGHTS`, and
    `_SEARCH_OVERFETCH`; without a reset between tests a leak would
    silently couple them and the next test's assertions would flap.
    Mirrors `tests/test_eval_retrieval._reset_memory_module`.
    """
    import kai.memory as mem_mod

    mem_mod._memory = None
    mem_mod._config = None
    mem_mod._SPEAKER_WEIGHTS.clear()
    mem_mod._SPEAKER_WEIGHTS.update(
        {
            "user": retrieval_scoped._PRODUCTION_USER_WEIGHT,
            "assistant": retrieval_scoped._PRODUCTION_ASSISTANT_WEIGHT,
            "episode_summary": retrieval_scoped._PRODUCTION_EPISODE_SUMMARY_WEIGHT,
        }
    )
    mem_mod._UNKNOWN_SPEAKER_WEIGHT = mem_mod._SPEAKER_WEIGHTS["assistant"]
    mem_mod._SEARCH_OVERFETCH = retrieval_scoped._PRODUCTION_OVERFETCH


@pytest.fixture(autouse=True)
def _clean_memory_state():
    _reset_memory_module()
    # The DB-layer registry cache is process-global; a stray entry from
    # another test would change detection behavior here. Clear before
    # AND after so the bootstrap tests below start from a known empty
    # DB layer regardless of test order.
    mp_mod._db_registry.clear()
    mp_mod._db_creators.clear()
    yield
    _reset_memory_module()
    mp_mod._db_registry.clear()
    mp_mod._db_creators.clear()


# ── Test helpers ────────────────────────────────────────────────────


def _hit(memory_id: str) -> SimpleNamespace:
    """Build a minimal ScopedMemoryHit stub.

    The harness only reads `.result.id` on the raw helper's hits, so a
    `SimpleNamespace` chain is sufficient and avoids constructing a
    full MemoryResult + ResolvedMemoryScope tower per row.
    """
    return SimpleNamespace(result=SimpleNamespace(id=memory_id))


def _scoped_result(hit_ids: list[str], reason: str = "ok") -> SimpleNamespace:
    """Build a minimal ScopedRetrievalResult stub.

    Carries just the surfaces the harness reads: `.hits` (a list of
    objects exposing `.result.id`) and `.debug.reason` (unused by the
    harness but mirrored for shape parity).
    """
    return SimpleNamespace(
        hits=[_hit(h) for h in hit_ids],
        debug=SimpleNamespace(reason=reason),
    )


def _rendered_payload(
    *,
    hit_ids: list[str] | None = None,
    lines_used: int = 5,
    latency_ms: int = 12,
    reason: str = "ok",
    active_project_id: str | None = None,
) -> dict:
    """Build a renderer recall_payload matching the rendered call's
    real shape: `hits` carries per-hit dicts with an `id` field
    (mirrors `_scoped_hit_to_payload` in kai.memory), plus
    `lines_used`, `latency_ms`, top-level `reason`, and a nested
    `scoped_debug` dict.

    Defaults to a non-empty list with `lines_used=5` so most tests can
    omit the parameters; the divergence tests override `hit_ids` to
    pin specific positions.
    """
    if hit_ids is None:
        hit_ids = []
    return {
        "hits": [{"id": h} for h in hit_ids],
        "lines_used": lines_used,
        "latency_ms": latency_ms,
        "reason": reason,
        "scoped_debug": {
            "active_project_id": active_project_id,
            "reason": reason,
        },
    }


def _rendered(
    *,
    hit_ids: list[str] | None = None,
    lines_used: int = 5,
    latency_ms: int = 12,
    reason: str = "ok",
    active_project_id: str | None = None,
) -> SimpleNamespace:
    """Wrap `_rendered_payload` in the ScopedRecallResult shape (a
    namespace with `.recall_payload`)."""
    return SimpleNamespace(
        rendered_context="",
        recall_payload=_rendered_payload(
            hit_ids=hit_ids,
            lines_used=lines_used,
            latency_ms=latency_ms,
            reason=reason,
            active_project_id=active_project_id,
        ),
    )


def _alive(memory_id: str, tags: tuple[str, ...] = ()) -> object:
    """Build a get_by_id "alive" MemoryResult shaped enough for drift
    detection: it reads `.metadata` and pulls `tags` out of it."""
    return SimpleNamespace(
        id=memory_id,
        metadata={"source": "extracted", "tags": list(tags)},
    )


# ── Test 1: Probe loading ──────────────────────────────────────────


class TestProbeLoading:
    """Loader contracts: per-line errors, polarity gate, element
    validation. Schema v1 backward compatibility is the load-bearing
    invariant for legacy probe files."""

    def test_v1_shaped_probe_loads_as_v2_defaults(self, tmp_path):
        path = tmp_path / "probes.jsonl"
        path.write_text(
            json.dumps({"question": "q1", "expected_fact_id": "fact-a"}) + "\n",
            encoding="utf-8",
        )
        probes = load_probes(path)
        assert len(probes) == 1
        p = probes[0]
        assert p.question == "q1"
        assert p.expected_fact_id == "fact-a"
        # v2 defaults must NOT require editing an existing v1 probe file.
        assert p.expected_excluded_fact_ids == ()
        assert p.workspace is None
        assert p.line_number == 1

    def test_comments_and_blank_lines_skipped(self, tmp_path):
        path = tmp_path / "probes.jsonl"
        path.write_text(
            "# comment\n"
            "    # indented comment\n"
            "\n" + json.dumps({"question": "q1", "expected_fact_id": "fact-a"}) + "\n",
            encoding="utf-8",
        )
        probes = load_probes(path)
        assert len(probes) == 1
        # Line number reflects the actual source line, not the index
        # among scored probes; an operator opening the file at the
        # reported line lands on the JSON row.
        assert probes[0].line_number == 4

    def test_missing_both_polarities_rejected(self, tmp_path):
        path = tmp_path / "probes.jsonl"
        path.write_text(json.dumps({"question": "q1"}) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="neither"):
            load_probes(path)

    def test_excluded_only_probe_loads(self, tmp_path):
        path = tmp_path / "probes.jsonl"
        path.write_text(
            json.dumps({"question": "q1", "expected_excluded_fact_ids": ["bad-1"]}) + "\n",
            encoding="utf-8",
        )
        probes = load_probes(path)
        assert probes[0].expected_fact_id is None
        assert probes[0].expected_excluded_fact_ids == ("bad-1",)

    def test_excluded_id_must_be_string(self, tmp_path):
        path = tmp_path / "probes.jsonl"
        path.write_text(
            json.dumps({"question": "q1", "expected_excluded_fact_ids": [42]}) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match=r"expected_excluded_fact_ids\[0\].*int"):
            load_probes(path)

    def test_excluded_id_must_be_nonempty(self, tmp_path):
        path = tmp_path / "probes.jsonl"
        path.write_text(
            json.dumps({"question": "q1", "expected_excluded_fact_ids": [""]}) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match=r"expected_excluded_fact_ids\[0\].*non-empty"):
            load_probes(path)

    def test_empty_question_rejected(self, tmp_path):
        path = tmp_path / "probes.jsonl"
        path.write_text(
            json.dumps({"question": "", "expected_fact_id": "fact-a"}) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="'question' missing or not a non-empty string"):
            load_probes(path)

    def test_error_includes_line_number(self, tmp_path):
        path = tmp_path / "probes.jsonl"
        path.write_text(
            "# preamble\n"
            "\n"
            + json.dumps({"question": "ok", "expected_fact_id": "fact-a"})
            + "\n"
            + json.dumps({"question": "bad"})
            + "\n",
            encoding="utf-8",
        )
        # The error must name line 4, not "the second probe" - operator
        # workflow opens the file by line and reads the cited row.
        with pytest.raises(ValueError, match=rf"{path}:4:"):
            load_probes(path)

    def test_workspace_must_be_nonempty(self, tmp_path):
        path = tmp_path / "probes.jsonl"
        path.write_text(
            json.dumps({"question": "q1", "expected_fact_id": "fact-a", "workspace": ""}) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="'workspace' must be non-empty"):
            load_probes(path)


# ── Test 1b: probe_set_hash ────────────────────────────────────────


class TestProbeSetHash:
    """Hash invariance under file reorder; hash changes when negative
    polarity changes. Both invariants matter for the operator-side
    comparison workflow (compare baselines only when hashes match)."""

    def test_hash_invariant_under_reorder(self):
        a = ScopedProbe(
            question="q1",
            expected_fact_id="fact-a",
            expected_excluded_fact_ids=(),
            workspace=None,
            line_number=1,
        )
        b = ScopedProbe(
            question="q2",
            expected_fact_id="fact-b",
            expected_excluded_fact_ids=(),
            workspace=None,
            line_number=2,
        )
        assert probe_set_hash([a, b]) == probe_set_hash([b, a])

    def test_hash_changes_when_excluded_changes(self):
        a = ScopedProbe(
            question="q1",
            expected_fact_id="fact-a",
            expected_excluded_fact_ids=(),
            workspace=None,
            line_number=1,
        )
        a_neg = ScopedProbe(
            question="q1",
            expected_fact_id="fact-a",
            expected_excluded_fact_ids=("bad-1",),
            workspace=None,
            line_number=1,
        )
        assert probe_set_hash([a]) != probe_set_hash([a_neg])


# ── Test 2: Drift detection ────────────────────────────────────────


class TestDriftDetection:
    """Per-polarity drift: a probe can drift positively, negatively,
    on both, or on neither."""

    def test_positive_and_negative_drift_independent(self):
        probes = [
            # Positive drifts; negative clean.
            ScopedProbe(
                question="q1",
                expected_fact_id="dead-pos",
                expected_excluded_fact_ids=("alive-neg",),
                workspace=None,
                line_number=1,
            ),
            # Positive clean; one negative drifts.
            ScopedProbe(
                question="q2",
                expected_fact_id="alive-pos",
                expected_excluded_fact_ids=("dead-neg", "alive-other"),
                workspace=None,
                line_number=2,
            ),
        ]

        def fake_get_by_id(*, user_id: str, memory_id: str):
            if memory_id in {"alive-pos", "alive-neg", "alive-other"}:
                return _alive(memory_id, tags=("preferences",) if memory_id == "alive-pos" else ())
            return None

        with patch("kai.memory.get_by_id", side_effect=fake_get_by_id):
            positive, negative, tags = detect_drift(probes, user_id="42")

        # Per-line buckets: line 1 has positive drift, line 2 does not.
        assert positive == {1: True, 2: False}
        # Line 1: alive-neg resolved -> no negative drift.
        # Line 2: dead-neg drifted, alive-other did not.
        assert negative == {1: [], 2: ["dead-neg"]}
        # Tags collected only for surviving positive ids.
        assert tags == {"alive-pos": ("preferences",)}


# ── Test 3: Positive scoring split ─────────────────────────────────


class TestPositiveScoringSplit:
    """The two-call shape exists so candidate_rank and prompt_position
    can diverge under section ordering. Pin both metrics against their
    correct source list; the test must fail if either is computed
    over the wrong list."""

    def test_candidate_rank_uses_raw_helper_prompt_position_uses_renderer(self):
        # Raw helper ranks "fact-project" FIRST in adjusted-score
        # order; renderer's prompt order puts it at position 6 after a
        # five-row global section, with lines_used=5.
        probe = ScopedProbe(
            question="q1",
            expected_fact_id="fact-project",
            expected_excluded_fact_ids=(),
            workspace="/work/kai",
            line_number=1,
        )

        async def fake_rendered(**kwargs):
            return _rendered(
                hit_ids=["g1", "g2", "g3", "g4", "g5", "fact-project"],
                lines_used=5,
                active_project_id="kai",
            )

        async def fake_raw(context):
            return _scoped_result(["fact-project", "g1", "g2", "g3"])

        with (
            patch("kai.memory.get_by_id", return_value=_alive("fact-project")),
            patch(
                "kai.memory.format_scoped_context_with_recall_payload",
                side_effect=fake_rendered,
            ),
            patch("kai.memory.retrieve_scoped_memories", side_effect=fake_raw),
        ):
            results, _ = asyncio.run(evaluate([probe], user_id="42"))

        r = results[0]
        # The exact divergence the spec's two-call shape exists to
        # capture. If candidate_rank were computed over the renderer's
        # hits, it would be 6; if prompt_position were computed over
        # the raw helper's hits, it would be 1.
        assert r.candidate_rank == 1
        assert r.prompt_position == 6
        assert r.in_prompt is False
        assert r.lines_used == 5


# ── Test 4: Negative scoring ───────────────────────────────────────


class TestNegativeScoring:
    """Three exclusion position cases: absent, in candidates only, in
    prompt slice. Both axes reported because they describe distinct
    safety failures."""

    def test_three_position_cases(self):
        probe = ScopedProbe(
            question="q1",
            expected_fact_id=None,
            expected_excluded_fact_ids=("absent", "in-candidates", "in-prompt"),
            workspace=None,
            line_number=1,
        )

        async def fake_rendered(**kwargs):
            # in-prompt at position 2 (<= lines_used=2 -> in prompt).
            # in-candidates at position 5 (> lines_used -> in candidates only).
            # absent does not appear.
            return _rendered(
                hit_ids=["x", "in-prompt", "y", "z", "in-candidates"],
                lines_used=2,
                active_project_id=None,
            )

        async def fake_raw(context):
            return _scoped_result([])

        def fake_get_by_id(*, user_id: str, memory_id: str):
            # All three excluded ids resolve - no negative drift.
            return _alive(memory_id)

        with (
            patch("kai.memory.get_by_id", side_effect=fake_get_by_id),
            patch(
                "kai.memory.format_scoped_context_with_recall_payload",
                side_effect=fake_rendered,
            ),
            patch("kai.memory.retrieve_scoped_memories", side_effect=fake_raw),
        ):
            results, _ = asyncio.run(evaluate([probe], user_id="42"))

        r = results[0]
        # in-prompt position 2 with lines_used=2 -> agent saw it.
        assert r.excluded_in_prompt == ["in-prompt"]
        # in-prompt also lives in the rendered hits list, so it's a
        # candidate too. in-candidates is at position 5, only in
        # candidates. "absent" is in neither. Result order follows the
        # probe author's expected_excluded_fact_ids order so a per-probe
        # report row reads in the same order as the probe file.
        assert r.excluded_in_candidates == ["in-candidates", "in-prompt"]


# ── Test 5: Aggregate metrics ──────────────────────────────────────


class TestAggregateMetrics:
    """Denominator contracts and distribution population."""

    def _make_result(
        self,
        *,
        line: int,
        expected: str | None,
        excluded: tuple[str, ...] = (),
        candidate_rank: int | None = None,
        prompt_position: int | None = None,
        lines_used: int = 5,
        excluded_in_prompt: list[str] | None = None,
        excluded_in_candidates: list[str] | None = None,
        positive_drift: bool = False,
        negative_drift_ids: list[str] | None = None,
        active_project_id: str | None = None,
        scoped_reason: str = "ok",
    ):
        from kai.eval.retrieval_scoped import ScopedProbeResult

        probe = ScopedProbe(
            question="q",
            expected_fact_id=expected,
            expected_excluded_fact_ids=excluded,
            workspace=None,
            line_number=line,
        )
        in_prompt = prompt_position is not None and prompt_position <= lines_used
        return ScopedProbeResult(
            probe=probe,
            candidate_rank=candidate_rank,
            prompt_position=prompt_position,
            in_prompt=in_prompt,
            lines_used=lines_used,
            latency_ms=10,
            tags=(),
            excluded_in_prompt=excluded_in_prompt or [],
            excluded_in_candidates=excluded_in_candidates or [],
            positive_drift=positive_drift,
            negative_drift_ids=negative_drift_ids or [],
            active_project_id=active_project_id,
            scoped_reason=scoped_reason,
        )

    def test_positive_denominator_excludes_drift(self):
        # Two probes with expected_fact_id: one hit at rank 1, one
        # drifted. Precision@5 must be 1.0 (1/1), not 0.5 (1/2).
        scored = self._make_result(line=1, expected="alive", candidate_rank=1, prompt_position=1)
        drifted = self._make_result(line=2, expected="dead", positive_drift=True)
        metrics = aggregate_metrics([scored, drifted], n_probes=2)
        assert metrics.n_probes == 2
        assert metrics.n_scored_positive == 1
        assert metrics.n_drift_positive == 1
        assert metrics.precision_at_k[5] == pytest.approx(1.0)
        assert metrics.fraction_in_prompt == pytest.approx(1.0)

    def test_negative_denominator_excludes_drift(self):
        # One probe with three excluded ids, one drifted. The non-
        # drifted pair is excluded_in_prompt=[] -> 2/2 pass; the
        # drifted id is NOT counted toward either numerator or
        # denominator.
        r = self._make_result(
            line=1,
            expected=None,
            excluded=("a", "b", "c"),
            negative_drift_ids=["a"],
            excluded_in_prompt=[],
            excluded_in_candidates=[],
        )
        metrics = aggregate_metrics([r], n_probes=1)
        assert metrics.n_scored_negative == 2
        assert metrics.n_drift_negative == 1
        assert metrics.exclusion_pass_in_prompt == pytest.approx(1.0)
        assert metrics.exclusion_pass_in_candidates == pytest.approx(1.0)

    def test_by_active_project_buckets_none_under_sentinel(self):
        r_kai = self._make_result(
            line=1,
            expected="x",
            candidate_rank=1,
            prompt_position=1,
            active_project_id="kai",
        )
        r_none = self._make_result(
            line=2,
            expected="y",
            candidate_rank=1,
            prompt_position=1,
            active_project_id=None,
        )
        metrics = aggregate_metrics([r_kai, r_none], n_probes=2)
        assert metrics.by_active_project == {"kai": 1, "__none__": 1}

    def test_by_scoped_reason_counts_across_probes(self):
        r1 = self._make_result(
            line=1,
            expected="x",
            candidate_rank=1,
            prompt_position=1,
            scoped_reason="ok",
        )
        r2 = self._make_result(line=2, expected="y", scoped_reason="no_results_after_scope")
        r3 = self._make_result(line=3, expected="z", scoped_reason="ok")
        metrics = aggregate_metrics([r1, r2, r3], n_probes=3)
        assert metrics.by_scoped_reason == {"ok": 2, "no_results_after_scope": 1}


# ── Test 6 + 9: Workspace handling and registry bootstrap ──────────


class TestWorkspaceHandling:
    """Workspace null routes global-only; registered workspace routes
    project. The harness exposes the routing decision through the
    rendered payload's `active_project_id`."""

    def test_workspace_path_routes_through_renderer(self):
        # Capture the kwargs the harness passes into the rendered call
        # so we can verify the Path conversion and per-probe routing.
        captured = []

        async def fake_rendered(**kwargs):
            captured.append(kwargs)
            return _rendered(
                hit_ids=[],
                lines_used=0,
                active_project_id=("kai" if kwargs.get("workspace") else None),
            )

        async def fake_raw(context):
            return _scoped_result([])

        probes = [
            ScopedProbe(
                question="q1",
                expected_fact_id=None,
                expected_excluded_fact_ids=("any",),
                workspace="/work/kai",
                line_number=1,
            ),
            ScopedProbe(
                question="q2",
                expected_fact_id=None,
                expected_excluded_fact_ids=("any",),
                workspace=None,
                line_number=2,
            ),
        ]

        with (
            patch("kai.memory.get_by_id", return_value=_alive("any")),
            patch(
                "kai.memory.format_scoped_context_with_recall_payload",
                side_effect=fake_rendered,
            ),
            patch("kai.memory.retrieve_scoped_memories", side_effect=fake_raw),
        ):
            _results, metrics = asyncio.run(evaluate(probes, user_id="42"))

        # First probe carried a Path; second carried None.
        assert captured[0]["workspace"] == Path("/work/kai")
        assert captured[1]["workspace"] is None
        # by_active_project surfaces both buckets, "__none__" for the
        # second probe's null workspace.
        assert metrics.by_active_project == {"kai": 1, "__none__": 1}


class TestProjectRegistryBootstrap:
    """The bootstrap must run BEFORE scoring; a probe whose workspace
    lives under a chat-registered root must resolve to the registered
    project."""

    @pytest.mark.asyncio
    async def test_chat_registered_root_resolves_after_bootstrap(self, tmp_path):
        root = tmp_path / "kaiproj"
        root.mkdir()
        rows = [
            {
                "project_id": "kaiproj",
                "display_name": "Kai project",
                "workspace_root": str(root),
                "memory_enabled": True,
                "default_scope_for_new_facts": "project",
                "created_by": 100,
            }
        ]

        # Patch sessions on the registry module so the bootstrap reads
        # OUR DB rows. The relocated load_project_registry runs against
        # the in-process registry cache modified by load_db_registry.
        with (
            patch.object(mp_mod.sessions, "init_db", new=AsyncMock()),
            patch.object(
                mp_mod.sessions,
                "get_memory_project_rows",
                new=AsyncMock(return_value=rows),
            ),
        ):
            registry = await mp_mod.load_project_registry(
                Config(
                    telegram_bot_token="t",
                    allowed_user_ids={1},
                )
            )

        assert "kaiproj" in registry
        active = mp_mod.detect_active_memory_project(root, registry)
        assert active is not None
        assert active.project_id == "kaiproj"

    @pytest.mark.asyncio
    async def test_bootstrap_imported_from_memory_projects_not_reclassify(self):
        # The relocation contract: `load_project_registry` lives on
        # kai.memory_projects. The reclassify module re-exports it for
        # backward compatibility, so its attribute must be the same
        # object (not a wrapper) - if a future refactor breaks that,
        # the test fires.
        from kai import memory_reclassify

        assert mp_mod.load_project_registry is memory_reclassify.load_project_registry


# ── Test 7: Sweep restoration ──────────────────────────────────────


class TestSweepRestoration:
    """`run_sweep` must restore `_SPEAKER_WEIGHTS`, `_SEARCH_OVERFETCH`,
    and `_config` even when a probe raises mid-sweep."""

    def test_state_restored_on_mid_sweep_raise(self):
        import kai.memory as mem_mod

        mem_mod._config = _BASE_CONFIG

        # Snapshot the entry-time state to compare against post-sweep.
        entry_speaker = dict(mem_mod._SPEAKER_WEIGHTS)
        entry_overfetch = mem_mod._SEARCH_OVERFETCH

        probe = ScopedProbe(
            question="q1",
            expected_fact_id="alive",
            expected_excluded_fact_ids=(),
            workspace=None,
            line_number=1,
        )

        async def fake_rendered(**kwargs):
            raise RuntimeError("simulated mid-sweep failure")

        grid = [
            ConfigOverride(floor=0.4, user_weight=0.5, assistant_weight=0.4, episode_summary_weight=0.6, overfetch=15),
        ]

        with (
            patch("kai.memory.get_by_id", return_value=_alive("alive")),
            patch(
                "kai.memory.format_scoped_context_with_recall_payload",
                side_effect=fake_rendered,
            ),
            pytest.raises(RuntimeError),
        ):
            asyncio.run(run_sweep([probe], user_id="42", grid=grid))

        # All three pieces of mutated state must be back to entry shape.
        assert entry_speaker == mem_mod._SPEAKER_WEIGHTS
        assert entry_overfetch == mem_mod._SEARCH_OVERFETCH
        assert mem_mod._config is _BASE_CONFIG


# ── Test 8: CLI filters ────────────────────────────────────────────


class TestCLIFilters:
    """Argparse enforces mutual exclusion; filter semantics drop the
    right probes."""

    def test_mutually_exclusive_at_argparse(self):
        parser = retrieval_scoped._build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    "42",
                    "--probes",
                    "x.jsonl",
                    "--non-project-only",
                    "--projects",
                    "kai",
                ]
            )

    @pytest.mark.asyncio
    async def test_non_project_only_drops_workspace_pinned(self):
        probes = [
            ScopedProbe(
                question="q1", expected_fact_id="a", expected_excluded_fact_ids=(), workspace=None, line_number=1
            ),
            ScopedProbe(
                question="q2", expected_fact_id="b", expected_excluded_fact_ids=(), workspace="/work/kai", line_number=2
            ),
        ]
        filtered = await retrieval_scoped._apply_cli_filters(
            probes,
            non_project_only=True,
            projects=None,
            config=_BASE_CONFIG,
        )
        assert [p.line_number for p in filtered] == [1]

    @pytest.mark.asyncio
    async def test_projects_filter_uses_detection(self, tmp_path):
        root = tmp_path / "kaiproj"
        root.mkdir()
        registry = {
            "kaiproj": MemoryProjectConfig(
                project_id="kaiproj",
                display_name="Kai",
                workspace_roots=(root.resolve(),),
                memory_enabled=True,
                default_scope_for_new_facts="project",
            ),
        }
        probes = [
            ScopedProbe(
                question="q1", expected_fact_id="a", expected_excluded_fact_ids=(), workspace=str(root), line_number=1
            ),
            ScopedProbe(
                question="q2",
                expected_fact_id="b",
                expected_excluded_fact_ids=(),
                workspace=str(tmp_path / "other"),
                line_number=2,
            ),
            ScopedProbe(
                question="q3", expected_fact_id="c", expected_excluded_fact_ids=(), workspace=None, line_number=3
            ),
        ]

        with patch(
            "kai.memory_projects.load_project_registry",
            new=AsyncMock(return_value=registry),
        ):
            filtered = await retrieval_scoped._apply_cli_filters(
                probes,
                non_project_only=False,
                projects=["kaiproj"],
                config=_BASE_CONFIG,
            )
        # Only the probe with workspace inside kaiproj survives.
        assert [p.line_number for p in filtered] == [1]


# ── Test 10: Output format ─────────────────────────────────────────


class TestOutputFormat:
    """Schema version is the scoped counter; per-probe details array
    follows the documented presence rules."""

    def _result_with_probe(self, **kwargs):
        from kai.eval.retrieval_scoped import ScopedProbeResult

        probe = ScopedProbe(
            question=kwargs.get("question", "q1"),
            expected_fact_id=kwargs.get("expected", "fact-a"),
            expected_excluded_fact_ids=kwargs.get("excluded", ()),
            workspace=kwargs.get("workspace"),
            line_number=kwargs.get("line", 1),
        )
        return ScopedProbeResult(
            probe=probe,
            candidate_rank=kwargs.get("candidate_rank", 1),
            prompt_position=kwargs.get("prompt_position", 1),
            in_prompt=kwargs.get("in_prompt", True),
            lines_used=kwargs.get("lines_used", 5),
            latency_ms=kwargs.get("latency_ms", 10),
            tags=kwargs.get("tags", ()),
            excluded_in_prompt=kwargs.get("excluded_in_prompt", []),
            excluded_in_candidates=kwargs.get("excluded_in_candidates", []),
            positive_drift=kwargs.get("positive_drift", False),
            negative_drift_ids=kwargs.get("negative_drift_ids", []),
            active_project_id=kwargs.get("active_project_id"),
            scoped_reason=kwargs.get("scoped_reason", "ok"),
        )

    def test_single_config_includes_probes_array(self):
        from kai.eval.retrieval_scoped import _build_single_config_json

        results = [self._result_with_probe()]
        metrics = aggregate_metrics(results, n_probes=1)
        cfg = ConfigOverride(
            floor=0.3,
            user_weight=0.85,
            assistant_weight=0.8,
            episode_summary_weight=0.85,
            overfetch=20,
        )
        # Same minimal probe set used to compute the hash; aggregate
        # metrics ignored the probe list directly, so reuse the one
        # we built the result for.
        probes = [results[0].probe]
        envelope = _build_single_config_json(probes, results, metrics, cfg)

        assert envelope["version"] == retrieval_scoped._BASELINE_SCHEMA_VERSION
        assert "probes" in envelope
        assert len(envelope["probes"]) == 1
        # Acceptance: per-probe entry carries all documented fields.
        entry = envelope["probes"][0]
        expected_keys = {
            "probe_index",
            "expected_fact_id",
            "expected_excluded_fact_ids",
            "workspace",
            "active_project_id",
            "candidate_rank",
            "prompt_position",
            "in_prompt",
            "excluded_in_prompt",
            "excluded_in_candidates",
            "positive_drift",
            "negative_drift_ids",
            "scoped_reason",
            "question_truncated",
        }
        assert expected_keys.issubset(entry.keys())

    def test_sweep_omits_probes_by_default(self):
        from kai.eval.retrieval_scoped import _build_sweep_json

        results = [self._result_with_probe()]
        metrics = aggregate_metrics(results, n_probes=1)
        cfg = ConfigOverride(
            floor=0.3,
            user_weight=0.85,
            assistant_weight=0.8,
            episode_summary_weight=0.85,
            overfetch=20,
        )
        probes = [results[0].probe]
        sweep_results = [(cfg, results, metrics)]
        envelope = _build_sweep_json(probes, sweep_results, include_details=False)

        assert "probes" not in envelope["sweep"][0]

    def test_sweep_includes_probes_under_flag(self):
        from kai.eval.retrieval_scoped import _build_sweep_json

        results = [self._result_with_probe()]
        metrics = aggregate_metrics(results, n_probes=1)
        cfg = ConfigOverride(
            floor=0.3,
            user_weight=0.85,
            assistant_weight=0.8,
            episode_summary_weight=0.85,
            overfetch=20,
        )
        probes = [results[0].probe]
        sweep_results = [(cfg, results, metrics)]
        envelope = _build_sweep_json(probes, sweep_results, include_details=True)

        assert "probes" in envelope["sweep"][0]
        assert envelope["sweep"][0]["probes"][0]["probe_index"] == 1


# ── Test 11: Stdout summary ────────────────────────────────────────


class TestStdoutSummary:
    """`by_active_project` and `by_scoped_reason` lines must surface on
    every run; a regression where the format string changes silently
    is invisible without an assertion."""

    def test_distribution_lines_render(self):
        from kai.eval.retrieval_scoped import _format_distribution

        line = _format_distribution("by_active_project", {"kai": 2, "__none__": 1})
        assert "by_active_project" in line
        # Stable sort: __none__ < kai alphabetically.
        assert line.endswith("__none__=1, kai=2")

    def test_distribution_line_handles_empty(self):
        from kai.eval.retrieval_scoped import _format_distribution

        line = _format_distribution("by_scoped_reason", {})
        assert line == "eval: by_scoped_reason: (empty)"

    def test_sweep_summary_uses_top_ranked_config_not_last_grid_point(self, capsys):
        """`by_scoped_reason` shifts with `--floor` because rows that
        fall below the floor flip from `ok` to `all_below_floor`. The
        sweep stdout summary must therefore be tagged to a specific
        config; emitting the last grid point's distribution after a
        precision-sorted table would silently misreport. The fix
        prints the top-ranked config (same sort key the table uses)
        with the config inline; this test pins that behavior.
        """
        from kai.eval.retrieval_scoped import (
            ScopedMetrics,
            _print_sweep_top_config_distributions,
        )

        # Two configs. The "winner" (precision@5 = 0.9) is NOT the last
        # grid point. The "loser" (precision@5 = 0.1) is last and
        # carries a different by_scoped_reason distribution; with the
        # buggy "last row wins" logic, the test would observe the
        # loser's distribution.
        winner_cfg = ConfigOverride(
            floor=0.15,
            user_weight=0.85,
            assistant_weight=0.8,
            episode_summary_weight=0.85,
            overfetch=20,
        )
        loser_cfg = ConfigOverride(
            floor=0.40,
            user_weight=0.85,
            assistant_weight=0.8,
            episode_summary_weight=0.85,
            overfetch=20,
        )

        def _metrics(p5: float, scoped_reason: dict[str, int]) -> ScopedMetrics:
            return ScopedMetrics(
                n_probes=10,
                n_scored_positive=10,
                n_drift_positive=0,
                precision_at_k={1: p5, 3: p5, 5: p5},
                recall_at_k={1: p5, 3: p5, 5: p5},
                mrr=p5,
                fraction_in_prompt=p5,
                n_scored_negative=0,
                n_drift_negative=0,
                exclusion_pass_in_prompt=1.0,
                exclusion_pass_in_candidates=1.0,
                latency_p50_ms=10.0,
                latency_p95_ms=20.0,
                by_scoped_reason=scoped_reason,
                by_active_project={"kai": 10},
            )

        sweep_results = [
            (winner_cfg, [], _metrics(0.9, {"ok": 10})),
            (loser_cfg, [], _metrics(0.1, {"all_below_floor": 10})),
        ]
        _print_sweep_top_config_distributions(sweep_results)
        out = capsys.readouterr().out
        # The winner's floor must appear (0.15), not the loser's (0.40).
        assert "floor=0.15" in out
        assert "floor=0.40" not in out
        # The winner's scoped_reason distribution must be printed, not
        # the loser's. The buggy "last row wins" code would have
        # printed `all_below_floor=10` instead.
        assert "ok=10" in out
        assert "all_below_floor" not in out


# ── Test 13: Fail-closed observation ────────────────────────────────


class TestFailClosedObservation:
    """A scoped_error payload produces a per-probe row with all-None
    ranks; the harness MUST NOT abort the run."""

    def test_scoped_error_records_row_without_aborting(self):
        # Two probes: the first triggers scoped_error in the rendered
        # call, the second runs cleanly. The aggregate must include
        # both rows; the failed one carries all-None ranks and
        # scoped_reason="scoped_error".
        probes = [
            ScopedProbe(
                question="boom",
                expected_fact_id="fact-a",
                expected_excluded_fact_ids=("bad-1",),
                workspace=None,
                line_number=1,
            ),
            ScopedProbe(
                question="ok", expected_fact_id="fact-b", expected_excluded_fact_ids=(), workspace=None, line_number=2
            ),
        ]

        async def fake_rendered(**kwargs):
            if kwargs["query"] == "boom":
                # Mirrors the production wrapper's fail-closed
                # collapse: empty rendered context and reason set.
                return SimpleNamespace(
                    rendered_context="",
                    recall_payload={
                        "hits": [],
                        "lines_used": 0,
                        "latency_ms": 0,
                        "reason": "scoped_error",
                        "scoped_debug": {"active_project_id": None, "reason": "scoped_error"},
                    },
                )
            return _rendered(hit_ids=["fact-b"], lines_used=1)

        raw_calls = []

        async def fake_raw(context):
            raw_calls.append(context.message)
            return _scoped_result(["fact-b"])

        def fake_get_by_id(*, user_id: str, memory_id: str):
            return _alive(memory_id)

        with (
            patch("kai.memory.get_by_id", side_effect=fake_get_by_id),
            patch(
                "kai.memory.format_scoped_context_with_recall_payload",
                side_effect=fake_rendered,
            ),
            patch("kai.memory.retrieve_scoped_memories", side_effect=fake_raw),
        ):
            results, metrics = asyncio.run(evaluate(probes, user_id="42"))

        # Both probes returned per-probe rows; neither aborted the run.
        assert len(results) == 2
        boom, ok = results
        assert boom.scoped_reason == "scoped_error"
        assert boom.candidate_rank is None
        assert boom.prompt_position is None
        assert boom.in_prompt is False
        assert ok.scoped_reason == "ok"
        assert ok.candidate_rank == 1
        assert ok.prompt_position == 1
        # The raw helper must be SKIPPED on the failed probe (calling
        # it on a path that already raised would re-raise out of the
        # harness and lose the per-probe row we just recorded). Only
        # the clean probe's query reaches the raw helper.
        assert raw_calls == ["ok"]
        # The positive denominator counts the clean probe only because
        # the failed probe could not be ranked; both are reflected in
        # by_scoped_reason though.
        assert metrics.by_scoped_reason == {"scoped_error": 1, "ok": 1}


# ── Bonus: score_positive / score_negative as pure functions ───────


class TestScoringPureFunctions:
    """Verify the math helpers behave correctly on hand-constructed
    inputs, independent of the harness's mock plumbing. Aggregate-
    level tests cover the integration; this class isolates the math."""

    def _r(self, **kwargs):
        from kai.eval.retrieval_scoped import ScopedProbeResult

        probe = ScopedProbe(
            question="q",
            expected_fact_id=kwargs.get("expected", "x"),
            expected_excluded_fact_ids=kwargs.get("excluded", ()),
            workspace=None,
            line_number=kwargs.get("line", 1),
        )
        lines_used = kwargs.get("lines_used", 5)
        prompt_position = kwargs.get("prompt_position")
        return ScopedProbeResult(
            probe=probe,
            candidate_rank=kwargs.get("candidate_rank"),
            prompt_position=prompt_position,
            in_prompt=(prompt_position is not None and prompt_position <= lines_used),
            lines_used=lines_used,
            latency_ms=10,
            tags=(),
            excluded_in_prompt=kwargs.get("excluded_in_prompt", []),
            excluded_in_candidates=kwargs.get("excluded_in_candidates", []),
            positive_drift=kwargs.get("positive_drift", False),
            negative_drift_ids=kwargs.get("negative_drift_ids", []),
            active_project_id=None,
            scoped_reason="ok",
        )

    def test_score_positive_mrr_treats_miss_as_zero(self):
        # Two probes: rank 2 -> 1/2; miss -> 0. MRR = (0.5 + 0) / 2.
        results = [
            self._r(candidate_rank=2, prompt_position=2),
            self._r(candidate_rank=None, prompt_position=None),
        ]
        block = score_positive(results)
        assert block["mrr"] == pytest.approx(0.25)

    def test_score_negative_empty_returns_pass_one(self):
        # No excluded ids in the probe set -> denominator zero; the
        # safety metric defaults to 1.0 ("no failures observed").
        results = [self._r(candidate_rank=1, prompt_position=1)]
        n_scored, n_drift, in_prompt, in_cand = score_negative(results)
        assert n_scored == 0
        assert n_drift == 0
        assert in_prompt == pytest.approx(1.0)
        assert in_cand == pytest.approx(1.0)


# ── _initialize_memory startup-line behaviour ──────────────────────


class TestInitMemoryStartupLine:
    """The scoped harness's startup line is the operator's signal for
    which evaluator is running; pin its presence so a future log
    refactor cannot drop it silently."""

    def test_startup_line_names_both_pipelines(self, capsys, monkeypatch):
        # Patch the imports `_initialize_memory` uses so we don't need
        # a real memory backend. is_enabled=True path is the one with
        # the startup line; the inner Mem0 setup is mocked out.
        from kai import memory as mem_mod

        monkeypatch.setattr(mem_mod, "init_memory", lambda cfg: None)
        monkeypatch.setattr(mem_mod, "is_enabled", lambda: True)
        cfg = Config(
            telegram_bot_token="t",
            allowed_user_ids={1},
            memory_enabled=True,
        )
        monkeypatch.setattr("kai.config.load_config", lambda: cfg)

        ok = retrieval_scoped._initialize_memory()
        assert ok is True
        captured = capsys.readouterr()
        # Both pipeline names appear so an operator running both
        # harnesses side by side can tell the log streams apart.
        assert "retrieve_scoped_memories" in captured.err
        assert "format_scoped_context_with_recall_payload" in captured.err
