"""
Unit tests for `src/kai/eval/gen_collision_probes.py`.

Regression coverage for the load-bearing invariants:
    - non-project exclusion probes go through legacy verification
    - probe_id_by_line_number survives the load_probes round-trip
    - self-grade destructures the (results, metrics) tuple
    - legacy-default selection uses resolve_memory_scope, not raw metadata
    - provider normalization handles single-provider backends
    - non-project quota is corpus-wide, not per-project
    - parent directory is created before exclusive create
    - exclusive-create blocks pre-existing canonical files
    - --allow-shortfalls does not relax the self-grade gate

No real Mem0, no real LLM calls, no real timeouts. The reasoner is a
fake object; embed_texts / get_all_facts / get_all_episodes /
get_by_id / resolve_memory_scope / format_context / evaluate are
monkeypatched; JSONL output lands in tmp_path.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from kai.config import MemoryProjectConfig
from kai.eval import gen_collision_probes as gcp

# ── Fixture builders ────────────────────────────────────────────────


@dataclass
class FakeRow:
    """
    Minimal MemoryResult substitute for tests.

    Carries the fields the generator actually reads (id, text,
    metadata) so we can build a row pool without instantiating the
    full MemoryResult dataclass (which carries more required fields
    than these tests need).
    """

    id: str
    text: str
    metadata: dict[str, Any]


def _row(rid: str, text: str = "fact text", **metadata: Any) -> FakeRow:
    """Build a FakeRow with optional metadata overrides."""
    md: dict[str, Any] = {"source": "extracted"}
    md.update(metadata)
    return FakeRow(id=rid, text=text, metadata=md)


def _project(pid: str, *, root: str | None = None) -> MemoryProjectConfig:
    """Build a MemoryProjectConfig with a single workspace root."""
    return MemoryProjectConfig(
        project_id=pid,
        display_name=pid,
        workspace_roots=(Path(root or f"/work/{pid}"),),
        memory_enabled=True,
        default_scope_for_new_facts="project",
    )


# ── Pure helper tests ──────────────────────────────────────────────


class TestTokenize:
    """The Pass-1 tokenization gate."""

    def test_drops_stopwords_and_short_tokens(self):
        toks = gcp._tokenize("the api endpoint is at /var/lib/foo")
        # 'the', 'is', 'at' are stopwords; '/', 'var', 'lib', 'foo'
        # are content tokens. The single-char split products are
        # filtered by the len>=2 rule.
        assert "the" not in toks
        assert "is" not in toks
        assert "at" not in toks
        assert "api" in toks
        assert "endpoint" in toks
        assert "var" in toks
        assert "foo" in toks

    def test_splits_on_paths_and_underscores(self):
        toks = gcp._tokenize("Users/kai/Projects/anvil_main")
        # Path separators, underscores, and dots all split on the
        # _TOKEN_SPLIT_RE alternation.
        assert "users" in toks
        assert "kai" in toks
        assert "anvil" in toks
        assert "main" in toks


class TestHasDistinctiveOverlap:
    """Pass-1 acceptance gate (at least one shared distinctive token)."""

    def test_accepts_when_one_distinctive_token_shared(self):
        source = {"alpha", "beta"}
        target = {"alpha", "gamma"}
        assert gcp._has_distinctive_overlap(source, target) is True

    def test_rejects_when_no_overlap(self):
        source = {"alpha", "beta"}
        target = {"gamma", "delta"}
        assert gcp._has_distinctive_overlap(source, target) is False


class TestCosineSimilarity:
    """Math sanity for the Pass-2 cosine helper."""

    def test_identical_vectors_one(self):
        v = [1.0, 2.0, 3.0]
        assert gcp._cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_zero(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert gcp._cosine_similarity(a, b) == pytest.approx(0.0)

    def test_zero_magnitude_safe(self):
        # Either-side zero vector returns 0.0 instead of NaN; downstream
        # threshold checks then treat as "no similarity."
        assert gcp._cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0
        assert gcp._cosine_similarity([1.0, 1.0], [0.0, 0.0]) == 0.0


class TestComputeCentroid:
    def test_per_dimension_mean(self):
        vecs = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        c = gcp._compute_centroid(vecs)
        assert c == pytest.approx([0.5, 0.5, 0.0])

    def test_empty_input_returns_empty(self):
        assert gcp._compute_centroid([]) == []


class TestRowMatchesProject:
    """Row -> project partition logic with workspace_root fallback."""

    def test_matches_by_project_id(self):
        proj = _project("anvil")
        row_md = {"project_id": "anvil"}
        assert gcp._row_matches_project(row_md, proj) is True

    def test_fallback_to_workspace_root(self):
        proj = _project("anvil", root="/work/anvil")
        # Row missing project_id but carrying workspace_root pointing
        # at the project's registered root.
        row_md = {"workspace_root": "/work/anvil"}
        assert gcp._row_matches_project(row_md, proj) is True

    def test_no_match(self):
        proj = _project("anvil")
        row_md = {"project_id": "phi"}
        assert gcp._row_matches_project(row_md, proj) is False

    def test_empty_metadata_no_match(self):
        proj = _project("anvil")
        assert gcp._row_matches_project(None, proj) is False
        assert gcp._row_matches_project({}, proj) is False

    def test_explicit_project_id_owns_row_even_if_workspace_root_matches_sibling(self):
        # A row whose project_id explicitly names project phi must NOT
        # leak into project anvil's pool, even if its (stale)
        # workspace_root happens to match anvil's registered root.
        # The workspace_root fallback exists for rows MISSING
        # project_id; it must not override an explicit assignment.
        proj_anvil = _project("anvil", root="/work/anvil")
        row_md = {"project_id": "phi", "workspace_root": "/work/anvil"}
        assert gcp._row_matches_project(row_md, proj_anvil) is False


class TestTFIDF:
    def test_rare_token_outranks_common_token(self):
        # Build two projects: rare term "diesel" only in project A;
        # common terms "the api system" in both. Highest-TF-IDF row
        # in project A is the one whose tokens are most distinctive
        # (the diesel row), not the generic api row.
        rows_a = [
            _row("a1", "the api system is generic"),
            _row("a2", "diesel pump pressure measurement"),
        ]
        rows_b = [
            _row("b1", "the api system handles requests"),
            _row("b2", "the api system stores data"),
        ]
        all_rows = {"a": rows_a, "b": rows_b}
        scores = gcp._project_tfidf_scores(rows_a, all_rows)
        # Diesel row scores strictly higher than the api row.
        assert scores["a2"] > scores["a1"]


# ── Provider normalization regression ─────────────────────────────


class TestProviderNormalization:
    """Single-provider backend resolution under inherited llm_provider.

    The generator must call get_effective_provider(backend, raw)
    BETWEEN resolve_classification_settings and resolve_user_model so
    a codex user inheriting llm_provider='anthropic' resolves to the
    (codex, openai, pr_review) registry entry, not the non-existent
    (codex, anthropic, pr_review) entry.
    """

    def test_codex_with_anthropic_cascade_resolves_to_openai(self, monkeypatch):
        # Stub resolve_classification_settings to return the raw
        # cascade unchanged so we exercise the normalization step.
        from kai import memory_reclassify

        monkeypatch.setattr(
            memory_reclassify,
            "resolve_classification_settings",
            lambda config, user_id, **kwargs: ("codex", None, "anthropic"),
        )
        monkeypatch.setattr(memory_reclassify, "_resolve_user_config", lambda *a, **k: None)

        captured: dict[str, Any] = {}

        def fake_resolve_user_model(role, user_cfg, config, *, backend, provider):
            captured["backend"] = backend
            captured["provider"] = provider
            return "stub-model"

        monkeypatch.setattr(gcp, "resolve_user_model", fake_resolve_user_model)

        # Minimal Config substitute exposing the fields the resolver reads.
        config = MagicMock()
        config.pr_review_timeout_s = 900

        args = MagicMock()
        args.user_id = "42"
        args.backend = None
        args.os_user = None
        args.provider = None
        args.model = ""
        args.timeout_s = None
        args.projects = None
        args.per_project_collisions = 5
        args.per_project_positive = 2
        args.non_project = 3
        args.per_project_legacy = 2
        args.similarity_threshold = 0.55
        args.fallback_cap = 200
        args.verify_top_k = 20
        args.output = Path("/tmp/test-canonical.jsonl")
        args.promote = False
        args.force = False
        args.allow_shortfalls = False
        args.reject = ""

        result = gcp._resolve_run_config(args, config)

        assert isinstance(result, gcp.GenerationConfig)
        # Provider normalized from anthropic to openai for the
        # codex backend before being passed to resolve_user_model.
        # Without this normalization, the registry lookup would
        # fail for any single-provider-backend user whose inherited
        # llm_provider does not match the backend's wire name.
        assert captured["backend"] == "codex"
        assert captured["provider"] == "openai"

    def test_claude_with_deepseek_cascade_resolves_to_anthropic(self, monkeypatch):
        from kai import memory_reclassify

        monkeypatch.setattr(
            memory_reclassify,
            "resolve_classification_settings",
            lambda config, user_id, **kwargs: ("claude", None, "deepseek"),
        )
        monkeypatch.setattr(memory_reclassify, "_resolve_user_config", lambda *a, **k: None)

        captured: dict[str, Any] = {}

        def fake_resolve_user_model(role, user_cfg, config, *, backend, provider):
            captured["backend"] = backend
            captured["provider"] = provider
            return "stub-model"

        monkeypatch.setattr(gcp, "resolve_user_model", fake_resolve_user_model)
        config = MagicMock()
        config.pr_review_timeout_s = 900

        args = MagicMock()
        args.user_id = "42"
        args.backend = None
        args.os_user = None
        args.provider = None
        args.model = ""
        args.timeout_s = None
        args.projects = None
        args.per_project_collisions = 5
        args.per_project_positive = 2
        args.non_project = 3
        args.per_project_legacy = 2
        args.similarity_threshold = 0.55
        args.fallback_cap = 200
        args.verify_top_k = 20
        args.output = Path("/tmp/test-canonical.jsonl")
        args.promote = False
        args.force = False
        args.allow_shortfalls = False
        args.reject = ""

        gcp._resolve_run_config(args, config)
        assert captured["backend"] == "claude"
        assert captured["provider"] == "anthropic"


# ── _build_drafting_reasoner ──────────────────────────────────────


class TestBuildDraftingReasoner:
    """Backend dispatch."""

    def test_each_backend_selects_right_class(self):
        from kai.oneshot import (
            ClaudeOneShotReasoner,
            CodexOneShotReasoner,
            GooseOneShotReasoner,
            OpenCodeOneShotReasoner,
            PiOneShotReasoner,
        )

        assert isinstance(
            gcp._build_drafting_reasoner("claude", os_user=None, provider="anthropic"),
            ClaudeOneShotReasoner,
        )
        assert isinstance(
            gcp._build_drafting_reasoner("codex", os_user=None, provider="openai"),
            CodexOneShotReasoner,
        )
        assert isinstance(
            gcp._build_drafting_reasoner("opencode", os_user=None, provider="anthropic"),
            OpenCodeOneShotReasoner,
        )
        assert isinstance(
            gcp._build_drafting_reasoner("goose", os_user=None, provider="anthropic"),
            GooseOneShotReasoner,
        )
        assert isinstance(
            gcp._build_drafting_reasoner("pi", os_user=None, provider="anthropic"),
            PiOneShotReasoner,
        )

    def test_unknown_backend_raises(self):
        with pytest.raises(RuntimeError, match="unsupported effective backend"):
            gcp._build_drafting_reasoner("invalid-backend", os_user=None, provider="")


# ── Drafting dispatch and abort guard ─────────────────────────────


class FakeReasoner:
    """Pops one canned outcome per run() call.

    Outcomes are either an `OneShotResult` (success) or a
    `OneShotError` subclass (failure). The reasoner does not honor
    timeout / model / purpose values; it just records them on the
    call_history so the test can assert the dispatcher passed the
    right parameters.
    """

    def __init__(self, outcomes: list[Any]):
        self._outcomes = list(outcomes)
        self.call_history: list[dict[str, Any]] = []

    async def run(self, **kwargs):
        self.call_history.append(kwargs)
        out = self._outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


class TestDraftQuestion:
    """The minimal drafting helper."""

    def test_passes_purpose_collision_probe_drafting(self):
        from kai.oneshot import OneShotResult

        result = OneShotResult(text="drafted question", backend="claude", model="m")
        reasoner = FakeReasoner(outcomes=[result])
        text = asyncio.run(gcp._draft_question("a prompt", reasoner=reasoner, model="m-x", timeout_s=300))
        assert text == "drafted question"
        assert reasoner.call_history[0]["purpose"] == "collision_probe_drafting"
        assert reasoner.call_history[0]["model"] == "m-x"
        assert reasoner.call_history[0]["timeout"] == 300

    def test_wraps_oneshot_error_into_drafting_failure(self):
        from kai.oneshot import OneShotTimeout

        reasoner = FakeReasoner(outcomes=[OneShotTimeout("boom")])
        with pytest.raises(gcp.DraftingFailure):
            asyncio.run(gcp._draft_question("a prompt", reasoner=reasoner, model="m", timeout_s=1))


class TestConsecutiveFailureAbort:
    """The shared abort state across drafting calls."""

    def test_fires_at_documented_threshold(self):
        from kai.oneshot import OneShotTimeout

        reasoner = FakeReasoner(outcomes=[OneShotTimeout("t1")] * gcp._CONSECUTIVE_FAILURE_ABORT)
        state = gcp._AbortState()
        gen_config = MagicMock()
        gen_config.model = "m"
        gen_config.timeout_s = 1
        gen_config.effective_backend = "claude"
        gen_config.effective_provider = "anthropic"
        gen_config.effective_os_user = None

        async def drive():
            for _ in range(gcp._CONSECUTIVE_FAILURE_ABORT):
                await gcp._draft_and_count("p", reasoner=reasoner, gen_config=gen_config, state=state)

        with pytest.raises(gcp._AbortException):
            asyncio.run(drive())

    def test_success_resets_counter(self):
        from kai.oneshot import OneShotResult, OneShotTimeout

        # Two failures, then a success: the counter must drop to 0 so
        # the next failure does not immediately trip the abort.
        reasoner = FakeReasoner(
            outcomes=[
                OneShotTimeout("t1"),
                OneShotTimeout("t2"),
                OneShotResult(text="ok", backend="claude", model="m"),
                OneShotTimeout("t3"),  # after success: counter at 1, not 3
            ]
        )
        state = gcp._AbortState()
        gen_config = MagicMock()
        gen_config.model = "m"
        gen_config.timeout_s = 1
        gen_config.effective_backend = "claude"
        gen_config.effective_provider = "anthropic"
        gen_config.effective_os_user = None

        async def drive():
            await gcp._draft_and_count("p", reasoner=reasoner, gen_config=gen_config, state=state)
            await gcp._draft_and_count("p", reasoner=reasoner, gen_config=gen_config, state=state)
            text = await gcp._draft_and_count("p", reasoner=reasoner, gen_config=gen_config, state=state)
            await gcp._draft_and_count("p", reasoner=reasoner, gen_config=gen_config, state=state)
            return text

        result = asyncio.run(drive())
        assert result == "ok"
        assert state.consecutive_failures == 1


# ── Verification ──────────────────────────────────────────────────


class TestVerifyExclusion:
    """The unscoped recall gate used by both collision and non-project."""

    def test_in_top_k_accepts(self, monkeypatch):
        # legacy_retrieve_hits returns ([hits], latency). The target id
        # appears at rank 3 (1-indexed), which is within the default
        # verify_top_k=20.
        async def fake_legacy(question, *, user_id):
            return (
                [{"id": "decoy-a"}, {"id": "decoy-b"}, {"id": "target-x"}],
                10,
            )

        monkeypatch.setattr(gcp, "legacy_retrieve_hits", fake_legacy)
        accepted, rank = asyncio.run(
            gcp._verify_exclusion("a q", user_id="42", excluded_row_id="target-x", verify_top_k=20)
        )
        assert accepted is True
        assert rank == 3

    def test_outside_top_k_drops(self, monkeypatch):
        # Target appears, but at rank 25 with verify_top_k=20: dropped.
        async def fake_legacy(question, *, user_id):
            hits = [{"id": f"decoy-{i}"} for i in range(24)] + [{"id": "target-x"}]
            return hits, 10

        monkeypatch.setattr(gcp, "legacy_retrieve_hits", fake_legacy)
        accepted, rank = asyncio.run(
            gcp._verify_exclusion("a q", user_id="42", excluded_row_id="target-x", verify_top_k=20)
        )
        assert accepted is False
        assert rank == 25

    def test_not_in_hits_drops_with_rank_none(self, monkeypatch):
        # Target absent: rank None, dropped.
        async def fake_legacy(question, *, user_id):
            return [{"id": "decoy-a"}, {"id": "decoy-b"}], 10

        monkeypatch.setattr(gcp, "legacy_retrieve_hits", fake_legacy)
        accepted, rank = asyncio.run(
            gcp._verify_exclusion("a q", user_id="42", excluded_row_id="target-x", verify_top_k=20)
        )
        assert accepted is False
        assert rank is None


# ── Legacy-default enumeration and allocation ─────────────────────


class TestLegacyDefaultEnumeration:
    """Selection MUST go through resolve_memory_scope, not raw metadata."""

    def test_row_without_scope_field_selected_as_legacy_default(self, monkeypatch):
        # Row with NO 'scope' field in raw metadata is resolver-legacy-
        # default. Raw metadata lookup would miss it; resolve_memory_scope
        # surfaces it. Selecting by raw metadata would miss
        # every row whose metadata pre-dates the scope_source key.
        from kai.memory import SCOPE_SOURCE_LEGACY_DEFAULT

        legacy_row = _row("legacy-1")  # no scope_md, no scope key
        scoped_row = _row("scoped-1", project_id="anvil", scope="project")

        monkeypatch.setattr("kai.memory.get_all_facts", lambda *, user_id: [legacy_row])
        monkeypatch.setattr("kai.memory.get_all_episodes", lambda *, user_id: [scoped_row])

        def fake_resolve(metadata):
            # Mirrors the real resolver behavior for the two test rows:
            # missing scope key -> legacy_default; scope=project ->
            # classifier/operator.
            scope_source = SCOPE_SOURCE_LEGACY_DEFAULT
            non_legacy_scope_source = "operator"
            return MagicMock(
                scope_source=(scope_source if "scope" not in (metadata or {}) else non_legacy_scope_source)
            )

        monkeypatch.setattr("kai.memory.resolve_memory_scope", fake_resolve)

        pool = gcp._enumerate_legacy_default_rows(user_id="42")
        assert [r.id for r in pool] == ["legacy-1"]


class TestLegacyDefaultAllocation:
    """Round-robin allocation across projects."""

    def test_three_rows_one_project_two_per_project(self):
        # Pool of 3 rows, single project, quota 2: project gets the
        # first 2 rows, third row unused.
        rows = [_row("r1"), _row("r2"), _row("r3")]
        projects = [_project("anvil")]
        allocation, repeated = gcp._allocate_legacy_default_round_robin(rows, projects, per_project=2)
        assert [r.id for r in allocation["anvil"]] == ["r1", "r2"]
        assert repeated is False

    def test_pool_smaller_than_total_slots_sets_repeated(self):
        # 2 rows, 2 projects, quota 2 each -> need 4 slots; pool
        # wraps. The flag must read True.
        rows = [_row("r1"), _row("r2")]
        projects = [_project("anvil"), _project("phi")]
        allocation, repeated = gcp._allocate_legacy_default_round_robin(rows, projects, per_project=2)
        assert repeated is True
        # First project gets r1, r2; second project wraps back to r1, r2.
        assert [r.id for r in allocation["anvil"]] == ["r1", "r2"]
        assert [r.id for r in allocation["phi"]] == ["r1", "r2"]

    def test_empty_pool_returns_empty_allocation(self):
        projects = [_project("anvil")]
        allocation, repeated = gcp._allocate_legacy_default_round_robin([], projects, per_project=2)
        assert allocation == {"anvil": []}
        assert repeated is False


# ── JSONL write and probe_id_by_line_number ───────────────────────


class TestJsonlWriteAndMapping:
    def test_round_trip_via_load_probes(self, tmp_path: Path):
        """probe_id_by_line_number aligns with load_probes line numbers.

        probe_id is written as a JSONL extra field, then
        load_probes loads only the ScopedProbe fields (line_number
        included). The mapping keyed by line_number must look up
        the same probe_id so the report can name leaking probes
        after self-grade.
        """
        probes = [
            gcp.VerifiedProbe(
                kind=gcp.KIND_COLLISION,
                probe_id="collision:phi:row-a",
                question="q1",
                expected_fact_id=None,
                expected_excluded_fact_ids=("row-a",),
                workspace="/work/phi",
                legacy_rank=3,
            ),
            gcp.VerifiedProbe(
                kind=gcp.KIND_POSITIVE,
                probe_id="positive:anvil:row-b",
                question="q2",
                expected_fact_id="row-b",
                expected_excluded_fact_ids=(),
                workspace="/work/anvil",
                legacy_rank=None,
            ),
            gcp.VerifiedProbe(
                kind=gcp.KIND_NON_PROJECT,
                probe_id="non_project:row-c",
                question="q3",
                expected_fact_id=None,
                expected_excluded_fact_ids=("row-c",),
                workspace=None,
                legacy_rank=5,
            ),
        ]
        path = tmp_path / "probes.jsonl"
        gcp._write_dryrun_jsonl(probes, path)

        # Each JSONL row carries the probe_id extra field.
        for raw in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(raw)
            assert "probe_id" in row

        # The mapping keyed by line_number aligns with load_probes.
        # Build a (question -> probe_id) reference from the JSONL
        # rows, then walk load_probes results and confirm each
        # loaded probe's line_number maps back to the right probe_id.
        # `question` is the join key because it round-trips through
        # load_probes (probe_id does not, which is the entire reason
        # the side-channel mapping exists).
        from kai.eval.retrieval_scoped import load_probes

        mapping = gcp._build_probe_id_by_line_number(path)
        loaded = load_probes(path)
        question_to_probe_id = {}
        for raw in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(raw)
            question_to_probe_id[row["question"]] = row["probe_id"]
        for probe in loaded:
            expected_pid = question_to_probe_id[probe.question]
            assert mapping[probe.line_number] == expected_pid

    def test_duplicate_probe_ids_get_dup_suffix(self, tmp_path: Path):
        probes = [
            gcp.VerifiedProbe(
                kind=gcp.KIND_LEGACY_DEFAULT,
                probe_id="legacy_default:anvil:row-x",
                question="q1",
                expected_fact_id="row-x",
                expected_excluded_fact_ids=(),
                workspace="/work/anvil",
                legacy_rank=None,
            ),
            gcp.VerifiedProbe(
                kind=gcp.KIND_LEGACY_DEFAULT,
                probe_id="legacy_default:anvil:row-x",  # duplicate
                question="q2",
                expected_fact_id="row-x",
                expected_excluded_fact_ids=(),
                workspace="/work/anvil",
                legacy_rank=None,
            ),
        ]
        path = tmp_path / "probes.jsonl"
        gcp._write_dryrun_jsonl(probes, path)
        ids = [json.loads(raw)["probe_id"] for raw in path.read_text().splitlines()]
        assert ids[0] == "legacy_default:anvil:row-x"
        assert ids[1] == "legacy_default:anvil:row-x:dup1"


# ── Self-grade destructure ────────────────────────────────────────


class TestSelfGrade:
    """`evaluate` returns (results, metrics); the generator destructures both."""

    def test_destructures_results_and_metrics(self, monkeypatch, tmp_path: Path):
        # Write a minimal valid probes file so load_probes can parse it.
        probes_path = tmp_path / "probes.jsonl"
        probes_path.write_text(
            json.dumps(
                {
                    "question": "q",
                    "expected_fact_id": None,
                    "expected_excluded_fact_ids": ["x"],
                    "workspace": "/w/a",
                }
            )
            + "\n",
            encoding="utf-8",
        )

        fake_metrics = MagicMock()
        fake_metrics.n_scored_negative = 1
        fake_metrics.exclusion_pass_in_prompt = 1.0
        fake_metrics.exclusion_pass_in_candidates = 1.0

        # AsyncMock returning the (results, metrics) 2-tuple.
        async def fake_evaluate(probes, user_id):
            return ([], fake_metrics)

        monkeypatch.setattr(gcp, "evaluate", fake_evaluate)
        result = asyncio.run(gcp._self_grade(probes_path, user_id="42"))
        assert result.verdict == "ship"
        assert result.n_scored_negative == 1

    def test_verdict_regenerate_when_no_negatives(self, monkeypatch, tmp_path: Path):
        probes_path = tmp_path / "probes.jsonl"
        probes_path.write_text(
            json.dumps(
                {
                    "question": "q",
                    "expected_fact_id": "x",
                    "expected_excluded_fact_ids": [],
                    "workspace": "/w/a",
                }
            )
            + "\n",
            encoding="utf-8",
        )

        fake_metrics = MagicMock()
        fake_metrics.n_scored_negative = 0
        fake_metrics.exclusion_pass_in_prompt = 1.0
        fake_metrics.exclusion_pass_in_candidates = 1.0

        async def fake_evaluate(probes, user_id):
            return ([], fake_metrics)

        monkeypatch.setattr(gcp, "evaluate", fake_evaluate)
        result = asyncio.run(gcp._self_grade(probes_path, user_id="42"))
        assert result.verdict == "regenerate"

    def test_verdict_investigate_on_leak(self, monkeypatch, tmp_path: Path):
        probes_path = tmp_path / "probes.jsonl"
        probes_path.write_text(
            json.dumps(
                {
                    "question": "q",
                    "expected_fact_id": None,
                    "expected_excluded_fact_ids": ["x"],
                    "workspace": "/w/a",
                }
            )
            + "\n",
            encoding="utf-8",
        )

        fake_metrics = MagicMock()
        fake_metrics.n_scored_negative = 1
        fake_metrics.exclusion_pass_in_prompt = 0.5  # leak
        fake_metrics.exclusion_pass_in_candidates = 1.0

        async def fake_evaluate(probes, user_id):
            return ([], fake_metrics)

        monkeypatch.setattr(gcp, "evaluate", fake_evaluate)
        result = asyncio.run(gcp._self_grade(probes_path, user_id="42"))
        assert result.verdict == "INVESTIGATE"


# ── Promote gates ─────────────────────────────────────────────────


def _gen_config(**overrides: Any) -> gcp.GenerationConfig:
    """Build a GenerationConfig with sensible defaults; override per test."""
    defaults: dict[str, Any] = dict(
        user_id="42",
        effective_backend="claude",
        effective_os_user=None,
        effective_provider="anthropic",
        model="stub-model",
        timeout_s=900,
        project_filter=None,
        per_project_collisions=5,
        per_project_positive=2,
        non_project_quota=3,
        per_project_legacy=2,
        similarity_threshold=0.55,
        fallback_cap=200,
        verify_top_k=20,
        output_path=Path("/tmp/test-canonical.jsonl"),
        promote=True,
        force=False,
        allow_shortfalls=False,
        reject_ids=set(),
    )
    defaults.update(overrides)
    return gcp.GenerationConfig(**defaults)


def _verified(kind: str, probe_id: str, workspace: str | None, exc=()) -> gcp.VerifiedProbe:
    return gcp.VerifiedProbe(
        kind=kind,
        probe_id=probe_id,
        question="q",
        expected_fact_id=None if exc else "x",
        expected_excluded_fact_ids=tuple(exc),
        workspace=workspace,
        legacy_rank=None,
    )


class TestPromoteGates:
    """Five-gate evaluation: self-grade, structural, legacy-default, parent dir, exclusive create."""

    def test_block_on_regenerate_verdict(self):
        sg = gcp._SelfGradeResult(
            verdict="regenerate",
            n_scored_negative=0,
            exclusion_pass_in_prompt=1.0,
            exclusion_pass_in_candidates=1.0,
            results=[],
        )
        gc = _gen_config()
        eval_ = gcp._evaluate_promote_gates(sg, [], [_project("anvil")], gc)
        assert eval_.self_grade == "block"

    def test_block_on_investigate_verdict(self):
        sg = gcp._SelfGradeResult(
            verdict="INVESTIGATE",
            n_scored_negative=1,
            exclusion_pass_in_prompt=0.5,
            exclusion_pass_in_candidates=1.0,
            results=[],
        )
        gc = _gen_config()
        eval_ = gcp._evaluate_promote_gates(sg, [], [_project("anvil")], gc)
        assert eval_.self_grade == "block"

    def test_block_on_structural_collision_shortfall(self):
        sg = gcp._SelfGradeResult(
            verdict="ship",
            n_scored_negative=1,
            exclusion_pass_in_prompt=1.0,
            exclusion_pass_in_candidates=1.0,
            results=[],
        )
        gc = _gen_config(per_project_collisions=5)
        # Accepted has 0 collisions for the project; shortfall == 5.
        eval_ = gcp._evaluate_promote_gates(sg, [], [_project("anvil")], gc)
        assert eval_.structural_coverage == "block"

    def test_block_on_legacy_default_shortfall(self):
        sg = gcp._SelfGradeResult(
            verdict="ship",
            n_scored_negative=1,
            exclusion_pass_in_prompt=1.0,
            exclusion_pass_in_candidates=1.0,
            results=[],
        )
        # All structural minimums met; legacy-default count zero -> block.
        gc = _gen_config(per_project_collisions=0, per_project_positive=0, non_project_quota=0)
        eval_ = gcp._evaluate_promote_gates(sg, [], [_project("anvil")], gc)
        assert eval_.legacy_default_coverage == "block"

    def test_allow_shortfalls_relaxes_structural(self):
        sg = gcp._SelfGradeResult(
            verdict="ship",
            n_scored_negative=1,
            exclusion_pass_in_prompt=1.0,
            exclusion_pass_in_candidates=1.0,
            results=[],
        )
        gc = _gen_config(allow_shortfalls=True, per_project_collisions=5, per_project_legacy=0)
        eval_ = gcp._evaluate_promote_gates(sg, [], [_project("anvil")], gc)
        assert eval_.structural_coverage == "relaxed"

    def test_allow_shortfalls_does_not_relax_self_grade(self):
        """--allow-shortfalls does NOT let an INVESTIGATE verdict promote.

        The two gates --allow-shortfalls relaxes are structural and
        legacy-default coverage. The self-grade gate is unconditional;
        a real leak is a real leak regardless of coverage minimums.
        """
        sg = gcp._SelfGradeResult(
            verdict="INVESTIGATE",
            n_scored_negative=1,
            exclusion_pass_in_prompt=0.5,
            exclusion_pass_in_candidates=1.0,
            results=[],
        )
        gc = _gen_config(allow_shortfalls=True)
        eval_ = gcp._evaluate_promote_gates(sg, [], [_project("anvil")], gc)
        assert eval_.self_grade == "block"


# ── Promote write: exclusive create and --force ──────────────────


class TestPromoteWrite:
    """The actual file-system write under the four file gates."""

    def test_creates_parent_directory_on_first_promote(self, tmp_path: Path):
        """First promote creates the canonical parent directory."""
        dryrun = tmp_path / "dry.jsonl"
        dryrun.write_text("{}\n", encoding="utf-8")
        canonical = tmp_path / "nested" / "eval" / "probes" / "collision_v1.jsonl"
        gc = _gen_config(output_path=canonical, force=False, allow_shortfalls=False)
        ge = gcp._GateEvaluation(
            self_grade="pass",
            structural_coverage="pass",
            legacy_default_coverage="pass",
            parent_directory="not_attempted",
            exclusive_create="not_attempted",
            block_reason=None,
            structural_shortfalls={},
            legacy_default_shortfalls={},
        )
        report = gcp.GeneratorReport(
            user_id="42",
            generated_at="now",
            effective_backend="claude",
            effective_provider="anthropic",
            effective_os_user=None,
            model="m",
            timeout_s=900,
        )
        code = gcp._execute_promote_write(dryrun, canonical, gc, ge, report)
        assert code == 0
        assert canonical.exists()
        assert ge.parent_directory == "pass"
        assert ge.exclusive_create == "pass"

    def test_pre_existing_target_blocks_without_force(self, tmp_path: Path):
        dryrun = tmp_path / "dry.jsonl"
        dryrun.write_text("new bytes\n", encoding="utf-8")
        canonical = tmp_path / "collision_v1.jsonl"
        canonical.write_text("old bytes\n", encoding="utf-8")

        gc = _gen_config(output_path=canonical, force=False)
        ge = gcp._GateEvaluation(
            self_grade="pass",
            structural_coverage="pass",
            legacy_default_coverage="pass",
            parent_directory="not_attempted",
            exclusive_create="not_attempted",
            block_reason=None,
            structural_shortfalls={},
            legacy_default_shortfalls={},
        )
        report = gcp.GeneratorReport(
            user_id="42",
            generated_at="now",
            effective_backend="claude",
            effective_provider="anthropic",
            effective_os_user=None,
            model="m",
            timeout_s=900,
        )
        code = gcp._execute_promote_write(dryrun, canonical, gc, ge, report)
        assert code == 2
        # The existing bytes must be unchanged.
        assert canonical.read_text() == "old bytes\n"
        assert ge.exclusive_create == "block"

    def test_force_replaces_atomically(self, tmp_path: Path):
        dryrun = tmp_path / "dry.jsonl"
        dryrun.write_text("new bytes\n", encoding="utf-8")
        canonical = tmp_path / "collision_v1.jsonl"
        canonical.write_text("old bytes\n", encoding="utf-8")

        gc = _gen_config(output_path=canonical, force=True)
        ge = gcp._GateEvaluation(
            self_grade="pass",
            structural_coverage="pass",
            legacy_default_coverage="pass",
            parent_directory="not_attempted",
            exclusive_create="not_attempted",
            block_reason=None,
            structural_shortfalls={},
            legacy_default_shortfalls={},
        )
        report = gcp.GeneratorReport(
            user_id="42",
            generated_at="now",
            effective_backend="claude",
            effective_provider="anthropic",
            effective_os_user=None,
            model="m",
            timeout_s=900,
        )
        code = gcp._execute_promote_write(dryrun, canonical, gc, ge, report)
        assert code == 0
        assert canonical.read_text() == "new bytes\n"
        assert report.forced_overwrite == canonical


# ── --reject path ──────────────────────────────────────────────────


class TestApplyRejects:
    def test_drops_matching_rows_atomically(self, tmp_path: Path):
        path = tmp_path / "probes.jsonl"
        path.write_text(
            "\n".join(
                [
                    json.dumps({"probe_id": "a:1", "question": "q1"}),
                    json.dumps({"probe_id": "b:2", "question": "q2"}),
                    json.dumps({"probe_id": "c:3", "question": "q3"}),
                    "",
                ]
            ),
            encoding="utf-8",
        )
        found, missing = gcp._apply_rejects_to_jsonl(path, {"b:2"})
        assert found == ["b:2"]
        assert missing == []
        remaining = path.read_text(encoding="utf-8").splitlines()
        ids = [json.loads(r)["probe_id"] for r in remaining if r.strip()]
        assert ids == ["a:1", "c:3"]

    def test_unknown_id_warning_not_abort(self, tmp_path: Path):
        path = tmp_path / "probes.jsonl"
        path.write_text(
            json.dumps({"probe_id": "a:1", "question": "q1"}) + "\n",
            encoding="utf-8",
        )
        found, missing = gcp._apply_rejects_to_jsonl(path, {"ghost"})
        assert found == []
        assert missing == ["ghost"]


# ── End-to-end smoke via main() ────────────────────────────────────


class TestMainSmoke:
    """End-to-end smoke: main() with everything mocked.

    Exercises CLI parsing, init, resolver chain, the orchestrator
    pipeline, JSONL write, self-grade, and report rendering. Not a
    correctness test for any single component (each is tested above)
    but verifies they compose without runtime errors.
    """

    def test_default_dry_run_completes(self, monkeypatch, tmp_path: Path):
        from kai.oneshot import OneShotResult

        # Stub init + memory: one project, two rows.
        config = MagicMock()
        config.memory_projects = {"anvil": _project("anvil"), "phi": _project("phi")}
        config.pr_review_timeout_s = 900

        monkeypatch.setattr(gcp, "_initialize_memory_or_exit", lambda: config)

        # Stub project registry side-effect call.
        async def fake_load_registry(cfg):
            return {}

        monkeypatch.setattr("kai.memory_projects.load_project_registry", fake_load_registry)

        # Resolver chain.
        from kai import memory_reclassify

        monkeypatch.setattr(
            memory_reclassify,
            "resolve_classification_settings",
            lambda config, user_id, **kwargs: ("claude", None, "anthropic"),
        )
        monkeypatch.setattr(memory_reclassify, "_resolve_user_config", lambda *a, **k: None)
        monkeypatch.setattr(gcp, "resolve_user_model", lambda *a, **k: "stub-model")

        # Memory rows: each project has two rows that share at least one
        # token with the other so the Pass-1 token filter accepts them.
        anvil_rows = [_row("anvil-1", "api endpoint", project_id="anvil")]
        phi_rows = [_row("phi-1", "api system", project_id="phi")]

        monkeypatch.setattr(
            "kai.memory.get_all_facts",
            lambda *, user_id: anvil_rows + phi_rows,
        )
        monkeypatch.setattr("kai.memory.get_all_episodes", lambda *, user_id: [])

        # Embeddings: identical 3-d vectors so Pass-2 similarity is 1.0
        # and the threshold gate (default 0.55) passes.
        monkeypatch.setattr(
            "kai.memory.embed_texts",
            lambda texts: [[1.0, 0.0, 0.0] for _ in texts],
        )

        # Legacy-default pool: empty (corpus has no legacy-default rows
        # in this smoke). resolve_memory_scope returns scope_source !=
        # legacy_default for every row.

        scope_source_non_legacy = "operator"
        monkeypatch.setattr(
            "kai.memory.resolve_memory_scope",
            lambda md: MagicMock(scope_source=scope_source_non_legacy),
        )

        # Reasoner: every drafting call succeeds with a canned question.
        result = OneShotResult(text="drafted question", backend="claude", model="m")

        async def fake_reasoner_run(**kwargs):
            return result

        fake_reasoner = MagicMock()
        fake_reasoner.run = fake_reasoner_run
        monkeypatch.setattr(gcp, "_build_drafting_reasoner", lambda *a, **k: fake_reasoner)

        # Legacy verify: every drafted exclusion probe passes.
        async def fake_legacy(question, *, user_id):
            return ([{"id": "anvil-1"}, {"id": "phi-1"}], 5)

        monkeypatch.setattr(gcp, "legacy_retrieve_hits", fake_legacy)

        # evaluate(): a ship verdict so the gate logic runs.
        fake_metrics = MagicMock()
        fake_metrics.n_scored_negative = 1
        fake_metrics.exclusion_pass_in_prompt = 1.0
        fake_metrics.exclusion_pass_in_candidates = 1.0

        async def fake_evaluate(probes, user_id):
            return ([], fake_metrics)

        monkeypatch.setattr(gcp, "evaluate", fake_evaluate)

        # Redirect /tmp/ artifacts into tmp_path.
        monkeypatch.setattr(gcp, "_DRYRUN_PROBES_PATH", tmp_path / "dry.jsonl")
        monkeypatch.setattr(gcp, "_DRYRUN_REPORT_PATH", tmp_path / "report.md")

        # Default invocation: dry-run, no --promote.
        code = gcp.main(["42"])
        assert code == 0
        # Default invocation MUST NOT touch the canonical path.
        assert not (tmp_path / "canonical-not-written.jsonl").exists()
        # Both dry-run artifacts exist.
        assert (tmp_path / "dry.jsonl").exists()
        assert (tmp_path / "report.md").exists()
        # Report carries the verdict.
        report_text = (tmp_path / "report.md").read_text(encoding="utf-8")
        assert "Self-grade verdict: ship" in report_text


# ── init failure path ─────────────────────────────────────────────


class TestInitFailure:
    def test_disabled_memory_exits_one(self, monkeypatch, capsys):
        monkeypatch.setattr(gcp, "_initialize_memory_or_exit", lambda: None)
        code = gcp.main(["42"])
        assert code == 1
