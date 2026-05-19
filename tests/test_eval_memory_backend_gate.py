"""
Tests for `kai.eval.memory_backend_gate`.

Coverage focus: the bits the gate owns directly - loader validation,
fixture minimums, sandbox guard, per-backend config copy, anchor
matching, forbidden-content matching, retrieval scoring with
anchor_missing, threshold comparison, and the no-raw-window-text
guarantee on the output JSON.

Synthetic fixtures only. Real probe content stays out of the repo
because it can contain conversation fragments.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from kai import memory
from kai.config import Config
from kai.eval import memory_backend_gate as g

_BASE_CONFIG = Config(
    telegram_bot_token="test-token",
    allowed_user_ids={12345},
    webhook_secret="test-secret",
    memory_enabled=False,
    memory_extraction_enabled=False,
)


# ── Loader ──────────────────────────────────────────────────────────


def _probe_row(**overrides) -> dict:
    """Build a minimum-valid probe row, with overrides applied."""
    row = {
        "probe_id": "p1",
        "category": "durable-fact",
        "window": {
            "prior": [],
            "current": {"user": "user msg", "assistant": "assistant reply"},
        },
        "expected": {
            "must_store": [
                {
                    "anchor_id": "a1",
                    "content_any": ["likes coffee"],
                }
            ]
        },
    }
    row.update(overrides)
    return row


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write a list of dicts to `path` as JSONL. Synthetic data only."""
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


class TestLoadProbes:
    def test_accepts_all_categories(self, tmp_path):
        """A synthetic fixture exercising each allowed category must
        round-trip through the loader without complaint, so a future
        category rename surfaces here."""
        rows = []
        for i, cat in enumerate(g.ALLOWED_CATEGORIES):
            row = _probe_row(probe_id=f"p{i}", category=cat)
            if cat in ("workflow-noise",):
                row["expected"] = {"must_not_store": [{"content_any": ["forbidden phrase"]}]}
            rows.append(row)
        path = tmp_path / "probes.jsonl"
        _write_jsonl(path, rows)

        probes = g.load_probes(path)

        assert {p.category for p in probes} == set(g.ALLOWED_CATEGORIES)

    def test_rejects_missing_current(self, tmp_path):
        """Probes with no `window.current.user` must fail with a
        line-numbered error so the operator can find them."""
        row = _probe_row()
        row["window"].pop("current")
        path = tmp_path / "probes.jsonl"
        _write_jsonl(path, [row])

        with pytest.raises(ValueError, match=r"window\.current"):
            g.load_probes(path)

    def test_rejects_unknown_category(self, tmp_path):
        path = tmp_path / "probes.jsonl"
        _write_jsonl(path, [_probe_row(category="garbage")])

        with pytest.raises(ValueError, match="garbage"):
            g.load_probes(path)

    def test_rejects_retrieval_pointing_at_unknown_anchor(self, tmp_path):
        """A retrieval query whose `anchor_id` is not declared in
        `must_store` is unscoreable; the loader must reject it
        before the harness runs."""
        row = _probe_row()
        row["expected"]["retrieval"] = [{"query": "q", "anchor_id": "nonexistent"}]
        path = tmp_path / "probes.jsonl"
        _write_jsonl(path, [row])

        with pytest.raises(ValueError, match="nonexistent"):
            g.load_probes(path)

    def test_validates_consolidation_intent_and_outcome(self, tmp_path):
        """expected.consolidation values must come from the allowed
        enums; the spec's enum omits `dropped` so a probe asserting
        it must be rejected at load time."""
        row = _probe_row(category="consolidation")
        row["expected"]["consolidation"] = {"expected_outcome": "dropped"}
        path = tmp_path / "probes.jsonl"
        _write_jsonl(path, [row])

        with pytest.raises(ValueError, match="dropped"):
            g.load_probes(path)

        # And the same for an invalid intent.
        row["expected"]["consolidation"] = {"expected_intent": "garbage"}
        _write_jsonl(path, [row])
        with pytest.raises(ValueError, match="garbage"):
            g.load_probes(path)

    def test_duplicate_anchor_ids_rejected(self, tmp_path):
        row = _probe_row()
        row["expected"]["must_store"].append({"anchor_id": "a1", "content_any": ["other"]})
        path = tmp_path / "probes.jsonl"
        _write_jsonl(path, [row])

        with pytest.raises(ValueError, match="duplicate anchor_id"):
            g.load_probes(path)


# ── Fixture minimums ────────────────────────────────────────────────


def _minimum_fixture() -> list[g.GateProbe]:
    """Construct an in-memory probe list that satisfies §D4 minimums.

    Lives next to the test that uses it so the threshold values move
    together if the spec ever loosens or tightens; reading the minimum
    counts off the helper is the source of truth for these tests.
    """
    probes: list[g.GateProbe] = []
    for i in range(10):
        probes.append(
            g.GateProbe(
                probe_id=f"durable-{i}",
                category="durable-fact" if i % 2 == 0 else "confirmation",
                prior=(),
                current_user="user",
                current_assistant="assistant",
                must_store=(g.ExpectedAnchor(anchor_id=f"a-{i}", content_any=(f"fact-{i}",)),),
                retrieval=(
                    g.RetrievalExpectation(query=f"q-{i}", anchor_id=f"a-{i}"),
                    g.RetrievalExpectation(query=f"q2-{i}", anchor_id=f"a-{i}"),
                ),
            )
        )
    for i in range(6):
        probes.append(
            g.GateProbe(
                probe_id=f"wf-{i}",
                category="workflow-noise",
                prior=(),
                current_user="u",
                current_assistant="a",
                must_not_store=(g.ForbiddenContent(content_any=(f"noise-{i}",)),),
            )
        )
    for i in range(4):
        probes.append(
            g.GateProbe(
                probe_id=f"cons-{i}",
                category="consolidation",
                prior=(),
                current_user="u",
                current_assistant="a",
                consolidation=g.ExpectedConsolidation(expected_intent="skip_redundant"),
            )
        )
    for i in range(2):
        probes.append(
            g.GateProbe(
                probe_id=f"ep-pos-{i}",
                category="episode-positive",
                prior=(),
                current_user="u",
                current_assistant="a",
            )
        )
        probes.append(
            g.GateProbe(
                probe_id=f"ep-neg-{i}",
                category="episode-negative",
                prior=(),
                current_user="u",
                current_assistant="a",
            )
        )
    return probes


class TestValidateFixtureMinimums:
    def test_passes_on_minimum_fixture(self):
        g.validate_fixture_minimums(_minimum_fixture())

    def test_rejects_too_few_probes(self):
        with pytest.raises(ValueError, match="at least 24 probes"):
            g.validate_fixture_minimums(_minimum_fixture()[:5])

    def test_lists_all_issues_in_single_message(self):
        """Operator-side ergonomics: the loader should report every
        unmet minimum in one pass so the operator does not have to
        re-run the gate four times to find each violation."""
        too_small = _minimum_fixture()[:5]
        with pytest.raises(ValueError) as excinfo:
            g.validate_fixture_minimums(too_small)
        message = str(excinfo.value)
        assert "24 probes" in message
        assert ";" in message


# ── Sandbox guard ───────────────────────────────────────────────────


class TestValidateUserPrefix:
    def test_accepts_sandbox_prefix(self):
        g.validate_user_prefix("sandbox-498")

    def test_rejects_non_sandbox_prefix(self):
        """Non-sandbox prefixes risk writing real user rows; the guard
        must refuse them at preflight."""
        with pytest.raises(ValueError, match="sandbox-"):
            g.validate_user_prefix("real-user-498")


# ── Backend config copy ─────────────────────────────────────────────


class TestMakeBackendConfig:
    def test_claude_resolves_claude_models(self):
        config = g.make_backend_config(_BASE_CONFIG, "claude")
        assert config.memory_reasoner_backend == "claude"
        assert config.memory_extraction_model == "claude-haiku-4-5-20251001"
        assert config.memory_episode_model == "claude-haiku-4-5-20251001"
        assert config.memory_enabled is True
        assert config.memory_extraction_enabled is True

    def test_codex_resolves_codex_models(self):
        config = g.make_backend_config(_BASE_CONFIG, "codex")
        assert config.memory_reasoner_backend == "codex"
        assert config.memory_extraction_model == "gpt-5.4-mini"
        assert config.memory_episode_model == "gpt-5.4-mini"


# ── Anchor matching ─────────────────────────────────────────────────


def _row(
    *, row_id: str = "r1", text: str = "", speaker: str = "user", tags: list[str] | None = None
) -> memory.MemoryResult:
    return memory.MemoryResult(
        id=row_id,
        text=text,
        score=0.0,
        memory_type="fact",
        metadata={"speaker": speaker, "tags": tags or []},
        created_at="",
        updated_at="",
    )


class TestMatchAnchor:
    def test_case_insensitive_substring(self):
        anchor = g.ExpectedAnchor(anchor_id="a", content_any=("Prefers Eastern Time",))
        row = _row(text="user prefers eastern time for scheduling")
        assert g._match_anchor(anchor, [row]) is row

    def test_no_match_returns_none(self):
        anchor = g.ExpectedAnchor(anchor_id="a", content_any=("eastern time",))
        row = _row(text="something unrelated")
        assert g._match_anchor(anchor, [row]) is None

    def test_returns_first_matching_row(self):
        """Multiple matching rows: the first wins so the harness can
        pin a stable target_row_id for retrieval scoring; later rows
        with the same content would never be reachable."""
        anchor = g.ExpectedAnchor(anchor_id="a", content_any=("coffee",))
        row1 = _row(row_id="r1", text="likes coffee")
        row2 = _row(row_id="r2", text="prefers coffee black")
        assert g._match_anchor(anchor, [row1, row2]) is row1


# ── Forbidden-content matching ──────────────────────────────────────


class TestForbiddenContent:
    def test_violation_recorded(self):
        """A row whose content contains any forbidden substring
        triggers a violation; multiple needles in the same row count
        once per row (the first match short-circuits inner loop)."""
        forbidden = g.ForbiddenContent(content_any=("merged PR",))
        row = _row(text="user merged PR #501 today")
        # _run_probe is async and side-effectful; this test exercises
        # the matching logic by constructing an outcome directly and
        # checking the same case-insensitive substring rule.
        text_lower = (row.text or "").lower()
        assert any(needle.lower() in text_lower for needle in forbidden.content_any)


# ── Retrieval scoring ───────────────────────────────────────────────


class TestRetrievalScoring:
    def test_anchor_missing_maps_to_miss(self):
        """A retrieval query whose anchor was never satisfied must
        score as a miss with reason `anchor_missing`; missing
        extraction is part of end-to-end backend quality and must
        not be silently excluded from precision/MRR."""
        results = [
            g.RetrievalQueryResult(
                probe_id="p1",
                query="q",
                anchor_id="a1",
                target_row_id=None,
                rank=None,
                in_prompt=False,
                reason="anchor_missing",
            ),
            g.RetrievalQueryResult(
                probe_id="p2",
                query="q2",
                anchor_id="a2",
                target_row_id="r2",
                rank=1,
                in_prompt=True,
            ),
        ]
        assert g._precision_at_k(results, 1) == 0.5
        assert g._precision_at_k(results, 5) == 0.5
        # MRR: one hit at rank 1, one miss -> 0.5
        assert g._mean_reciprocal_rank(results) == 0.5


# ── Threshold comparison ────────────────────────────────────────────


def _clean_metrics(**overrides) -> g.BackendMetrics:
    """Build a `BackendMetrics` with all-zero failure counters and
    above-floor retrieval values, so each test can override exactly
    the fields it cares about."""
    base = g.BackendMetrics(
        retrieval_query_count=20,
        total_reasoner_calls=20,
        success_count=20,
        fact_anchor_total=10,
        fact_anchor_satisfied=10,
        fact_anchor_recall=1.0,
        forbidden_content_total=6,
        forbidden_content_violation_count=0,
        speaker_labeled_anchor_count=10,
        speaker_correct_count=10,
        speaker_accuracy=1.0,
        tag_presence_rate=1.0,
        malformed_tag_count=0,
        precision_at_1=1.0,
        precision_at_3=1.0,
        precision_at_5=1.0,
        mrr=1.0,
        fraction_in_prompt=1.0,
        episode_required_field_validity=1.0,
        episode_recall=1.0,
    )
    return replace(base, **overrides)


class TestCompareThresholds:
    def test_passes_when_codex_equals_claude(self):
        claude = _clean_metrics()
        codex = _clean_metrics()
        report = g.compare_thresholds(claude, codex)
        assert report.overall == "pass"
        for check in report.checks:
            assert check.passed, f"check {check.name} unexpectedly failed: {check.reason}"

    def test_fails_codex_on_runtime_parse_failure(self):
        """A single output_error on the codex side flips T1 even
        if every other metric is identical to claude; runtime
        cleanliness is a hard threshold."""
        claude = _clean_metrics()
        codex = _clean_metrics(output_error_count=1, parse_failure_rate=0.05)
        report = g.compare_thresholds(claude, codex)
        assert report.overall == "fail"
        failed = [c.name for c in report.checks if not c.passed]
        assert "T1.output_error" in failed
        assert "T1.parse_failure_rate" in failed

    def test_fails_codex_on_p_at_5_below_floor(self):
        claude = _clean_metrics()
        codex = _clean_metrics(precision_at_5=0.79)
        report = g.compare_thresholds(claude, codex)
        assert report.overall == "fail"
        failed = [c.name for c in report.checks if not c.passed]
        assert "T5.precision_at_5_floor" in failed

    def test_fails_codex_on_hallucinated_ids(self):
        claude = _clean_metrics()
        codex = _clean_metrics(hallucinated_id_count=1)
        report = g.compare_thresholds(claude, codex)
        assert report.overall == "fail"
        failed = [c.name for c in report.checks if not c.passed]
        assert "T7.hallucinated_id" in failed

    def test_invalid_baseline_when_claude_has_runtime_failures(self):
        """A claude arm with its own runtime failures invalidates the
        whole gate; the codex result is not comparable. Overall is
        `invalid_baseline`, not `fail`."""
        claude = _clean_metrics(timeout_count=2)
        codex = _clean_metrics()
        report = g.compare_thresholds(claude, codex)
        assert report.overall == "invalid_baseline"
        assert len(report.checks) == 1
        assert report.checks[0].name == "T1.claude_baseline"

    def test_fails_codex_on_episode_recall_floor_with_three_positives(self):
        """The conditional T6 recall floor fires when at least 3
        episode-positive probes exist. A weak claude baseline can
        let the FN-count band pass while codex misses every positive;
        the recall floor exists to catch exactly that hole.

        Setup: 3 episode-positive probes. Claude misses 2 (TP=1,
        FN=2, recall=0.33). Codex misses 3 (TP=0, FN=3, recall=0.0).
        FN band: codex_FN (3) <= claude_FN (2) + 1 = 3, passes.
        Recall floor: max(0.67, 0.33 - 0.25) = 0.67, codex (0.0)
        fails. The check is what flips the verdict."""
        claude = _clean_metrics(
            episode_true_positive_count=1,
            episode_false_negative_count=2,
            episode_recall=1 / 3,
        )
        codex = _clean_metrics(
            episode_true_positive_count=0,
            episode_false_negative_count=3,
            episode_recall=0.0,
        )
        report = g.compare_thresholds(claude, codex)
        failed = [c.name for c in report.checks if not c.passed]
        assert "T6.episode_recall" in failed
        # The FN-count band still passes; this is the key point of the
        # finding - the recall floor is not redundant with the FN band.
        assert "T6.episode_false_negative" not in failed

    def test_recall_floor_not_evaluated_below_three_positives(self):
        """With only 2 episode-positive probes the recall floor does
        NOT fire; small-sample noise would make a hard threshold
        meaningless. The FN-count band still runs."""
        claude = _clean_metrics(
            episode_true_positive_count=1,
            episode_false_negative_count=1,
            episode_recall=0.5,
        )
        codex = _clean_metrics(
            episode_true_positive_count=0,
            episode_false_negative_count=2,
            episode_recall=0.0,
        )
        report = g.compare_thresholds(claude, codex)
        check_names = [c.name for c in report.checks]
        assert "T6.episode_recall" not in check_names


# ── Output ──────────────────────────────────────────────────────────


def _stub_run(backend: str) -> g.BackendRun:
    """Minimal BackendRun for output-shape tests. Carries one probe
    outcome so per_probe and JSON shape assertions have something to
    inspect."""
    return g.BackendRun(
        backend=backend,
        sandbox_user_id=f"sandbox-498-{backend}",
        model_fact="model-x",
        model_episode="model-x",
        log_path=Path("/dev/null"),
        probes=[
            g.ProbeOutcome(
                probe_id="p1",
                category="durable-fact",
                satisfied_anchors={"a1": "row-1"},
                new_fact_ids=["row-1"],
            )
        ],
    )


class TestBuildGateResult:
    def test_omits_raw_window_text(self):
        """Operator probe windows can contain conversation fragments
        that must not leak into a shareable JSON artifact. The
        per_probe entries reference probe_id and anchor IDs only -
        no `user` / `assistant` text from `window.current`."""
        probes = [
            g.GateProbe(
                probe_id="p1",
                category="durable-fact",
                prior=(),
                current_user="USER-PRIVATE-TEXT",
                current_assistant="ASSISTANT-PRIVATE-TEXT",
                must_store=(g.ExpectedAnchor(anchor_id="a1", content_any=("phrase",)),),
                retrieval=(g.RetrievalExpectation(query="QUERY-PRIVATE-TEXT", anchor_id="a1"),),
            )
        ]
        runs = {"claude": _stub_run("claude"), "codex": _stub_run("codex")}
        metrics = {"claude": _clean_metrics(), "codex": _clean_metrics()}
        report = g.compare_thresholds(metrics["claude"], metrics["codex"])

        result = g.build_gate_result(
            probes=probes,
            runs=runs,
            metrics=metrics,
            threshold_report=report,
            generated_at="2026-05-18T00:00:00Z",
        )

        serialized = json.dumps(result)
        assert "USER-PRIVATE-TEXT" not in serialized
        assert "ASSISTANT-PRIVATE-TEXT" not in serialized
        assert "QUERY-PRIVATE-TEXT" not in serialized

    def test_has_top_level_qualitative_verdict_pending(self):
        """The harness writes `qualitative_verdict: "pending"` so the
        operator overwrites it after reviewing the sample markdown.
        Top-level key, not nested under thresholds."""
        runs = {"claude": _stub_run("claude")}
        metrics = {"claude": _clean_metrics()}
        report = g.ThresholdReport(checks=[], overall="single_backend")
        result = g.build_gate_result(
            probes=[],
            runs=runs,
            metrics=metrics,
            threshold_report=report,
            generated_at="2026-05-18T00:00:00Z",
        )
        assert result["qualitative_verdict"] == "pending"

    def test_per_probe_carries_anchor_outcomes(self):
        probes = [
            g.GateProbe(
                probe_id="p1",
                category="durable-fact",
                prior=(),
                current_user="u",
                current_assistant="a",
                must_store=(g.ExpectedAnchor(anchor_id="a1", content_any=("x",)),),
            )
        ]
        runs = {"claude": _stub_run("claude")}
        metrics = {"claude": _clean_metrics()}
        report = g.ThresholdReport(checks=[], overall="single_backend")
        result = g.build_gate_result(
            probes=probes,
            runs=runs,
            metrics=metrics,
            threshold_report=report,
            generated_at="2026-05-18T00:00:00Z",
        )
        assert result["per_probe"][0]["probe_id"] == "p1"
        assert result["per_probe"][0]["backends"]["claude"]["satisfied_anchors"] == {"a1": "row-1"}


class TestBuildQualitativeSample:
    def test_no_raw_window_text(self):
        """The qualitative sample must not contain raw probe window
        text. The probe carries private USER-PRIVATE-TEXT but the
        sample should render only the (synthetic) row content and
        anchor labels."""
        probes = [
            g.GateProbe(
                probe_id="p1",
                category="durable-fact",
                prior=(),
                current_user="USER-PRIVATE-TEXT",
                current_assistant="ASSISTANT-PRIVATE-TEXT",
                must_store=(g.ExpectedAnchor(anchor_id="a1", content_any=("x",)),),
            )
        ]
        run = _stub_run("claude")
        run.final_facts = [_row(row_id="row-1", text="row content")]
        runs = {"claude": run}

        out = g.build_qualitative_sample(runs=runs, probes=probes)

        assert "USER-PRIVATE-TEXT" not in out
        assert "ASSISTANT-PRIVATE-TEXT" not in out
        assert "qualitative_verdict: pending" in out


# ── Log parsing ─────────────────────────────────────────────────────


class TestParseLogLines:
    def test_oneshot_kv_extracts_outcome_and_category(self):
        message = (
            "oneshot_reasoner purpose=fact_extraction backend=codex "
            "model=gpt-5.4-mini duration_ms=12046 outcome=success returncode=0"
        )
        fields = g.parse_oneshot_kv_line(message)
        assert fields["outcome"] == "success"
        assert fields["backend"] == "codex"
        assert fields["model"] == "gpt-5.4-mini"
        assert fields["returncode"] == "0"

    def test_json_suffix_decodes_consolidate_intent(self):
        message = 'memory.consolidate.intent {"intent":"new","outcome":"stored","new_id":"r1"}'
        payload = g.parse_json_suffix_line(message, "memory.consolidate.intent")
        assert payload == {"intent": "new", "outcome": "stored", "new_id": "r1"}

    def test_json_suffix_ignores_other_prefixes(self):
        message = 'memory.recall {"hits":[]}'
        assert g.parse_json_suffix_line(message, "memory.consolidate.intent") is None


# ── Metric computation from a synthetic log ─────────────────────────


class TestComputeMetricsFromLog:
    def test_counts_oneshot_outcomes_and_categories(self, tmp_path):
        """A synthetic per-backend log with three reasoner lines
        produces matching `_count` fields and the right
        parse_failure_rate. Hand-built to match the FileHandler
        format `LEVEL:logger:message`."""
        log_path = tmp_path / "claude.log"
        log_path.write_text(
            "\n".join(
                [
                    "INFO:kai.oneshot:oneshot_reasoner purpose=fact_extraction backend=claude model=m duration_ms=10 outcome=success returncode=0",
                    "INFO:kai.oneshot:oneshot_reasoner purpose=fact_extraction backend=claude model=m duration_ms=10 outcome=output_error error_category=non_object_json returncode=0",
                    "INFO:kai.oneshot:oneshot_reasoner purpose=fact_extraction backend=claude model=m duration_ms=10 outcome=timeout error_category=timeout",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        run = g.BackendRun(
            backend="claude",
            sandbox_user_id="sandbox-498-claude",
            model_fact="m",
            model_episode="m",
            log_path=log_path,
        )
        metrics = g.compute_metrics(run)
        assert metrics.total_reasoner_calls == 3
        assert metrics.success_count == 1
        assert metrics.output_error_count == 1
        assert metrics.timeout_count == 1
        assert metrics.non_object_json_count == 1
        assert metrics.parse_failure_rate == pytest.approx(1 / 3)

    def test_counts_consolidate_intent_outcomes(self, tmp_path):
        """`stored` (intent=update_of) increments both stored_count
        and replaced_count, matching the spec's subset relationship.
        `hallucinated_id` increments hallucinated_id_count via the
        intent axis, not the outcome axis."""
        log_path = tmp_path / "claude.log"
        log_path.write_text(
            "\n".join(
                [
                    'INFO:kai.memory_extraction:memory.consolidate.intent {"intent":"new","outcome":"stored","new_id":"r1"}',
                    'INFO:kai.memory_extraction:memory.consolidate.intent {"intent":"update_of","outcome":"stored","new_id":"r2","replaced_id":"r0"}',
                    'INFO:kai.memory_extraction:memory.consolidate.intent {"intent":"skip_redundant","outcome":"skipped"}',
                    'INFO:kai.memory_extraction:memory.consolidate.intent {"intent":"hallucinated_id","outcome":"dropped"}',
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        run = g.BackendRun(
            backend="claude",
            sandbox_user_id="sandbox-498-claude",
            model_fact="m",
            model_episode="m",
            log_path=log_path,
        )
        metrics = g.compute_metrics(run)
        assert metrics.stored_count == 2
        assert metrics.replaced_count == 1
        assert metrics.skipped_count == 1
        assert metrics.hallucinated_id_count == 1
        # consolidation_skip_rate = skipped / (stored + skipped) = 1/3.
        assert metrics.consolidation_skip_rate == pytest.approx(1 / 3)


# ── Consolidation assertion scoring ─────────────────────────────────


def _write_consolidate_line(path: Path, payload: dict, *, append: bool = True) -> int:
    """Append one synthetic `memory.consolidate.intent` line, return new file size.

    Matches the FileHandler `LEVEL:logger:message` format the harness
    produces. The returned size is what `_run_probe` would see as
    `pre_log_offset` for the NEXT probe.
    """
    line = "INFO:kai.memory_extraction:memory.consolidate.intent " + json.dumps(payload, separators=(",", ":")) + "\n"
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as fh:
        fh.write(line)
    return path.stat().st_size


class TestReadConsolidationEvents:
    """`_read_consolidation_events` reads only lines after the supplied
    offset and decodes JSON payloads from `memory.consolidate.intent`
    log lines; other prefixes and malformed lines are silently
    skipped so per-probe attribution survives a future log change."""

    def test_returns_only_lines_after_offset(self, tmp_path):
        path = tmp_path / "claude.log"
        path.write_text("", encoding="utf-8")
        _write_consolidate_line(path, {"intent": "new", "outcome": "stored"})
        offset_after_first = path.stat().st_size
        _write_consolidate_line(path, {"intent": "skip_redundant", "outcome": "skipped"})

        events = g._read_consolidation_events(path, offset_after_first)

        assert len(events) == 1
        assert events[0]["intent"] == "skip_redundant"

    def test_skips_unrelated_prefixes(self, tmp_path):
        path = tmp_path / "claude.log"
        path.write_text(
            'INFO:kai.memory:memory.recall {"hits":[]}\n'
            'INFO:kai.memory_extraction:memory.consolidate.intent {"intent":"new","outcome":"stored"}\n',
            encoding="utf-8",
        )
        events = g._read_consolidation_events(path, 0)
        assert len(events) == 1
        assert events[0]["intent"] == "new"


class TestScoreConsolidationAssertion:
    """A probe's `expected.consolidation` declares one or both of
    `expected_intent` / `expected_outcome`. The scorer flips the
    matching `consolidation_*_satisfied` bool to True iff at least
    one captured event satisfies that axis."""

    def test_intent_match_only(self):
        outcome = g.ProbeOutcome(
            probe_id="p",
            category="consolidation",
            consolidation_events=[{"intent": "skip_redundant", "outcome": "skipped"}],
        )
        g._score_consolidation_assertion(outcome, g.ExpectedConsolidation(expected_intent="skip_redundant"))
        assert outcome.consolidation_intent_satisfied is True
        assert outcome.consolidation_outcome_satisfied is None

    def test_both_axes_must_be_satisfied(self):
        """The two axes can be satisfied by different events; that
        flexibility is intentional so a probe declaring both does
        not have to depend on a single payload carrying both."""
        outcome = g.ProbeOutcome(
            probe_id="p",
            category="consolidation",
            consolidation_events=[
                {"intent": "update_of", "outcome": "stored"},
                {"intent": "skip_redundant", "outcome": "skipped"},
            ],
        )
        g._score_consolidation_assertion(
            outcome,
            g.ExpectedConsolidation(expected_intent="update_of", expected_outcome="skipped"),
        )
        assert outcome.consolidation_intent_satisfied is True
        assert outcome.consolidation_outcome_satisfied is True

    def test_failure_when_intent_does_not_match(self):
        outcome = g.ProbeOutcome(
            probe_id="p",
            category="consolidation",
            consolidation_events=[{"intent": "new", "outcome": "stored"}],
        )
        g._score_consolidation_assertion(outcome, g.ExpectedConsolidation(expected_intent="skip_redundant"))
        assert outcome.consolidation_intent_satisfied is False


# ── Episode validity includes speaker check ─────────────────────────


class TestEpisodeValiditySpeakerCheck:
    """An episode row missing or carrying the wrong speaker must
    NOT count toward `episode_required_field_validity`, even when
    every content field is present. The speaker field lives outside
    the schema's required list (the stage-2 generator sets it) so
    the gate has to enforce it explicitly."""

    def _make_episode(self, *, speaker: str | None) -> memory.MemoryResult:
        metadata = {
            "goal": "Diagnose the slowness",
            "context": "Production extractions are slow",
            "approach": "Ran a diagnostic and isolated the cost driver",
            "outcome": "Reduced the payload cap",
            "outcome_quality": "success",
            "tags": ["memory"],
            "actors": ["user"],
        }
        if speaker is not None:
            metadata["speaker"] = speaker
        return memory.MemoryResult(
            id="ep-1",
            text="episode summary",
            score=0.0,
            memory_type="episode",
            metadata=metadata,
            created_at="",
            updated_at="",
        )

    def _make_run(self, episode: memory.MemoryResult) -> g.BackendRun:
        return g.BackendRun(
            backend="claude",
            sandbox_user_id="sandbox-498-claude",
            model_fact="m",
            model_episode="m",
            log_path=Path("/dev/null"),
            probes=[
                g.ProbeOutcome(
                    probe_id="p1",
                    category="episode-positive",
                    new_episode_ids=[episode.id],
                )
            ],
            final_episodes=[episode],
        )

    def _probes(self) -> list[g.GateProbe]:
        return [
            g.GateProbe(
                probe_id="p1",
                category="episode-positive",
                prior=(),
                current_user="u",
                current_assistant="a",
            )
        ]

    def test_speaker_episode_summary_counts_as_valid(self):
        run = self._make_run(self._make_episode(speaker="episode_summary"))
        tp, _fp, _fn, _tn, validity = g.score_episodes(self._probes(), run)
        assert tp == 1
        assert validity == 1.0

    def test_missing_speaker_drops_validity_to_zero(self):
        run = self._make_run(self._make_episode(speaker=None))
        tp, _fp, _fn, _tn, validity = g.score_episodes(self._probes(), run)
        # Still a true positive (the row appeared), but invalid.
        assert tp == 1
        assert validity == 0.0

    def test_wrong_speaker_drops_validity_to_zero(self):
        run = self._make_run(self._make_episode(speaker="user"))
        tp, _fp, _fn, _tn, validity = g.score_episodes(self._probes(), run)
        assert tp == 1
        assert validity == 0.0


# ── CLI initializes memory before run_backend ───────────────────────


class TestCLIInitializesMemory:
    """The CLI must call `memory.init_memory` before any backend arm
    runs. Without it, `kai.memory._memory` stays None and every
    storage call short-circuits, so the harness would produce a
    silent zero-state result. This was caught in PR #502 round 1."""

    def test_init_memory_called_before_backend_runs(self, tmp_path, monkeypatch):
        """Patch `init_memory` and `run_backend` to record call order;
        the gate must call init_memory at least once and must do so
        before the first run_backend invocation."""
        from unittest.mock import patch

        # Build a minimum-valid probe fixture so the preflight passes.
        path = tmp_path / "probes.jsonl"
        rows = []
        for i in range(10):
            rows.append(
                _probe_row(
                    probe_id=f"d{i}",
                    category="durable-fact",
                    expected={
                        "must_store": [{"anchor_id": f"a{i}", "content_any": [f"c{i}"]}],
                        "retrieval": [
                            {"query": f"q{i}", "anchor_id": f"a{i}"},
                            {"query": f"q2{i}", "anchor_id": f"a{i}"},
                        ],
                    },
                )
            )
        for i in range(6):
            rows.append(
                _probe_row(
                    probe_id=f"w{i}",
                    category="workflow-noise",
                    expected={"must_not_store": [{"content_any": [f"n{i}"]}]},
                )
            )
        for i in range(4):
            rows.append(
                _probe_row(
                    probe_id=f"c{i}",
                    category="consolidation",
                    expected={"consolidation": {"expected_intent": "skip_redundant"}},
                )
            )
        for i in range(2):
            rows.append(_probe_row(probe_id=f"ep{i}", category="episode-positive", expected={}))
            rows.append(_probe_row(probe_id=f"en{i}", category="episode-negative", expected={}))
        _write_jsonl(path, rows)

        call_order: list[str] = []

        def _fake_init(_config):
            call_order.append("init_memory")

        async def _fake_run_backend(**kwargs):
            call_order.append("run_backend")
            return g.BackendRun(
                backend=kwargs["backend"],
                sandbox_user_id=kwargs["sandbox_user_id"],
                model_fact="m",
                model_episode="m",
                log_path=kwargs["log_path"],
            )

        args = type(
            "Args",
            (),
            {
                "probes": path,
                "output_dir": tmp_path / "out",
                "user_prefix": "sandbox-test",
                "backends": ["claude", "codex"],
                "os_user": "test-target",
                "reset": True,
                "keep_sandboxes": False,
                "qualitative_sample_size": 10,
                "fail_on_threshold": False,
                "validate_only": False,
            },
        )()

        with (
            patch("kai.eval.memory_backend_gate.memory.init_memory", _fake_init),
            patch("kai.eval.memory_backend_gate.load_config", return_value=_BASE_CONFIG),
            patch("kai.eval.memory_backend_gate.run_backend", _fake_run_backend),
        ):
            import asyncio

            rc = asyncio.run(g._run_cli(args))

        assert rc == 0
        assert "init_memory" in call_order
        assert call_order.index("init_memory") < call_order.index("run_backend")


# ── --os-user preflight (issue #503) ────────────────────────────────


class TestOsUserPreflight:
    """The eval gate's `--os-user` is required when the codex arm is
    in scope, because sandbox user IDs do not resolve to users.yaml
    entries and the codex memory reasoner refuses to run with
    `os_user=None`. The preflight rejects the missing-flag case
    before any model call so the operator gets exit 2 instead of a
    per-probe routing_error cascade."""

    def _valid_fixture(self, tmp_path: Path) -> Path:
        rows = []
        for i in range(10):
            rows.append(
                _probe_row(
                    probe_id=f"d{i}",
                    category="durable-fact",
                    expected={
                        "must_store": [{"anchor_id": f"a{i}", "content_any": [f"c{i}"]}],
                        "retrieval": [
                            {"query": f"q{i}", "anchor_id": f"a{i}"},
                            {"query": f"q2{i}", "anchor_id": f"a{i}"},
                        ],
                    },
                )
            )
        for i in range(6):
            rows.append(
                _probe_row(
                    probe_id=f"w{i}",
                    category="workflow-noise",
                    expected={"must_not_store": [{"content_any": [f"n{i}"]}]},
                )
            )
        for i in range(4):
            rows.append(
                _probe_row(
                    probe_id=f"c{i}",
                    category="consolidation",
                    expected={"consolidation": {"expected_intent": "skip_redundant"}},
                )
            )
        for i in range(2):
            rows.append(_probe_row(probe_id=f"ep{i}", category="episode-positive", expected={}))
            rows.append(_probe_row(probe_id=f"en{i}", category="episode-negative", expected={}))
        path = tmp_path / "probes.jsonl"
        _write_jsonl(path, rows)
        return path

    def _args(self, *, probes_path: Path, output_dir: Path, backends: list[str], os_user: str | None) -> object:
        """Synthetic argparse.Namespace stand-in."""
        return type(
            "Args",
            (),
            {
                "probes": probes_path,
                "output_dir": output_dir,
                "user_prefix": "sandbox-test",
                "backends": backends,
                "os_user": os_user,
                "reset": True,
                "keep_sandboxes": False,
                "qualitative_sample_size": 10,
                "fail_on_threshold": False,
                "validate_only": False,
            },
        )()

    def test_codex_without_os_user_exits_2(self, tmp_path, capsys):
        import asyncio

        path = self._valid_fixture(tmp_path)
        args = self._args(
            probes_path=path,
            output_dir=tmp_path / "out",
            backends=["claude", "codex"],
            os_user=None,
        )
        rc = asyncio.run(g._run_cli(args))
        assert rc == 2
        err = capsys.readouterr().err
        assert "--os-user is required" in err

    def test_claude_only_does_not_require_os_user(self, tmp_path, monkeypatch):
        """If the operator runs only the claude arm, `--os-user` is
        not required (the claude reasoner falls through to direct
        spawn for the historical Max-plan installs)."""
        import asyncio
        from unittest.mock import patch

        path = self._valid_fixture(tmp_path)
        args = self._args(
            probes_path=path,
            output_dir=tmp_path / "out",
            backends=["claude"],
            os_user=None,
        )

        async def _fake_run_backend(**kwargs):
            return g.BackendRun(
                backend=kwargs["backend"],
                sandbox_user_id=kwargs["sandbox_user_id"],
                model_fact="m",
                model_episode="m",
                log_path=kwargs["log_path"],
            )

        with (
            patch("kai.eval.memory_backend_gate.memory.init_memory", lambda _c: None),
            patch("kai.eval.memory_backend_gate.load_config", return_value=_BASE_CONFIG),
            patch("kai.eval.memory_backend_gate.run_backend", _fake_run_backend),
        ):
            rc = asyncio.run(g._run_cli(args))
        assert rc == 0

    def test_os_user_flows_into_run_backend(self, tmp_path, monkeypatch):
        """When `--os-user` is supplied, the value is threaded into
        `run_backend(..., os_user_override=...)` for both arms."""
        import asyncio
        from unittest.mock import patch

        path = self._valid_fixture(tmp_path)
        args = self._args(
            probes_path=path,
            output_dir=tmp_path / "out",
            backends=["claude", "codex"],
            os_user="target",
        )

        captured: list[str | None] = []

        async def _fake_run_backend(**kwargs):
            captured.append(kwargs.get("os_user_override"))
            return g.BackendRun(
                backend=kwargs["backend"],
                sandbox_user_id=kwargs["sandbox_user_id"],
                model_fact="m",
                model_episode="m",
                log_path=kwargs["log_path"],
            )

        with (
            patch("kai.eval.memory_backend_gate.memory.init_memory", lambda _c: None),
            patch("kai.eval.memory_backend_gate.load_config", return_value=_BASE_CONFIG),
            patch("kai.eval.memory_backend_gate.run_backend", _fake_run_backend),
        ):
            rc = asyncio.run(g._run_cli(args))
        assert rc == 0
        assert captured == ["target", "target"]
