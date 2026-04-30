"""Integration tests for `src/kai/eval/extraction.py`.

These tests do NOT call the live extractor (cost + non-determinism);
the harness's full end-to-end behavior runs on demand via the CLI.
What we pin here:

- The pinned v5 prompt has not drifted from the v5 hash captured at
  the time #426 landed.
- The example probe fixture parses cleanly and matches the
  documented schema.
- `_classify_outcome` returns the right label for each
  (category, v5_facts, v6_facts) triple.
- The harness aggregate arithmetic produces the documented rates.
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import pytest

from kai import memory_extraction
from kai.config import Config
from kai.eval import extraction

# Minimal Config for the harness tests. Mirrors _BASE_CONFIG in
# test_memory_extraction.py. The harness stubs `_run_extractor` so
# the Config fields are never actually exercised; this exists to
# satisfy the typed `config: Config` parameter on
# `_run_one_probe` without needing real credentials.
_TEST_CONFIG = Config(
    telegram_bot_token="test-token",
    allowed_user_ids={12345},
    webhook_secret="test-secret",
)

# Hash of `_PROMPT_V5_PINNED` captured at #426 landing. If the
# pinned constant drifts (intentional or not), this test fails and
# either the constant must be reverted or this hash must be updated
# to a new captured baseline.
#
# Stored as the raw hexdigest; the JSON report emitted by the
# harness CLI prefixes the same value with "sha256:" via the
# `_hash()` helper. An operator cross-checking a report's
# `v5_prompt_hash` against this constant should strip the
# prefix from the report value (or compare report value to
# `f"sha256:{_V5_PROMPT_HASH}"`).
_V5_PROMPT_HASH = "764f249d2556a6e00489ac7ba5eac265f4a4d09f27d21dc76f612b84f0874c13"


def test_v5_pinned_drift():
    """The pinned baseline must remain byte-identical to the v5
    prompt as captured at #426 landing. A future intentional update
    to a new baseline (capturing v6, etc.) requires updating both
    the pinned constant AND `_V5_PROMPT_HASH` above; an accidental
    edit fails this test."""
    h = hashlib.sha256(extraction._PROMPT_V5_PINNED.encode()).hexdigest()
    assert h == _V5_PROMPT_HASH, (
        "Pinned v5 prompt drifted. If this is intentional (capturing "
        "a new baseline), update _V5_PROMPT_HASH above to match."
    )


def test_example_probe_fixture_loads():
    """The tracked example fixture parses end-to-end and contains
    the documented mix (2 workflow-noise + 2 durable-content)."""
    path = Path(__file__).parent.parent / "home" / "evals" / "extraction-probes.example.jsonl"
    probes = extraction.load_probes(path)
    assert len(probes) == 4
    cats = [p.category for p in probes]
    assert cats.count("workflow-noise") == 2
    assert cats.count("durable-content") == 2
    # Schema sanity: every probe has window.current.{user,assistant}
    # and an `expected` block.
    for p in probes:
        assert "current" in p.window
        assert "user" in p.window["current"]
        assert "assistant" in p.window["current"]
        assert isinstance(p.expected, dict)


def test_load_probes_malformed_json_raises_with_path_and_line(tmp_path: Path):
    """A typo in a fixture line surfaces the file path and 1-based
    line number, not a bare JSONDecodeError. Operators curate this
    fixture by hand; the diagnostic context shortens the
    typo-find-fix loop."""
    f = tmp_path / "probes.jsonl"
    f.write_text(
        '{"probe_id":"p1","category":"workflow-noise",'
        '"window":{"prior":[],"current":{"user":"u","assistant":"a"}},"expected":{}}\n'
        "this is not json\n"
    )
    with pytest.raises(ValueError) as exc_info:
        extraction.load_probes(f)
    msg = str(exc_info.value)
    # Path is in the message.
    assert str(f) in msg
    # Line number is 1-based; the bad line is the second one.
    assert ":2:" in msg
    # The underlying JSONDecodeError detail is preserved via the
    # f-string so the operator sees what was wrong, not just where.
    assert "invalid JSON" in msg


def test_load_probes_missing_window_current_raises_with_path_and_line(tmp_path: Path):
    """A fixture line that is valid JSON and has the top-level
    Probe fields but is missing `window.current` should fail at
    load time, not silently route through `_run_one_probe`'s
    error-bucket path. The load-time check is the operator's
    diagnostic anchor; runtime tolerance for an empty current is
    a separate concern handled by `_window_to_extractor_args`."""
    f = tmp_path / "probes.jsonl"
    f.write_text('{"probe_id":"p1","category":"workflow-noise","window":{"prior":[]},"expected":{}}\n')
    with pytest.raises(ValueError) as exc_info:
        extraction.load_probes(f)
    msg = str(exc_info.value)
    assert str(f) in msg
    assert ":1:" in msg
    assert "window.current is required" in msg


def test_load_probes_unknown_category_raises_with_path_and_line(tmp_path: Path):
    """A category typo (e.g. underscore instead of hyphen) would
    otherwise silently route every probe in the file to
    `ambiguous` in `_classify_outcome`, yielding `None` rates
    with no indication of the cause. The load-time check raises
    a ValueError naming the offending category and the allowed
    set so the operator can find and fix the typo."""
    f = tmp_path / "probes.jsonl"
    f.write_text(
        '{"probe_id":"p1","category":"workflow_noise",'
        '"window":{"prior":[],"current":{"user":"u","assistant":"a"}},'
        '"expected":{}}\n'
    )
    with pytest.raises(ValueError) as exc_info:
        extraction.load_probes(f)
    msg = str(exc_info.value)
    assert str(f) in msg
    assert ":1:" in msg
    assert "unknown category" in msg
    # The error names the typo'd value AND the allowed set, so the
    # operator does not have to grep the source for the right names.
    assert "workflow_noise" in msg
    assert "workflow-noise" in msg
    assert "durable-content" in msg


def test_load_probes_typo_in_window_current_user_raises(tmp_path: Path):
    """A fixture line that misspells `user` (e.g., `typo_user`)
    inside `window.current` passes the outer `current` existence
    check but produces empty extractor inputs at run time. The
    deeper validation catches the typo class at load time so the
    operator does not waste subprocess cost on a probe that will
    silently route to `error`."""
    f = tmp_path / "probes.jsonl"
    f.write_text(
        '{"probe_id":"p1","category":"workflow-noise",'
        '"window":{"prior":[],"current":{"typo_user":"u","assistant":"a"}},'
        '"expected":{}}\n'
    )
    with pytest.raises(ValueError) as exc_info:
        extraction.load_probes(f)
    msg = str(exc_info.value)
    assert str(f) in msg
    assert ":1:" in msg
    assert "non-empty user and assistant" in msg


def test_load_probes_empty_assistant_raises(tmp_path: Path):
    """Empty-string assistant is rejected too. An empty turn would
    short-circuit the extractor's confirmation rules and silently
    produce a meaningless run."""
    f = tmp_path / "probes.jsonl"
    f.write_text(
        '{"probe_id":"p1","category":"workflow-noise",'
        '"window":{"prior":[],"current":{"user":"u","assistant":""}},'
        '"expected":{}}\n'
    )
    with pytest.raises(ValueError) as exc_info:
        extraction.load_probes(f)
    assert "non-empty user and assistant" in str(exc_info.value)


def test_load_probes_missing_required_field_raises_with_path_and_line(tmp_path: Path):
    """Same diagnostic context applies when a line is valid JSON
    but lacks one of the required Probe fields. The KeyError raised
    by the dict access is converted into the same ValueError shape
    so the operator does not need to distinguish failure modes."""
    f = tmp_path / "probes.jsonl"
    f.write_text('{"probe_id":"p1","category":"workflow-noise"}\n')
    with pytest.raises(ValueError) as exc_info:
        extraction.load_probes(f)
    msg = str(exc_info.value)
    assert str(f) in msg
    assert ":1:" in msg
    assert "missing required field" in msg


def test_load_probes_skips_comments_and_blanks(tmp_path: Path):
    """`#`-prefixed lines and blank lines must be ignored so the
    fixture format can carry inline documentation."""
    f = tmp_path / "probes.jsonl"
    f.write_text(
        "# header comment\n"
        "\n"
        '{"probe_id":"p1","category":"workflow-noise",'
        '"window":{"prior":[],"current":{"user":"u","assistant":"a"}},'
        '"expected":{}}\n'
        "# trailing comment\n"
        "\n"
    )
    probes = extraction.load_probes(f)
    assert len(probes) == 1
    assert probes[0].probe_id == "p1"


@pytest.fixture
def workflow_probe() -> extraction.Probe:
    return extraction.Probe(
        probe_id="wf-1",
        category="workflow-noise",
        window={"prior": [], "current": {"user": "u", "assistant": "a"}},
        expected={"should_extract_any": False, "must_not_contain": ["filed", "approved"]},
    )


@pytest.fixture
def durable_probe() -> extraction.Probe:
    return extraction.Probe(
        probe_id="dur-1",
        category="durable-content",
        window={"prior": [], "current": {"user": "u", "assistant": "a"}},
        expected={"should_extract_any": True, "must_contain": ["earl grey"]},
    )


class TestClassifyOutcome:
    def test_workflow_noise_v5_extracts_v6_drops(self, workflow_probe):
        """The win condition for a workflow-noise probe: v5 produced
        facts, v6 produced none. Labeled `workflow_dropped`."""
        outcome = extraction._classify_outcome(
            workflow_probe,
            v5_facts=["User decided to file an issue."],
            v6_facts=[],
        )
        assert outcome == "workflow_dropped"

    def test_workflow_noise_v5_empty_is_ambiguous(self, workflow_probe):
        """If v5 was already empty, v6 cannot have "dropped"
        anything. Treat as ambiguous so the rate denominator
        excludes it."""
        outcome = extraction._classify_outcome(
            workflow_probe,
            v5_facts=[],
            v6_facts=[],
        )
        assert outcome == "ambiguous"

    def test_workflow_noise_v6_violates_must_not(self, workflow_probe):
        """v6 still emits a fact carrying a forbidden substring
        => regression."""
        outcome = extraction._classify_outcome(
            workflow_probe,
            v5_facts=["User decided to file an issue."],
            v6_facts=["User filed an issue about Z."],
        )
        assert outcome == "regression"

    def test_workflow_noise_partial_drop_is_regression_when_should_extract_any_false(self, workflow_probe):
        """Pin the strict classification: when the probe declares
        `should_extract_any: false`, v6 must produce zero facts to
        score `workflow_dropped`. A partial drop where v6 emits
        non-forbidden content is a regression because the probe
        asked for nothing and v6 did not deliver nothing."""
        outcome = extraction._classify_outcome(
            workflow_probe,
            v5_facts=["User decided to file an issue.", "Spec X v3 was approved."],
            v6_facts=["User mentioned a project name."],
        )
        assert outcome == "regression"

    def test_durable_v6_preserves_with_must_contain(self, durable_probe):
        """v6 produced a fact carrying the required substring =>
        durable_preserved."""
        outcome = extraction._classify_outcome(
            durable_probe,
            v5_facts=["User prefers Earl Grey over English Breakfast."],
            v6_facts=["User prefers Earl Grey."],
        )
        assert outcome == "durable_preserved"

    def test_durable_v6_drops_required_substring(self, durable_probe):
        """v6 produced a fact but it lacks the required substring =>
        regression."""
        outcome = extraction._classify_outcome(
            durable_probe,
            v5_facts=["User prefers Earl Grey."],
            v6_facts=["User prefers tea."],
        )
        assert outcome == "regression"

    def test_durable_v5_extracts_v6_empty_is_regression(self, durable_probe):
        """Durable category cares about preservation; when v5
        produced a fact but v6 did not, that is a regression
        because the durable fact must come through under v6."""
        outcome = extraction._classify_outcome(
            durable_probe,
            v5_facts=["User prefers Earl Grey."],
            v6_facts=[],
        )
        assert outcome == "regression"

    def test_durable_both_empty_is_ambiguous(self, durable_probe):
        """Symmetric to workflow-noise: a durable-content probe
        whose window is too sparse to produce extractions on either
        arm is uninformative, not a v6 failure. Treating it as a
        regression silently inflates the regression count and
        deflates `durable_preservation_rate`."""
        outcome = extraction._classify_outcome(
            durable_probe,
            v5_facts=[],
            v6_facts=[],
        )
        assert outcome == "ambiguous"

    def test_durable_v5_empty_v6_extracts_is_ambiguous(self, durable_probe):
        """v5 produced nothing; v6 found a fact carrying the
        must_contain anchor. Nothing was "preserved" because v5
        had nothing to preserve; counting this as
        `durable_preserved` would inflate the rate by classifying
        a v6 strict-improvement as preservation. Mirrors the
        workflow-noise branch's symmetric `not v5_facts ->
        ambiguous` guard."""
        outcome = extraction._classify_outcome(
            durable_probe,
            v5_facts=[],
            v6_facts=["User prefers Earl Grey."],
        )
        assert outcome == "ambiguous"


class TestAggregate:
    def test_aggregate_rates_skip_ambiguous(self):
        """Ambiguous outcomes drop out of the denominator so a few
        uninformative probes do not drag the rate down."""
        outcomes = [
            # v5 fired Rule 6 four times across the workflow probes
            # (v5 produces workflow-event content); v6 fired it once
            # (defense-in-depth on a leak the prompt missed).
            extraction.ProbeOutcome(
                probe_id="wf-1",
                category="workflow-noise",
                v5_facts=["x"],
                v6_facts=[],
                outcome="workflow_dropped",
                v5_rule_6_rejections_delta=2,
                v6_rule_6_rejections_delta=0,
            ),
            extraction.ProbeOutcome(
                probe_id="wf-2",
                category="workflow-noise",
                v5_facts=["x"],
                v6_facts=["x"],
                outcome="regression",
                v5_rule_6_rejections_delta=1,
                v6_rule_6_rejections_delta=0,
            ),
            extraction.ProbeOutcome(
                probe_id="wf-3",
                category="workflow-noise",
                v5_facts=[],
                v6_facts=[],
                outcome="ambiguous",
                v5_rule_6_rejections_delta=1,
                v6_rule_6_rejections_delta=0,
            ),
            extraction.ProbeOutcome(
                probe_id="dur-1",
                category="durable-content",
                v5_facts=["x"],
                v6_facts=["x"],
                outcome="durable_preserved",
                v5_rule_6_rejections_delta=0,
                v6_rule_6_rejections_delta=0,
            ),
            extraction.ProbeOutcome(
                probe_id="dur-2",
                category="durable-content",
                v5_facts=["x"],
                v6_facts=[],
                outcome="regression",
                v5_rule_6_rejections_delta=0,
                v6_rule_6_rejections_delta=1,
            ),
        ]
        agg = extraction._aggregate(outcomes)
        # 1 workflow_dropped of 2 scorable workflow probes (the
        # ambiguous one is excluded).
        assert agg["workflow_drop_rate"] == 0.5
        assert agg["scorable_workflow_count"] == 2
        # 1 durable_preserved of 2 scorable durable probes.
        assert agg["durable_preservation_rate"] == 0.5
        assert agg["scorable_durable_count"] == 2
        # Per-arm counter sums attributed correctly: v5 fired four
        # times (the workflow-noise probes), v6 fired once.
        assert agg["v5_rule_6_rejections"] == 4
        assert agg["v6_rule_6_rejections"] == 1

    def test_aggregate_zero_scorable_returns_none(self):
        """All-ambiguous category => rate is None, not 0.0, so a
        downstream consumer can distinguish "no data" from "all
        regressions"."""
        outcomes = [
            extraction.ProbeOutcome(
                probe_id="wf-1",
                category="workflow-noise",
                v5_facts=[],
                v6_facts=[],
                outcome="ambiguous",
                v5_rule_6_rejections_delta=0,
                v6_rule_6_rejections_delta=0,
            ),
        ]
        agg = extraction._aggregate(outcomes)
        assert agg["workflow_drop_rate"] is None
        assert agg["durable_preservation_rate"] is None

    def test_aggregate_excludes_error_outcomes_from_denominators(self):
        """`error` outcomes are recorded in the report (operator
        sees the failure) but skipped in the rate denominators
        (parallel to `ambiguous`). Pins the per-probe-exception
        behavior added so a single mid-run subprocess crash does
        not poison the workflow_drop_rate or
        durable_preservation_rate metrics."""
        outcomes = [
            extraction.ProbeOutcome(
                probe_id="wf-1",
                category="workflow-noise",
                v5_facts=["x"],
                v6_facts=[],
                outcome="workflow_dropped",
                v5_rule_6_rejections_delta=0,
                v6_rule_6_rejections_delta=0,
            ),
            extraction.ProbeOutcome(
                probe_id="wf-2",
                category="workflow-noise",
                v5_facts=[],
                v6_facts=[],
                outcome="error",
                v5_rule_6_rejections_delta=0,
                v6_rule_6_rejections_delta=0,
            ),
            extraction.ProbeOutcome(
                probe_id="dur-1",
                category="durable-content",
                v5_facts=[],
                v6_facts=[],
                outcome="error",
                v5_rule_6_rejections_delta=0,
                v6_rule_6_rejections_delta=0,
            ),
        ]
        agg = extraction._aggregate(outcomes)
        # 1 of 1 scorable workflow probe; the error probe is excluded.
        assert agg["workflow_drop_rate"] == 1.0
        assert agg["scorable_workflow_count"] == 1
        # Both durable probes are errors; the rate is None (not 0.0).
        assert agg["durable_preservation_rate"] is None
        assert agg["scorable_durable_count"] == 0


class TestRunOneProbeErrorPath:
    """`_run_one_probe` never raises. On exception it returns a
    ProbeOutcome with outcome=`error` and whatever per-arm Rule 6
    deltas it captured before the failure. Pinned because the
    error-path delta accuracy is operator-facing: a v5 arm that
    completed should report its real rejection delta even when v6
    crashes mid-run, otherwise `_aggregate`'s v5_rule_6_rejections
    total is silently under-counted."""

    def test_v6_exception_preserves_v5_delta(self, monkeypatch):
        """Drive v5 to completion (returning a fact that fires
        Rule 6) and v6 to raise. The returned ProbeOutcome must
        carry outcome=`error`, the v5 delta from the real arm, and
        v6 delta = 0 (v6 never produced facts)."""
        # Reset the counter's contents in-place so the test's delta
        # arithmetic is independent of preceding tests' state.
        # `_Counter._reset` is the test-only entry point; reassigning
        # `_RULE_6_REJECTIONS` would leave other test modules
        # (which import the counter by name) holding a stale
        # reference to the old object.
        memory_extraction._RULE_6_REJECTIONS._reset()

        call_count = {"n": 0}

        async def fake_run_extractor(payload, config, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                # v5 arm: emit a workflow-event fact so Rule 6
                # fires and the counter increments.
                memory_extraction._RULE_6_REJECTIONS.increment(user_id="probe-x")
                return memory_extraction.ExtractionResult(
                    facts=[{"content": "User prefers tea.", "tags": ["preference"]}],
                    has_episode=False,
                )
            # v6 arm: simulate a subprocess crash.
            raise RuntimeError("v6 subprocess crashed")

        monkeypatch.setattr(memory_extraction, "_run_extractor", fake_run_extractor)

        probe = extraction.Probe(
            probe_id="probe-x",
            category="workflow-noise",
            window={"prior": [], "current": {"user": "u", "assistant": "a"}},
            expected={"should_extract_any": False, "must_not_contain": []},
        )

        outcome = asyncio.run(extraction._run_one_probe(probe, config=_TEST_CONFIG, user_id="probe-x"))

        assert outcome.outcome == "error"
        # v5 arm completed: the increment we drove in fake_run_extractor
        # should be visible as a real per-arm delta.
        assert outcome.v5_rule_6_rejections_delta == 1
        # v6 arm raised before completing: delta stays at 0.
        assert outcome.v6_rule_6_rejections_delta == 0
        # v5_facts captured pre-exception; v6_facts empty.
        assert outcome.v5_facts == ["User prefers tea."]
        assert outcome.v6_facts == []


class TestWindowToExtractorArgs:
    def test_pairs_user_assistant_sequentially(self):
        """Two complete prior turns + a current exchange => two
        prior pairs and the current user/assistant strings."""
        window = {
            "prior": [
                {"role": "user", "text": "u1"},
                {"role": "assistant", "text": "a1"},
                {"role": "user", "text": "u2"},
                {"role": "assistant", "text": "a2"},
            ],
            "current": {"user": "uc", "assistant": "ac"},
        }
        user, assistant, prior_pairs = extraction._window_to_extractor_args(window)
        assert user == "uc"
        assert assistant == "ac"
        assert prior_pairs == [("u1", "a1"), ("u2", "a2")]

    def test_orphan_user_without_following_assistant_dropped(self):
        """A user turn with no matching assistant reply is omitted
        from the pair list. The schema allows asymmetric prior turns
        but the production payload expects pairs."""
        window = {
            "prior": [
                {"role": "user", "text": "u1"},
                {"role": "assistant", "text": "a1"},
                {"role": "user", "text": "u2-orphan"},
            ],
            "current": {"user": "uc", "assistant": "ac"},
        }
        _, _, prior_pairs = extraction._window_to_extractor_args(window)
        assert prior_pairs == [("u1", "a1")]

    def test_orphan_assistant_without_preceding_user_dropped(self):
        """An assistant turn that arrives before any user turn is
        also dropped. Pinned alongside the user-orphan case so the
        BOTH-DIRECTIONS contract documented in the function's
        docstring is enforced symmetrically."""
        window = {
            "prior": [
                {"role": "assistant", "text": "a-orphan"},
                {"role": "user", "text": "u1"},
                {"role": "assistant", "text": "a1"},
            ],
            "current": {"user": "uc", "assistant": "ac"},
        }
        _, _, prior_pairs = extraction._window_to_extractor_args(window)
        assert prior_pairs == [("u1", "a1")]
