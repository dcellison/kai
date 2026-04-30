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

import hashlib
from pathlib import Path

import pytest

from kai.eval import extraction

# Hash of `_PROMPT_V5_PINNED` captured at #426 landing. If the
# pinned constant drifts (intentional or not), this test fails and
# either the constant must be reverted or this hash must be updated
# to a new captured baseline.
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

    def test_durable_v6_empty_is_regression(self, durable_probe):
        """Durable category cares about preservation; an empty v6 is
        a regression even if v5 was also empty."""
        outcome = extraction._classify_outcome(
            durable_probe,
            v5_facts=["User prefers Earl Grey."],
            v6_facts=[],
        )
        assert outcome == "regression"


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
                expected_outcome="workflow_dropped",
                v5_rule_6_rejections_delta=2,
                v6_rule_6_rejections_delta=0,
            ),
            extraction.ProbeOutcome(
                probe_id="wf-2",
                category="workflow-noise",
                v5_facts=["x"],
                v6_facts=["x"],
                expected_outcome="regression",
                v5_rule_6_rejections_delta=1,
                v6_rule_6_rejections_delta=0,
            ),
            extraction.ProbeOutcome(
                probe_id="wf-3",
                category="workflow-noise",
                v5_facts=[],
                v6_facts=[],
                expected_outcome="ambiguous",
                v5_rule_6_rejections_delta=1,
                v6_rule_6_rejections_delta=0,
            ),
            extraction.ProbeOutcome(
                probe_id="dur-1",
                category="durable-content",
                v5_facts=["x"],
                v6_facts=["x"],
                expected_outcome="durable_preserved",
                v5_rule_6_rejections_delta=0,
                v6_rule_6_rejections_delta=0,
            ),
            extraction.ProbeOutcome(
                probe_id="dur-2",
                category="durable-content",
                v5_facts=["x"],
                v6_facts=[],
                expected_outcome="regression",
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
                expected_outcome="ambiguous",
                v5_rule_6_rejections_delta=0,
                v6_rule_6_rejections_delta=0,
            ),
        ]
        agg = extraction._aggregate(outcomes)
        assert agg["workflow_drop_rate"] is None
        assert agg["durable_preservation_rate"] is None


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
