"""Tests for the Layer 2 behavioral A/B eval harness (kai.eval.behavioral).

Ten tests covering the harness's load-bearing seams, organized by concern:

1. TestArmPromptIsolation - byte-level guarantee that memory-on and
   memory-off arms differ in EXACTLY one place (the memory block).
   This is the test that defends the entire hypothesis: if the two arms
   diverge in any other way (whitespace, ordering, system context),
   every metric the harness produces is invalidated.
2. TestJudgePromptRendering - snapshot the rendered judge user payload
   so a future template tweak that silently re-orders fields or drops
   a separator breaks loudly here rather than corrupting verdicts.
3. TestJudgeOutputParsingValid - all four `choice` strings parse and
   roll up correctly through the structured-output envelope.
4. TestJudgeOutputParsingMalformed - every malformed shape (bad JSON,
   wrong type, is_error, missing fields, invalid enum) returns None,
   which the caller buckets as judge_error.
5. TestAnonymizationDeterministic - same seed -> same letter sequence;
   different seeds -> divergent sequences. Both directions matter:
   determinism is for debuggability, divergence is for non-pathological
   coverage of A and B positions.
6. TestRollupOutcome - the eight-cell truth table mapping
   (judge_choice, memory_arm_letter) to memory_outcome.
7. TestSubprocessTimeout - end-to-end through _run_one_probe with the
   subprocess mocked to mimic the timeout return shape; verifies the
   probe lands in generation_error AND the judge call is skipped (cost
   discipline).
8. TestDriftExclusion - drift outcomes do not contribute to the
   win/loss/tie/both_wrong rate denominators; the divide-by-zero guard
   on a fully-error run produces 0% rates rather than crashing.
9. TestSubprocessExplicitCwd - locks the cwd= argument is passed
   explicitly to asyncio.create_subprocess_exec, preventing a future
   refactor from accidentally inheriting the harness's cwd (which would
   leak the per-user home_workspace/.claude/CLAUDE.md into both arms).
10. TestGeneratorEmptySystemPrompt - locks --system-prompt is followed
    by literally "" (not omitted), so the CLI's default system prompt
    cannot leak into the generator and confound the A/B measurement.

Every subprocess call is mocked. No real claude binary is invoked. The
arm-isolation test mocks format_context; the timeout test mocks the
shared _run_subprocess; the cwd test mocks asyncio.create_subprocess_exec
at the lowest level so we can introspect what kwargs were actually
forwarded.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kai.eval import behavioral
from kai.eval.behavioral import (
    BehavioralConfig,
    ProbeOutcome,
    _aggregate_outcomes,
    _arm_letter_for_memory,
    _build_gen_cmd,
    _build_gen_cmd_codex,
    _build_judge_cmd,
    _build_judge_cmd_codex,
    _capture_agent_cli_version,
    _compute_rates,
    _inject_synthetic_pollution,
    _load_user_history_messages,
    _make_drift_outcome,
    _non_negative_int,
    _parse_codex_gen_stdout,
    _parse_codex_judge_stdout,
    _parse_judge_stdout,
    _positive_int,
    _render_codex_stdin,
    _render_judge_user_payload,
    _render_summary,
    _resolve_seed,
    _rollup_outcome,
    _run_subprocess,
    _sample_pollution_for_probe,
    _validate_judge_envelope,
    build_arm_prompt,
    build_output_json,
)
from kai.eval.retrieval import Probe

# ── Shared helpers ──────────────────────────────────────────────────


def _make_config(**overrides) -> BehavioralConfig:
    """Construct a BehavioralConfig with stable defaults.

    Tests that only care about one field (e.g. gen_timeout_s) override
    that field and let the rest match production defaults. Keeps each
    test focused on the one knob it is exercising.
    """
    base: dict = {
        "judge_model": "claude-haiku-4-5-20251001",
        "judge_budget_usd": 0.05,
        "judge_timeout_s": 60,
        "gen_model": "sonnet",
        "gen_budget_usd": 0.10,
        "gen_timeout_s": 120,
        "seed": "0123456789abcdef",
        "max_concurrency": 4,
    }
    base.update(overrides)
    return BehavioralConfig(**base)


def _make_outcome(*, probe: Probe, memory_outcome: str) -> ProbeOutcome:
    """Construct a synthetic ProbeOutcome for aggregation tests.

    The aggregation paths only read `memory_outcome` and `tags`, but
    `judge_choice` is set consistently with `memory_outcome` so the
    fixture is trustworthy for any future test that reads both fields
    (e.g. _outcome_to_per_probe_dict serialization checks). Mapping:
    memory_arm_letter is fixed at "A", so memory_wins -> A_wins,
    memory_loses -> B_wins, and the passthrough buckets (tie,
    both_wrong, judge_error, generation_error) carry their own name
    as judge_choice, matching the production rollup.
    """
    judge_choice_by_outcome = {
        "memory_wins": "A_wins",
        "memory_loses": "B_wins",
        "tie": "tie",
        "both_wrong": "both_wrong",
        "judge_error": "judge_error",
        "generation_error": "generation_error",
    }
    return ProbeOutcome(
        probe=probe,
        tags=(),
        memory_arm_letter="A",
        responses={"A": "ra", "B": "rb"},
        judge_choice=judge_choice_by_outcome[memory_outcome],
        judge_reasoning="reason",
        memory_outcome=memory_outcome,
        latency_ms={"A": 1.0, "B": 1.0, "judge": 1.0},
        cost_usd={"A": None, "B": None, "judge": 0.001},
    )


# ── Test 1: A/B isolation guarantee ────────────────────────────────


class TestArmPromptIsolation:
    """Byte-level isolation between memory-on and memory-off arms.

    The hypothesis "memory helps" is only meaningful if the two arms
    differ in EXACTLY one place: presence vs absence of the memory
    block. This test class is the regression guard against any future
    refactor that adds a stray newline, system-prompt fragment, or
    history snippet to one arm but not the other.
    """

    def test_arm_prompts_differ_only_in_memory_block(self):
        # Mock format_context to return a known memory block. Patching
        # at the module level (kai.memory.format_context) works because
        # build_arm_prompt does a function-local `from kai.memory import
        # format_context` that re-resolves the attribute on each call.
        memory_block = "[Relevant memories]\n- the user's favorite color is blue"
        question = "What's my favorite color?"

        with patch("kai.memory.format_context", new=AsyncMock(return_value=memory_block)):
            arm_on = asyncio.run(build_arm_prompt(question, "user-1", memory_enabled=True))
            arm_off = asyncio.run(build_arm_prompt(question, "user-1", memory_enabled=False))

        # Memory-off arm is the bare question, no decoration.
        assert arm_off == question

        # Memory-on arm exactly equals the production join helper's
        # output. Comparing against the helper (rather than a hand-
        # rolled string) means a future change to the join separator
        # propagates into the test automatically — what we are locking
        # is "both arms go through the same join", NOT a specific
        # separator value.
        from kai.backend import prepend_to_prompt

        expected_on = prepend_to_prompt(question, memory_block)
        assert arm_on == expected_on

    def test_arm_on_with_empty_memory_equals_arm_off(self):
        # When format_context returns the empty string (no relevant
        # memories), the production path skips prepend_to_prompt and
        # both arms collapse to the bare question. This matches the
        # bot's runtime behavior and avoids a confound where the arms
        # would differ only by leading whitespace.
        question = "What's my favorite color?"
        with patch("kai.memory.format_context", new=AsyncMock(return_value="")):
            arm_on = asyncio.run(build_arm_prompt(question, "user-1", memory_enabled=True))
            arm_off = asyncio.run(build_arm_prompt(question, "user-1", memory_enabled=False))
        assert arm_on == arm_off == question


# ── Test 2: Judge prompt rendering ─────────────────────────────────


class TestJudgePromptRendering:
    """Snapshot the rendered judge user payload against template drift.

    The judge prompt's structure (field order, separators, header
    casing) is load-bearing: prompt-caching across calls only fires if
    the rubric portion is byte-identical, and a silent reordering would
    invalidate the bias mitigations baked into the rubric.
    """

    def test_judge_prompt_renders_correctly(self):
        rendered = _render_judge_user_payload(
            question="What's my favorite color?",
            ground_truth_text="The user's favorite color is blue.",
            response_a="Your favorite color is blue.",
            response_b="I don't have that information.",
        )
        # Snapshot the exact rendering. If a future PR changes a header
        # or the field order, this test breaks loudly so the change can
        # be evaluated against the rubric's bias-mitigation design.
        assert rendered == (
            "USER QUESTION:\n"
            "What's my favorite color?\n"
            "\n"
            "GROUND-TRUTH FACT (what the operator has previously told the assistant):\n"
            "The user's favorite color is blue.\n"
            "\n"
            "RESPONSE A:\n"
            "Your favorite color is blue.\n"
            "\n"
            "RESPONSE B:\n"
            "I don't have that information."
        )

    def test_curly_braces_in_response_are_passed_through_literally(self):
        # Regression guard: response_a and response_b are
        # MODEL-GENERATED text. An earlier implementation used
        # str.format() to interpolate them, which would either
        # silently substitute `{response_b}` from one arm into another
        # (corrupting the prompt) or raise KeyError on unknown braces
        # (crashing the probe with no diagnostic). The current
        # implementation uses concatenation; this test locks that
        # property by feeding literal `{...}` content through the
        # render and asserting it survives unmodified.
        rendered = _render_judge_user_payload(
            question="Does {anything_unknown} crash?",
            ground_truth_text="Ground {response_a} truth.",
            response_a="Response with {response_b} placeholder.",
            response_b="Other response with {missing_key} brace.",
        )
        # Each curly-braced fragment appears in the output verbatim.
        # If the renderer ever switches back to .format(), one of these
        # asserts breaks loudly (substitution) or the call raises before
        # the assert (KeyError on missing_key).
        assert "{anything_unknown}" in rendered
        assert "{response_a}" in rendered
        assert "{response_b}" in rendered
        assert "{missing_key}" in rendered


# ── Test 3: Judge output parsing (valid choices) ───────────────────


class TestJudgeOutputParsingValid:
    """All four valid `choice` strings parse via the structured envelope."""

    @pytest.mark.parametrize("choice", ["A_wins", "B_wins", "tie", "both_wrong"])
    def test_parse_valid_choices(self, choice: str):
        # Build the envelope shape that `claude --print --output-format
        # json --json-schema ...` produces: outer dict with structured
        # output nested, plus the cost field used by _extract_judge_cost.
        envelope = {
            "is_error": False,
            "structured_output": {"choice": choice, "reasoning": "test reasoning"},
            "total_cost_usd": 0.001,
        }
        parsed = _parse_judge_stdout(json.dumps(envelope).encode("utf-8"))
        assert parsed is not None
        assert parsed[0] == choice
        assert parsed[1] == "test reasoning"

    def test_envelope_without_is_error_field(self):
        # Older CLI versions omit `is_error` entirely on success.
        # Parser must treat missing-key as not-error, not as failure.
        envelope = {
            "structured_output": {"choice": "tie", "reasoning": "equiv"},
        }
        parsed = _parse_judge_stdout(json.dumps(envelope).encode("utf-8"))
        assert parsed == ("tie", "equiv")


# ── Test 4: Judge output parsing (malformed) ───────────────────────


class TestJudgeOutputParsingMalformed:
    """Every malformed shape returns None (caller buckets as judge_error).

    Defense-in-depth: the JSON schema passed to claude --json-schema
    should already enforce the choice enum at the CLI side, but a
    future schema-validator regression must not leak invalid choices
    into the rollup. Each test below exercises one failure mode the
    parser handles in its 5-step chain.
    """

    def test_invalid_json(self):
        # Garbage stdout (model went off the rails despite --json-schema).
        assert _parse_judge_stdout(b"not json at all") is None

    def test_empty_bytes(self):
        # Subprocess produced no output (broken pipe, killed early).
        assert _parse_judge_stdout(b"") is None

    def test_top_level_not_dict(self):
        # CLI envelope is always a dict; an array means the schema
        # enforcement skipped or the wrong stream was captured.
        assert _parse_judge_stdout(b'["a", "b"]') is None

    def test_is_error_true(self):
        # CLI exits 0 with is_error=true on budget-cap mid-retry.
        # Treat as failure rather than partial-success-with-noise.
        env = {
            "is_error": True,
            "structured_output": {"choice": "A_wins", "reasoning": "x"},
        }
        assert _parse_judge_stdout(json.dumps(env).encode("utf-8")) is None

    def test_missing_structured_output(self):
        # Top-level envelope present but the nested payload is missing.
        env = {"total_cost_usd": 0.001}
        assert _parse_judge_stdout(json.dumps(env).encode("utf-8")) is None

    def test_structured_output_not_dict(self):
        # structured_output is the key the CLI nests the validated
        # payload under. If it shows up as a string or list, the schema
        # was bypassed entirely — bucket as failure.
        env = {"structured_output": "A_wins"}
        assert _parse_judge_stdout(json.dumps(env).encode("utf-8")) is None

    def test_invalid_choice_value(self):
        # Choice outside the four-string enum.
        env = {"structured_output": {"choice": "MAYBE_A", "reasoning": "x"}}
        assert _parse_judge_stdout(json.dumps(env).encode("utf-8")) is None

    def test_missing_reasoning(self):
        # Schema requires both `choice` and `reasoning`; missing
        # reasoning is a schema violation we must not silently accept.
        env = {"structured_output": {"choice": "A_wins"}}
        assert _parse_judge_stdout(json.dumps(env).encode("utf-8")) is None


# ── Test 5: Anonymization is deterministic with seed ───────────────


class TestAnonymizationDeterministic:
    """Same seed produces the same arm-letter sequence; different seeds diverge.

    Determinism matters for debugging ("why did probe 17 swing from
    win to loss between two runs?"); divergence across seeds matters
    so the harness does not develop a systematic preference for arm A
    at any position across all of an operator's runs.
    """

    def test_same_seed_produces_same_sequence(self):
        # Two RNGs seeded identically must emit the same letter
        # sequence over a long enough draw to catch a per-call drift.
        rng1 = random.Random(0xDEADBEEF)
        rng2 = random.Random(0xDEADBEEF)
        letters1 = [_arm_letter_for_memory(rng1) for _ in range(50)]
        letters2 = [_arm_letter_for_memory(rng2) for _ in range(50)]
        assert letters1 == letters2

    def test_seed_produces_both_letters(self):
        # Sanity: the coin is not stuck on one side. A fixed seed
        # might happen to draw all-A or all-B over a small N, so we
        # use 50 draws — vanishingly small probability of all-same
        # for a fair coin.
        rng = random.Random(0xDEADBEEF)
        letters = [_arm_letter_for_memory(rng) for _ in range(50)]
        assert "A" in letters
        assert "B" in letters

    def test_different_seeds_diverge(self):
        # Two distinct seeds should not produce identical sequences;
        # if they do, something is wrong with the RNG plumbing (e.g.
        # silently shared global state).
        rng1 = random.Random(1)
        rng2 = random.Random(2)
        letters1 = [_arm_letter_for_memory(rng1) for _ in range(50)]
        letters2 = [_arm_letter_for_memory(rng2) for _ in range(50)]
        assert letters1 != letters2

    def test_resolve_seed_deterministic_across_calls(self):
        # _resolve_seed itself must be a pure function of probes +
        # user_id; calling it twice returns the same hex.
        probes = [
            Probe(question="q1", expected_fact_id="f1"),
            Probe(question="q2", expected_fact_id="f2"),
        ]
        s1 = _resolve_seed(cli_seed=None, probes=probes, user_id="user-1")
        s2 = _resolve_seed(cli_seed=None, probes=probes, user_id="user-1")
        assert s1 == s2

    def test_resolve_seed_cli_override_wins(self):
        # When the operator passes --seed, that value is used
        # verbatim and the hash-of-(probes,user_id) is ignored.
        probes = [Probe(question="q1", expected_fact_id="f1")]
        s = _resolve_seed(cli_seed="abc123", probes=probes, user_id="user-1")
        assert s == "abc123"

    def test_resolve_seed_empty_string_is_explicit_override(self):
        # `--seed ""` is an unusual but unambiguous explicit override:
        # argparse only yields None when the flag is absent, so any
        # string the operator passes (including "") must be honored
        # verbatim rather than silently falling through to the hash
        # default. Locks the round-9 contract change from a truthy
        # guard (`if cli_seed:`) to an explicit None-check.
        probes = [Probe(question="q1", expected_fact_id="f1")]
        s = _resolve_seed(cli_seed="", probes=probes, user_id="user-1")
        assert s == ""

    def test_resolve_seed_changes_with_user(self):
        # Different operator -> different shuffle, even on the same
        # probe set. Ensures the harness does not bake in a single
        # preference across all users.
        probes = [Probe(question="q1", expected_fact_id="f1")]
        s_a = _resolve_seed(cli_seed=None, probes=probes, user_id="user-a")
        s_b = _resolve_seed(cli_seed=None, probes=probes, user_id="user-b")
        assert s_a != s_b


# ── Test 6: Per-probe rollup truth table ───────────────────────────


class TestRollupOutcome:
    """Truth table: judge_choice + memory_arm_letter -> memory_outcome.

    All eight combinations of the four scoring choices crossed with the
    two arm letters. The two error states are tested separately because
    they bypass the A/B mapping (no information about memory wins or
    loses when one arm could not produce a clean response).
    """

    @pytest.mark.parametrize(
        "choice,letter,expected",
        [
            ("A_wins", "A", "memory_wins"),
            ("A_wins", "B", "memory_loses"),
            ("B_wins", "A", "memory_loses"),
            ("B_wins", "B", "memory_wins"),
            ("tie", "A", "tie"),
            ("tie", "B", "tie"),
            ("both_wrong", "A", "both_wrong"),
            ("both_wrong", "B", "both_wrong"),
        ],
    )
    def test_rollup_truth_table(self, choice: str, letter: str, expected: str):
        assert _rollup_outcome(judge_choice=choice, memory_arm_letter=letter) == expected

    @pytest.mark.parametrize("err", ["judge_error", "generation_error"])
    @pytest.mark.parametrize("letter", ["A", "B"])
    def test_error_states_pass_through(self, err: str, letter: str):
        # Error states must pass through unchanged regardless of which
        # arm carried memory — the arm letter has no scoring
        # significance when an arm is broken.
        assert _rollup_outcome(judge_choice=err, memory_arm_letter=letter) == err


# ── Test 7: Subprocess timeout buckets as generation_error ─────────


class TestSubprocessTimeout:
    """Generator subprocess timeout -> probe lands in generation_error.

    Verifies the cost-saving short-circuit also fires: when both arms
    fail, the judge call is skipped (no point spending money to score
    nothing against nothing).
    """

    def test_subprocess_timeout_buckets_as_generation_error(self):
        # Mock _run_subprocess to mimic the timeout return shape:
        # rc=-1, empty stdout/stderr, recorded latency. Mocking at this
        # level avoids depending on a real `sleep` binary or a real
        # asyncio timeout — the behavior under test is the bucketing
        # logic, not the timeout mechanics themselves (those are
        # exercised by the cwd test below which mocks one level deeper).
        async def fake_run_subprocess(*, cmd, stdin_payload, timeout_s, cwd, env):
            # Returns the exact tuple shape _run_subprocess returns on
            # asyncio.wait_for raising TimeoutError. The latency_ms
            # value is arbitrary; we just check it propagates.
            return -1, b"", b"", 50.0

        cfg = _make_config(gen_timeout_s=1)

        with (
            patch.object(behavioral, "_run_subprocess", new=fake_run_subprocess),
            patch("kai.memory.format_context", new=AsyncMock(return_value="mem block")),
        ):
            outcome = asyncio.run(
                behavioral._run_one_probe(
                    probe=Probe(question="q", expected_fact_id="f1"),
                    tags=("tag1",),
                    user_id="u1",
                    config=cfg,
                    rng=random.Random(1),
                    ground_truth_text="gold",
                    cwd=Path("/tmp"),
                    env={},
                )
            )

        # Both arms broken -> generation_error rollup, raw judge_choice
        # also generation_error (judge call was skipped).
        assert outcome.memory_outcome == "generation_error"
        assert outcome.judge_choice == "generation_error"

        # Per-arm latencies are still recorded so the operator can see
        # which side timed out. Judge latency is zero because the call
        # was short-circuited (cost discipline).
        assert outcome.latency_ms["A"] == 50.0
        assert outcome.latency_ms["B"] == 50.0
        assert outcome.latency_ms["judge"] == 0.0

    def test_one_arm_failure_also_buckets_as_generation_error(self):
        # If only one arm fails (e.g. transient rate-limit on B), the
        # probe is still unscorable — the judge needs both responses.
        # Toggle return between None and a real response by tracking
        # call count.
        call_count = {"n": 0}

        async def fake_run_subprocess(*, cmd, stdin_payload, timeout_s, cwd, env):
            call_count["n"] += 1
            # First call (arm A): clean text response.
            # Second call (arm B): timeout shape.
            # No third call expected — judge must be skipped.
            if call_count["n"] == 1:
                return 0, b"good response from arm A", b"", 100.0
            return -1, b"", b"", 50.0

        cfg = _make_config()

        with (
            patch.object(behavioral, "_run_subprocess", new=fake_run_subprocess),
            patch("kai.memory.format_context", new=AsyncMock(return_value="mem")),
        ):
            outcome = asyncio.run(
                behavioral._run_one_probe(
                    probe=Probe(question="q", expected_fact_id="f1"),
                    tags=(),
                    user_id="u1",
                    config=cfg,
                    rng=random.Random(1),
                    ground_truth_text="gold",
                    cwd=Path("/tmp"),
                    env={},
                )
            )
        assert outcome.memory_outcome == "generation_error"
        # Exactly two subprocess calls: gen-A and gen-B. No judge call.
        assert call_count["n"] == 2


# ── Test 8: Drift bucket excluded from rates ───────────────────────


class TestDriftExclusion:
    """Drift outcomes do not contribute to win/loss/tie/both_wrong rates."""

    def test_drift_excluded_from_aggregation(self):
        probe = Probe(question="q", expected_fact_id="f1")
        outcomes = [
            _make_outcome(probe=probe, memory_outcome="memory_wins"),
            _make_outcome(probe=probe, memory_outcome="memory_wins"),
            _make_outcome(probe=probe, memory_outcome="memory_loses"),
            _make_drift_outcome(probe, ()),
            _make_drift_outcome(probe, ()),
        ]
        counts = _aggregate_outcomes(outcomes)

        # Drift not counted into any bucket; the wins/loses tallies
        # reflect only the scorable outcomes.
        assert counts["memory_wins"] == 2
        assert counts["memory_loses"] == 1
        assert "drift" not in counts

        # Denominator is scorable = 2+1+0+0 = 3, NOT 5. If drift were
        # included, win_rate would be 40% instead of the correct 66.67%.
        rates = _compute_rates(counts)
        assert rates["win_rate_pct"] == round(100.0 * 2 / 3, 2)
        assert rates["loss_rate_pct"] == round(100.0 * 1 / 3, 2)

    def test_all_errors_no_div_by_zero(self):
        # Fully-error run: the max(scorable, 1) guard kicks in. Rates
        # emit as 0.0 so downstream parsers do not crash; the failure
        # mode is surfaced through the visible error counts in the
        # bucket dict.
        counts = {b: 0 for b in behavioral._OUTCOME_BUCKETS}
        counts["judge_error"] = 5
        counts["generation_error"] = 3
        rates = _compute_rates(counts)
        assert rates["win_rate_pct"] == 0.0
        assert rates["loss_rate_pct"] == 0.0
        assert rates["tie_rate_pct"] == 0.0
        assert rates["both_wrong_rate_pct"] == 0.0

    def test_tie_and_both_wrong_reported_separately(self):
        # Combining tie and both_wrong would hide the distinction
        # between "memory had no effect" (tie: both responses used the
        # fact) and "memory failure mode" (both_wrong: NEITHER used it).
        probe = Probe(question="q", expected_fact_id="f1")
        outcomes = [
            _make_outcome(probe=probe, memory_outcome="tie"),
            _make_outcome(probe=probe, memory_outcome="tie"),
            _make_outcome(probe=probe, memory_outcome="both_wrong"),
        ]
        counts = _aggregate_outcomes(outcomes)
        assert counts["tie"] == 2
        assert counts["both_wrong"] == 1
        rates = _compute_rates(counts)
        assert rates["tie_rate_pct"] == round(100.0 * 2 / 3, 2)
        assert rates["both_wrong_rate_pct"] == round(100.0 * 1 / 3, 2)


# ── Test 9: Subprocess uses explicit cwd ───────────────────────────


class TestSubprocessExplicitCwd:
    """The cwd= kwarg is passed explicitly to create_subprocess_exec.

    Without an explicit cwd, claude --print walks up from the harness's
    cwd and picks up the per-user home_workspace/.claude/CLAUDE.md
    (Kai's bot identity: voice rules, persona, scheduling API docs).
    The judge would then filter every verdict through Kai's persona,
    and the generator would receive bot-voice priming. Both confound
    the spec's minimal-prompt measurement. This test locks the parameter
    plumbing so a future refactor cannot quietly drop the cwd= argument.
    """

    def test_subprocess_passes_explicit_cwd_string(self):
        # Capture the kwargs forwarded to create_subprocess_exec.
        captured: dict = {}

        async def fake_communicate(input):
            # Return a valid envelope so _run_subprocess does not
            # bucket as failure on this path; the assertion is on the
            # call's cwd argument, not the parsed result.
            return (
                b'{"is_error": false, "structured_output": {"choice": "tie", "reasoning": "x"}}',
                b"",
            )

        async def fake_exec(*cmd, **kwargs):
            captured.update(kwargs)
            proc = MagicMock()
            proc.communicate = fake_communicate
            proc.returncode = 0
            return proc

        cwd = Path("/some/explicit/path")
        with patch("asyncio.create_subprocess_exec", new=fake_exec):
            asyncio.run(
                _run_subprocess(
                    cmd=["claude", "--print"],
                    stdin_payload="x",
                    timeout_s=10,
                    cwd=cwd,
                    env={},
                )
            )

        # cwd must be passed as str(Path), not Path itself, matching
        # the extractor's pattern. If this assertion ever fails, the
        # subprocess call is inheriting the harness's cwd implicitly,
        # which means the per-user home_workspace/.claude/CLAUDE.md is
        # leaking into both arms.
        assert "cwd" in captured
        assert captured["cwd"] == str(cwd)


# ── Test 10: Generator passes empty system prompt ──────────────────


class TestGeneratorEmptySystemPrompt:
    """The generator passes --system-prompt with a literal empty string.

    Omitting the flag would let the CLI's default system prompt confound
    the A/B measurement (a different default between two CLI versions
    would silently change generator behavior). Passing "" replaces the
    default with literally nothing. Some CLI default context still flows
    (~340 tokens at CLI 2.1.118), but it is identical between arms in
    one run; cross-run drift in that residual is detectable via the
    claude_cli_version field captured in the output JSON.
    """

    def test_generator_passes_empty_system_prompt(self):
        cfg = _make_config()
        cmd = _build_gen_cmd(cfg)
        # Locate the --system-prompt flag and verify its argument is "".
        assert "--system-prompt" in cmd, "Generator must pass --system-prompt explicitly"
        idx = cmd.index("--system-prompt")
        assert cmd[idx + 1] == "", (
            "Generator must pass --system-prompt with a literal empty "
            "string. Omitting the flag would let the CLI's default "
            "system prompt confound the A/B measurement."
        )

    def test_generator_uses_text_output_format(self):
        # Generator emits free-form text for the judge to read; locks
        # the design choice that --output-format=text (no JSON envelope,
        # which is why generator cost is None in the output JSON).
        cmd = _build_gen_cmd(_make_config())
        assert "--output-format" in cmd
        assert cmd[cmd.index("--output-format") + 1] == "text"
        # Generator does NOT pass --json-schema (it would be ignored
        # with text format anyway, but its presence would suggest a
        # design confusion).
        assert "--json-schema" not in cmd

    def test_judge_uses_json_output_format(self):
        # Symmetric assertion on the judge side: locks the parsing
        # chain's input shape. _parse_judge_stdout depends on the
        # structured_output envelope, which only exists with
        # --output-format=json + --json-schema.
        cmd = _build_judge_cmd(_make_config())
        assert "--output-format" in cmd
        assert cmd[cmd.index("--output-format") + 1] == "json"
        assert "--json-schema" in cmd

    def test_both_subprocesses_disable_tools_and_persistence(self):
        # Defense in depth: both subprocesses must run with tools
        # disabled and session persistence off. A regression in either
        # would hand the model the parent's full env (via tool calls)
        # or leak state across runs.
        for cmd in [_build_gen_cmd(_make_config()), _build_judge_cmd(_make_config())]:
            assert "--tools" in cmd
            assert cmd[cmd.index("--tools") + 1] == ""
            assert "--no-session-persistence" in cmd
            assert "--permission-mode" in cmd
            assert cmd[cmd.index("--permission-mode") + 1] == "bypassPermissions"


# ── Test 11: RNG stable across drift-set changes (W2 fix) ──────────


class TestRngStableUnderDrift:
    """Per-probe arm assignment must survive changes in the drift set.

    The seed for each probe's RNG hashes (config.seed, expected_fact_id)
    rather than (config.seed, positional_index). This means that when an
    earlier probe in the file drifts between two runs, the surviving
    probes still get the same arm letter — which is what makes the
    "same probe set + user pair -> same arm assignments" guarantee in
    _resolve_seed actually hold.
    """

    def test_arm_assignment_invariant_under_drift_change(self):
        # Two scenarios:
        #   Run 1: probes [P1, P2, P3] all scored
        #   Run 2: P1 drifts, scored = [P2, P3]
        # In Run 2, P2 is at positional index 0 and P3 at 1, whereas
        # in Run 1 P2 was at 1 and P3 at 2. If the seed hashes the
        # positional index, P2 and P3 get different arm letters
        # between the two runs — flipping the de-anonymization and
        # making cross-run per-probe comparison impossible.
        #
        # The fix seeds off expected_fact_id, which is invariant.
        # This test reproduces the seed math and asserts P2/P3 get
        # the same letter regardless of the position change.
        seed = "0123456789abcdef"

        def letter_for(probe_id: str) -> str:
            # Mirror the seed math in _run_under_semaphore exactly so
            # this test breaks if the production formula is changed.
            import hashlib
            import random as _rand

            h = hashlib.sha256()
            h.update(seed.encode("utf-8"))
            h.update(probe_id.encode("utf-8"))
            rng = _rand.Random(int(h.hexdigest()[:16], 16))
            return _arm_letter_for_memory(rng)

        # Same probe ID always maps to the same letter, regardless of
        # what other probes happen to flank it in the scored subset.
        p2_letter_run1 = letter_for("fact-P2")
        p3_letter_run1 = letter_for("fact-P3")
        # In Run 2, P1 has drifted; P2 and P3 are still asked but at
        # different positions. They still hash by their own ID, so:
        p2_letter_run2 = letter_for("fact-P2")
        p3_letter_run2 = letter_for("fact-P3")
        assert p2_letter_run1 == p2_letter_run2
        assert p3_letter_run1 == p3_letter_run2

    def test_different_probe_ids_diverge(self):
        # Sanity: distinct probe IDs hash to (statistically) distinct
        # arm letters, so the seed isn't accidentally collapsing every
        # probe to the same coin flip.
        seed = "0123456789abcdef"

        def letter_for(probe_id: str) -> str:
            import hashlib
            import random as _rand

            h = hashlib.sha256()
            h.update(seed.encode("utf-8"))
            h.update(probe_id.encode("utf-8"))
            return _arm_letter_for_memory(_rand.Random(int(h.hexdigest()[:16], 16)))

        # Across 50 distinct probe IDs we expect roughly half A, half
        # B; assert at least both letters appear, which guards against
        # a stuck RNG.
        letters = {letter_for(f"fact-{i}") for i in range(50)}
        assert letters == {"A", "B"}


# ── Test 12: Output write failure returns exit 1 (W3 fix) ──────────


class TestOutputWriteFailure:
    """An OSError on the output JSON write is surfaced as exit code 1.

    Without this guard, an unwritable path or a full disk would crash
    the process and lose the entire run's results (78 subprocess calls
    of work). The summary is already on stdout before the write
    attempt, so the operator still sees results — but the exit code
    flip lets a wrapper script detect the partial-success state.
    """

    def test_write_failure_returns_exit_1(self, tmp_path):
        # Point --output at a path the write CANNOT succeed against:
        # a directory (write_text on a directory raises IsADirectoryError,
        # which is an OSError subclass — the same except clause catches
        # both real-disk and adversarial cases).
        bad_output = tmp_path  # tmp_path itself is a directory

        # Stub out the heavyweight pieces of _run_cli so the test only
        # exercises the failure-handling branch around args.output.
        # Build a minimal probes file so load_behavioral_probes succeeds.
        probes_file = tmp_path / "probes.jsonl"
        probes_file.write_text(
            json.dumps({"question": "q1", "expected_fact_id": "f1"}) + "\n",
            encoding="utf-8",
        )

        args = MagicMock()
        args.probes = probes_file
        args.user_id = "user-1"
        args.output = bad_output
        args.judge_model = "claude-haiku-4-5-20251001"
        args.judge_budget_usd = 0.05
        args.judge_timeout_s = 60
        args.gen_model = "sonnet"
        args.gen_budget_usd = 0.10
        args.gen_timeout_s = 120
        args.seed = None
        args.max_concurrency = 1
        args.pollution_lines = 0

        # Mock the heavy machinery: init succeeds, drift detection
        # produces one scored probe, all_probes returns one
        # generation_error outcome. The interesting path is the
        # write-failure handling, not the run itself.
        async def fake_run_all_probes(**kwargs):
            return [
                ProbeOutcome(
                    probe=Probe(question="q1", expected_fact_id="f1"),
                    tags=(),
                    memory_arm_letter="A",
                    responses={"A": "", "B": ""},
                    judge_choice="generation_error",
                    judge_reasoning="",
                    memory_outcome="generation_error",
                    latency_ms={"A": 0.0, "B": 0.0, "judge": 0.0},
                    cost_usd={"A": None, "B": None, "judge": None},
                ),
            ]

        # _resolve_ground_truth is a sync function (no awaits inside;
        # see its docstring). Mock it as a plain return_value so the
        # caller's non-await call site works correctly.
        with (
            patch.object(behavioral, "_initialize_memory", return_value=tmp_path),
            patch.object(
                behavioral,
                "detect_drift",
                return_value=(
                    [Probe(question="q1", expected_fact_id="f1")],
                    [],
                    {"f1": ()},
                ),
            ),
            patch.object(behavioral, "_run_all_probes", new=fake_run_all_probes),
            patch.object(behavioral, "_resolve_ground_truth", return_value={"f1": "gold"}),
            patch.object(behavioral, "_capture_agent_cli_version", return_value=("claude_cli_version", "2.1.118")),
        ):
            exit_code = asyncio.run(behavioral._run_cli(args))

        # Write to a directory raises IsADirectoryError -> caught as
        # OSError -> exit 1 with the summary already printed.
        assert exit_code == 1


class TestBackendAwareModelResolution:
    """
    Verify the backend-aware dispatch in _run_cli for judge/gen models.

    The resolution shape:
    - On claude / codex, get_model_for(role, backend, override=args.x or "")
      handles the lookup. Unset flag yields the registry default;
      explicit flag wins via override.
    - On goose (and any future non-registry backend), the legacy
      _DEFAULT_JUDGE_MODEL / _DEFAULT_GEN_MODEL constants are used
      when the flag is unset, mirroring pre-Phase-1 behavior. Without
      this fallback, get_model_for would raise LookupError on the
      missing (goose, BEHAVIORAL_*) row and crash the eval CLI.
    """

    @staticmethod
    def _make_args(tmp_path: Path, *, judge_model=None, gen_model=None):
        """Minimal args.Namespace mock for _run_cli."""
        probes_file = tmp_path / "probes.jsonl"
        probes_file.write_text(
            json.dumps({"question": "q1", "expected_fact_id": "f1"}) + "\n",
            encoding="utf-8",
        )
        args = MagicMock()
        args.probes = probes_file
        args.user_id = "user-1"
        args.output = None
        args.judge_model = judge_model
        args.judge_budget_usd = 0.05
        args.judge_timeout_s = 60
        args.gen_model = gen_model
        args.gen_budget_usd = 0.10
        args.gen_timeout_s = 120
        args.seed = None
        args.max_concurrency = 1
        args.pollution_lines = 0
        return args

    @staticmethod
    def _run_with_captured_config(args):
        """
        Invoke _run_cli with heavyweight machinery mocked, capturing
        the BehavioralConfig that gets handed to _run_all_probes.

        Callers apply monkeypatch.setenv("AGENT_BACKEND", ...) before
        calling this helper; the env is already set by the time
        _run_cli reads it.
        """
        captured: dict[str, BehavioralConfig] = {}

        async def fake_run_all_probes(**kwargs):
            captured["config"] = kwargs["config"]
            return []

        with (
            patch.object(behavioral, "_initialize_memory", return_value=Path("/tmp")),
            patch.object(
                behavioral,
                "detect_drift",
                return_value=([Probe(question="q1", expected_fact_id="f1")], [], {"f1": ()}),
            ),
            patch.object(behavioral, "_run_all_probes", new=fake_run_all_probes),
            patch.object(behavioral, "_resolve_ground_truth", return_value={"f1": "gold"}),
            patch.object(behavioral, "_capture_agent_cli_version", return_value=("claude_cli_version", "2.1.118")),
        ):
            asyncio.run(behavioral._run_cli(args))
        return captured["config"]

    def test_goose_unset_flag_falls_back_to_legacy_constant(self, tmp_path, monkeypatch):
        """
        On a goose-backed install, an unset --judge-model must NOT trigger
        get_model_for("goose", BEHAVIORAL_JUDGE) (no registry row, would
        raise LookupError). The fallback path uses _DEFAULT_JUDGE_MODEL,
        matching pre-Phase-1 behavior.
        """
        monkeypatch.setenv("AGENT_BACKEND", "goose")
        args = self._make_args(tmp_path)  # judge_model=None, gen_model=None
        config = self._run_with_captured_config(args)
        assert config.judge_model == behavioral._DEFAULT_JUDGE_MODEL
        assert config.gen_model == behavioral._DEFAULT_GEN_MODEL

    def test_goose_explicit_flag_wins(self, tmp_path, monkeypatch):
        """
        Explicit --judge-model still wins on goose. The fallback only
        kicks in for an unset (None) flag value.
        """
        monkeypatch.setenv("AGENT_BACKEND", "goose")
        args = self._make_args(tmp_path, judge_model="opus", gen_model="haiku")
        config = self._run_with_captured_config(args)
        assert config.judge_model == "opus"
        assert config.gen_model == "haiku"

    def test_claude_unset_flag_uses_registry(self, tmp_path, monkeypatch):
        """
        On the claude backend, an unset flag resolves through the
        registry to the BEHAVIORAL_JUDGE / BEHAVIORAL_GEN row. The
        registry rows are byte-identical to the legacy constants by
        Phase 1's no-behavior-change invariant, so the resolved values
        must equal the pre-Phase-1 defaults.
        """
        monkeypatch.setenv("AGENT_BACKEND", "claude")
        args = self._make_args(tmp_path)
        config = self._run_with_captured_config(args)
        assert config.judge_model == behavioral._DEFAULT_JUDGE_MODEL
        assert config.gen_model == behavioral._DEFAULT_GEN_MODEL

    def test_claude_explicit_flag_wins(self, tmp_path, monkeypatch):
        """Explicit flag wins on claude too, exercising the override path."""
        monkeypatch.setenv("AGENT_BACKEND", "claude")
        args = self._make_args(tmp_path, judge_model="opus", gen_model="sonnet")
        config = self._run_with_captured_config(args)
        assert config.judge_model == "opus"
        assert config.gen_model == "sonnet"


class TestProbeIdMatchesSourcePosition:
    """per_probe[].probe_id reflects position in the source probes file,
    not position in the (scored + drift) concatenated outcomes list.

    This matters because outcomes are stored as `scored_outcomes +
    drift_outcomes`, which is NOT the same order as the probes file
    when any middle-of-file probe drifts. Using the iteration index
    would falsely label a drifted probe with the wrong source line,
    breaking any downstream tooling that round-trips probe_id back to
    the input file (e.g. failure-bisection scripts or cluster heat
    maps that assume probe_id == row number).
    """

    def test_drifted_middle_probe_keeps_source_position(self):
        # Three probes in source-file order. f2 drifts; the outcomes
        # list ends up as [f1_scored, f3_scored, f2_drift]. Iteration
        # index would assign probe_id = 1, 2, 3, which is wrong for
        # f3 and f2. The fix keys probe_id off expected_fact_id ->
        # source position.
        probes = [
            Probe(question="q1", expected_fact_id="f1"),
            Probe(question="q2", expected_fact_id="f2"),
            Probe(question="q3", expected_fact_id="f3"),
        ]
        outcomes = [
            _make_outcome(probe=probes[0], memory_outcome="memory_wins"),
            _make_outcome(probe=probes[2], memory_outcome="memory_loses"),
            _make_drift_outcome(probes[1], ()),
        ]
        config = _make_config()

        output = behavioral.build_output_json(
            probes=probes,
            outcomes=outcomes,
            drift_count=1,
            config=config,
            cli_version_field="claude_cli_version",
            cli_version_value="2.1.118",
        )

        # The per_probe rows preserve outcomes-list ORDER (scored then
        # drift) but each row's probe_id is the SOURCE-FILE position.
        per_probe = output["per_probe"]
        assert len(per_probe) == 3
        assert per_probe[0]["expected_fact_id"] == "f1"
        assert per_probe[0]["probe_id"] == 1
        assert per_probe[1]["expected_fact_id"] == "f3"
        assert per_probe[1]["probe_id"] == 3  # NOT 2
        assert per_probe[2]["expected_fact_id"] == "f2"
        assert per_probe[2]["probe_id"] == 2  # NOT 3


class TestPositiveIntCLIArg:
    """`--max-concurrency` rejects non-positive values at parse time.

    Without the guard, `--max-concurrency 0` builds an
    asyncio.Semaphore(0) whose first acquire() blocks forever (no
    task can ever release a slot it never held). The harness would
    hang silently with no diagnostic; an operator who fat-fingered
    a zero would only see "no output" until they killed the process.
    Catching the bad value at the argparse boundary turns the hang
    into a one-line ArgumentTypeError before any subprocess starts.
    """

    @pytest.mark.parametrize("bad", ["0", "-1", "-100"])
    def test_rejects_non_positive(self, bad: str):
        with pytest.raises(argparse.ArgumentTypeError, match=r"must be >= 1"):
            _positive_int(bad)

    def test_rejects_non_integer(self):
        with pytest.raises(argparse.ArgumentTypeError, match=r"expected integer"):
            _positive_int("four")

    @pytest.mark.parametrize("good,expected", [("1", 1), ("4", 4), ("128", 128)])
    def test_accepts_positive(self, good: str, expected: int):
        assert _positive_int(good) == expected


# ── Pollution injection (eval-only Track 1 reconstruction) ─────────


class TestPollutionInjection:
    """Synthetic pollution: helpers, determinism, A/B-isolation respect.

    The pollution feature exists to test the Layer 2 hypothesis that
    removing Track 1 raw-utterance noise mattered more than adding
    semantic retrieval. The tests below lock down the invariants the
    feature MUST preserve for that hypothesis test to be meaningful:

    1. Default (pollution_lines=0) is byte-identical to today's
       behavior; the existing A/B isolation guarantee in
       TestArmPromptIsolation still holds, so the no-pollution baseline
       is interpretable as "exactly the production memory-on prompt".
    2. The pollution RNG is independent of the A/B coin-flip RNG, so
       enabling pollution does NOT shift arm assignments. Without this,
       a pollution-on vs pollution-off comparison would conflate the
       pollution effect with arbitrary arm-letter changes.
    3. The memory-off arm is never polluted; the asymmetry IS the
       experimental contrast.
    """

    def test_inject_with_empty_lines_is_noop(self):
        # Byte-level no-op when no lines are sampled (the pollution-off
        # case). This is what protects test 1's isolation invariant from
        # accidentally regressing once the pollution code path is wired
        # into build_arm_prompt; if the helper added even a trailing
        # newline here, every existing memory-on arm would silently
        # diverge from production format_context output.
        ctx = "[Relevant memories]\n- foo\n- bar"
        assert _inject_synthetic_pollution(ctx, []) == ctx

    def test_inject_appends_user_said_bullets(self):
        # The "User said:" prefix is the literal Track 1 prompt shape
        # we are reconstructing. Whitespace is stripped per-line so that
        # history entries with trailing newlines do not produce ragged
        # bullets that would tip off the judge that something synthetic
        # is happening.
        ctx = "[Relevant memories]\n- favorite color is blue"
        out = _inject_synthetic_pollution(ctx, ["hello world  ", "  what time is it"])
        # Original facts preserved verbatim, bullets appended with one
        # newline separator (rstrip on ctx tolerates an existing trailing
        # newline without producing a blank-line gap).
        assert out == (
            "[Relevant memories]\n- favorite color is blue\n- User said: hello world\n- User said: what time is it"
        )

    def test_inject_strips_existing_trailing_newline(self):
        # format_context returns text ending in "\n" in some paths; the
        # rstrip + explicit "\n" join produces a single separator
        # regardless. This test is the regression guard against a future
        # change that drops the rstrip and produces double-newlines.
        ctx = "[Relevant memories]\n- a\n"
        out = _inject_synthetic_pollution(ctx, ["x"])
        assert "\n\n" not in out
        assert out.endswith("- User said: x")

    def test_sample_returns_empty_for_zero_n(self):
        assert _sample_pollution_for_probe(["a", "b", "c"], 0, base_seed="seed", expected_fact_id="fid") == []

    def test_sample_returns_empty_for_empty_history(self):
        assert _sample_pollution_for_probe([], 5, base_seed="seed", expected_fact_id="fid") == []

    def test_sample_is_deterministic_for_same_seed_and_fact_id(self):
        # Same (seed, fact_id) -> same sample. This is the property that
        # lets a re-run produce identical pollution per probe, which is
        # required for any "did the fix help?" comparison to be valid.
        history = [f"msg-{i}" for i in range(50)]
        a = _sample_pollution_for_probe(history, 5, base_seed="s", expected_fact_id="f")
        b = _sample_pollution_for_probe(history, 5, base_seed="s", expected_fact_id="f")
        assert a == b
        assert len(a) == 5

    def test_sample_differs_across_probes(self):
        # Different fact_ids -> different samples (with high
        # probability). 30 lines / 5 sampled gives ~3.4M unique combos;
        # collision is essentially impossible. If this ever fails, the
        # hash domain separation is broken.
        history = [f"msg-{i}" for i in range(30)]
        a = _sample_pollution_for_probe(history, 5, base_seed="s", expected_fact_id="fact-A")
        b = _sample_pollution_for_probe(history, 5, base_seed="s", expected_fact_id="fact-B")
        assert a != b

    def test_sample_independent_of_ab_coin_flip_rng(self):
        # CRITICAL invariant: enabling pollution must NOT shift arm
        # assignments. The pollution RNG seeds with `+ b"pollution"`
        # salt; the A/B RNG (in _run_all_probes) seeds without it. This
        # test reproduces both seedings and asserts the salt actually
        # changes the byte stream, which is the only thing keeping
        # pollution-on vs pollution-off comparisons valid against the
        # same (probes, user_id, seed) triple.
        import hashlib

        base_seed = "seed-xyz"
        fact_id = "fact-1"

        # The A/B RNG seeding (mirrors _run_all_probes:_run_under_semaphore).
        ab_h = hashlib.sha256()
        ab_h.update(base_seed.encode("utf-8"))
        ab_h.update(fact_id.encode("utf-8"))
        ab_seed_int = int(ab_h.hexdigest()[:16], 16)

        # The pollution RNG seeding (mirrors _sample_pollution_for_probe).
        # NUL delimiters between fields prevent concatenation collisions;
        # the A/B side intentionally does not use them because changing
        # that would shift every recorded baseline's arm assignments.
        poll_h = hashlib.sha256()
        poll_h.update(base_seed.encode("utf-8"))
        poll_h.update(b"\x00")
        poll_h.update(fact_id.encode("utf-8"))
        poll_h.update(b"\x00")
        poll_h.update(b"pollution")
        poll_seed_int = int(poll_h.hexdigest()[:16], 16)

        assert ab_seed_int != poll_seed_int

    def test_sample_no_collision_under_concatenation_ambiguity(self):
        # Domain-separation guard: without NUL delimiters between fields,
        # SHA256.update() is purely concatenative, so (seed="ab",
        # fact_id="cde") and (seed="abc", fact_id="de") would draw the
        # same pollution sample because both feed b"abcde" + b"pollution"
        # into the hash. The delimiter byte makes those two cases
        # distinguishable. Without this property, two probes whose IDs
        # happen to share a string boundary with a different seed could
        # silently get identical pollution sets, breaking the per-probe
        # determinism property the rest of this class locks down.
        history = [f"msg-{i}" for i in range(40)]
        sample_a = _sample_pollution_for_probe(history, 5, base_seed="ab", expected_fact_id="cde")
        sample_b = _sample_pollution_for_probe(history, 5, base_seed="abc", expected_fact_id="de")
        assert sample_a != sample_b

    def test_sample_returns_all_when_n_exceeds_history(self):
        # When n >= len(history), sampling returns the full history
        # (in shuffled order). The harness must not crash on a small
        # history with --pollution-lines set high; this exercises the
        # min(n, len(history)) clamp inside _sample_pollution_for_probe.
        history = ["a", "b", "c"]
        out = _sample_pollution_for_probe(history, 100, base_seed="s", expected_fact_id="f")
        assert sorted(out) == ["a", "b", "c"]

    def test_load_history_returns_empty_when_dir_missing(self, tmp_path: Path):
        # Running against a fresh KAI_DATA_DIR with no history directory
        # is a legitimate path (e.g. a new operator bringing up Layer 2).
        # The helper returns [] and the CLI emits a stderr warning;
        # nothing raises.
        assert _load_user_history_messages("does-not-exist", tmp_path) == []

    @pytest.mark.parametrize(
        "bad",
        ["../etc", "../../etc/passwd", "foo/bar", "", ".", "..", "/abs/path"],
    )
    def test_load_history_rejects_path_traversal(self, tmp_path: Path, bad: str):
        # Defense-in-depth against an operator passing --user-id with a
        # traversal payload (or accidentally with an empty/dot value
        # that would resolve to data_dir/history itself, returning every
        # user's messages). The helper raises before touching disk.
        with pytest.raises(ValueError, match="invalid user_id"):
            _load_user_history_messages(bad, tmp_path)

    def test_load_history_filters_to_user_role_and_truncates(self, tmp_path: Path):
        # End-to-end through the JSONL parser: round-trip a synthetic
        # history file with a mix of user/assistant rows, blank lines,
        # corrupt lines, and one over-length user row. The expected
        # output exercises every filter branch in one fixture.
        history_dir = tmp_path / "history" / "user-1"
        history_dir.mkdir(parents=True)
        long_text = "x" * 2500
        rows = [
            json.dumps({"ts": 1, "dir": "user", "chat_id": 1, "text": "first user msg"}),
            json.dumps({"ts": 2, "dir": "assistant", "chat_id": 1, "text": "ignored bot reply"}),
            json.dumps({"ts": 3, "dir": "user", "chat_id": 1, "text": "  "}),  # whitespace-only, dropped
            "",  # blank line, dropped
            "{not valid json",  # corrupt row, skipped silently
            json.dumps({"ts": 4, "dir": "user", "chat_id": 1, "text": long_text}),
            json.dumps({"ts": 5, "dir": "user", "chat_id": 1, "text": "second user msg"}),
        ]
        (history_dir / "001.jsonl").write_text("\n".join(rows), encoding="utf-8")

        out = _load_user_history_messages("user-1", tmp_path)
        # Three user-role messages: assistant + whitespace + blank +
        # corrupt all dropped.
        assert len(out) == 3
        assert out[0] == "first user msg"
        # Long row truncated to _MAX_POLLUTION_LINE_CHARS (2000) + "...".
        assert out[1].startswith("x" * 2000)
        assert out[1].endswith("...")
        assert len(out[1]) == 2003
        assert out[2] == "second user msg"

    def test_load_history_reads_multiple_jsonl_files_in_sorted_order(self, tmp_path: Path):
        # Production history rotates into multiple JSONL files; the
        # loader concatenates them in lexicographic filename order
        # (matching the bot's own reader). Test asserts both ordering
        # and that the loader does not stop at the first file.
        history_dir = tmp_path / "history" / "user-2"
        history_dir.mkdir(parents=True)
        (history_dir / "001.jsonl").write_text(
            json.dumps({"ts": 1, "dir": "user", "chat_id": 2, "text": "from-001"}) + "\n",
            encoding="utf-8",
        )
        (history_dir / "002.jsonl").write_text(
            json.dumps({"ts": 2, "dir": "user", "chat_id": 2, "text": "from-002"}) + "\n",
            encoding="utf-8",
        )
        out = _load_user_history_messages("user-2", tmp_path)
        assert out == ["from-001", "from-002"]

    @pytest.mark.parametrize("bad", ["-1", "-100"])
    def test_non_negative_int_rejects_negatives(self, bad: str):
        with pytest.raises(argparse.ArgumentTypeError, match=r"must be >= 0"):
            _non_negative_int(bad)

    def test_non_negative_int_rejects_non_integer(self):
        with pytest.raises(argparse.ArgumentTypeError, match=r"expected integer"):
            _non_negative_int("ten")

    @pytest.mark.parametrize("good,expected", [("0", 0), ("1", 1), ("100", 100)])
    def test_non_negative_int_accepts_zero_and_positives(self, good: str, expected: int):
        # 0 is the legitimate default (pollution off); the divergence
        # from _positive_int's contract is exactly what makes this a
        # separate validator rather than a parameter on the existing one.
        assert _non_negative_int(good) == expected

    def test_build_arm_prompt_with_pollution_none_matches_today(self):
        # Byte-level isolation extends to the new pollution_lines arg:
        # the default (None) MUST produce the exact prompt today's code
        # produces. This is the regression guard that the existing
        # TestArmPromptIsolation cases continue to hold once the
        # pollution code path is wired in.
        memory_block = "[Relevant memories]\n- the user's favorite color is blue"
        question = "What's my favorite color?"

        with patch("kai.memory.format_context", new=AsyncMock(return_value=memory_block)):
            arm_default = asyncio.run(build_arm_prompt(question, "user-1", memory_enabled=True))
            arm_explicit_none = asyncio.run(
                build_arm_prompt(question, "user-1", memory_enabled=True, pollution_lines=None)
            )
            arm_empty_list = asyncio.run(build_arm_prompt(question, "user-1", memory_enabled=True, pollution_lines=[]))

        # Default, explicit None, and empty list all produce the exact
        # same prompt as the production memory-on path: no separator,
        # no marker, no whitespace difference.
        assert arm_default == arm_explicit_none == arm_empty_list

    def test_build_arm_prompt_injects_when_pollution_lines_supplied(self):
        # Positive case: with pollution_lines passed and memory enabled,
        # the bullets land inside the memory block (before
        # prepend_to_prompt fuses it with the question), so the user-
        # message marker added by prepend_to_prompt still cleanly
        # separates the polluted memory block from the question.
        memory_block = "[Relevant memories]\n- favorite color is blue"
        question = "What's my favorite color?"

        with patch("kai.memory.format_context", new=AsyncMock(return_value=memory_block)):
            arm_polluted = asyncio.run(
                build_arm_prompt(
                    question,
                    "user-1",
                    memory_enabled=True,
                    pollution_lines=["I had pizza yesterday", "where did I park"],
                )
            )

        # Pollution bullets present.
        assert "- User said: I had pizza yesterday" in arm_polluted
        assert "- User said: where did I park" in arm_polluted
        # Original fact preserved.
        assert "favorite color is blue" in arm_polluted
        # Question still in there (memory-on arm always ends with the question).
        assert "What's my favorite color?" in arm_polluted

    def test_build_arm_prompt_memory_off_ignores_pollution_lines(self):
        # The memory-off arm is the no-injection control by definition;
        # passing pollution_lines into a memory_enabled=False call must
        # be a no-op. If this ever fails, the experimental contrast
        # collapses (both arms polluted equally -> nothing being
        # measured).
        question = "What's my favorite color?"
        with patch("kai.memory.format_context", new=AsyncMock(return_value="ignored")):
            arm_off = asyncio.run(
                build_arm_prompt(
                    question,
                    "user-1",
                    memory_enabled=False,
                    pollution_lines=["should-not-appear"],
                )
            )
        assert arm_off == question
        assert "should-not-appear" not in arm_off


class TestInitializeMemoryDataDirContract:
    """`_initialize_memory` returns the live DATA_DIR, not a Config attr.

    Regression guard for a bug shipped in the synthetic-pollution PR:
    the function originally returned `load_config().data_dir`, but
    `Config` is a dataclass with no `data_dir` field; DATA_DIR is a
    module-level constant in kai.config. The bug was masked in tests
    because every behavioral test that exercises _initialize_memory
    mocks it with `return_value=tmp_path`, so the real attribute access
    never ran. This class is the integration-style check that the
    function's import contract still holds against the actual
    kai.config module.
    """

    def test_kai_config_exposes_data_dir_constant(self):
        # Import contract: kai.config.DATA_DIR is a Path. If a future
        # config refactor renames or removes it, the import inside
        # _initialize_memory raises ImportError at run time and the
        # whole eval CLI fails to launch. This one-line assertion
        # catches that earlier than the broken-CLI symptom.
        from kai.config import DATA_DIR

        assert isinstance(DATA_DIR, Path)

    def test_initialize_memory_returns_path_when_memory_enabled(self, tmp_path: Path, monkeypatch):
        # End-to-end through _initialize_memory with the heavy init
        # mocked but the actual import path exercised. The previous
        # bug shipped because every test in the suite stubbed
        # _initialize_memory itself; this test stubs only its
        # collaborators (init_memory, is_enabled) so the function's
        # own body, including `return DATA_DIR`, runs under test.
        # If the function tries to access a non-existent attr (the
        # exact bug we shipped), it lands in the except branch and
        # returns None, which the assertion catches loudly.
        #
        # Mechanics note for future readers extending this pattern:
        # _initialize_memory uses a function-local `from kai.config
        # import DATA_DIR`, which Python resolves by reading
        # sys.modules["kai.config"].DATA_DIR at call time, NOT at
        # test-module import time. monkeypatch.setattr on the module
        # object therefore takes effect for every subsequent
        # function-local import in the same process. The same logic
        # applies to `patch("kai.memory.init_memory")` and
        # `patch("kai.memory.is_enabled")`: those patch the module
        # attribute, which the function-local `from kai.memory
        # import ...` re-reads each call.
        from kai import config as _cfg

        monkeypatch.setattr(_cfg, "DATA_DIR", tmp_path)
        with (
            patch.object(_cfg, "load_config", return_value=MagicMock()),
            patch("kai.memory.init_memory"),
            patch("kai.memory.is_enabled", return_value=True),
        ):
            result = behavioral._initialize_memory()
        assert result == tmp_path

    def test_initialize_memory_returns_none_when_memory_disabled(self, monkeypatch):
        # The disabled-path is the other live branch; locked to keep
        # the bool-vs-Path return-type contract honest after the
        # type change from `bool` to `Path | None`.
        from kai import config as _cfg

        with (
            patch.object(_cfg, "load_config", return_value=MagicMock()),
            patch("kai.memory.init_memory"),
            patch("kai.memory.is_enabled", return_value=False),
        ):
            result = behavioral._initialize_memory()
        assert result is None


# ── Codex vertical (Phase 5 of codex epic #480) ─────────────────────


class TestBuildJudgeCmdCodex:
    """Codex judge argv is `codex exec --json --model <model>` with no
    claude-side flags, and `CODEX_BIN` overrides the binary path."""

    def test_argv_shape(self, monkeypatch):
        monkeypatch.delenv("CODEX_BIN", raising=False)
        config = _make_config(judge_model="gpt-5.4-mini", backend="codex")
        cmd = _build_judge_cmd_codex(config)
        assert cmd[0] == "codex"
        assert cmd[1] == "exec"
        assert "--json" in cmd
        # `--skip-git-repo-check` is load-bearing: codex exec refuses
        # to spawn unless the cwd is on the user's trusted-directories
        # list or this flag is passed. Without it the eval bucketed
        # every probe as generation_error.
        assert "--skip-git-repo-check" in cmd
        i = cmd.index("--model")
        assert cmd[i + 1] == "gpt-5.4-mini"
        # No claude flags.
        assert "--print" not in cmd
        assert "--json-schema" not in cmd
        assert "--system-prompt" not in cmd
        assert "--permission-mode" not in cmd
        assert "--tools" not in cmd
        assert "--max-budget-usd" not in cmd

    def test_codex_bin_env_override(self, monkeypatch):
        monkeypatch.setenv("CODEX_BIN", "/tmp/fake-codex")
        config = _make_config(judge_model="gpt-5.4-mini", backend="codex")
        cmd = _build_judge_cmd_codex(config)
        assert cmd[0] == "/tmp/fake-codex"


class TestBuildGenCmdCodex:
    """Codex generator argv mirrors the judge shape; same flags absent."""

    def test_argv_shape(self, monkeypatch):
        monkeypatch.delenv("CODEX_BIN", raising=False)
        config = _make_config(gen_model="gpt-5.4-mini", backend="codex")
        cmd = _build_gen_cmd_codex(config)
        assert cmd[0] == "codex"
        assert cmd[1] == "exec"
        assert "--json" in cmd
        # Same `--skip-git-repo-check` invariant as the judge builder;
        # see TestBuildJudgeCmdCodex.test_argv_shape for the rationale.
        assert "--skip-git-repo-check" in cmd
        i = cmd.index("--model")
        assert cmd[i + 1] == "gpt-5.4-mini"
        assert "--system-prompt" not in cmd
        assert "--tools" not in cmd
        assert "--max-budget-usd" not in cmd

    def test_codex_bin_env_override(self, monkeypatch):
        monkeypatch.setenv("CODEX_BIN", "/tmp/fake-codex")
        config = _make_config(gen_model="gpt-5.4-mini", backend="codex")
        cmd = _build_gen_cmd_codex(config)
        assert cmd[0] == "/tmp/fake-codex"


class TestRenderCodexStdin:
    """The stdin renderer prepends a boundary-delimited SYSTEM block when
    a system prompt is set, and emits the user payload unchanged when
    the system prompt is empty (generator's --system-prompt "" analog)."""

    def test_empty_system_prompt_returns_user_payload_unchanged(self):
        result = _render_codex_stdin("", "hello world")
        assert result == "hello world"

    def test_system_prompt_wraps_with_boundary_block(self):
        result = _render_codex_stdin("be impartial", "rate the responses")
        # SYSTEM block at the head, then a blank line, then the user payload.
        assert result.startswith("--- BEGIN SYSTEM")
        assert "be impartial" in result
        assert "--- END SYSTEM" in result
        assert result.endswith("rate the responses")
        # The SYSTEM block comes BEFORE the user payload.
        sys_end = result.index("--- END SYSTEM")
        user_start = result.index("rate the responses")
        assert sys_end < user_start


class TestValidateJudgeEnvelope:
    """Post-hoc validator that mirrors the claude `--json-schema` contract."""

    def test_valid_envelope_returns_true(self):
        assert _validate_judge_envelope({"choice": "A_wins", "reasoning": "because"})

    def test_top_level_not_dict_returns_false(self):
        assert not _validate_judge_envelope([])
        assert not _validate_judge_envelope("string")
        assert not _validate_judge_envelope(None)

    def test_missing_choice_returns_false(self):
        assert not _validate_judge_envelope({"reasoning": "r"})

    def test_missing_reasoning_returns_false(self):
        assert not _validate_judge_envelope({"choice": "A_wins"})

    def test_choice_not_in_enum_returns_false(self):
        assert not _validate_judge_envelope({"choice": "X_wins", "reasoning": "r"})

    def test_choice_non_string_returns_false(self):
        assert not _validate_judge_envelope({"choice": 1, "reasoning": "r"})

    def test_reasoning_non_string_returns_false(self):
        assert not _validate_judge_envelope({"choice": "A_wins", "reasoning": 42})

    def test_extra_property_returns_false(self):
        assert not _validate_judge_envelope({"choice": "A_wins", "reasoning": "r", "extra": "x"})


def _codex_ndjson_envelope(text: str, item_id: str = "i1") -> bytes:
    """Build a minimal codex `exec --json` NDJSON stream wrapping a single
    agent_message with `text` as its consolidated content."""
    events = [
        {"type": "thread.started", "thread_id": "thr_t"},
        {"type": "turn.started"},
        {"type": "item.completed", "item": {"id": item_id, "type": "agent_message", "text": text}},
        {"type": "turn.completed", "usage": {"input_tokens": 0, "output_tokens": 0}},
    ]
    return ("\n".join(json.dumps(e) for e in events) + "\n").encode()


class TestParseCodexJudgeStdout:
    """Judge parser: codex NDJSON -> agent_message JSON -> schema validate
    -> (choice, reasoning). join_items=False because the judge contract
    is exactly one final JSON object."""

    def test_happy_path(self):
        envelope_json = '{"choice": "A_wins", "reasoning": "the gold fact is present in A"}'
        stdout = _codex_ndjson_envelope(envelope_json)
        result = _parse_codex_judge_stdout(stdout)
        assert result == ("A_wins", "the gold fact is present in A")

    def test_malformed_json_returns_none(self):
        stdout = _codex_ndjson_envelope("not-json-at-all")
        assert _parse_codex_judge_stdout(stdout) is None

    def test_missing_choice_returns_none(self):
        stdout = _codex_ndjson_envelope('{"reasoning": "no choice key"}')
        assert _parse_codex_judge_stdout(stdout) is None

    def test_invalid_choice_enum_returns_none(self):
        stdout = _codex_ndjson_envelope('{"choice": "X_wins", "reasoning": "r"}')
        assert _parse_codex_judge_stdout(stdout) is None

    def test_extra_property_returns_none(self):
        stdout = _codex_ndjson_envelope('{"choice": "A_wins", "reasoning": "r", "extra": "x"}')
        assert _parse_codex_judge_stdout(stdout) is None

    def test_empty_stdout_returns_none(self):
        assert _parse_codex_judge_stdout(b"") is None

    def test_join_items_false_keeps_last_for_preamble_plus_json_safety(self):
        """If codex emits a preamble agent_message before the JSON body,
        last-wins selects the JSON. Joining would corrupt the parse."""
        envelope_json = '{"choice": "tie", "reasoning": "r"}'
        events = [
            {"type": "thread.started", "thread_id": "thr_t"},
            {"type": "turn.started"},
            {"type": "item.completed", "item": {"id": "i1", "type": "agent_message", "text": "Let me think..."}},
            {"type": "item.completed", "item": {"id": "i2", "type": "agent_message", "text": envelope_json}},
            {"type": "turn.completed", "usage": {"input_tokens": 0, "output_tokens": 0}},
        ]
        stdout = ("\n".join(json.dumps(e) for e in events) + "\n").encode()
        result = _parse_codex_judge_stdout(stdout)
        assert result == ("tie", "r")


class TestParseCodexGenStdout:
    """Generator parser: codex NDJSON -> joined agent_message text. Multi-
    item turns are preserved with a blank-line separator (B-1 regression)."""

    def test_single_item(self):
        stdout = _codex_ndjson_envelope("a single response")
        assert _parse_codex_gen_stdout(stdout) == "a single response"

    def test_multi_item_joined_with_blank_line(self):
        events = [
            {"type": "thread.started", "thread_id": "thr_t"},
            {"type": "turn.started"},
            {"type": "item.completed", "item": {"id": "i1", "type": "agent_message", "text": "first part"}},
            {"type": "item.completed", "item": {"id": "i2", "type": "agent_message", "text": "second part"}},
            {"type": "turn.completed", "usage": {"input_tokens": 0, "output_tokens": 0}},
        ]
        stdout = ("\n".join(json.dumps(e) for e in events) + "\n").encode()
        # B-1 regression guard: BOTH items present, separated by \n\n. The
        # prior implementation (join_items=False on generator) would have
        # returned only "second part" and silently dropped the first.
        assert _parse_codex_gen_stdout(stdout) == "first part\n\nsecond part"

    def test_empty_stream_returns_empty_string(self):
        assert _parse_codex_gen_stdout(b"") == ""


class TestCaptureAgentCliVersionCodex:
    """Backend dispatch for the version-capture helper, plus the
    `CODEX_BIN` override for the codex branch."""

    def test_claude_branch_uses_claude_version(self, monkeypatch):
        monkeypatch.delenv("CODEX_BIN", raising=False)
        fake_result = MagicMock()
        fake_result.returncode = 0
        fake_result.stdout = "claude 2.1.118\n"
        with patch("kai.eval.behavioral.subprocess.run", return_value=fake_result) as mock_run:
            field, value = _capture_agent_cli_version("claude")
        assert field == "claude_cli_version"
        assert value == "claude 2.1.118"
        assert mock_run.call_args[0][0] == ["claude", "--version"]

    def test_codex_branch_uses_codex_version(self, monkeypatch):
        monkeypatch.delenv("CODEX_BIN", raising=False)
        fake_result = MagicMock()
        fake_result.returncode = 0
        fake_result.stdout = "codex 0.130.0\n"
        with patch("kai.eval.behavioral.subprocess.run", return_value=fake_result) as mock_run:
            field, value = _capture_agent_cli_version("codex")
        assert field == "codex_cli_version"
        assert value == "codex 0.130.0"
        assert mock_run.call_args[0][0] == ["codex", "--version"]

    def test_codex_branch_honors_codex_bin(self, monkeypatch):
        monkeypatch.setenv("CODEX_BIN", "/tmp/fake-codex")
        fake_result = MagicMock()
        fake_result.returncode = 0
        fake_result.stdout = "codex 0.130.0\n"
        with patch("kai.eval.behavioral.subprocess.run", return_value=fake_result) as mock_run:
            field, _value = _capture_agent_cli_version("codex")
        assert field == "codex_cli_version"
        assert mock_run.call_args[0][0] == ["/tmp/fake-codex", "--version"]

    def test_unknown_backend_falls_back_to_claude(self, monkeypatch):
        monkeypatch.delenv("CODEX_BIN", raising=False)
        fake_result = MagicMock()
        fake_result.returncode = 0
        fake_result.stdout = "claude 2.1.118\n"
        with patch("kai.eval.behavioral.subprocess.run", return_value=fake_result) as mock_run:
            field, _value = _capture_agent_cli_version("goose")
        assert field == "claude_cli_version"
        assert mock_run.call_args[0][0] == ["claude", "--version"]

    def test_subprocess_failure_returns_unknown(self):
        fake_result = MagicMock()
        fake_result.returncode = 1
        fake_result.stdout = ""
        with patch("kai.eval.behavioral.subprocess.run", return_value=fake_result):
            _field, value = _capture_agent_cli_version("codex")
        assert value == "unknown"


class TestOutputJsonMutuallyExclusiveVersionFields:
    """`claude_cli_version` and `codex_cli_version` never co-occur, and no
    `backend` field is added to the output JSON."""

    def _outcomes_and_probes(self):
        probe = Probe(question="q", expected_fact_id="f1")
        outcome = _make_outcome(probe=probe, memory_outcome="memory_wins")
        return [probe], [outcome]

    def test_codex_run_writes_codex_field_only(self):
        probes, outcomes = self._outcomes_and_probes()
        output = build_output_json(
            probes=probes,
            outcomes=outcomes,
            drift_count=0,
            config=_make_config(backend="codex"),
            cli_version_field="codex_cli_version",
            cli_version_value="codex 0.130.0",
        )
        assert output["codex_cli_version"] == "codex 0.130.0"
        assert "claude_cli_version" not in output
        assert "backend" not in output

    def test_claude_run_writes_claude_field_only(self):
        probes, outcomes = self._outcomes_and_probes()
        output = build_output_json(
            probes=probes,
            outcomes=outcomes,
            drift_count=0,
            config=_make_config(backend="claude"),
            cli_version_field="claude_cli_version",
            cli_version_value="claude 2.1.118",
        )
        assert output["claude_cli_version"] == "claude 2.1.118"
        assert "codex_cli_version" not in output
        assert "backend" not in output


class TestRenderSummaryVersionDispatch:
    """The summary line picks the right label based on which version key is
    present in the output dict. Reading the literal claude key (the pre-
    codex shape) would KeyError on a codex run."""

    def _outcomes_and_probes(self):
        probe = Probe(question="q", expected_fact_id="f1")
        outcome = _make_outcome(probe=probe, memory_outcome="memory_wins")
        return [probe], [outcome]

    def test_claude_run_renders_claude_label(self):
        probes, outcomes = self._outcomes_and_probes()
        output = build_output_json(
            probes=probes,
            outcomes=outcomes,
            drift_count=0,
            config=_make_config(backend="claude"),
            cli_version_field="claude_cli_version",
            cli_version_value="claude 2.1.118",
        )
        summary = _render_summary(output)
        assert "Claude CLI: claude 2.1.118" in summary
        assert "Codex CLI:" not in summary

    def test_codex_run_renders_codex_label_without_keyerror(self):
        probes, outcomes = self._outcomes_and_probes()
        output = build_output_json(
            probes=probes,
            outcomes=outcomes,
            drift_count=0,
            config=_make_config(backend="codex"),
            cli_version_field="codex_cli_version",
            cli_version_value="codex 0.130.0",
        )
        # Pre-fix this would have KeyError'd on output['claude_cli_version'].
        summary = _render_summary(output)
        assert "Codex CLI: codex 0.130.0" in summary
        assert "Claude CLI:" not in summary

    def test_shared_summary_lines_match_between_backends(self):
        probes, outcomes = self._outcomes_and_probes()
        config = _make_config(backend="claude")
        claude_output = build_output_json(
            probes=probes,
            outcomes=outcomes,
            drift_count=0,
            config=config,
            cli_version_field="claude_cli_version",
            cli_version_value="v",
        )
        codex_output = build_output_json(
            probes=probes,
            outcomes=outcomes,
            drift_count=0,
            config=_make_config(backend="codex"),
            cli_version_field="codex_cli_version",
            cli_version_value="v",
        )
        claude_summary = _render_summary(claude_output)
        codex_summary = _render_summary(codex_output)
        # Probe counts, models, seed lines render identically between
        # the two runs; only the CLI label differs.
        for snippet in (
            f"Models: gen={config.gen_model}, judge={config.judge_model}",
            f"Seed: {config.seed}",
        ):
            assert snippet in claude_summary
            assert snippet in codex_summary
