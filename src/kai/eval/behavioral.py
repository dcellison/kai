"""
Layer 2 end-to-end behavioral A/B evaluation harness.

Reachable as `python -m kai.eval.behavioral`.

Layer 1 (`kai.eval.retrieval`) measures whether the retrieval step
surfaces the right facts. It does NOT answer the question that matters
to the operator: when retrieval works, does memory injection actually
make the model's response BETTER than a memory-off response would be?
And when retrieval surfaces irrelevant facts (false positives, paraphrase
clusters), does it make the response WORSE?

This module answers that. For each probe, it runs the question through
the model TWICE: once with the memory block prepended (arm A or B,
chosen randomly), once with the memory block omitted. The two responses
are paired and scored by a separate LLM-judge call against the gold-fact
text. The judge picks A_wins / B_wins / tie / both_wrong; the harness
de-anonymizes that into memory_wins / memory_loses / tie / both_wrong
and aggregates win-rate, loss-rate, tie-rate, both_wrong-rate.

Design rationale (the parts that are not obvious from the code):

- A/B isolation. The hypothesis "memory helps" only holds if the two
  arms differ in EXACTLY one place: presence vs absence of the memory
  block. Any other delta (whitespace, system prompt, history, workspace
  CLAUDE.md, message ordering) is a confound. The harness builds both
  arms through the SAME `build_arm_prompt` helper which calls the same
  production `prepend_to_prompt` join, and the only branch in the code
  is the `format_context` call itself. A regression test (test 1) asserts
  byte-level equality against `prepend_to_prompt(question, memory_block)`
  so a future refactor of the production helper cannot quietly diverge
  the two arms.

- Subprocess sandboxing. Both judge and generator run via `claude --print`
  subprocesses, lifted from the extractor's pattern in
  `kai.memory_extraction._run_extractor`. The flag vector, env allow-list,
  neutral cwd (no CLAUDE.md auto-discovery), and kill-on-timeout shape
  are all reused as-is. Generator deliberately passes `--system-prompt ""`
  (literally empty, not omitted) so the CLI's default system prompt does
  not confound the measurement. Some default context still flows from
  the CLI no matter what (~340 input tokens at CLI 2.1.118), but it is
  identical between arms within one run, so it cancels for A/B purposes.
  Cross-run drift in that residual is detectable via the
  `claude_cli_version` field captured in the output JSON.

- Anonymization. The per-probe arm assignment (memory-on -> A vs B) is
  randomized so the judge cannot infer arm semantics from position.
  Random is seeded deterministically from `sha256(probe_set_hash + user_id)`
  by default, so a rerun against the same store produces identical arm
  assignments — that property matters when debugging "why did probe 17
  swing from win to loss." Operators can override with `--seed`.

- Drift bucket. Probes whose `expected_fact_id` does not resolve via
  `memory.get_by_id` are bucketed as `drift` (fact deleted, ownership
  mismatch, source not "extracted", typo at probe authoring) and
  excluded from win/loss/tie denominators. Same discipline as Layer 1.
  Reusing `kai.eval.retrieval.detect_drift` keeps the two layers in
  agreement about which probes are scorable on a given snapshot.

- Failure buckets. `judge_error` and `generation_error` are deliberately
  separate so a noisy run is diagnosable: a 30% generation_error rate
  means the gen subprocess is broken (rate-limited, timed out, model
  unavailable); a 30% judge_error rate means the judge subprocess is
  broken (bad budget cap, schema regression). Mixing them would hide
  which side needs operator attention.

PII posture: probe questions and gold-fact text remain in the gitignored
probes file plus the operator-named output JSON. The committed test
fixtures use synthetic data only.

Cost: per run, on the existing 26-probe set, ~26 generations + 26
generations + 26 judge calls = 78 Claude calls. At Sonnet generator +
Haiku judge that lands around $5-10 per run; cheap enough for weekly
operator-driven iteration, too expensive for per-PR CI.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import random
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Import from the Layer 1 sibling module so the two layers stay aligned
# on probe schema, drift detection, and probe-set-hash semantics. If
# Layer 1's drift bucketing changes, Layer 2 follows automatically.
from kai.eval.retrieval import (
    Probe,
    detect_drift,
    load_probes,
    probe_set_hash,
)

log = logging.getLogger(__name__)


# ── Constants ───────────────────────────────────────────────────────


# Schema version written into the output JSON. Bump when the output
# shape changes in a way that downstream consumers (jq filters,
# comparison scripts) would misinterpret. The probe_set_hash is the
# orthogonal "is this comparable to that other run" check; the
# version is the "can my parser still read this" check.
_OUTPUT_SCHEMA_VERSION = 1


# Default judge model. Haiku is the cheapest path that has held up
# in practice for short structured-output classification tasks. If the
# judge starts producing biased verdicts (verbosity preference, falling
# for confident wrong answers), swap to Sonnet via `--judge-model`.
_DEFAULT_JUDGE_MODEL = "claude-haiku-4-5-20251001"


# Default generation model. The generator is the system-under-test:
# the operator typically wants this to match whatever the bot runs in
# production. Sonnet is the production default; override via `--gen-model`.
_DEFAULT_GEN_MODEL = "sonnet"


# Per-call cost ceilings. The judge call is small (~3-4k input tokens,
# tens of output tokens), so 0.05 USD is generous. The generator can
# emit longer responses for multi-fact questions; 0.10 USD covers a
# 1-2k output-token reply at Sonnet rates.
_DEFAULT_JUDGE_BUDGET_USD = 0.05
_DEFAULT_GEN_BUDGET_USD = 0.10


# Subprocess timeout ceilings. Match the extractor's 60s default; a
# generator producing more text might run longer, so it gets 120s.
# These are wall-clock kill caps, not soft hints — exceeding either
# kills the subprocess and buckets the probe as judge_error or
# generation_error respectively.
_DEFAULT_JUDGE_TIMEOUT_S = 60
_DEFAULT_GEN_TIMEOUT_S = 120


# Default concurrency for the per-probe Semaphore. Each probe slot
# runs both arms plus the judge call sequentially (gen-A awaited,
# then gen-B awaited, then judge awaited), so a slot is only ever
# tying up ONE claude subprocess at a time. N in flight therefore
# means up to N simultaneous claude processes, NOT 3N; the second
# subprocess in a slot cannot start until the first has returned.
# Higher concurrency hits rate limits on Anthropic's API and burns
# laptop fans without speeding up the wall-clock meaningfully.
_DEFAULT_MAX_CONCURRENCY = 4


# JSON Schema passed to `claude --print --json-schema` for the judge
# call. The four-value `choice` enum maps directly to the harness's
# outcome buckets (after de-anonymization). `additionalProperties=false`
# closes the schema so the judge cannot smuggle extra fields the
# harness would silently ignore. The judge's `reasoning` field is kept
# permissive (just a string) — the prompt caps its length verbally but
# the schema does not enforce, because schema-side max-length on string
# fields is not portably supported across CLI versions.
_JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "choice": {
            "type": "string",
            "enum": ["A_wins", "B_wins", "tie", "both_wrong"],
        },
        "reasoning": {"type": "string"},
    },
    "required": ["choice", "reasoning"],
    "additionalProperties": False,
}


# The judge prompt rubric, embedded as the system prompt. Per-probe
# substitutions (question, ground_truth_text, response_a, response_b)
# are interpolated into this template at call time and fed via stdin.
# Keeping the rubric in the system slot rather than stdin lets the
# Claude CLI's prompt-caching reuse it across the dozens of judge calls
# in one run.
#
# CRITICAL: the rubric's "Do NOT score on" block is load-bearing. LLM
# judges drift toward verbosity / confidence preference by default;
# spelling out what NOT to weight pushes back. The narrowing sentence
# on `both_wrong` ("Use 'both_wrong' only when NEITHER response uses
# the GROUND-TRUTH FACT") prevents the judge from over-firing on
# responses that use the fact imperfectly. Past internal review caught
# both of these as biases worth fixing at prompt time, not in code.
_JUDGE_SYSTEM_PROMPT = """You are an impartial evaluator comparing two assistant responses to the same user question. Your goal is to identify which response more accurately uses the GROUND-TRUTH FACT to answer the question.

Score on these criteria, in order:
1. Factual accuracy: which response is more consistent with the GROUND-TRUTH FACT? A response that contradicts the ground truth loses regardless of style.
2. Relevance: which response actually addresses the user's question rather than tangentially related material?

Do NOT score on:
- Verbosity (longer is not better)
- Confidence or hedging
- Formatting, markdown, bullet points
- Conversational tone

Use "tie" when both responses are roughly equivalent in accuracy and relevance. Use "both_wrong" only when NEITHER response uses the GROUND-TRUTH FACT (a response that uses the fact but adds tangential noise is still better than one that ignores it).

Output ONLY a single JSON object on one line, no prose:
{"choice": "A_wins"|"B_wins"|"tie"|"both_wrong", "reasoning": "<one sentence, max 30 words>"}"""


# The per-probe stdin payload is built by string concatenation in
# `_render_judge_user_payload` rather than from a `{...}` template,
# so that response_a / response_b (model-generated text) cannot
# trigger str.format substitution or KeyError. The system slot above
# carries the rubric and is cached across calls; only the per-probe
# payload varies, which is what makes the CLI's prompt cache amortize.


# Bucket names used in the output JSON's `outcomes` block. Defined as
# constants so the test that asserts the sum-check (`sum(outcomes) +
# drift == probe_count`) and the aggregation logic below stay aligned.
_OUTCOME_BUCKETS: tuple[str, ...] = (
    "memory_wins",
    "memory_loses",
    "tie",
    "both_wrong",
    "judge_error",
    "generation_error",
)


# Mapping from raw judge `choice` string to per-probe `memory_outcome`.
# The mapping depends on which arm letter was assigned to memory-on
# for that probe; see `_rollup_outcome` for the full truth table.
# Defined inline rather than as a module-level dict because it depends
# on the per-probe arm letter; no point pre-computing.


# ── Data classes ────────────────────────────────────────────────────


@dataclass(frozen=True)
class BehavioralConfig:
    """Per-run configuration, materialized from CLI args.

    Frozen so a probe slot cannot mutate it under another slot.
    Carries everything needed to drive both subprocesses plus the
    seed/concurrency knobs that govern run-level reproducibility.
    """

    judge_model: str
    judge_budget_usd: float
    judge_timeout_s: int
    gen_model: str
    gen_budget_usd: float
    gen_timeout_s: int
    seed: str
    max_concurrency: int


@dataclass
class ProbeOutcome:
    """Result of one probe's full A/B + judge cycle.

    `memory_arm_letter` is the post-hoc record of which arm carried
    the memory block — needed at aggregation time to translate the
    judge's A/B verdict back into memory-on/memory-off terms. The raw
    `judge_choice` is preserved alongside the rolled-up `memory_outcome`
    so a future debugger can see what the judge actually said before
    the de-anonymization step.

    `responses` and `latency_ms`/`cost_usd` are dicts keyed by arm
    letter ("A", "B") so the per-probe row in the output JSON can
    surface both arms symmetrically.
    """

    probe: Probe
    tags: tuple[str, ...]
    memory_arm_letter: str  # "A" or "B"; the arm that received the memory block
    responses: dict[str, str]  # {"A": "...", "B": "..."}
    judge_choice: str  # one of A_wins|B_wins|tie|both_wrong|judge_error|generation_error|drift
    judge_reasoning: str
    memory_outcome: str  # rolled-up bucket name (one of _OUTCOME_BUCKETS or "drift")
    latency_ms: dict[str, float]  # {"A": ..., "B": ..., "judge": ...}
    cost_usd: dict[str, float | None]  # {"A": None, "B": None, "judge": ...}


# ── Probe loading ──────────────────────────────────────────────────


def load_behavioral_probes(path: Path) -> tuple[list[Probe], dict[str, str]]:
    """Load probes plus optional per-probe ground_truth_text overrides.

    Reuses Layer 1's `load_probes` for the base schema (question,
    expected_fact_id, source_turn_ts, notes) and then re-parses the
    same file to pick up the optional `ground_truth_text` field,
    returned as a dict keyed by `expected_fact_id`. Probes without
    the field will have their gold text resolved at scoring time via
    `memory.get_by_id`.

    Two-pass design: Layer 1's loader is the source of truth for the
    base probe schema (validation, comment skipping, line numbering),
    and re-parsing here is cheap (probes files are small JSONL). The
    alternative — extending `load_probes` with an optional return —
    would couple Layer 1 to a Layer 2-only field. Independent layers
    over a shared loader is the cleaner separation.
    """
    probes = load_probes(path)
    overrides: dict[str, str] = {}
    raw = path.read_text(encoding="utf-8").splitlines()
    for line in raw:
        stripped = line.lstrip()
        # Mirror Layer 1's comment/blank skip exactly so we touch the
        # same set of lines. A divergence here would silently mean
        # Layer 1 sees probes Layer 2 doesn't (or vice versa).
        if not stripped or stripped.startswith("#"):
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            # Already validated by load_probes above; if a line passed
            # there, json.loads here cannot fail. The except guards the
            # comment-line edge case where a probe file has a malformed
            # commented-out probe — ignored as a comment, not raised.
            continue
        gt = obj.get("ground_truth_text")
        # Only record string overrides; non-strings (None, numbers)
        # are treated as absence rather than as a coercion target. An
        # operator who wrote `"ground_truth_text": null` clearly meant
        # "fall back to the store"; honor that rather than silently
        # stringifying.
        if isinstance(gt, str) and gt.strip():
            expected = obj.get("expected_fact_id")
            if isinstance(expected, str):
                overrides[expected] = gt
    return probes, overrides


# ── Subprocess construction (judge + generator) ────────────────────


def _build_judge_cmd(config: BehavioralConfig) -> list[str]:
    """Construct the argv vector for one judge subprocess invocation.

    Lifted from `kai.memory_extraction._run_extractor` (the extractor's
    flag vector). Reuses every invariant: `--print` for non-interactive
    one-shot, `--output-format json` to get the structured envelope,
    `--json-schema` for CLI-side shape validation, `--permission-mode
    bypassPermissions` + `--tools ""` to disable every tool the model
    could call, `--no-session-persistence` to avoid leaving state on
    disk. The one extractor flag we deliberately differ on: model
    defaults to Haiku here regardless of MEMORY_EXTRACTION_MODEL,
    because the judge is conceptually a separate role from the
    extractor and operators may want to tune them independently.

    Flag verification: every flag in the vector below was confirmed
    present in `claude --help` for CLI 2.1.118 before this code shipped.
    A missing flag would NOT raise — claude --print silently treats
    unknown options as argv and the subprocess would bucket every
    probe as judge_error with no diagnostic. Re-run the verification
    if upgrading the CLI past 2.x.
    """
    return [
        "claude",
        "--print",
        "--model",
        config.judge_model,
        "--output-format",
        "json",
        "--json-schema",
        json.dumps(_JUDGE_SCHEMA),
        "--max-budget-usd",
        str(config.judge_budget_usd),
        "--system-prompt",
        _JUDGE_SYSTEM_PROMPT,
        "--permission-mode",
        "bypassPermissions",
        "--tools",
        "",
        "--no-session-persistence",
    ]


def _build_gen_cmd(config: BehavioralConfig) -> list[str]:
    """Construct the argv vector for one generator subprocess invocation.

    Same shape as `_build_judge_cmd` with three deliberate differences:

    - `--output-format text` (not `json`): the generator's output is a
      free-form reply that the judge will read; the JSON envelope would
      need to be unwrapped, and we don't need the structured-output
      validator path. Side effect: the per-call cost is NOT recoverable
      from text output (the JSON envelope carries it). The harness
      records `cost_usd` as None for the generator arms; this is a
      documented limitation, not a bug.

    - No `--json-schema`: text output, no schema to validate against.

    - `--system-prompt ""` (literally empty, not omitted). Omitting the
      flag would let the CLI fall back to its default system prompt,
      which is itself a confound. Passing the empty string replaces
      the default with literally nothing. Some default CLI context still
      flows (~340 input tokens), but it is identical between arms
      within one run, so it does not bias A vs B. The
      `claude_cli_version` field in the output JSON lets cross-run
      comparisons detect drift in that residual.
    """
    return [
        "claude",
        "--print",
        "--model",
        config.gen_model,
        "--output-format",
        "text",
        "--max-budget-usd",
        str(config.gen_budget_usd),
        # Literally empty, not omitted. Omitting would let the CLI's
        # default system prompt confound the measurement; passing ""
        # replaces it with the empty string. Verified accepted by
        # claude CLI 2.1.118.
        "--system-prompt",
        "",
        "--permission-mode",
        "bypassPermissions",
        "--tools",
        "",
        "--no-session-persistence",
    ]


def _subprocess_env() -> dict[str, str]:
    """Build the allow-listed env for both judge and generator.

    Reuses the extractor's `_SUBPROCESS_ENV_ALLOWLIST` directly so the
    two subprocess paths cannot drift on which secrets they expose.
    The allow-list deliberately excludes DATABASE_URL, GitHub tokens,
    webhook secrets, etc.: defense in depth against a regression in
    `--tools ""` that would otherwise hand the model the parent's full
    env. Vars absent from the parent (e.g. ANTHROPIC_API_KEY when the
    operator runs on Max-plan OAuth) are simply not forwarded.
    """
    from kai.memory_extraction import _SUBPROCESS_ENV_ALLOWLIST

    return {key: os.environ[key] for key in _SUBPROCESS_ENV_ALLOWLIST if key in os.environ}


async def _run_subprocess(
    *,
    cmd: list[str],
    stdin_payload: str,
    timeout_s: int,
    cwd: Path,
    env: dict[str, str],
) -> tuple[int, bytes, bytes, float]:
    """Run a claude subprocess and return (returncode, stdout, stderr, elapsed_ms).

    Shared by both judge and generator. The kill-on-timeout shape is
    lifted from `kai.memory_extraction._run_extractor:872-884`: on
    `TimeoutError` we kill the process, await its termination, and
    return a non-zero returncode (-1) so the caller's parsing chain
    treats it as a failure without needing a separate timeout signal.
    Returning the returncode rather than raising lets the caller bucket
    the outcome (`judge_error` vs `generation_error`) without nested
    try/except scaffolding.

    elapsed_ms is wall-clock time around `proc.communicate`, captured
    by the loop's monotonic clock so it is unaffected by NTP jumps or
    timezone changes during a long-running sweep.
    """
    # get_running_loop, not get_event_loop: the latter emits a
    # DeprecationWarning in Python 3.10+ when called from inside a
    # coroutine, and this function is only ever called from an async
    # context. Same loop object either way; the explicit form just
    # documents the requirement.
    loop = asyncio.get_running_loop()
    start = loop.time()
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        # Neutral cwd: no CLAUDE.md discovery at subprocess startup.
        # Without this, claude --print would walk up from the harness's
        # cwd and pick up home/.claude/CLAUDE.md (Kai's bot identity:
        # voice rules, persona, scheduling API docs). The judge's
        # evaluations would then be filtered through Kai's persona, and
        # the generator would receive bot-voice priming — both confounds
        # the spec explicitly rules out. Reuses the extractor's neutral
        # cwd rather than introducing a sibling so the two stay aligned.
        cwd=str(cwd),
        env=env,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=stdin_payload.encode("utf-8")),
            timeout=timeout_s,
        )
    except TimeoutError:
        # Catches asyncio.wait_for's timeout. Safe because pyproject.toml
        # pins requires-python = ">=3.13", and asyncio.TimeoutError has
        # been an alias for the builtin TimeoutError since 3.11. On any
        # earlier interpreter (which we do not support) the two classes
        # are distinct and this clause would never fire, leaving the
        # subprocess running. Anyone backporting must change this to
        # `except asyncio.TimeoutError:`.
        # Kill and reap so the subprocess does not become a zombie if
        # the harness keeps running. wait() after kill() is mandatory:
        # without it, asyncio's child-watcher leaks the PID until next
        # event loop tick.
        proc.kill()
        await proc.wait()
        elapsed_ms = (loop.time() - start) * 1000.0
        log.warning("subprocess timed out after %ds: %s", timeout_s, cmd[0])
        # Return -1 + empty buffers so the caller's "non-zero -> error"
        # branch fires uniformly. The actual signal (timeout vs
        # process-level error) is recorded in the log line above; the
        # bucket distinction (judge_error vs generation_error) lives
        # at the caller, which knows which subprocess this was.
        return -1, b"", b"", elapsed_ms
    elapsed_ms = (loop.time() - start) * 1000.0
    # After communicate() returns, returncode is always set (the
    # process has terminated). The explicit None-check defends only
    # against a future asyncio change to that contract; phrasing it
    # this way (rather than `or 0`) makes the intent unambiguous,
    # since `or 0` could read as "coerce a 0 exit code to 0" rather
    # than "treat unset as 0", and a negative exit code from a SIGNAL
    # death must pass through untouched.
    rc = proc.returncode if proc.returncode is not None else 0
    if rc != 0:
        # Surface the subprocess's own diagnostic on a clean failure.
        # Without this, the caller buckets the probe as
        # judge_error / generation_error but the operator has no way to
        # see WHAT the CLI complained about (rate limit, missing flag,
        # OOM kill, auth expiry). Truncate to keep one line per failed
        # call; the full tail is one debug-level rerun away. errors=
        # "replace" because stderr can carry partial UTF-8 from a
        # SIGKILL mid-write.
        log.warning(
            "subprocess exited rc=%d: %s | stderr: %s",
            rc,
            cmd[0],
            stderr[:500].decode("utf-8", errors="replace").strip(),
        )
    return rc, stdout, stderr, elapsed_ms


# ── Judge call + parsing ───────────────────────────────────────────


def _render_judge_user_payload(
    *,
    question: str,
    ground_truth_text: str,
    response_a: str,
    response_b: str,
) -> str:
    """Render the per-probe stdin payload for the judge.

    Pure function so the snapshot test can pin the rendering against
    drift. Built via string concatenation rather than str.format()
    because response_a / response_b are MODEL-GENERATED text and
    str.format walks the template looking for `{name}` placeholders.
    A response containing literal `{response_b}` would silently
    substitute the other arm into itself, producing a corrupted
    judge prompt; `{anything_else}` would raise KeyError mid-format
    and crash the probe with no signal that a real bug occurred. Both
    failure modes are reachable by adversarial memory content. The
    concatenation form has no template substitution at all, so the
    risk is eliminated by construction rather than by trusting that
    no probe will ever hit the edge case.
    """
    return (
        "USER QUESTION:\n"
        + question
        + "\n\n"
        + "GROUND-TRUTH FACT (what the operator has previously told the assistant):\n"
        + ground_truth_text
        + "\n\n"
        + "RESPONSE A:\n"
        + response_a
        + "\n\n"
        + "RESPONSE B:\n"
        + response_b
    )


def _decode_judge_envelope(stdout: bytes) -> dict | None:
    """Decode the claude --output-format json envelope to a dict.

    Single source of truth for the judge-stdout JSON parse; both the
    choice extractor and the cost extractor build on this. Returns
    None on JSONDecodeError or when the top-level value is not a dict
    (claude can emit a JSON list under some error paths). The caller
    decides what an empty/None envelope means for the bucket choice.
    """
    try:
        envelope = json.loads(stdout)
    except json.JSONDecodeError:
        return None
    return envelope if isinstance(envelope, dict) else None


def _extract_judge_choice(envelope: dict) -> tuple[str, str] | None:
    """Validate envelope contents and extract (choice, reasoning).

    Returns None on any contents-level failure (is_error sentinel,
    missing structured_output, invalid enum, missing reasoning).
    Caller buckets None as `judge_error`.

    1. `is_error: true` -> None. The CLI can exit 0 with is_error set
       when a budget cap was hit mid-retry; we treat that as failure
       rather than partial-success-with-noise.
    2. Schema-validated payload nests under `structured_output` (this
       is what `--json-schema` produces, not at the top level;
       discovered by the extractor's smoke test for spec 320 §13.2
       step 5). Missing key or wrong shape -> None.
    3. `choice` not in the four-string enum -> None. The schema's
       enum should already enforce this at the CLI side, but the
       harness defends in depth: future schema-validator regressions
       must not leak invalid choices into the rollup.
    """
    if envelope.get("is_error") is True:
        return None
    structured = envelope.get("structured_output")
    if not isinstance(structured, dict):
        return None
    choice = structured.get("choice")
    reasoning = structured.get("reasoning")
    if not isinstance(choice, str) or choice not in {"A_wins", "B_wins", "tie", "both_wrong"}:
        return None
    if not isinstance(reasoning, str):
        return None
    return choice, reasoning


def _parse_judge_stdout(stdout: bytes) -> tuple[str, str] | None:
    """Convenience wrapper: decode envelope + extract (choice, reasoning).

    Used by the test suite (which passes raw bytes) and as a single
    call point when the caller does not also need the cost field.
    Production's `_run_one_judge` bypasses this and decodes the
    envelope once so the cost extractor can reuse it; see that
    function for the rationale on avoiding the duplicate decode.
    """
    envelope = _decode_judge_envelope(stdout)
    if envelope is None:
        return None
    return _extract_judge_choice(envelope)


def _extract_judge_cost_usd(envelope: dict) -> float | None:
    """Pull the judge's per-call cost from an already-decoded envelope.

    Takes a dict (not raw bytes) so callers that already decoded the
    envelope for the choice extractor do not pay for a second
    json.loads. Returns None if `total_cost_usd` is missing or not
    numeric. The extractor uses `total_cost_usd` at the envelope top
    level (verified via the smoke test referenced in
    memory_extraction's spec 320 §13.2). Different from the generator
    case where text-format output has no cost field at all.
    """
    cost = envelope.get("total_cost_usd")
    return float(cost) if isinstance(cost, (int, float)) else None


# ── Anonymization + outcome rollup ─────────────────────────────────


def _resolve_seed(*, cli_seed: str | None, probes: list[Probe], user_id: str) -> str:
    """Pick the anonymization seed from CLI or default.

    Default: hex prefix of `sha256(probe_set_hash + user_id)`. Two
    properties matter:

    1. Reproducibility within a probe set + user pair: rerunning the
       harness against the same store produces identical arm
       assignments per probe, so a debugger can compare two run JSONs
       probe-by-probe.
    2. Variability across probe sets / users: a different probe set
       (different probe_set_hash) or a different operator (different
       user_id) draws a different shuffle, so the harness does not
       develop a systematic preference for one arm at one position
       across all the operator's runs.

    Operators can override with `--seed <hex>` to force a specific
    shuffle (useful when bisecting "did the swing come from the
    seed change or from a real measurement change?").
    """
    # `is not None` rather than truthy: argparse's contract is that an
    # absent --seed flag yields None, so that is the only value that
    # should fall through to the hash default. An empty string from
    # `--seed ""` is unusual but unambiguous as an explicit override
    # and is honored as such; the hash branch only fires when no flag
    # was passed at all.
    if cli_seed is not None:
        return cli_seed
    h = hashlib.sha256()
    h.update(probe_set_hash(probes).encode("utf-8"))
    h.update(user_id.encode("utf-8"))
    return h.hexdigest()[:16]


def _arm_letter_for_memory(rng: random.Random) -> str:
    """Coin-flip the arm letter ("A" or "B") that carries memory.

    Pulled into a function so tests can patch the RNG and assert that
    a fixed seed produces a fixed sequence. The coin is fair: Python's
    random.choice on a 2-element sequence reduces to one Mersenne-
    Twister call and returns each element with exactly 0.5 probability,
    independent of seed magnitude. Per-probe skew in the win-rate is
    not attributable to this function.
    """
    return rng.choice(("A", "B"))


def _rollup_outcome(*, judge_choice: str, memory_arm_letter: str) -> str:
    """Translate the judge's A/B verdict into memory-on/memory-off terms.

    Truth table:
                          memory in A          memory in B
        A_wins         -> memory_wins          memory_loses
        B_wins         -> memory_loses         memory_wins
        tie            -> tie                  tie
        both_wrong     -> both_wrong           both_wrong
        judge_error    -> judge_error          judge_error
        generation_error -> generation_error   generation_error

    Implemented as nested if/else rather than a lookup table for
    readability. The two error states bypass the A/B mapping (both
    arms lose information together; there is no "memory_wins" when
    we don't have a clean response from one arm).
    """
    if judge_choice in {"judge_error", "generation_error"}:
        return judge_choice
    if judge_choice == "tie":
        return "tie"
    if judge_choice == "both_wrong":
        return "both_wrong"
    if judge_choice == "A_wins":
        return "memory_wins" if memory_arm_letter == "A" else "memory_loses"
    if judge_choice == "B_wins":
        return "memory_wins" if memory_arm_letter == "B" else "memory_loses"
    # Unreachable in practice (caller validates judge_choice via the
    # parsing chain) but a defensive default keeps the type-checker
    # happy and means a future bug in the caller cannot smuggle a
    # garbage string into the aggregation.
    return "judge_error"


# ── Arm prompt assembly ────────────────────────────────────────────


async def build_arm_prompt(question: str, user_id: str, *, memory_enabled: bool) -> str:
    """Build the user prompt for one A/B arm.

    The hypothesis "memory helps generation" only holds if the two
    arms differ in EXACTLY one place: the presence or absence of the
    memory block. Both arms run through this same function; the only
    branch is `memory_enabled`. Every other detail (whitespace, message
    ordering, system context, history) is held identical between the
    two calls so any measurable difference in the responses is
    attributable to memory injection alone.

    Memory-on arm returns either `prefix + "\n\n" + question` (when
    format_context produces a non-empty block) or just `question` (when
    no relevant memories were found — the production path also no-ops
    in that case). Memory-off arm always returns just `question`.

    `prepend_to_prompt` is the production join helper at
    `src/kai/backend.py`; reusing it rather than reimplementing the
    join means the byte-level isolation test (test 1) stays valid
    even if the production join's separator ever changes.
    """
    from kai.backend import prepend_to_prompt
    from kai.memory import format_context

    # Bare user message; nothing else. Deliberately no system prompt,
    # no history, no workspace context — see module docstring for the
    # scope rationale.
    prompt: str = question
    if memory_enabled:
        memory_ctx = await format_context(question, user_id=user_id)
        if memory_ctx:
            # prepend_to_prompt returns str | list; for a str input it
            # always returns str. The cast via assignment is safe because
            # we just passed a str in. type: ignore would be the more
            # honest spelling, but mypy is OK with the runtime type.
            prompt = prepend_to_prompt(prompt, memory_ctx)  # type: ignore[assignment]
    return prompt


# ── Per-probe execution ────────────────────────────────────────────


async def _run_one_arm(
    *,
    config: BehavioralConfig,
    arm_prompt: str,
    cwd: Path,
    env: dict[str, str],
) -> tuple[str | None, float, float | None]:
    """Run one generator arm; return (response_text, latency_ms, cost_usd).

    Returns response_text=None on any failure (non-zero exit, timeout,
    empty stdout); the caller buckets None as `generation_error`.
    cost_usd is always None for text-format output (the JSON envelope
    that carries cost is not produced when --output-format=text), so
    the field is included for schema symmetry with the judge call but
    not populated. Documented limitation per spec §3.3.
    """
    cmd = _build_gen_cmd(config)
    rc, stdout, _stderr, elapsed_ms = await _run_subprocess(
        cmd=cmd,
        stdin_payload=arm_prompt,
        timeout_s=config.gen_timeout_s,
        cwd=cwd,
        env=env,
    )
    if rc != 0:
        return None, elapsed_ms, None
    text = stdout.decode("utf-8", errors="replace").strip()
    if not text:
        # Successful exit with no output is still a broken response —
        # the judge cannot score nothing against the gold fact. Bucket
        # as generation_error rather than feeding empty string into
        # the comparison.
        return None, elapsed_ms, None
    return text, elapsed_ms, None


async def _run_one_judge(
    *,
    config: BehavioralConfig,
    question: str,
    ground_truth_text: str,
    response_a: str,
    response_b: str,
    cwd: Path,
    env: dict[str, str],
) -> tuple[tuple[str, str] | None, float, float | None]:
    """Run one judge call; return ((choice, reasoning) or None, latency_ms, cost_usd).

    None signals any failure in the parsing chain (timeout, non-zero
    exit, malformed JSON, missing structured_output, invalid choice).
    Caller buckets None as `judge_error`.
    """
    cmd = _build_judge_cmd(config)
    payload = _render_judge_user_payload(
        question=question,
        ground_truth_text=ground_truth_text,
        response_a=response_a,
        response_b=response_b,
    )
    rc, stdout, _stderr, elapsed_ms = await _run_subprocess(
        cmd=cmd,
        stdin_payload=payload,
        timeout_s=config.judge_timeout_s,
        cwd=cwd,
        env=env,
    )
    if rc != 0:
        return None, elapsed_ms, None
    # Decode the envelope once and feed it to BOTH the choice extractor
    # and the cost extractor; previously each function called
    # json.loads(stdout) independently, doubling the parse work per
    # judge call. Cost is still recorded when the envelope is
    # well-formed JSON but the contents fail validation (e.g.
    # is_error=True or invalid choice enum) so that operator-side
    # accounting stays accurate even on bucketed-as-judge_error probes.
    envelope = _decode_judge_envelope(stdout)
    if envelope is None:
        return None, elapsed_ms, None
    cost = _extract_judge_cost_usd(envelope)
    parsed = _extract_judge_choice(envelope)
    if parsed is None:
        return None, elapsed_ms, cost
    return parsed, elapsed_ms, cost


async def _run_one_probe(
    *,
    probe: Probe,
    tags: tuple[str, ...],
    user_id: str,
    config: BehavioralConfig,
    rng: random.Random,
    ground_truth_text: str,
    cwd: Path,
    env: dict[str, str],
) -> ProbeOutcome:
    """Drive one probe through both arms + judge; return the rolled-up outcome.

    Concurrency note: this function is called inside a Semaphore
    `acquire/release` by `_run_all_probes`, so the three subprocess
    calls (gen A, gen B, judge) run sequentially within one slot.
    Running the two arms in parallel would shave wall-clock time but
    would also halve the per-probe rate-limit headroom and cost twice
    as much in burst pressure on Anthropic's API; the Semaphore-of-N
    over sequential-within-slot pattern matches the extractor's
    existing per-user discipline.

    Order of operations: pick arm letter -> assemble both arm prompts
    -> run both arms (sequentially) -> if either failed, short-circuit
    to generation_error; otherwise run judge. The short-circuit avoids
    spending a judge call on a probe where one arm is missing a
    response.
    """
    # Coin-flip BEFORE building the prompts so the assignment is
    # logged-deterministic per probe; if both arms fail, we still
    # know which letter would have been memory-on.
    memory_arm_letter = _arm_letter_for_memory(rng)

    # Assemble prompts. Only the memory-on arm calls format_context;
    # the off arm is just the bare question. format_context itself
    # may fail (Mem0 not initialized, store unreadable) — that
    # surfaces as an exception here, propagating to the gather() in
    # _run_all_probes which wraps it as a generation_error outcome.
    memory_on_prompt = await build_arm_prompt(probe.question, user_id, memory_enabled=True)
    memory_off_prompt = await build_arm_prompt(probe.question, user_id, memory_enabled=False)
    arm_prompts: dict[str, str] = {
        memory_arm_letter: memory_on_prompt,
        ("B" if memory_arm_letter == "A" else "A"): memory_off_prompt,
    }

    # Run both arms sequentially (same slot). Order is A then B for
    # log readability, NOT memory-on then memory-off — the latter
    # would leak the arm assignment via timing patterns visible in
    # the bot's API logs.
    response_a, lat_a, cost_a = await _run_one_arm(config=config, arm_prompt=arm_prompts["A"], cwd=cwd, env=env)
    response_b, lat_b, cost_b = await _run_one_arm(config=config, arm_prompt=arm_prompts["B"], cwd=cwd, env=env)

    if response_a is None or response_b is None:
        # One or both arms broken; skip the judge call (cost-saving)
        # and bucket as generation_error. Per-arm latencies are still
        # recorded so the operator can see which side timed out.
        return ProbeOutcome(
            probe=probe,
            tags=tags,
            memory_arm_letter=memory_arm_letter,
            responses={"A": response_a or "", "B": response_b or ""},
            judge_choice="generation_error",
            judge_reasoning="",
            memory_outcome="generation_error",
            latency_ms={"A": lat_a, "B": lat_b, "judge": 0.0},
            cost_usd={"A": cost_a, "B": cost_b, "judge": None},
        )

    judged, lat_judge, cost_judge = await _run_one_judge(
        config=config,
        question=probe.question,
        ground_truth_text=ground_truth_text,
        response_a=response_a,
        response_b=response_b,
        cwd=cwd,
        env=env,
    )
    if judged is None:
        return ProbeOutcome(
            probe=probe,
            tags=tags,
            memory_arm_letter=memory_arm_letter,
            responses={"A": response_a, "B": response_b},
            judge_choice="judge_error",
            judge_reasoning="",
            memory_outcome="judge_error",
            latency_ms={"A": lat_a, "B": lat_b, "judge": lat_judge},
            cost_usd={"A": cost_a, "B": cost_b, "judge": cost_judge},
        )

    choice, reasoning = judged
    return ProbeOutcome(
        probe=probe,
        tags=tags,
        memory_arm_letter=memory_arm_letter,
        responses={"A": response_a, "B": response_b},
        judge_choice=choice,
        judge_reasoning=reasoning,
        memory_outcome=_rollup_outcome(judge_choice=choice, memory_arm_letter=memory_arm_letter),
        latency_ms={"A": lat_a, "B": lat_b, "judge": lat_judge},
        cost_usd={"A": cost_a, "B": cost_b, "judge": cost_judge},
    )


def _make_drift_outcome(probe: Probe, tags: tuple[str, ...]) -> ProbeOutcome:
    """Construct the placeholder ProbeOutcome for a drifted probe.

    Drifted probes (expected_fact_id no longer resolves via get_by_id)
    cannot be scored honestly: there is no gold fact to compare
    against. Surface them as a separate `drift` row so the per-probe
    output preserves the probe identity and the aggregation logic
    knows to exclude them from rate denominators. All response /
    latency / cost fields are zero/None — no subprocess call ran for
    this probe.
    """
    return ProbeOutcome(
        probe=probe,
        tags=tags,
        memory_arm_letter="",
        responses={"A": "", "B": ""},
        judge_choice="drift",
        judge_reasoning="",
        memory_outcome="drift",
        latency_ms={"A": 0.0, "B": 0.0, "judge": 0.0},
        cost_usd={"A": None, "B": None, "judge": None},
    )


# ── Concurrency wrapper ────────────────────────────────────────────


async def _run_all_probes(
    *,
    probes: list[Probe],
    tags_by_id: dict[str, tuple[str, ...]],
    ground_truth_by_id: dict[str, str],
    user_id: str,
    config: BehavioralConfig,
) -> list[ProbeOutcome]:
    """Schedule every probe through a bounded Semaphore and gather.

    Concurrency model: the Semaphore caps how many probes are mid-
    flight at any time. Each acquired slot runs both gen arms plus the
    judge call sequentially (see `_run_one_probe`), so a Semaphore
    capacity of N translates to up to N concurrent claude subprocesses;
    the second subprocess in a slot cannot start until the first has
    returned. The default cap (4) is a tradeoff between throughput and
    Anthropic-side rate-limit headroom; operators on a tight budget
    can drop it to 1 for fully serial execution.

    Per-probe RNG note: each probe gets its own seeded `random.Random`
    derived from (run_seed, expected_fact_id). This keeps arm assignment
    deterministic per probe regardless of the order in which probes
    finish (gather completes in launch order, but completion order in
    a Semaphore-bounded pool is non-deterministic). A single shared
    RNG would couple the assignment to scheduling order, which is
    flaky and undebuggable.

    cwd is set up once at run start (idempotent mkdir) so the per-
    probe subprocess calls do not re-stat the directory each time.
    """
    from kai.memory_extraction import _EXTRACTOR_CWD, _ensure_extractor_cwd

    # Idempotent; matches the extractor's lazy-init convention so a
    # permission failure surfaces as a logged subprocess miss rather
    # than an import-time crash.
    _ensure_extractor_cwd()
    env = _subprocess_env()

    sem = asyncio.Semaphore(config.max_concurrency)

    async def _run_under_semaphore(probe: Probe) -> ProbeOutcome:
        # Per-probe RNG seeded from (run_seed, expected_fact_id), NOT
        # from any positional index. expected_fact_id is the stable
        # per-probe identity that survives drift; seeding off the
        # loop position would silently break the "same probe set +
        # user pair -> same arm assignments" guarantee promised by
        # _resolve_seed (an earlier probe drifting between two runs
        # would shift every subsequent probe's seed, flipping arm
        # letters and making per-probe comparison across runs
        # meaningless). Drifted probes are excluded from this loop
        # entirely; surviving probes keep their own ID regardless of
        # what their neighbors did.
        h = hashlib.sha256()
        h.update(config.seed.encode("utf-8"))
        h.update(probe.expected_fact_id.encode("utf-8"))
        rng = random.Random(int(h.hexdigest()[:16], 16))

        tags = tags_by_id.get(probe.expected_fact_id, ())
        gt = ground_truth_by_id.get(probe.expected_fact_id, "")
        # Defensive: if the operator omitted ground_truth_text AND
        # get_by_id returned a fact whose .text was empty, the judge
        # would receive a blank gold-fact field. Fall back to the
        # probe's notes field (often a one-line gold-fact reminder
        # the operator wrote at probe-authoring time) so the judge
        # has SOMETHING to compare against. This is best-effort; an
        # empty-notes-empty-text combination is genuinely unrecoverable
        # and should be flagged as drift, but Layer 1's detect_drift
        # only checks fact existence, not content.
        if not gt:
            gt = probe.notes
        # Surface the unrecoverable case to the operator: an empty
        # gold field means the judge will produce a verdict that lands
        # in a real outcome bucket (memory_wins / loses / tie /
        # both_wrong) but is unreliable because there is nothing to
        # compare the responses against. detect_drift cannot catch
        # this — it only checks fact existence — so the warning here
        # is the only signal the operator gets that the bucket count
        # is contaminated. No fallback to a synthetic outcome bucket
        # because there is no honest one to choose; the operator must
        # decide whether to fix the probe or accept the noise.
        if not gt:
            log.warning(
                "probe expected_fact_id=%s has empty gold-fact (no ground_truth_text "
                "override, fact.text empty, probe.notes empty); judge verdict will "
                "be unreliable",
                probe.expected_fact_id,
            )

        async with sem:
            try:
                return await _run_one_probe(
                    probe=probe,
                    tags=tags,
                    user_id=user_id,
                    config=config,
                    rng=rng,
                    ground_truth_text=gt,
                    cwd=_EXTRACTOR_CWD,
                    env=env,
                )
            except Exception as e:
                # An unexpected exception (Mem0 not initialized, OS
                # error from the subprocess machinery, etc.) becomes
                # a generation_error rather than crashing the whole
                # run. The harness is meant to produce a result for
                # every probe; one bad probe should not lose the
                # other 25.
                # Log by expected_fact_id (the stable cross-run probe
                # identifier) rather than probe_index. probe_index is
                # the loop index over the drift-FILTERED scored subset,
                # so it does not map cleanly back to the source probes
                # file when any probe drifts. expected_fact_id is also
                # the same key the JSON output uses, so an operator can
                # grep this warning straight against per_probe[].
                log.warning("probe %s raised: %s", probe.expected_fact_id, e)
                return ProbeOutcome(
                    probe=probe,
                    tags=tags,
                    memory_arm_letter="",
                    responses={"A": "", "B": ""},
                    judge_choice="generation_error",
                    judge_reasoning=f"probe execution raised: {type(e).__name__}",
                    memory_outcome="generation_error",
                    latency_ms={"A": 0.0, "B": 0.0, "judge": 0.0},
                    cost_usd={"A": None, "B": None, "judge": None},
                )

    tasks = [_run_under_semaphore(p) for p in probes]
    return await asyncio.gather(*tasks)


# ── Aggregation ────────────────────────────────────────────────────


def _aggregate_outcomes(outcomes: list[ProbeOutcome]) -> dict[str, int]:
    """Tally the six outcome buckets across all per-probe results.

    Drift outcomes are NOT included here — they go to a top-level
    drift_count instead so the sum check (`sum(outcomes) + drift ==
    probe_count`) holds. Initializing every bucket to zero before the
    loop guarantees every key is present in the output JSON regardless
    of whether any probe landed in that bucket; downstream parsers can
    rely on `outcomes.tie` always existing rather than coding `.get(...)`
    everywhere.
    """
    counts: dict[str, int] = {bucket: 0 for bucket in _OUTCOME_BUCKETS}
    for o in outcomes:
        if o.memory_outcome == "drift":
            continue
        counts[o.memory_outcome] += 1
    return counts


def _compute_rates(outcomes_counts: dict[str, int]) -> dict[str, float]:
    """Compute the four rate percentages from the outcome counts.

    Denominator is `scorable = wins + loses + tie + both_wrong`;
    judge_error and generation_error are excluded so a noisy run does
    not artificially deflate the rates that the operator cares about.
    The `max(scorable, 1)` guard handles a fully-error run: a divide-by-
    zero would crash downstream parsers, so we emit zero rates with all
    error counts visible in the bucket dict — the failure mode is
    surfaced through the counts, not through a missing field.

    Tie and both_wrong are reported separately. Combining them would
    hide a critical distinction: 50% both_wrong is a memory failure
    mode (NEITHER response used the fact); 50% tie is memory having
    no effect (both responses used the fact equivalently).
    """
    scorable = (
        outcomes_counts["memory_wins"]
        + outcomes_counts["memory_loses"]
        + outcomes_counts["tie"]
        + outcomes_counts["both_wrong"]
    )
    denom = max(scorable, 1)
    return {
        "win_rate_pct": round(100.0 * outcomes_counts["memory_wins"] / denom, 2),
        "loss_rate_pct": round(100.0 * outcomes_counts["memory_loses"] / denom, 2),
        "tie_rate_pct": round(100.0 * outcomes_counts["tie"] / denom, 2),
        "both_wrong_rate_pct": round(100.0 * outcomes_counts["both_wrong"] / denom, 2),
    }


def _aggregate_by_tag(outcomes: list[ProbeOutcome]) -> dict[str, dict[str, int]]:
    """Per-tag rollup of the four scorable outcome buckets.

    Only includes the four scorable buckets (wins/loses/tie/both_wrong)
    — judge_error and generation_error per tag are not actionable
    information; the operator wants to know "for probes about
    preferences, did memory help?" not "for probes about preferences,
    did the judge fail?" A multi-tag fact appears in every one of its
    tag buckets (mirrors Layer 1's per-tag discipline).
    """
    by_tag: dict[str, dict[str, int]] = {}
    for o in outcomes:
        # Skip drift, error rows — they have no meaningful tag bucket.
        if o.memory_outcome not in {"memory_wins", "memory_loses", "tie", "both_wrong"}:
            continue
        # Probes without tags appear under the "untagged" bucket so
        # operators with mixed-tagging probe sets can still see the
        # sliced view without losing the un-tagged half.
        tags = o.tags if o.tags else ("untagged",)
        for tag in tags:
            bucket = by_tag.setdefault(
                tag,
                {"memory_wins": 0, "memory_loses": 0, "tie": 0, "both_wrong": 0},
            )
            bucket[o.memory_outcome] += 1
    return by_tag


def _pick_qualitative_examples(outcomes: list[ProbeOutcome]) -> dict[str, dict[str, Any]]:
    """Pick one example each for best help / worst hurt / clearest tie.

    Surfaces concrete probes the operator can read in a few seconds.
    Selection is the FIRST matching probe in source order — not the
    "best" by some judge-confidence proxy, because the judge does not
    return a confidence score. Source-order selection is deterministic
    (stable across runs given the same probes file) which matters more
    than picking the most extreme example.

    Empty buckets just emit empty dicts; the operator sees that no
    probe of that category landed in the run. Better than synthesizing
    a placeholder.
    """

    def _example_dict(o: ProbeOutcome) -> dict[str, Any]:
        # Translate arm letters back to memory-on/memory-off response
        # so the operator does not need to mentally apply the
        # de-anonymization. memory_response is the one with memory;
        # no_memory_response is the bare-question response.
        if o.memory_arm_letter == "A":
            mem_resp = o.responses.get("A", "")
            no_mem_resp = o.responses.get("B", "")
        else:
            mem_resp = o.responses.get("B", "")
            no_mem_resp = o.responses.get("A", "")
        return {
            "question": o.probe.question,
            "expected_fact_id": o.probe.expected_fact_id,
            "memory_response": mem_resp,
            "no_memory_response": no_mem_resp,
            "judge_reasoning": o.judge_reasoning,
        }

    examples: dict[str, dict[str, Any]] = {
        "best_help": {},
        "worst_hurt": {},
        "clearest_tie": {},
    }
    for o in outcomes:
        # Independent `if` branches rather than `elif`: each outcome
        # value is mutually exclusive today, but the reader should not
        # have to verify that to confirm correctness. The three buckets
        # are independent slots, and writing them as independent guards
        # makes that obvious without coupling the branch structure to
        # the (unrelated) shape of the memory_outcome enum.
        if o.memory_outcome == "memory_wins" and not examples["best_help"]:
            examples["best_help"] = _example_dict(o)
        if o.memory_outcome == "memory_loses" and not examples["worst_hurt"]:
            examples["worst_hurt"] = _example_dict(o)
        if o.memory_outcome == "tie" and not examples["clearest_tie"]:
            examples["clearest_tie"] = _example_dict(o)
        # Stop scanning once all three are filled; typical case on
        # any non-trivial probe set, and avoids re-walking the list.
        if all(examples.values()):
            break
    return examples


# ── Output JSON envelope ───────────────────────────────────────────


def _capture_claude_cli_version() -> str:
    """Capture `claude --version` output for the run record.

    Used to detect cross-run drift in the residual default context the
    CLI injects even with `--system-prompt ""`. A CLI upgrade between
    runs that changes that default is the most likely cause of a swing
    in win-rate that does not correspond to any code change in Kai or
    in the probe set. Recording the version makes that diagnosable
    without bisecting the CLI's release notes.

    Best-effort: if the CLI is missing, mis-aliased, or `--version`
    fails for any reason, return the literal "unknown". The harness
    can still run (the CLI subprocess invocations that follow will
    surface the real failure with a clearer error) and the output JSON
    just records that the version could not be determined.
    """
    try:
        result = subprocess.run(
            ["claude", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _outcome_to_per_probe_dict(outcome: ProbeOutcome, *, probe_id: int) -> dict[str, Any]:
    """Serialize one ProbeOutcome as the per_probe[] row in the JSON output.

    `probe_id` is the 1-indexed position in the source probes file.
    Carried explicitly rather than derived from the outcomes list index
    because outcomes are stored as `scored + drift` concatenated; using
    the iteration index would mis-label any middle-of-file probe that
    drifted (it would falsely claim end-of-file position). The caller
    must look up the source-file position via expected_fact_id.
    """
    return {
        "probe_id": probe_id,
        "question": outcome.probe.question,
        "expected_fact_id": outcome.probe.expected_fact_id,
        "tags": list(outcome.tags),
        "memory_arm_letter": outcome.memory_arm_letter,
        "responses": dict(outcome.responses),
        "judge_choice": outcome.judge_choice,
        "judge_reasoning": outcome.judge_reasoning,
        "memory_outcome": outcome.memory_outcome,
        # Round latencies to 1ms precision to keep the JSON readable;
        # sub-millisecond timing is meaningless for whole-subprocess
        # calls anyway. cost_usd round to 6 places (sub-cent precision
        # is what Anthropic's billing API reports).
        "latency_ms": {k: round(v, 1) for k, v in outcome.latency_ms.items()},
        "cost_usd": {k: (round(v, 6) if isinstance(v, (int, float)) else None) for k, v in outcome.cost_usd.items()},
    }


def build_output_json(
    *,
    probes: list[Probe],
    outcomes: list[ProbeOutcome],
    drift_count: int,
    config: BehavioralConfig,
    claude_cli_version: str,
) -> dict[str, Any]:
    """Assemble the full output JSON envelope.

    Pure function so tests can construct synthetic ProbeOutcome lists
    and assert the envelope shape without running the subprocess
    machinery. Schema versioned via _OUTPUT_SCHEMA_VERSION; bump it
    when downstream parsers would need to be re-tested.
    """
    counts = _aggregate_outcomes(outcomes)
    rates = _compute_rates(counts)
    by_tag = _aggregate_by_tag(outcomes)
    examples = _pick_qualitative_examples(outcomes)
    # Build a stable {expected_fact_id -> 1-indexed source-file position}
    # map from the ORIGINAL probes list. probe_id needs to identify the
    # row in the operator's probe file, not the position in `outcomes`
    # (which is `scored + drift` concatenated; a middle-of-file probe
    # that drifts would otherwise get an incorrect end-of-file probe_id).
    source_position_by_fact_id = {p.expected_fact_id: i + 1 for i, p in enumerate(probes)}
    return {
        "version": _OUTPUT_SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "probe_set_hash": probe_set_hash(probes),
        "probe_count": len(probes),
        # `attempted_count` is every probe that had a generation
        # subprocess launched, NOT just probes that received a
        # win/loss/tie/both_wrong verdict. It includes judge_error and
        # generation_error buckets, so attempted = probe_count - drift.
        # An operator comparing two runs with very different error
        # counts should look at the per-bucket numbers under `outcomes`
        # rather than treat this as a verdict count.
        "attempted_count": sum(counts.values()),
        "drift_count": drift_count,
        "claude_cli_version": claude_cli_version,
        "judge_model": config.judge_model,
        "gen_model": config.gen_model,
        "seed": config.seed,
        "max_concurrency": config.max_concurrency,
        "outcomes": counts,
        **rates,
        "by_tag": by_tag,
        "qualitative_examples": examples,
        "per_probe": [
            _outcome_to_per_probe_dict(o, probe_id=source_position_by_fact_id[o.probe.expected_fact_id])
            for o in outcomes
        ],
    }


# ── Stdout rendering ───────────────────────────────────────────────


def _render_summary(output: dict[str, Any]) -> str:
    """Human-readable summary of the run.

    Plain text, no color codes — the harness output is meant to be
    pasteable into a bug report or chat without ANSI escapes. Mirrors
    Layer 1's `_render_single_metrics` style.
    """
    counts = output["outcomes"]
    lines = [
        f"Probes: {output['probe_count']} total, {output['attempted_count']} attempted, {output['drift_count']} drifted",
        f"Claude CLI: {output['claude_cli_version']}",
        f"Models: gen={output['gen_model']}, judge={output['judge_model']}",
        f"Seed: {output['seed']} (max_concurrency={output['max_concurrency']})",
        "",
        "Outcomes:",
        f"  memory_wins:      {counts['memory_wins']}",
        f"  memory_loses:     {counts['memory_loses']}",
        f"  tie:              {counts['tie']}",
        f"  both_wrong:       {counts['both_wrong']}",
        f"  judge_error:      {counts['judge_error']}",
        f"  generation_error: {counts['generation_error']}",
        "",
        "Rates (over scorable = wins+loses+tie+both_wrong):",
        f"  win_rate_pct:        {output['win_rate_pct']:.2f}",
        f"  loss_rate_pct:       {output['loss_rate_pct']:.2f}",
        f"  tie_rate_pct:        {output['tie_rate_pct']:.2f}",
        f"  both_wrong_rate_pct: {output['both_wrong_rate_pct']:.2f}",
    ]
    by_tag = output.get("by_tag", {})
    if by_tag:
        lines.append("")
        lines.append("Per-tag (memory_wins / memory_loses / tie / both_wrong):")
        for tag, vals in sorted(by_tag.items()):
            lines.append(f"  {tag}: {vals['memory_wins']}/{vals['memory_loses']}/{vals['tie']}/{vals['both_wrong']}")
    # Surface latency stats if any probe ran subprocesses; skip if
    # everything drifted (zero latencies would be misleading).
    # Exclude judge_error too: a timed-out judge call records the full
    # _DEFAULT_JUDGE_TIMEOUT_S (60s) as its latency, which would
    # ceiling-shift the reported p50/max away from real happy-path
    # judge performance. An operator tuning the timeout or comparing
    # judge models needs the no-failure number; the failure count is
    # already visible under outcomes["judge_error"].
    judge_latencies = [
        p["latency_ms"]["judge"]
        for p in output.get("per_probe", [])
        if p["judge_choice"] not in {"drift", "generation_error", "judge_error"}
    ]
    if judge_latencies:
        lines.append("")
        lines.append(f"Judge latency: p50={statistics.median(judge_latencies):.0f}ms, max={max(judge_latencies):.0f}ms")
    return "\n".join(lines)


# ── CLI ────────────────────────────────────────────────────────────


def _positive_int(raw: str) -> int:
    """argparse `type=` callable that rejects non-positive integers.

    Used for --max-concurrency. asyncio.Semaphore(0) starts with an
    internal counter of zero and the first acquire() blocks forever
    because no task can ever release a slot that was never held; the
    harness would hang silently with no diagnostic. Catching the bad
    value at parse time turns the hang into a one-line argparse error
    before any subprocess is launched. asyncio.Semaphore(-1) raises
    on construction so it is caught later, but rejecting at the
    boundary is cleaner and gives both cases the same error path.
    """
    try:
        value = int(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected integer, got {raw!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1 (got {value})")
    return value


def _build_parser() -> argparse.ArgumentParser:
    """Top-level argparse for `python -m kai.eval.behavioral`."""
    parser = argparse.ArgumentParser(
        prog="python -m kai.eval.behavioral",
        description="Layer 2 end-to-end behavioral A/B evaluation harness.",
    )
    parser.add_argument(
        "--probes",
        required=True,
        type=Path,
        help="Path to probes.jsonl (JSONL format; #-prefixed lines are comments).",
    )
    parser.add_argument(
        "--user-id",
        required=True,
        help="Telegram chat_id (string) whose memory store to evaluate against.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the run JSON to this path (in addition to stdout summary).",
    )
    parser.add_argument(
        "--judge-model",
        default=_DEFAULT_JUDGE_MODEL,
        help=f"Model for the judge call (default: {_DEFAULT_JUDGE_MODEL}).",
    )
    parser.add_argument(
        "--judge-budget-usd",
        type=float,
        default=_DEFAULT_JUDGE_BUDGET_USD,
        help=f"Max USD per judge call (default: {_DEFAULT_JUDGE_BUDGET_USD}).",
    )
    parser.add_argument(
        "--judge-timeout-s",
        type=int,
        default=_DEFAULT_JUDGE_TIMEOUT_S,
        help=f"Wall-clock timeout per judge call (default: {_DEFAULT_JUDGE_TIMEOUT_S}s).",
    )
    parser.add_argument(
        "--gen-model",
        default=_DEFAULT_GEN_MODEL,
        help=f"Model for the generation arms (default: {_DEFAULT_GEN_MODEL}).",
    )
    parser.add_argument(
        "--gen-budget-usd",
        type=float,
        default=_DEFAULT_GEN_BUDGET_USD,
        help=f"Max USD per generation arm (default: {_DEFAULT_GEN_BUDGET_USD}).",
    )
    parser.add_argument(
        "--gen-timeout-s",
        type=int,
        default=_DEFAULT_GEN_TIMEOUT_S,
        help=f"Wall-clock timeout per generation arm (default: {_DEFAULT_GEN_TIMEOUT_S}s).",
    )
    parser.add_argument(
        "--seed",
        default=None,
        help=(
            "Hex seed for A/B coin-flip per probe. Default: "
            "sha256(probe_set_hash + user_id)[:16] for reproducibility "
            "across reruns of the same probe set + user pair."
        ),
    )
    parser.add_argument(
        "--max-concurrency",
        type=_positive_int,
        default=_DEFAULT_MAX_CONCURRENCY,
        help=(
            f"Maximum probes in flight simultaneously; subprocesses "
            f"within a probe run sequentially, so this is also the cap "
            f"on concurrent claude subprocesses "
            f"(default: {_DEFAULT_MAX_CONCURRENCY})."
        ),
    )
    return parser


def _initialize_memory() -> bool:
    """Load config and call init_memory(); return True on success.

    Mirrors Layer 1's discipline. The harness runs against the live
    store, so the same init path keeps embedding model, Qdrant
    directory, and history DB settings consistent with the bot
    runtime. Errors print to stderr and the caller exits with status 1.
    """
    try:
        from kai.config import load_config
        from kai.memory import init_memory, is_enabled

        config = load_config()
        init_memory(config)
        if not is_enabled():
            print(
                "eval: memory is not enabled. Set MEMORY_ENABLED=true and verify the store is readable.",
                file=sys.stderr,
            )
            return False
        return True
    except Exception as e:
        print(f"eval: init failed: {e}", file=sys.stderr)
        return False


def _resolve_ground_truth(
    *,
    scored_probes: list[Probe],
    overrides: dict[str, str],
    user_id: str,
) -> dict[str, str]:
    """Resolve ground_truth_text for every scored probe.

    Two-step lookup: first check the per-probe overrides (operator
    explicitly wrote `ground_truth_text` in the probe row), then fall
    back to `memory.get_by_id(...).text` for probes without an override.
    Drifted probes are NOT included here — the caller filters them
    upstream and they would have been excluded from `scored_probes`.

    The duplicate get_by_id call (same fact already fetched once during
    drift detection) is the documented cost of keeping Layer 2 read-only
    against Layer 1's API. At 26 probes and Layer 1's measured p50=9ms
    per get_by_id, the overhead is ~234ms, which is dominated by a
    single subprocess call. Acceptable.

    Synchronous because get_by_id is synchronous; declaring this `async`
    would be misleading (no awaits) and would block the event loop for
    the same ~234ms anyway. The caller invokes it directly (no await)
    before launching the per-probe gather.
    """
    from kai import memory as _mem

    resolved: dict[str, str] = {}
    for p in scored_probes:
        if p.expected_fact_id in overrides:
            resolved[p.expected_fact_id] = overrides[p.expected_fact_id]
            continue
        fact = _mem.get_by_id(user_id=user_id, memory_id=p.expected_fact_id)
        # detect_drift already filtered fact-not-found, so a None here
        # means a race (fact deleted between drift detection and this
        # lookup). Fall back to empty string; the inner _run_one_probe
        # will further fall back to probe.notes before bothering the
        # judge with an empty gold field.
        if fact is not None:
            resolved[p.expected_fact_id] = fact.text or ""
        else:
            resolved[p.expected_fact_id] = ""
    return resolved


async def _run_cli(args: argparse.Namespace) -> int:
    """CLI dispatch.

    Returns process exit code: 0 on success, 1 on init/IO failure,
    2 on probe-file validation failure (distinct so a calling script
    can tell the two apart, mirroring Layer 1's exit-code convention).
    """
    try:
        probes, overrides = load_behavioral_probes(args.probes)
    except (OSError, ValueError) as e:
        print(f"eval: failed to load probes: {e}", file=sys.stderr)
        return 2
    if not probes:
        print(f"eval: no probes loaded from {args.probes}", file=sys.stderr)
        return 2

    if not _initialize_memory():
        return 1

    # Drift detection + tag mapping — reuses Layer 1's helper so the
    # two layers agree on which probes are scorable on a given snapshot.
    scored_probes, drifted_probes, tags_by_id = detect_drift(probes, args.user_id)

    # Ground-truth text lookup for every scored probe, with overrides
    # taking precedence. Done before launching subprocesses so a
    # configuration error (e.g. embedder regression deleting facts
    # mid-run) surfaces here rather than as 26 silent generation errors.
    ground_truth_by_id = _resolve_ground_truth(
        scored_probes=scored_probes,
        overrides=overrides,
        user_id=args.user_id,
    )

    # Materialize the per-run config from CLI args. Done here (not in
    # _build_parser) because _resolve_seed needs the loaded probes.
    config = BehavioralConfig(
        judge_model=args.judge_model,
        judge_budget_usd=args.judge_budget_usd,
        judge_timeout_s=args.judge_timeout_s,
        gen_model=args.gen_model,
        gen_budget_usd=args.gen_budget_usd,
        gen_timeout_s=args.gen_timeout_s,
        seed=_resolve_seed(cli_seed=args.seed, probes=probes, user_id=args.user_id),
        max_concurrency=args.max_concurrency,
    )

    claude_cli_version = _capture_claude_cli_version()

    print(
        f"Running {len(scored_probes)} probes "
        f"({len(drifted_probes)} drifted) with seed={config.seed} "
        f"max_concurrency={config.max_concurrency}",
        file=sys.stderr,
    )

    scored_outcomes = await _run_all_probes(
        probes=scored_probes,
        tags_by_id=tags_by_id,
        ground_truth_by_id=ground_truth_by_id,
        user_id=args.user_id,
        config=config,
    )

    # Concatenate scored + drifted outcomes; the per_probe[] order in
    # the output matches probe-file order WITHIN each bucket but the
    # buckets are concatenated rather than interleaved, because
    # interleaving would require re-keying outcomes by source line
    # and the drift placeholder is cheap to scan past.
    drift_outcomes = [_make_drift_outcome(p, tags_by_id.get(p.expected_fact_id, ())) for p in drifted_probes]
    all_outcomes = scored_outcomes + drift_outcomes

    output = build_output_json(
        probes=probes,
        outcomes=all_outcomes,
        drift_count=len(drifted_probes),
        config=config,
        claude_cli_version=claude_cli_version,
    )

    # Sum check: surface a loud failure here rather than at downstream
    # parsing time, where a missing probe could be silently lost.
    summed = sum(output["outcomes"].values()) + output["drift_count"]
    if summed != output["probe_count"]:
        print(
            f"eval: BUG: outcome counts sum to {summed} but probe_count is {output['probe_count']}",
            file=sys.stderr,
        )
        return 1

    print(_render_summary(output))
    if args.output:
        # The summary above is already on stdout, so the operator sees
        # the run's results even if persistence fails. The exit code
        # flips to 1 so a wrapper script can detect the partial-success
        # state (results computed, but the JSON file was not written:
        # full disk, unwritable path, permissions). This runs after all
        # 78 subprocess calls — losing a full run's worth of data to
        # an unhandled OSError would be a bad-day failure mode.
        try:
            args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
        except OSError as e:
            print(f"\neval: failed to write {args.output}: {e}", file=sys.stderr)
            return 1
        print(f"\nWrote {args.output}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entrypoint for `python -m kai.eval.behavioral`.

    `argv` defaults to `sys.argv[1:]`. Returns the exit code for the
    caller; the `__main__` block below propagates it via sys.exit.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    return asyncio.run(_run_cli(args))


if __name__ == "__main__":
    sys.exit(main())
