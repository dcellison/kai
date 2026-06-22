"""Layer 4 evaluation: extractor prompt pre/post comparison.

Subprocess-driven head-to-head of the active extractor prompt against
a pinned baseline. For each probe in a JSONL fixture, the harness
runs `_run_extractor` twice (one arm per prompt) and reports a
confusion matrix: workflow-noise dropped (the v5 → v6 win condition)
versus durable-fact preservation (the regression we must avoid).

The pinned baseline (`_PROMPT_V5_PINNED` below) is the v5 prompt
captured verbatim before #426's prompt edits landed. It is checked
in so the eval is reproducible across machines and across future
prompt revisions; updating it requires capturing a new baseline at
that revision's landing.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from kai import memory_extraction
from kai.config import Config, load_config

log = logging.getLogger(__name__)


# ── Pinned baseline prompt ──────────────────────────────────────────
#
# Verbatim copy of `_EXTRACTION_SYSTEM_PROMPT` as it stood at
# `_EXTRACTION_PROMPT_VERSION = "5"` (PR #415, 2026-04-30). The
# eval harness threads this string into `_run_extractor` via the
# `system_prompt` keyword-only parameter so the v5 arm runs without
# reverting `memory_extraction.py`.
#
# Update this constant ONLY when adding a new pinned baseline (e.g.,
# capturing v6 to compare against a future v7). Do NOT track it with
# the active prompt; that defeats the head-to-head purpose. The
# `tests/test_eval_extraction.py::test_v5_pinned_drift` integration
# test hashes this string against a known-good digest and fails on
# any silent edit.
_PROMPT_V5_PINNED = """You are a memory extraction assistant for Kai, a personal AI agent.
You receive a short conversation window: zero or more PRIOR CONTEXT
exchanges followed by ONE current exchange (a USER message and an
ASSISTANT reply, marked with >>>).

Fact extraction operates ONLY on the current exchange. PRIOR CONTEXT
is for episode classification only; do NOT extract facts from prior
turns.

Your job is to extract stable, high-signal facts worth remembering
across sessions. Return a JSON object matching the provided schema.

STORE these fact types:
- User-stated preferences (e.g., units, timezone, style preferences,
  what the user likes or dislikes).
- Stable facts about the user (e.g., location, role, project names,
  repo names, hardware).
- Decisions the user made in this exchange ("we're going with X",
  "let's use Y", "I decided Z").
- Actions the user confirmed happened, BUT with strict evidence
  rules to prevent laundered hallucinations:
  1. A confirmed_action fact must include a `confirmation_quote`
     field that quotes the exact user text demonstrating the
     confirmation.
  2. The confirmation_quote must be at least 20 characters long
     and must explicitly reference the action being confirmed.
     "I see PR #299 is merged, thanks" is valid evidence.
     "thanks", "ok", "good", "nice", a single emoji, or any
     one-word or two-word affirmation is NOT valid evidence.
     If the user's text is a generic acknowledgment, DO NOT emit
     a confirmed_action fact, regardless of how clearly the
     assistant claimed the action.
  3. The assistant's prior claim alone is never sufficient.
     Without specific, quotable user confirmation that names the
     action, extract nothing.
- Constraints or requirements the user stated ("must be local",
  "must not use external APIs", "never commit without running tests").

IGNORE:
- Assistant self-reports of completed actions unless user-confirmed
  ("I saved the file", "Done", "Created X", "Pushed to main").
  Treat these as unverified until the user confirms.
- Assistant speculation, hypotheticals, or hedging ("I think...",
  "it might be...", "probably...").
- Intermediate reasoning, tool-output summaries, step-by-step plans.
- Transient conversation state (what you are mid-doing, open
  questions, clarifying requests).
- Casual chat, greetings, acknowledgments, thanks without content.
- Anything that contradicts a stated user preference.
- Code snippets, file contents, error messages (store facts about
  them if needed, not the raw text).

CONFIDENCE:
- Only store facts you can phrase as a single clear sentence.
- Each fact must have a concrete subject and predicate. Vague
  impressions do not qualify.
- If nothing qualifies, return {"facts": []}. An empty result is
  correct and preferred over low-quality facts.

FORMAT each fact as:
- content: one sentence, third-person where possible
  ("User prefers Celsius"), past tense for confirmed actions
  ("User confirmed PR #299 was merged on 2026-04-12").
- tags: 1 to 5 lowercase topical tags. Use these preferred tags
  when one fits the fact: preference, decision, fact, constraint,
  confirmed_action, project, location, schedule, relationship.
  Only invent new tags when none of these capture the fact's
  topic; new tags should be single lowercase words or short
  underscore-joined compounds, no punctuation beyond underscores.

  The tag `confirmed_action` is structurally significant: when
  the fact represents a user-confirmed action (with a
  confirmation_quote), you MUST use the literal tag
  `confirmed_action`. NEVER substitute synonyms like
  `confirmation`, `confirmed`, `user_confirmed`, or `confirm` -
  they are NOT recognized by the storage system and will cause
  the fact to be rejected.
- confidence: a number in [0, 1]. Use 0.9+ for direct user statements,
  0.7 for clear user confirmation of an assistant claim, 0.5 for
  paraphrased or implied facts. Do not store below 0.5.
- confirmation_quote: REQUIRED when tags include "confirmed_action",
  MUST be absent otherwise. Must be the verbatim user text that
  confirms the action, minimum 20 characters, and must reference
  the action specifically (not a generic "thanks"). If no such
  quote exists, do not emit the fact.

EPISODE CLASSIFICATION (windowed):
Decide whether the CURRENT exchange (marked with >>>) is the closing
turn of an episode. PRIOR CONTEXT shows the lead-up; it is background,
NEVER the unit being classified.

Set `has_episode: true` ONLY when ALL of the following hold:
1. The CURRENT exchange contains a stated decision, lesson, outcome,
   or resolution. Closure must be visible in the current turn itself.
2. The PRIOR CONTEXT (if non-empty) sets up that closure: a problem,
   a question, a deliberation, an incident in progress.
3. You can quote a fragment from the CURRENT exchange (not from
   prior turns) that signals the closure.

Set `has_episode: false` when:
- The current exchange is itself a question, a request, an analytical
  reply, or a status update with no resolution. Even if prior context
  is rich, an unresolved current turn is not an episode close.
- The current exchange is routine: a single fact lookup, an
  acknowledgment, casual chat.
- Closure exists in prior turns but the current turn moved on to a
  new topic.

When in doubt, prefer false. The cost of a false negative is one
missed episode; the cost of a false positive is a hallucinated
episode entering the memory store.

CONSOLIDATION:
You will sometimes receive an EXISTING FACTS block before the USER/ASSISTANT
exchange. Each existing fact is shown with its id in square brackets,
provenance, and confidence. For each fact you are about to emit, choose one
of three intents:

- "new": the proposed fact is genuinely net-new information. Use this when
  no existing fact covers the same underlying claim, even paraphrased.
  Most facts are new; do not over-eagerly tie facts to existing ids.
  When no EXISTING FACTS block is present, "new" is always the correct
  intent.

- "update_of": the proposed fact ASSERTS THE SAME UNDERLYING CLAIM as an
  existing fact, but with a value that differs (a path changed, a tunable
  was retuned, a project name was renamed) OR with strictly more specific
  information (a confirmed timestamp where there was a vague reference).
  Cite the existing id in `existing_id`. The new fact will REPLACE the
  cited fact. Use this conservatively: only when one fact rendering the
  other obsolete is clearly correct.

- "skip_redundant": the proposed fact is a paraphrase of an existing fact
  with no new information and no contradictory value. Cite the existing id
  in `existing_id`. The new fact will NOT be stored. Prefer this over
  "update_of" when the existing fact is already adequate; only use
  "update_of" when the new wording carries information the old wording
  lacks.

Important constraints:
- existing_id MUST be one of the ids shown in the EXISTING FACTS block.
  Do NOT invent ids. If no EXISTING FACTS block is present, or if no
  existing fact matches, use intent "new".
- Each existing id may be referenced by AT MOST ONE proposed fact in this
  batch. Do not split a single update across two proposed facts, and do
  not have two proposed facts both update the same existing fact.
- A confirmed_action fact is always "new" (a confirmation is a fresh
  observation about reality, even if the wording paraphrases an existing
  fact). Never emit "skip_redundant" or "update_of" for a confirmed_action;
  always store it as a separate "new" fact so the timestamp record stays
  intact.
"""


# ── Pinned baseline prompt: v6 ──────────────────────────────────────
#
# Verbatim copy of `_EXTRACTION_SYSTEM_PROMPT` as it stood at
# `_EXTRACTION_PROMPT_VERSION = "6"` (PR #427, 2026-04-30), captured
# immediately before the v7 EPISODE CLASSIFICATION edits land. Allows
# v6-vs-v7 head-to-head measurement of the episode-classifier
# tightening without reverting `memory_extraction.py`.
#
# Selected at runtime via the `--baseline {v5,v6}` CLI flag in
# `main()`; default is `v6` so each new prompt revision compares
# against the immediately prior one. The v5 baseline stays available
# for cross-revision sanity checks.
#
# Same maintenance contract as `_PROMPT_V5_PINNED`: do NOT update to
# track the active prompt; capture a new pinned constant when adding
# a new baseline. The
# `tests/test_eval_extraction.py::test_v6_pinned_drift` integration
# test hashes this string against a known-good digest and fails on
# any silent edit.
_PROMPT_V6_PINNED = """You are a memory extraction assistant for Kai, a personal AI agent.
You receive a short conversation window: zero or more PRIOR CONTEXT
exchanges followed by ONE current exchange (a USER message and an
ASSISTANT reply, marked with >>>).

Fact extraction operates ONLY on the current exchange. PRIOR CONTEXT
is for episode classification only; do NOT extract facts from prior
turns.

Your job is to extract stable, high-signal facts worth remembering
across sessions. Return a JSON object matching the provided schema.

STORE these fact types:
- User-stated preferences (e.g., units, timezone, style preferences,
  what the user likes or dislikes).
- Stable facts about the user (e.g., location, role, project names,
  repo names, hardware).
- Architectural or design decisions the user made in this exchange
  with durable scope ("we're going with async architecture",
  "default to option B in v1 of the spec", "the home workspace
  must be per-user, not shared"). NOT workflow micro-decisions
  about which task to do next, which spec to evaluate, or which
  issue to file - those are transient session activity. Apply the
  DURABILITY TEST below.
- Actions the user confirmed happened, BUT with strict evidence
  rules to prevent laundered hallucinations:
  1. A confirmed_action fact must include a `confirmation_quote`
     field that quotes the exact user text demonstrating the
     confirmation.
  2. The confirmation_quote must be at least 20 characters long
     and must explicitly reference the action being confirmed.
     "I see PR #299 is merged, thanks" is valid evidence.
     "thanks", "ok", "good", "nice", a single emoji, or any
     one-word or two-word affirmation is NOT valid evidence.
     If the user's text is a generic acknowledgment, DO NOT emit
     a confirmed_action fact, regardless of how clearly the
     assistant claimed the action.
  3. The assistant's prior claim alone is never sufficient.
     Without specific, quotable user confirmation that names the
     action, extract nothing.
- Constraints or requirements the user stated ("must be local",
  "must not use external APIs", "never commit without running tests").

IGNORE:
- Assistant self-reports of completed actions unless user-confirmed
  ("I saved the file", "Done", "Created X", "Pushed to main").
  Treat these as unverified until the user confirms.
- Assistant speculation, hypotheticals, or hedging ("I think...",
  "it might be...", "probably...").
- Intermediate reasoning, tool-output summaries, step-by-step plans.
- Transient conversation state (what you are mid-doing, open
  questions, clarifying requests).
- Workflow-event metadata: spec/PR/issue lifecycle events. Examples
  to NOT extract: "Spec X v3 was approved", "PR Y received a review
  verdict of Z", "All N findings were closed in vM",
  "The evaluation of spec Q produced a verdict of R". The durable
  artifact is the spec/PR/issue itself; events around it are
  transient session activity that loses meaning once the event
  closes.
- "Decisions to do" workflow actions: a decision to file an issue,
  create a spec, request an evaluation, run a test, or perform
  any other workflow action. The artifact produced (the issue,
  the spec, the test result) is the durable fact; the
  decision-to-do is workflow noise. Examples to NOT extract:
  "User decided to file an issue about X", "User requested
  evaluation of spec Y", "User decided to address issue #Z",
  "User confirmed test input triggered the pipeline".
- Casual chat, greetings, acknowledgments, thanks without content.
- Anything that contradicts a stated user preference.
- Code snippets, file contents, error messages (store facts about
  them if needed, not the raw text).

DURABILITY TEST:

Before emitting any fact, ask: "would this still be useful context
in 30 days?" If the answer is no - because the fact captures a
workflow event, a one-off task decision, a status of work in
progress that will have shipped or moved on, or a session-event
metadata fragment - do not emit it. The 30-day test catches the
most common low-quality extraction: session-event metadata that
reads as fact-shaped but loses meaning once the event closes.

If a fact passes IGNORE rules but fails the durability test,
do not emit it.

CONFIDENCE:
- Only store facts you can phrase as a single clear sentence.
- Each fact must have a concrete subject and predicate. Vague
  impressions do not qualify.
- If nothing qualifies, return {"facts": []}. An empty result is
  correct and preferred over low-quality facts.

FORMAT each fact as:
- content: one sentence, third-person where possible
  ("User prefers Celsius"), past tense for confirmed actions
  ("User confirmed PR #299 was merged on 2026-04-12").
- tags: 1 to 5 lowercase topical tags. Use these preferred tags
  when one fits the fact: preference, decision, fact, constraint,
  confirmed_action, project, location, schedule, relationship.
  Only invent new tags when none of these capture the fact's
  topic; new tags should be single lowercase words or short
  underscore-joined compounds, no punctuation beyond underscores.

  The tag `confirmed_action` is structurally significant: when
  the fact represents a user-confirmed action (with a
  confirmation_quote), you MUST use the literal tag
  `confirmed_action`. NEVER substitute synonyms like
  `confirmation`, `confirmed`, `user_confirmed`, or `confirm` -
  they are NOT recognized by the storage system and will cause
  the fact to be rejected.
- confidence: a number in [0, 1]. Use 0.9+ for direct user statements,
  0.7 for clear user confirmation of an assistant claim, 0.5 for
  paraphrased or implied facts. Do not store below 0.5.
- confirmation_quote: REQUIRED when tags include "confirmed_action",
  MUST be absent otherwise. Must be the verbatim user text that
  confirms the action, minimum 20 characters, and must reference
  the action specifically (not a generic "thanks"). If no such
  quote exists, do not emit the fact.

EPISODE CLASSIFICATION (windowed):
Decide whether the CURRENT exchange (marked with >>>) is the closing
turn of an episode. PRIOR CONTEXT shows the lead-up; it is background,
NEVER the unit being classified.

Set `has_episode: true` ONLY when ALL of the following hold:
1. The CURRENT exchange contains a stated decision, lesson, outcome,
   or resolution. Closure must be visible in the current turn itself.
2. The PRIOR CONTEXT (if non-empty) sets up that closure: a problem,
   a question, a deliberation, an incident in progress.
3. You can quote a fragment from the CURRENT exchange (not from
   prior turns) that signals the closure.

Set `has_episode: false` when:
- The current exchange is itself a question, a request, an analytical
  reply, or a status update with no resolution. Even if prior context
  is rich, an unresolved current turn is not an episode close.
- The current exchange is routine: a single fact lookup, an
  acknowledgment, casual chat.
- Closure exists in prior turns but the current turn moved on to a
  new topic.

When in doubt, prefer false. The cost of a false negative is one
missed episode; the cost of a false positive is a hallucinated
episode entering the memory store.

CONSOLIDATION:
You will sometimes receive an EXISTING FACTS block before the USER/ASSISTANT
exchange. Each existing fact is shown with its id in square brackets,
provenance, and confidence. For each fact you are about to emit, choose one
of three intents:

- "new": the proposed fact is genuinely net-new information. Use this when
  no existing fact covers the same underlying claim, even paraphrased.
  Most facts are new; do not over-eagerly tie facts to existing ids.
  When no EXISTING FACTS block is present, "new" is always the correct
  intent.

- "update_of": the proposed fact ASSERTS THE SAME UNDERLYING CLAIM as an
  existing fact, but with a value that differs (a path changed, a tunable
  was retuned, a project name was renamed) OR with strictly more specific
  information (a confirmed timestamp where there was a vague reference).
  Cite the existing id in `existing_id`. The new fact will REPLACE the
  cited fact. Use this conservatively: only when one fact rendering the
  other obsolete is clearly correct.

- "skip_redundant": the proposed fact is a paraphrase of an existing fact
  with no new information and no contradictory value. Cite the existing id
  in `existing_id`. The new fact will NOT be stored. Prefer this over
  "update_of" when the existing fact is already adequate; only use
  "update_of" when the new wording carries information the old wording
  lacks.

Important constraints:
- existing_id MUST be one of the ids shown in the EXISTING FACTS block.
  Do NOT invent ids. If no EXISTING FACTS block is present, or if no
  existing fact matches, use intent "new".
- Each existing id may be referenced by AT MOST ONE proposed fact in this
  batch. Do not split a single update across two proposed facts, and do
  not have two proposed facts both update the same existing fact.
- A confirmed_action fact is always "new" (a confirmation is a fresh
  observation about reality, even if the wording paraphrases an existing
  fact). Never emit "skip_redundant" or "update_of" for a confirmed_action;
  always store it as a separate "new" fact so the timestamp record stays
  intact.
"""


# ── Harness ─────────────────────────────────────────────────────────


# Output schema version. Bumped when the per_probe or aggregate shape
# changes in a way that would break a tool reading prior runs. v2
# (issue #428) renamed `v5_prompt_hash`/`v6_prompt_hash` to
# `baseline_prompt_hash`/`active_prompt_hash` because the historical
# names lied at the new `--baseline v6` default (the field "v5" was
# carrying a v6 hash).
_OUTPUT_SCHEMA_VERSION = "2"


@dataclass
class Probe:
    """One labelled probe from `extraction-probes.jsonl`.

    `category` selects the success criterion: workflow-noise probes
    are scored on whether v6 dropped extractions that v5 produced;
    durable-content probes are scored on whether v6 preserved them.
    `expected.must_not_contain` (workflow-noise) and
    `expected.must_contain` (durable-content) carry per-probe
    substring assertions for finer-grained outcome classification.
    """

    probe_id: str
    category: str
    window: dict
    expected: dict


# Closed set of categories recognised by `_classify_outcome`. Used
# at load time by `load_probes` to reject typo'd category strings
# before the harness starts running, since the classifier silently
# returns "ambiguous" for unknown categories.
_VALID_CATEGORIES: frozenset[str] = frozenset({"workflow-noise", "durable-content"})


@dataclass
class ProbeOutcome:
    """Per-probe result of running v5 and v6 extractor arms.

    `outcome` is the v6 classification, NOT what the probe expected.
    Possible values: `"workflow_dropped"`, `"durable_preserved"`,
    `"regression"`, `"ambiguous"`, `"error"`. The first two are
    wins; `regression` is a v6 failure; `ambiguous` is an
    uninformative probe (e.g., v5 was empty so we cannot say v6
    "dropped" anything); `error` is a probe that raised mid-run
    and was captured rather than aborting the whole batch.
    """

    probe_id: str
    category: str
    v5_facts: list[str]
    v6_facts: list[str]
    outcome: str
    # Rule 6 fires per-arm; the harness reports both deltas
    # separately so the v6 number is not inflated by v5's
    # higher rejection rate (v5 produces more workflow-event
    # content by design).
    v5_rule_6_rejections_delta: int
    v6_rule_6_rejections_delta: int


def load_probes(path: Path) -> list[Probe]:
    """Parse a JSONL probe file. `#`-prefixed lines are comments."""
    probes: list[Probe] = []
    with path.open() as f:
        for line_number, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            # Wrap JSON parsing so a malformed fixture line surfaces
            # the file path and line number rather than a bare
            # JSONDecodeError. Operators curate this fixture by
            # hand from session history, so a typo on probe 12 of
            # 20 should not produce a traceback that buries which
            # probe broke. Same shape applies to a missing required
            # field (probe_id / category / window).
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            try:
                probe = Probe(
                    probe_id=obj["probe_id"],
                    category=obj["category"],
                    window=obj["window"],
                    expected=obj.get("expected", {}),
                )
            except KeyError as exc:
                raise ValueError(f"{path}:{line_number}: missing required field {exc}") from exc
            # Validate the window structure at load time so a fixture
            # missing `current` (or with a typo'd inner field like
            # `typo_user` instead of `user`) raises an actionable
            # error here instead of silently routing through
            # `_run_one_probe`'s error-bucket path at run time with
            # empty extractor inputs and an unrecoverable subprocess
            # cost. The harness's `_window_to_extractor_args`
            # tolerates missing fields via `.get("user", "")` for
            # defensive runtime behavior; the load-time check is the
            # operator's diagnostic anchor and catches the typo
            # class the runtime fallback can't surface.
            if not isinstance(probe.window, dict) or "current" not in probe.window:
                raise ValueError(f"{path}:{line_number}: window.current is required")
            current = probe.window["current"]
            if not isinstance(current, dict) or not current.get("user") or not current.get("assistant"):
                raise ValueError(f"{path}:{line_number}: window.current must have non-empty user and assistant")
            # Validate category against the closed set so a typo
            # (e.g. underscore-vs-hyphen `workflow_noise`) raises at
            # load time. The classifier's `if probe.category ==`
            # comparisons would otherwise silently fall through to
            # `ambiguous` for every probe in the file, yielding
            # `None` rates with no indication of why.
            if probe.category not in _VALID_CATEGORIES:
                raise ValueError(
                    f"{path}:{line_number}: unknown category {probe.category!r}"
                    f" (must be one of {sorted(_VALID_CATEGORIES)})"
                )
            probes.append(probe)
    return probes


def _window_to_extractor_args(window: dict) -> tuple[str, str, list[tuple[str, str]]]:
    """Convert a probe window into the positional args
    `_build_extraction_payload` expects.

    `prior` is rendered as a list of `(user_text, assistant_text)`
    pairs, mirroring the production `prior_pairs` shape. Orphan
    turns are silently dropped in BOTH directions: a user turn with
    no following assistant reply is omitted (the user reply has
    nothing to pair with), and an assistant turn that arrives
    before any user turn (`pending_user is None`) is also omitted.
    The schema allows asymmetric prior turns at the JSON layer for
    fixture-author flexibility; the function commits to a paired
    rendering for the production payload, so a fixture author who
    wants an orphan to surface should restructure the window.
    `current.user` and `current.assistant` map directly onto the
    function's first two positional args.
    """
    prior_raw = window.get("prior", []) or []
    prior_pairs: list[tuple[str, str]] = []
    pending_user: str | None = None
    for entry in prior_raw:
        role = entry.get("role")
        text = entry.get("text", "")
        if role == "user":
            pending_user = text
        elif role == "assistant" and pending_user is not None:
            prior_pairs.append((pending_user, text))
            pending_user = None
    current = window.get("current") or {}
    return current.get("user", ""), current.get("assistant", ""), prior_pairs


async def _run_one_probe(
    probe: Probe,
    config: Config,
    *,
    user_id: str,
    baseline_prompt: str = _PROMPT_V6_PINNED,
) -> ProbeOutcome:
    """Run both prompt arms against a single probe and classify the
    outcome. Each arm is a sequential subprocess call so the active
    `_RULE_6_REJECTIONS` counter delta can be attributed per-arm.

    `baseline_prompt` selects which pinned constant the first arm
    runs against. Default tracks the CLI's `--baseline` default
    (currently `v6`) so direct programmatic callers and ad-hoc test
    fixtures get the same baseline as the CLI rather than silently
    regressing to v5. The `v5_facts` / `v6_facts` field names on
    `ProbeOutcome` are historical (the original design measured v5
    vs active=v6); when `baseline_prompt=_PROMPT_V6_PINNED` the
    first-arm fields carry v6 facts and the second-arm fields carry
    v7 (the new active) facts. Operators reading the report should
    consult the per-run `baseline_choice` field for which baseline
    was actually selected.

    Never raises: a subprocess crash, JSON parse failure, or any
    other exception inside either arm is caught and reported as an
    `outcome="error"` ProbeOutcome with whatever per-arm deltas
    were captured before the failure. Specifically, if v5 completes
    cleanly but v6 raises, `v5_rule_6_rejections_delta` carries the
    v5 arm's real delta (already snapshotted before v6 ran) rather
    than zero. This is the right shape for the operator-facing
    report: an attributable accounting of partial progress, not a
    silent loss of v5's rejections to the error bucket.
    """
    # Initialize all per-arm state to zero/empty so an exception in
    # either arm (or the payload-build below) leaves the
    # partial-progress accounting accurate. The control flow inside
    # the try block updates these in place as each step completes.
    v5_facts: list[str] = []
    v6_facts: list[str] = []
    v5_rule_6_delta = 0
    v6_rule_6_delta = 0

    try:
        # Payload construction lives inside the try block so a
        # malformed probe (e.g., a window field that violates an
        # invariant `_build_extraction_payload` does not check
        # defensively) routes through the same error-bucket path
        # as a subprocess crash. Without the guard the docstring's
        # "Never raises" contract would have a hole - the exact
        # failure mode the per-probe try/except exists to prevent.
        user_text, assistant_text, prior_pairs = _window_to_extractor_args(probe.window)
        payload = memory_extraction._build_extraction_payload(
            user_text, assistant_text, candidates=None, prior_pairs=prior_pairs
        )
        # v5 arm: thread the pinned prompt. The `confirmed_action`
        # skip in Rule 6 still applies on this arm since the regex
        # is part of the active validator regardless of which prompt
        # produced the candidate facts; that is the right behavior
        # because Rule 6 is the SAME validator the harness is
        # measuring against.
        #
        # Rule 6 fires on BOTH arms (the active validator runs
        # against whatever facts each arm emits), and v5 is expected
        # to fire it more because v5 produces more workflow-event
        # content. We snapshot the counter around each arm
        # separately so the v6 delta is attributed to v6 only -
        # lumping the two arms together would inflate the v6 metric
        # with v5's more numerous rejections.
        # Per-user dispatch (issue #515): `_run_extractor` now requires
        # an explicit `effective_backend`. The eval harness runs in
        # single-backend mode (one prompt-pair comparison per run, no
        # mixed-backend cascade), so the global `default_backend` is the
        # correct backend for both arms; threading it explicitly avoids
        # relying on a default we no longer have.
        effective_backend = config.default_backend
        pre_v5 = sum(memory_extraction._RULE_6_REJECTIONS.snapshot().values())
        v5_result = await memory_extraction._run_extractor(
            payload,
            config,
            candidate_ids=set(),
            candidate_metadata={},
            user_id=user_id,
            effective_backend=effective_backend,
            system_prompt=baseline_prompt,
        )
        post_v5 = sum(memory_extraction._RULE_6_REJECTIONS.snapshot().values())
        v5_rule_6_delta = post_v5 - pre_v5
        v5_facts = [str(f.get("content", "")) for f in v5_result.facts]

        # v6 arm: default kwarg, inherits the active `_EXTRACTION_SYSTEM_PROMPT`.
        pre_v6 = post_v5
        v6_result = await memory_extraction._run_extractor(
            payload,
            config,
            candidate_ids=set(),
            candidate_metadata={},
            user_id=user_id,
            effective_backend=effective_backend,
        )
        post_v6 = sum(memory_extraction._RULE_6_REJECTIONS.snapshot().values())
        v6_rule_6_delta = post_v6 - pre_v6
        v6_facts = [str(f.get("content", "")) for f in v6_result.facts]

        classified = _classify_outcome(probe, v5_facts=v5_facts, v6_facts=v6_facts)
    except Exception as exc:
        # One probe failure must not abort the whole run; a live
        # 20-probe run wastes meaningful subprocess cost if it
        # bails on a transient JSON parse error or a timeout.
        # Preserve whatever per-arm state we captured before the
        # exception (see the local-variable initialization above)
        # so the report attributes Rule 6 rejections accurately
        # even on partial-completion failures.
        log.exception("probe %s failed: %s", probe.probe_id, exc)
        classified = "error"

    return ProbeOutcome(
        probe_id=probe.probe_id,
        category=probe.category,
        v5_facts=v5_facts,
        v6_facts=v6_facts,
        outcome=classified,
        v5_rule_6_rejections_delta=v5_rule_6_delta,
        v6_rule_6_rejections_delta=v6_rule_6_delta,
    )


def _classify_outcome(probe: Probe, *, v5_facts: list[str], v6_facts: list[str]) -> str:
    """Map (category, v5_facts, v6_facts, expected) to a labelled
    outcome string. The labels feed the aggregate counters and
    drive the workflow_drop_rate / durable_preservation_rate
    arithmetic.

    Substring-matching semantics:

    - `must_not_contain` is OR: v6 fails if ANY listed substring
      appears in any v6 fact. The list enumerates banned
      substrings; one hit is enough to flag a regression.
    - `must_contain` is also OR: v6 succeeds if ANY listed
      substring appears in any v6 fact. The list enumerates
      acceptable anchors and the probe scores `durable_preserved`
      on the first hit. A future fixture author who needs strict
      AND semantics ("v6 must mention BOTH home AND per-user")
      should split into two probes or extend the schema with a
      `must_contain_all` field; bare `any()` is the documented
      behavior here.
    """
    must_not_contain = probe.expected.get("must_not_contain") or []
    must_contain = probe.expected.get("must_contain") or []
    should_extract_any = probe.expected.get("should_extract_any")

    v6_text_lower = " ".join(v6_facts).lower()
    v6_violates_must_not = any(s.lower() in v6_text_lower for s in must_not_contain)
    v6_satisfies_must = not must_contain or any(s.lower() in v6_text_lower for s in must_contain)

    if probe.category == "workflow-noise":
        # The win condition: v6 produced no facts (or no facts
        # carrying the workflow-event substrings the probe forbids).
        # If v5 itself was empty the probe is uninformative ("ambiguous"):
        # we cannot say v6 "dropped" something v5 also did not produce.
        if not v5_facts:
            return "ambiguous"
        # Strict win condition: when the probe declares
        # `should_extract_any: false`, only a fully-empty v6 counts
        # as `workflow_dropped`. Earlier wording let a partial drop
        # (v6 still emitted facts, just none containing the
        # forbidden substrings) score as a win, which inflated
        # workflow_drop_rate on probes where v6 partially-suppresses
        # but the operator asked for zero extraction.
        if should_extract_any is False:
            if not v6_facts:
                return "workflow_dropped"
            return "regression"
        # When the probe does NOT declare should_extract_any: false
        # (no opinion on whether v6 must be empty), a v6 that
        # produced fewer facts than v5 AND avoided every forbidden
        # substring still counts as `workflow_dropped`. This is the
        # softer win condition for probes where some durable signal
        # may legitimately remain.
        if not v6_violates_must_not and len(v6_facts) < len(v5_facts):
            return "workflow_dropped"
        return "regression"
    if probe.category == "durable-content":
        # The win condition: v6 PRESERVED a fact v5 also produced.
        # Symmetric ambiguity checks come first:
        #   - both arms empty: probe window too sparse to inform.
        #   - v5 empty, v6 non-empty: v6 found something v5 missed.
        #     Nothing was "preserved" because v5 had nothing to
        #     start with; treat as ambiguous rather than counting
        #     a v6 strict-improvement as preservation. Mirrors the
        #     workflow-noise branch's `if not v5_facts: return
        #     "ambiguous"` symmetric guard.
        # When v5 produced facts but v6 did not, that IS a
        # regression because the durable fact must come through
        # under v6.
        if not v5_facts:
            return "ambiguous"
        if not v6_facts:
            return "regression"
        if v6_satisfies_must:
            return "durable_preserved"
        return "regression"
    return "ambiguous"


def _aggregate(outcomes: list[ProbeOutcome]) -> dict:
    """Collapse per-probe outcomes into the documented aggregate
    fields (workflow_drop_rate, durable_preservation_rate,
    rule_6_rejections). Rates are computed over the scorable subset
    of each category (excluding `ambiguous`) so a fixture with a few
    uninformative probes does not drag the rate down.
    """
    workflow_total = 0
    workflow_drops = 0
    durable_total = 0
    durable_preserves = 0
    v5_rule_6_total = 0
    v6_rule_6_total = 0
    # Outcomes excluded from rate denominators: `ambiguous` (probe
    # was uninformative) and `error` (probe raised mid-run; see
    # `_run_async` for the per-probe try/except). Both go in the
    # report so the operator sees them, but they do not feed the
    # workflow_drop_rate or durable_preservation_rate arithmetic.
    skip_outcomes = {"ambiguous", "error"}
    for o in outcomes:
        v5_rule_6_total += o.v5_rule_6_rejections_delta
        v6_rule_6_total += o.v6_rule_6_rejections_delta
        if o.category == "workflow-noise" and o.outcome not in skip_outcomes:
            workflow_total += 1
            if o.outcome == "workflow_dropped":
                workflow_drops += 1
        elif o.category == "durable-content" and o.outcome not in skip_outcomes:
            durable_total += 1
            if o.outcome == "durable_preserved":
                durable_preserves += 1
    return {
        "workflow_drop_rate": (workflow_drops / workflow_total) if workflow_total else None,
        "durable_preservation_rate": (durable_preserves / durable_total) if durable_total else None,
        "v5_rule_6_rejections": v5_rule_6_total,
        "v6_rule_6_rejections": v6_rule_6_total,
        "scorable_workflow_count": workflow_total,
        "scorable_durable_count": durable_total,
    }


def _hash(s: str) -> str:
    return "sha256:" + hashlib.sha256(s.encode()).hexdigest()


async def _run_async(args: argparse.Namespace) -> int:
    config = load_config()

    probes_path = Path(args.probes)
    if not probes_path.exists():
        log.error("probes file not found: %s", probes_path)
        return 1
    # `load_probes` raises ValueError on a malformed fixture
    # (invalid JSON, missing required field, missing
    # `window.current`, unknown category). The message already
    # carries `{path}:{line}: ...` context, so logging the
    # exception text and returning non-zero is the right shape
    # for the operator-facing CLI: a parse failure looks the
    # same as the file-not-found and empty-fixture cases above
    # rather than surfacing as a Python traceback through
    # `asyncio.run`.
    try:
        probes = load_probes(probes_path)
    except ValueError as exc:
        log.error("failed to parse probes from %s: %s", probes_path, exc)
        return 1
    if not probes:
        log.error("no probes loaded from %s", probes_path)
        return 1

    # Resolve the baseline arm's prompt from the CLI flag. The dict
    # is the single source of truth for which pinned constants are
    # selectable; the argparse `choices` list mirrors its keys so a
    # mismatch is caught at parse time rather than at execution.
    baseline_prompts: dict[str, str] = {
        "v5": _PROMPT_V5_PINNED,
        "v6": _PROMPT_V6_PINNED,
    }
    baseline_prompt = baseline_prompts[args.baseline]

    log.info("running %d probes (baseline=%s)", len(probes), args.baseline)

    outcomes: list[ProbeOutcome] = []
    for probe in probes:
        # `_run_one_probe` never raises; on exception it returns a
        # ProbeOutcome with `outcome="error"` and whatever per-arm
        # state it captured before the failure. The aggregate skip
        # set keeps error rows out of rate denominators.
        result = await _run_one_probe(probe, config, user_id=args.user_id, baseline_prompt=baseline_prompt)
        log.info(
            "probe %s [%s] -> %s",
            result.probe_id,
            result.category,
            result.outcome,
        )
        outcomes.append(result)

    aggregate = _aggregate(outcomes)
    probe_set_text = "\n".join(json.dumps(asdict(p), sort_keys=True) for p in probes)

    report = {
        "version": _OUTPUT_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "probe_set_hash": _hash(probe_set_text),
        # `baseline_choice` records which pinned constant the baseline
        # arm ran (`v5` or `v6`); the two hash fields below report the
        # baseline arm and active arm prompts respectively. Earlier
        # output-schema-v1 reports carried `v5_prompt_hash` /
        # `v6_prompt_hash` keys; v2 renamed them because the v5/v6
        # labels lied at the new `--baseline v6` default. A downstream
        # tool reading this output should branch on `version` (the
        # schema version) and choose the matching key set.
        "baseline_choice": args.baseline,
        "baseline_prompt_hash": _hash(baseline_prompt),
        "active_prompt_hash": _hash(memory_extraction._EXTRACTION_SYSTEM_PROMPT),
        **aggregate,
        "per_probe": [asdict(o) for o in outcomes],
    }

    out_path = Path(args.output)
    # Guard the write: at $0.32 / 20-probe run, losing the report
    # to a missing parent directory or a read-only output path is
    # an expensive failure. Convert the OSError into a logged
    # error and a non-zero return so the operator sees what went
    # wrong AND can inspect the in-memory aggregate (still printed
    # below) before re-running.
    try:
        out_path.write_text(json.dumps(report, indent=2))
    except OSError as exc:
        log.error("failed to write %s: %s", out_path, exc)
        # Print the aggregate before returning so the caller has
        # the rate numbers even though the JSON file did not land.
        print(json.dumps({k: v for k, v in report.items() if k != "per_probe"}, indent=2))
        return 1
    print(json.dumps({k: v for k, v in report.items() if k != "per_probe"}, indent=2))
    log.info("wrote %s", out_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probes",
        required=True,
        help="Path to the JSONL probe fixture (one probe per line; `#`-prefixed lines are comments).",
    )
    parser.add_argument(
        "--user-id",
        default="eval-harness",
        help="user_id passed through to the extractor (cosmetic for "
        "this harness; the harness does not read or write the live "
        "memory store). The Rule 6 rejection counter is keyed by "
        "user_id, so setting this to a stable label makes a long-"
        "running session's rejection counts easy to attribute.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the JSON report.",
    )
    parser.add_argument(
        "--baseline",
        choices=("v5", "v6"),
        default="v6",
        help="Which pinned prompt the baseline arm runs. Default v6 "
        "compares the active prompt against the immediately prior "
        "revision (v6 vs v7); v5 is retained for cross-revision "
        "sanity checks against the original baseline.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    return asyncio.run(_run_async(args))


# Public API: dataclasses, the JSONL probe loader, and the CLI
# entry point. `_PROMPT_V5_PINNED`, `_OUTPUT_SCHEMA_VERSION`, and
# the helper functions retain their leading-underscore privacy and
# are not exported via `__all__`. Tests reach in via attribute
# access (e.g. `extraction._PROMPT_V5_PINNED`) which bypasses
# `__all__` and is the supported pattern for cross-module pinning.
__all__ = [
    "Probe",
    "ProbeOutcome",
    "load_probes",
    "main",
]


if __name__ == "__main__":
    sys.exit(main())
