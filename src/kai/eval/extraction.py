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
from kai.config import load_config

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


# ── Harness ─────────────────────────────────────────────────────────


# Output schema version. Bumped when the per_probe or aggregate shape
# changes in a way that would break a tool reading prior runs.
_OUTPUT_SCHEMA_VERSION = "1"


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


@dataclass
class ProbeOutcome:
    probe_id: str
    category: str
    v5_facts: list[str]
    v6_facts: list[str]
    expected_outcome: str
    v6_rule_6_rejections_delta: int


def load_probes(path: Path) -> list[Probe]:
    """Parse a JSONL probe file. `#`-prefixed lines are comments."""
    probes: list[Probe] = []
    with path.open() as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            obj = json.loads(line)
            probes.append(
                Probe(
                    probe_id=obj["probe_id"],
                    category=obj["category"],
                    window=obj["window"],
                    expected=obj.get("expected", {}),
                )
            )
    return probes


def _window_to_extractor_args(window: dict) -> tuple[str, str, list[tuple[str, str]]]:
    """Convert a probe window into the positional args
    `_build_extraction_payload` expects.

    `prior` is rendered as a list of `(user_text, assistant_text)`
    pairs, mirroring the production `prior_pairs` shape; an entry
    is omitted when its role's text is missing (the schema allows
    asymmetric prior turns). `current.user` and `current.assistant`
    map directly onto the function's first two positional args.
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
    config,
    *,
    user_id: str,
) -> ProbeOutcome:
    """Run both prompt arms against a single probe and classify the
    outcome. Each arm is a sequential subprocess call so the active
    `_RULE_6_REJECTIONS` counter delta can be attributed to v6.
    """
    user_text, assistant_text, prior_pairs = _window_to_extractor_args(probe.window)
    payload = memory_extraction._build_extraction_payload(
        user_text, assistant_text, candidates=None, prior_pairs=prior_pairs
    )

    pre_count = sum(memory_extraction._RULE_6_REJECTIONS.snapshot().values())

    # v5 arm: thread the pinned prompt. The `confirmed_action` skip
    # in Rule 6 still applies on this arm since the regex is part of
    # the active validator regardless of which prompt produced the
    # candidate facts; that is the right behavior because Rule 6 is
    # the SAME validator the harness is measuring against.
    v5_result = await memory_extraction._run_extractor(
        payload,
        config,
        candidate_ids=set(),
        candidate_metadata={},
        user_id=user_id,
        system_prompt=_PROMPT_V5_PINNED,
    )

    # v6 arm: default kwarg, inherits the active `_EXTRACTION_SYSTEM_PROMPT`.
    v6_result = await memory_extraction._run_extractor(
        payload,
        config,
        candidate_ids=set(),
        candidate_metadata={},
        user_id=user_id,
    )

    post_count = sum(memory_extraction._RULE_6_REJECTIONS.snapshot().values())

    v5_facts = [str(f.get("content", "")) for f in v5_result.facts]
    v6_facts = [str(f.get("content", "")) for f in v6_result.facts]

    expected_outcome = _classify_outcome(probe, v5_facts=v5_facts, v6_facts=v6_facts)

    return ProbeOutcome(
        probe_id=probe.probe_id,
        category=probe.category,
        v5_facts=v5_facts,
        v6_facts=v6_facts,
        expected_outcome=expected_outcome,
        v6_rule_6_rejections_delta=post_count - pre_count,
    )


def _classify_outcome(probe: Probe, *, v5_facts: list[str], v6_facts: list[str]) -> str:
    """Map (category, v5_facts, v6_facts, expected) to a labelled
    outcome string. The labels feed the aggregate counters and
    drive the workflow_drop_rate / durable_preservation_rate
    arithmetic.
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
        if should_extract_any is False and not v6_facts:
            return "workflow_dropped"
        if not v6_violates_must_not and len(v6_facts) < len(v5_facts):
            return "workflow_dropped"
        return "regression"
    if probe.category == "durable-content":
        # The win condition: v6 produced at least one fact carrying
        # the required substrings. A v5-empty probe is not informative
        # for this category either, but a v6-empty probe IS a
        # regression because the durable fact must come through.
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
    rule_6_total = 0
    for o in outcomes:
        rule_6_total += o.v6_rule_6_rejections_delta
        if o.category == "workflow-noise" and o.expected_outcome != "ambiguous":
            workflow_total += 1
            if o.expected_outcome == "workflow_dropped":
                workflow_drops += 1
        elif o.category == "durable-content" and o.expected_outcome != "ambiguous":
            durable_total += 1
            if o.expected_outcome == "durable_preserved":
                durable_preserves += 1
    return {
        "workflow_drop_rate": (workflow_drops / workflow_total) if workflow_total else None,
        "durable_preservation_rate": (durable_preserves / durable_total) if durable_total else None,
        "rule_6_rejections": rule_6_total,
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
    probes = load_probes(probes_path)
    if not probes:
        log.error("no probes loaded from %s", probes_path)
        return 1

    log.info("running %d probes", len(probes))

    outcomes: list[ProbeOutcome] = []
    for probe in probes:
        outcome = await _run_one_probe(probe, config, user_id=args.user_id)
        log.info(
            "probe %s [%s] -> %s",
            outcome.probe_id,
            outcome.category,
            outcome.expected_outcome,
        )
        outcomes.append(outcome)

    aggregate = _aggregate(outcomes)
    probe_set_text = "\n".join(json.dumps(asdict(p), sort_keys=True) for p in probes)

    report = {
        "version": _OUTPUT_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "probe_set_hash": _hash(probe_set_text),
        "v5_prompt_hash": _hash(_PROMPT_V5_PINNED),
        "v6_prompt_hash": _hash(memory_extraction._EXTRACTION_SYSTEM_PROMPT),
        **aggregate,
        "per_probe": [asdict(o) for o in outcomes],
    }

    out_path = Path(args.output)
    out_path.write_text(json.dumps(report, indent=2))
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
        default="2114582497",
        help="user_id passed through to the extractor (cosmetic for "
        "this harness; the harness does not read or write the live "
        "memory store).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the JSON report.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    return asyncio.run(_run_async(args))


__all__ = [
    "_OUTPUT_SCHEMA_VERSION",
    "_PROMPT_V5_PINNED",
    "Probe",
    "ProbeOutcome",
    "load_probes",
    "main",
]


if __name__ == "__main__":
    sys.exit(main())
