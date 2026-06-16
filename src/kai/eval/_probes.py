"""
Shared probe primitives for the eval surface.

Used by the behavioral evaluator (`kai.eval.behavioral`) for its
own probes and by the collision-probe generator
(`kai.eval.gen_collision_probes`) as its unscoped verifier's
ranking helper. The scoped retrieval harness
(`kai.eval.retrieval_scoped`) carries its own parallel
implementations on the richer `ScopedProbe` schema (which adds
`expected_excluded_fact_ids` and `workspace`); the two probe
shapes coexist because the behavioral evaluator and the collision
generator do not need scope information, and a unified type would
either bloat the behavioral loader or strip scope from the scoped
harness.

Five exported names:

- `Probe`: dataclass for one eval probe (`question` +
  `expected_fact_id`, plus optional `source_turn_ts` and `notes`
  bookkeeping).
- `load_probes`: JSONL parser with the `#`-comment extension.
- `probe_set_hash`: stable SHA-256 over sorted (question,
  expected_fact_id) pairs so baseline comparisons can detect
  probe-set drift.
- `detect_drift`: bucket probes by whether their expected fact
  still resolves via `kai.memory.get_by_id`; returns the
  scorable probes, the drifted probes, and a tag mapping for
  downstream per-tag rollups.
- `compute_rank`: 1-indexed position of an expected fact ID in a
  recall payload's hit list, or None if absent.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Probe:
    """A single eval probe.

    `question` is the text fed through the recall pipeline;
    `expected_fact_id` is the Mem0 row ID the question should
    have surfaced. The optional fields (`source_turn_ts`, `notes`)
    are operator bookkeeping - they let an author trace a failing
    probe back to the conversation it was drawn from but do not
    feed scoring.
    """

    question: str
    expected_fact_id: str
    source_turn_ts: str = ""
    notes: str = ""


def load_probes(path: Path) -> list[Probe]:
    """Load and validate probes from a JSONL file.

    Documented format extension: lines that, after lstrip, begin with
    `#` are treated as comments and skipped. This lets operators
    annotate a probe file in-place without an external metadata file.
    Empty/whitespace-only lines are also skipped. Every other line is
    parsed as JSON and must carry `question` (non-empty string) and
    `expected_fact_id` (non-empty string); `source_turn_ts` and
    `notes` are optional.

    Validation errors include the file path and 1-indexed line number
    so an operator with a multi-hundred-line probe set can find the
    bad row immediately. Line order is preserved in the returned list
    so per-probe output rows align with the probe file.
    """
    probes: list[Probe] = []
    raw = path.read_text(encoding="utf-8").splitlines()
    for lineno, line in enumerate(raw, start=1):
        stripped = line.lstrip()
        # Skip comment and blank lines BEFORE attempting JSON parse.
        # Putting the comment check on `lstrip()` (not the raw line)
        # accepts indented annotations like `    # category notes`,
        # which operators may use when grouping probes visually.
        if not stripped or stripped.startswith("#"):
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{lineno}: malformed JSON ({e.msg})") from e
        if not isinstance(obj, dict):
            raise ValueError(f"{path}:{lineno}: expected JSON object, got {type(obj).__name__}")
        # Required fields. Type check both presence and string-ness so
        # an int slip-through (e.g. `expected_fact_id: 42` from a
        # spreadsheet export) fails at load time rather than on the
        # first hit comparison where the type mismatch would silently
        # prevent every match.
        question = obj.get("question")
        expected = obj.get("expected_fact_id")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{path}:{lineno}: 'question' missing or not a non-empty string")
        if not isinstance(expected, str) or not expected.strip():
            raise ValueError(f"{path}:{lineno}: 'expected_fact_id' missing or not a non-empty string")
        source_turn_ts = obj.get("source_turn_ts", "")
        notes = obj.get("notes", "")
        if not isinstance(source_turn_ts, str):
            raise ValueError(f"{path}:{lineno}: 'source_turn_ts' must be a string when present")
        if not isinstance(notes, str):
            raise ValueError(f"{path}:{lineno}: 'notes' must be a string when present")
        probes.append(
            Probe(
                question=question,
                expected_fact_id=expected,
                source_turn_ts=source_turn_ts,
                notes=notes,
            )
        )
    return probes


def probe_set_hash(probes: list[Probe]) -> str:
    """SHA-256 of the sorted (question, expected_fact_id) list.

    Locks a baseline measurement to its probe set: comparing two
    baseline files with different hashes is meaningless even if the
    metric numbers look similar. Sorted to make the hash invariant
    under reordering of the probe file. Source turn timestamps and
    notes are excluded - they are bookkeeping, not scoring inputs.
    """
    pairs = sorted((p.question, p.expected_fact_id) for p in probes)
    blob = json.dumps(pairs, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(blob).hexdigest()


def compute_rank(hits: list[dict[str, Any]], expected_fact_id: str) -> int | None:
    """Return the 1-indexed position of expected_fact_id in hits, or None.

    Relies on the per-hit `id` field in the recall payload (carried
    from MemoryResult.id by `kai.memory.format_context`). Without
    it, this match would have to fall back to fragile
    snippet-substring comparison against the 80-char truncated text.
    """
    for i, h in enumerate(hits):
        if h.get("id") == expected_fact_id:
            return i + 1
    return None


def detect_drift(probes: list[Probe], user_id: str) -> tuple[list[Probe], list[Probe], dict[str, tuple[str, ...]]]:
    """Bucket probes into (scored, drifted) and collect tag mapping.

    Calls `kai.memory.get_by_id` for each probe's expected_fact_id.
    A None return means the fact has been deleted, the probe was
    authored against the wrong user_id (ownership mismatch), the
    fact's source is not "extracted" (filtered by get_by_id by
    design), or the recorded ID was never valid (typo at probe
    authoring). All four collapse to "drift": the probe cannot be
    scored honestly. The tag mapping is built here rather than at
    scoring time so we don't re-fetch the same fact twice; the cost
    is one get_by_id call per probe regardless.

    Returns:
        (scored_probes, drifted_probes, tags_by_probe_id) where
        tags_by_probe_id maps each scored probe's expected_fact_id
        to the tuple of tags the fact carries (empty tuple if the
        fact has none).
    """
    from kai import memory as _mem

    scored: list[Probe] = []
    drifted: list[Probe] = []
    tags_by_id: dict[str, tuple[str, ...]] = {}
    for p in probes:
        fact = _mem.get_by_id(user_id=user_id, memory_id=p.expected_fact_id)
        if fact is None:
            drifted.append(p)
            continue
        scored.append(p)
        # `tags` may be absent or None on older Mem0 rows; defensive
        # `or []` matches the shape used elsewhere in `kai.memory`.
        # Convert to tuple for hash-friendly storage by callers.
        raw_tags = (fact.metadata.get("tags") if fact.metadata else None) or []
        tags_by_id[p.expected_fact_id] = tuple(raw_tags)
    return scored, drifted, tags_by_id
