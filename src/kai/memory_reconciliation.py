"""Private, review-gated reconciliation of existing semantic memory.

The audit is deliberately heuristic and read-only.  It identifies candidates;
it never promotes its own guesses to truth.  A human must complete and seal a
review before ``apply`` will emit canonical fact/episode lifecycle events.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

from kai import memory
from kai.memory import MemoryResult
from kai.workshop.domain import MemoryEpisodeId, PrincipalId, RuntimeProfileId
from kai.workshop.episode_history import EpisodeInput, MemoryEpisodeHistoryService
from kai.workshop.fact_lifecycle import FactRevisionInput, MemoryFactLifecycleService
from kai.workshop.store import WorkshopEventStore
from kai.workshop.temporal_memory import EpisodeFollowupRelationship, classify_legacy_temporal_metadata

AUDIT_KIND = "kai.memory_reconciliation.audit"
REVIEW_KIND = "kai.memory_reconciliation.review"
RECEIPT_KIND = "kai.memory_reconciliation.receipt"
FORMAT_VERSION = 1

DECISIONS = frozenset({"pending", "approve", "reject", "defer"})
APPLICABLE_ACTIONS = frozenset(
    {"adopt_as_current", "adopt_corrected", "keep_first_retract_rest", "expire_all", "record_episode_chain"}
)
_TOKEN = re.compile(r"[a-z0-9]+")
_CURRENT_WORDS = frozenset({"currently", "current", "now", "today", "latest"})


class MemoryReconciliationError(RuntimeError):
    """A reconciliation artifact or requested mutation is unsafe."""


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _document_digest(document: dict[str, Any]) -> str:
    return _digest({key: value for key, value in document.items() if key != "sha256"})


def _private_write(path: Path, value: dict[str, Any], *, immutable: bool) -> None:
    from kai.config import PROJECT_ROOT

    resolved = path.resolve(strict=False)
    if resolved == PROJECT_ROOT.resolve() or resolved.is_relative_to(PROJECT_ROOT.resolve()):
        raise MemoryReconciliationError(
            "Private memory-reconciliation artifacts cannot be written inside the source tree"
        )
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o400 if immutable else 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
    except Exception:
        path.unlink(missing_ok=True)
        raise
    os.chmod(path, 0o400 if immutable else 0o600)


def load_document(path: Path, *, kind: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MemoryReconciliationError(f"Cannot read {kind} artifact: {exc}") from exc
    if not isinstance(value, dict) or value.get("kind") != kind or value.get("version") != FORMAT_VERSION:
        raise MemoryReconciliationError(f"Unsupported {kind} artifact")
    if value.get("sha256") != _document_digest(value):
        raise MemoryReconciliationError(f"{kind} artifact digest does not match its content")
    return value


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _number(value: object, default: float = 0.5) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return max(0.0, min(1.0, float(value)))
    return default


def _tokens(text: str) -> frozenset[str]:
    return frozenset(_TOKEN.findall(text.casefold()))


def _normalized(text: str) -> str:
    return " ".join(_TOKEN.findall(text.casefold()))


def _timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def _row_kind(row: MemoryResult) -> str:
    metadata = row.metadata or {}
    return "episode" if metadata.get("type") == "episode" or metadata.get("source") == "episode" else "fact"


def _row_snapshot(row: MemoryResult) -> dict[str, Any]:
    metadata = dict(row.metadata or {})
    kind = _row_kind(row)
    classification = classify_legacy_temporal_metadata(metadata, memory_kind=kind)
    scope = memory.resolve_memory_scope(metadata)
    return {
        "memory_id": row.id,
        "kind": kind,
        "text": row.text,
        "text_sha256": hashlib.sha256(row.text.encode()).hexdigest(),
        "created_at": row.created_at,
        "updated_at": row.updated_at,
        "scope": scope.scope,
        "project_id": scope.project_id,
        "source": _text(metadata.get("source")),
        "confidence": _number(metadata.get("confidence")),
        "source_receipt_id": _text(metadata.get("source_receipt_id")),
        "source_run_id": _text(metadata.get("source_run_id")),
        "source_message_id": _text(metadata.get("source_message_id")),
        "result_message_id": _text(metadata.get("result_message_id")),
        "backend": _text(metadata.get("backend")),
        "provider": _text(metadata.get("provider")),
        "model": _text(metadata.get("model")),
        "prompt_version": _text(metadata.get("prompt_version")),
        "schema_version": _text(metadata.get("schema_version")),
        "valid_from": _text(metadata.get("valid_from")),
        "valid_until": _text(metadata.get("valid_until")),
        "asserted_at": _text(metadata.get("asserted_at")),
        "observed_at": _text(metadata.get("observed_at")),
        "occurred_from": _text(metadata.get("occurred_from")),
        "occurred_until": _text(metadata.get("occurred_until")),
        "migration_classification": classification.classification.value,
        "migration_gaps": [gap.value for gap in classification.gaps],
        "metadata": metadata,
    }


def _candidate(
    category: str,
    rows: Iterable[dict[str, Any]],
    *,
    proposed_action: dict[str, Any],
    uncertainty: str,
    rationale: str,
) -> dict[str, Any]:
    evidence = sorted(rows, key=lambda item: str(item["memory_id"]))
    state = {
        "category": category,
        "evidence": evidence,
        "proposed_action": proposed_action,
        "uncertainty": uncertainty,
        "rationale": rationale,
    }
    state_digest = _digest(state)
    return {"candidate_id": f"mrc_{state_digest[:32]}", "state_sha256": state_digest, **state}


def _prior_dispositions(directory: Path) -> set[tuple[str, str]]:
    terminal: set[tuple[str, str]] = set()
    if not directory.exists():
        return terminal
    for path in directory.glob("receipt-*.json"):
        try:
            receipt = load_document(path, kind=RECEIPT_KIND)
        except MemoryReconciliationError:
            continue
        for decision in receipt.get("decisions", []):
            if isinstance(decision, dict) and decision.get("disposition") in {"reject", "defer"}:
                candidate_id = decision.get("candidate_id")
                state_sha256 = decision.get("state_sha256")
                if isinstance(candidate_id, str) and isinstance(state_sha256, str):
                    terminal.add((candidate_id, state_sha256))
    return terminal


def build_audit(
    *,
    principal_id: str,
    runtime_profile_id: str,
    rows: list[MemoryResult],
    prior_decisions: set[tuple[str, str]] = frozenset(),
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build a deterministic candidate report without mutating memory."""
    observed_at = (now or datetime.now(UTC)).astimezone(UTC)
    snapshots = [_row_snapshot(row) for row in sorted(rows, key=lambda item: item.id)]
    legacy = [
        row
        for row in snapshots
        if not row["metadata"].get("canonical_memory_revision_id")
        and not row["metadata"].get("canonical_memory_episode_id")
    ]
    facts = [row for row in legacy if row["kind"] == "fact"]
    episodes = [row for row in legacy if row["kind"] == "episode"]
    candidates: list[dict[str, Any]] = []

    exact: dict[tuple[str, str, str | None], list[dict[str, Any]]] = {}
    for row in facts:
        exact.setdefault((_normalized(str(row["text"])), str(row["scope"]), row["project_id"]), []).append(row)
    for grouped in exact.values():
        if len(grouped) > 1:
            keeper = min(grouped, key=lambda item: (str(item["created_at"]), str(item["memory_id"])))
            candidates.append(
                _candidate(
                    "likely_duplicate",
                    grouped,
                    proposed_action={"kind": "keep_first_retract_rest", "keeper_memory_id": keeper["memory_id"]},
                    uncertainty="low",
                    rationale="Normalized content and scope are identical.",
                )
            )

    provenance_groups: dict[str, list[dict[str, Any]]] = {}
    for row in facts:
        key = row["source_receipt_id"] or row["source_message_id"]
        if isinstance(key, str):
            provenance_groups.setdefault(key, []).append(row)
    for grouped in provenance_groups.values():
        if len(grouped) > 1 and len({_normalized(str(item["text"])) for item in grouped}) > 1:
            candidates.append(
                _candidate(
                    "fragmented_claims",
                    grouped,
                    proposed_action={"kind": "manual_edit_required"},
                    uncertainty="medium",
                    rationale="Several distinct facts were extracted from the same source evidence.",
                )
            )

    by_polarity: dict[str, dict[bool, list[dict[str, Any]]]] = {}
    for row in facts:
        tokens = list(_TOKEN.findall(str(row["text"]).casefold()))
        negative = "not" in tokens or "no" in tokens or "never" in tokens
        key = " ".join(token for token in tokens if token not in {"not", "no", "never"})
        by_polarity.setdefault(key, {True: [], False: []})[negative].append(row)
    for values in by_polarity.values():
        if values[True] and values[False]:
            candidates.append(
                _candidate(
                    "possible_contradiction",
                    [*values[False], *values[True]],
                    proposed_action={"kind": "manual_edit_required"},
                    uncertainty="high",
                    rationale="Statements share normalized terms but differ in explicit negation.",
                )
            )

    for row in facts:
        valid_until = _timestamp(row["valid_until"])
        created = _timestamp(row["created_at"])
        tokens = _tokens(str(row["text"]))
        expired = valid_until is not None and valid_until <= observed_at
        aging_current = created is not None and bool(tokens & _CURRENT_WORDS) and (observed_at - created).days >= 180
        if expired or aging_current:
            candidates.append(
                _candidate(
                    "stale_current_state",
                    [row],
                    proposed_action={"kind": "expire_all"},
                    uncertainty="low" if expired else "medium",
                    rationale="The fact is explicitly expired or uses time-sensitive wording and is at least 180 days old.",
                )
            )

    for row in legacy:
        if row["migration_gaps"]:
            candidates.append(
                _candidate(
                    "malformed_provenance",
                    [row],
                    proposed_action={
                        "kind": "record_episode_chain" if row["kind"] == "episode" else "adopt_as_current"
                    },
                    uncertainty="high",
                    rationale="Legacy temporal or provenance fields are incomplete; explicit review is required.",
                )
            )

    episode_groups: dict[str, list[dict[str, Any]]] = {}
    for row in episodes:
        key = row["source_run_id"] or row["source_message_id"]
        if isinstance(key, str):
            episode_groups.setdefault(key, []).append(row)
    for grouped in episode_groups.values():
        if len(grouped) > 1:
            candidates.append(
                _candidate(
                    "related_episodes",
                    grouped,
                    proposed_action={"kind": "record_episode_chain"},
                    uncertainty="medium",
                    rationale="Episodes share the same source run or source message.",
                )
            )

    unique = {candidate["candidate_id"]: candidate for candidate in candidates}
    visible = [
        candidate
        for candidate in sorted(unique.values(), key=lambda item: (item["category"], item["candidate_id"]))
        if (candidate["candidate_id"], candidate["state_sha256"]) not in prior_decisions
    ]
    corpus_state = [
        {"memory_id": row["memory_id"], "text_sha256": row["text_sha256"], "metadata_sha256": _digest(row["metadata"])}
        for row in snapshots
    ]
    audit_id = f"mra_{_digest({'principal': principal_id, 'runtime': runtime_profile_id, 'corpus': corpus_state})[:32]}"
    document: dict[str, Any] = {
        "kind": AUDIT_KIND,
        "version": FORMAT_VERSION,
        "audit_id": audit_id,
        "principal_id": principal_id,
        "runtime_profile_id": runtime_profile_id,
        "generated_at": observed_at.isoformat(timespec="seconds"),
        "read_only": True,
        "corpus_sha256": _digest(corpus_state),
        "corpus_count": len(snapshots),
        "candidate_count": len(visible),
        "suppressed_unchanged_count": len(unique) - len(visible),
        "candidates": visible,
    }
    document["sha256"] = _document_digest(document)
    return document


def corpus_sha256(rows: list[MemoryResult]) -> str:
    """Return the same immutable corpus fingerprint recorded by an audit."""
    snapshots = [_row_snapshot(row) for row in sorted(rows, key=lambda item: item.id)]
    state = [
        {"memory_id": row["memory_id"], "text_sha256": row["text_sha256"], "metadata_sha256": _digest(row["metadata"])}
        for row in snapshots
    ]
    return _digest(state)


def build_review_template(audit: dict[str, Any]) -> dict[str, Any]:
    validate_audit(audit)
    return {
        "kind": REVIEW_KIND,
        "version": FORMAT_VERSION,
        "audit_id": audit["audit_id"],
        "audit_sha256": audit["sha256"],
        "reviewer": None,
        "sealed_at": None,
        "decisions": [
            {
                "candidate_id": candidate["candidate_id"],
                "state_sha256": candidate["state_sha256"],
                "disposition": "pending",
                "action": candidate["proposed_action"],
                "operator_note": "",
            }
            for candidate in audit["candidates"]
        ],
    }


def validate_audit(audit: dict[str, Any]) -> None:
    if audit.get("kind") != AUDIT_KIND or audit.get("version") != FORMAT_VERSION:
        raise MemoryReconciliationError("Unsupported reconciliation audit")
    if audit.get("sha256") != _document_digest(audit):
        raise MemoryReconciliationError("Reconciliation audit digest does not match its content")
    candidates = audit.get("candidates")
    if not isinstance(candidates, list) or len({item.get("candidate_id") for item in candidates}) != len(candidates):
        raise MemoryReconciliationError("Reconciliation audit candidates are malformed")


def seal_review(audit: dict[str, Any], review: dict[str, Any], *, reviewer: str) -> dict[str, Any]:
    validate_audit(audit)
    if review.get("kind") != REVIEW_KIND or review.get("version") != FORMAT_VERSION:
        raise MemoryReconciliationError("Unsupported reconciliation review")
    if review.get("audit_id") != audit["audit_id"] or review.get("audit_sha256") != audit["sha256"]:
        raise MemoryReconciliationError("Review is not bound to this immutable audit")
    decisions = review.get("decisions")
    if not isinstance(decisions, list):
        raise MemoryReconciliationError("Review decisions must be a list")
    expected = {item["candidate_id"]: item for item in audit["candidates"]}
    if len(decisions) != len(expected) or {
        item.get("candidate_id") for item in decisions if isinstance(item, dict)
    } != set(expected):
        raise MemoryReconciliationError("Review must contain exactly one decision for every candidate")
    approved_memory: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for raw in decisions:
        if not isinstance(raw, dict):
            raise MemoryReconciliationError("Review decision is malformed")
        candidate = expected[str(raw["candidate_id"])]
        if raw.get("state_sha256") != candidate["state_sha256"]:
            raise MemoryReconciliationError("Review candidate state drifted from the audit")
        disposition = raw.get("disposition")
        if disposition not in DECISIONS - {"pending"}:
            raise MemoryReconciliationError("Every candidate must be approved, rejected, or deferred")
        action = raw.get("action")
        if not isinstance(action, dict) or not isinstance(action.get("kind"), str):
            raise MemoryReconciliationError("Every decision requires an action object")
        if disposition == "approve":
            if action["kind"] not in APPLICABLE_ACTIONS:
                raise MemoryReconciliationError("Approved candidates require an applicable lifecycle action")
            _validate_action(candidate, action)
            memory_ids = {str(item["memory_id"]) for item in candidate["evidence"]}
            overlap = approved_memory & memory_ids
            if overlap:
                raise MemoryReconciliationError(f"Approved candidates overlap on memory rows: {sorted(overlap)}")
            approved_memory.update(memory_ids)
        note = raw.get("operator_note", "")
        if not isinstance(note, str):
            raise MemoryReconciliationError("Operator notes must be strings")
        normalized.append(
            {
                "candidate_id": candidate["candidate_id"],
                "state_sha256": candidate["state_sha256"],
                "disposition": disposition,
                "action": action,
                "operator_note": note,
            }
        )
    sealed = {
        "kind": REVIEW_KIND,
        "version": FORMAT_VERSION,
        "audit_id": audit["audit_id"],
        "audit_sha256": audit["sha256"],
        "reviewer": reviewer.strip(),
        "sealed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "decisions": sorted(normalized, key=lambda item: item["candidate_id"]),
    }
    if not sealed["reviewer"]:
        raise MemoryReconciliationError("Reviewer must be non-empty")
    sealed["sha256"] = _document_digest(sealed)
    return sealed


def validate_sealed_review(audit: dict[str, Any], review: dict[str, Any]) -> None:
    """Validate a sealed review independently of the command that minted it."""
    validate_audit(audit)
    if review.get("kind") != REVIEW_KIND or review.get("version") != FORMAT_VERSION:
        raise MemoryReconciliationError("Unsupported sealed reconciliation review")
    if review.get("sha256") != _document_digest(review):
        raise MemoryReconciliationError("Sealed review digest does not match its content")
    if review.get("audit_id") != audit["audit_id"] or review.get("audit_sha256") != audit["sha256"]:
        raise MemoryReconciliationError("Sealed review is not bound to this audit")
    if not _text(review.get("reviewer")) or not _text(review.get("sealed_at")):
        raise MemoryReconciliationError("Review has not been sealed by a human reviewer")
    decisions = review.get("decisions")
    if not isinstance(decisions, list):
        raise MemoryReconciliationError("Sealed review decisions must be a list")
    candidates = {item["candidate_id"]: item for item in audit["candidates"]}
    if len(decisions) != len(candidates) or {
        item.get("candidate_id") for item in decisions if isinstance(item, dict)
    } != set(candidates):
        raise MemoryReconciliationError("Sealed review must cover every audit candidate exactly once")
    approved_memory: set[str] = set()
    for decision in decisions:
        if not isinstance(decision, dict):
            raise MemoryReconciliationError("Sealed review decision is malformed")
        candidate = candidates[str(decision["candidate_id"])]
        if decision.get("state_sha256") != candidate["state_sha256"]:
            raise MemoryReconciliationError("Sealed review candidate state drifted from the audit")
        disposition = decision.get("disposition")
        if disposition not in DECISIONS - {"pending"}:
            raise MemoryReconciliationError("Sealed review contains an incomplete disposition")
        action = decision.get("action")
        if not isinstance(action, dict) or not isinstance(action.get("kind"), str):
            raise MemoryReconciliationError("Sealed review action is malformed")
        if disposition == "approve":
            if action["kind"] not in APPLICABLE_ACTIONS:
                raise MemoryReconciliationError("Approved review action is not applicable")
            _validate_action(candidate, action)
            memory_ids = {str(item["memory_id"]) for item in candidate["evidence"]}
            overlap = approved_memory & memory_ids
            if overlap:
                raise MemoryReconciliationError(f"Approved candidates overlap on memory rows: {sorted(overlap)}")
            approved_memory.update(memory_ids)


def _validate_action(candidate: dict[str, Any], action: dict[str, Any]) -> None:
    rows = candidate["evidence"]
    kind = action["kind"]
    row_kinds = {row["kind"] for row in rows}
    if kind == "record_episode_chain":
        if row_kinds != {"episode"}:
            raise MemoryReconciliationError("Episode action may contain only episode evidence")
        return
    if row_kinds != {"fact"}:
        raise MemoryReconciliationError("Fact lifecycle action may contain only fact evidence")
    if kind == "adopt_corrected":
        replacement = action.get("replacement")
        if not isinstance(replacement, dict):
            raise MemoryReconciliationError("Corrected adoption requires replacement fields")
        permitted = {
            "content",
            "scope_kind",
            "scope_key",
            "confidence",
            "asserted_at",
            "observed_at",
            "valid_from",
            "valid_until",
        }
        if not set(replacement).issubset(permitted):
            raise MemoryReconciliationError("Corrected adoption contains unsupported replacement fields")
        content = replacement.get("content")
        if not isinstance(content, str) or not content.strip() or len(content.strip()) > 16_384:
            raise MemoryReconciliationError("Corrected adoption requires bounded replacement content")
        scope_kind = replacement.get("scope_kind", rows[0].get("scope"))
        scope_key = replacement.get("scope_key", rows[0].get("project_id") or "")
        if scope_kind not in {"global", "project"}:
            raise MemoryReconciliationError("Corrected adoption scope must be global or project")
        if scope_kind == "global" and scope_key not in {None, ""}:
            raise MemoryReconciliationError("Global corrected adoption cannot include a project")
        if scope_kind == "project" and (not isinstance(scope_key, str) or not scope_key):
            raise MemoryReconciliationError("Project corrected adoption requires a project")
        confidence = replacement.get("confidence", rows[0].get("confidence", 0.5))
        if not isinstance(confidence, int | float) or isinstance(confidence, bool) or not 0 <= confidence <= 1:
            raise MemoryReconciliationError("Corrected adoption confidence must be between zero and one")
        for field in ("asserted_at", "observed_at", "valid_from", "valid_until"):
            value = replacement.get(field)
            if value is not None and _timestamp(value) is None:
                raise MemoryReconciliationError(f"Corrected adoption {field} must be timezone-aware")
        valid_from = _timestamp(replacement.get("valid_from"))
        valid_until = _timestamp(replacement.get("valid_until"))
        if valid_from is not None and valid_until is not None and valid_until <= valid_from:
            raise MemoryReconciliationError("Corrected adoption validity end must follow its start")
        source_memory_id = action.get("source_memory_id")
        if source_memory_id is not None and source_memory_id not in {row["memory_id"] for row in rows}:
            raise MemoryReconciliationError("Corrected adoption source must belong to the candidate")
        return
    if kind == "keep_first_retract_rest":
        memory_ids = {row["memory_id"] for row in rows}
        if len(memory_ids) < 2 or action.get("keeper_memory_id") not in memory_ids:
            raise MemoryReconciliationError("Duplicate action requires a keeper from a multi-row candidate")


def validate_candidate_action(candidate: dict[str, Any], action: dict[str, Any]) -> None:
    """Validate one operator-selected action before sealing a complete review."""
    if not isinstance(action, dict) or action.get("kind") not in APPLICABLE_ACTIONS:
        raise MemoryReconciliationError("Approved candidates require an applicable lifecycle action")
    _validate_action(candidate, action)


def _parse_time(value: object) -> datetime | None:
    return _timestamp(value)


def _fact_spec(
    row: dict[str, Any],
    *,
    receipt_id: str,
    reason: str,
    action: dict[str, Any] | None = None,
    evidence_rows: list[dict[str, Any]] | None = None,
) -> FactRevisionInput:
    now = datetime.now(UTC)
    metadata = dict(row["metadata"])
    metadata["_canonical_adopt_memory_id"] = row["memory_id"]
    replacement = action.get("replacement", {}) if action is not None else {}
    scope = replacement.get("scope_kind", row["scope"])
    scope = scope if scope in {"global", "project"} else "global"
    project = replacement.get("scope_key", row["project_id"]) if scope == "project" else None
    source_rows = evidence_rows or [row]
    return FactRevisionInput(
        content=str(replacement.get("content", row["text"])).strip(),
        scope_kind=scope,
        scope_key=str(project or ""),
        reason=reason,
        evidence=tuple(
            {"kind": "legacy", "reference_id": str(item["memory_id"]), "sha256": str(item["text_sha256"])}
            for item in source_rows
        )
        + ({"kind": "operator", "reference_id": receipt_id, "sha256": None},),
        vector_metadata=metadata,
        confidence=_number(replacement.get("confidence", row["confidence"])),
        asserted_at=_parse_time(replacement.get("asserted_at", row["asserted_at"])) or now,
        observed_at=_parse_time(replacement.get("observed_at", row["observed_at"])) or now,
        valid_from=_parse_time(replacement.get("valid_from", row["valid_from"])),
        valid_until=_parse_time(replacement.get("valid_until", row["valid_until"])),
        backend=row["backend"],
        provider=row["provider"],
        model=row["model"],
        prompt_version=row["prompt_version"],
        schema_version=row["schema_version"],
        migration_classification="legacy_complete",
    )


def _episode_spec(row: dict[str, Any], *, receipt_id: str) -> EpisodeInput:
    metadata = dict(row["metadata"])
    content = str(row["text"])
    scope = row["scope"] if row["scope"] in {"global", "project"} else "global"
    return EpisodeInput(
        goal=str(metadata.get("goal") or content),
        context=str(metadata.get("context") or content),
        approach=str(metadata.get("approach") or "Legacy episode; approach was not recorded."),
        outcome=str(metadata.get("outcome") or "Legacy episode; outcome was not recorded."),
        outcome_quality=str(metadata.get("outcome_quality") or "unknown"),
        lessons=_text(metadata.get("lessons")),
        tags=tuple(str(value) for value in metadata.get("tags", []) if isinstance(value, str)),
        actors=tuple(str(value) for value in metadata.get("actors", []) if isinstance(value, str)),
        scope_kind=scope,
        scope_key=str(row["project_id"] or "") if scope == "project" else "",
        reason="Operator-reviewed adoption of legacy episode history.",
        evidence=(
            {"kind": "legacy", "reference_id": str(row["memory_id"]), "sha256": str(row["text_sha256"])},
            {"kind": "operator", "reference_id": receipt_id, "sha256": None},
        ),
        vector_metadata=metadata,
        occurred_from=_parse_time(row["occurred_from"]) or _parse_time(row["created_at"]),
        occurred_until=_parse_time(row["occurred_until"]),
        observed_at=_parse_time(row["observed_at"]) or datetime.now(UTC),
        backend=row["backend"],
        provider=row["provider"],
        model=row["model"],
        prompt_version=row["prompt_version"],
        schema_version=row["schema_version"],
        migration_classification="legacy_complete",
    )


async def apply_review(
    *,
    db_path: Path,
    audit: dict[str, Any],
    review: dict[str, Any],
) -> dict[str, Any]:
    """Apply only sealed approvals through canonical lifecycle services."""
    validate_sealed_review(audit, review)
    candidates = {item["candidate_id"]: item for item in audit["candidates"]}
    receipt_id = f"mrr_{str(review['sha256'])[:32]}"
    store = await WorkshopEventStore.open(db_path)
    applied: list[dict[str, Any]] = []
    try:
        fact_service = MemoryFactLifecycleService(store)
        episode_service = MemoryEpisodeHistoryService(store)
        principal = PrincipalId(str(audit["principal_id"]))
        runtime = RuntimeProfileId(str(audit["runtime_profile_id"]))
        fact_authority = await fact_service.authority_for(principal, runtime)
        episode_authority = await episode_service.authority_for(principal, runtime)
        for decision in review["decisions"]:
            if decision["disposition"] != "approve":
                continue
            candidate = candidates[decision["candidate_id"]]
            action = decision["action"]
            rows = list(candidate["evidence"])
            kind = str(action["kind"])
            if kind == "record_episode_chain":
                if any(row["kind"] != "episode" for row in rows):
                    raise MemoryReconciliationError("Episode action contains a fact row")
                results = []
                for index, row in enumerate(rows):
                    result = await episode_service.record(
                        episode_authority,
                        _episode_spec(row, receipt_id=receipt_id),
                        idempotency_key=f"memory-reconcile:{receipt_id}:{decision['candidate_id']}:{index}",
                    )
                    results.append(str(result.episode_id))
                for index, (source, target) in enumerate(pairwise(results)):
                    await episode_service.followup(
                        episode_authority,
                        MemoryEpisodeId(source),
                        MemoryEpisodeId(target),
                        EpisodeFollowupRelationship.REVISITED,
                        reason="Operator-reviewed legacy episodes belong to the same historical sequence.",
                        idempotency_key=(f"memory-reconcile:{receipt_id}:{decision['candidate_id']}:followup:{index}"),
                    )
                applied.append({"candidate_id": decision["candidate_id"], "events": results})
                continue
            if any(row["kind"] != "fact" for row in rows):
                raise MemoryReconciliationError("Fact action contains an episode row")
            mutations = []
            if kind == "adopt_corrected":
                source_id = action.get("source_memory_id")
                source = next(
                    (row for row in rows if source_id is None or row["memory_id"] == source_id),
                    rows[0],
                )
                adopted = await fact_service.create(
                    fact_authority,
                    _fact_spec(
                        source,
                        receipt_id=receipt_id,
                        reason="Operator-corrected reconciliation of existing semantic memory.",
                        action=action,
                        evidence_rows=rows,
                    ),
                    idempotency_key=f"memory-reconcile:{receipt_id}:{decision['candidate_id']}:corrected",
                    stable_claim_key=f"reconciled:{decision['candidate_id']}",
                )
                applied.append({"candidate_id": decision["candidate_id"], "events": [str(adopted.revision_id)]})
                continue
            keeper = action.get("keeper_memory_id")
            if kind == "keep_first_retract_rest" and keeper not in {row["memory_id"] for row in rows}:
                raise MemoryReconciliationError("Duplicate action keeper is not part of the candidate")
            for index, row in enumerate(rows):
                adopted = await fact_service.create(
                    fact_authority,
                    _fact_spec(
                        row,
                        receipt_id=receipt_id,
                        reason="Operator-reviewed reconciliation of existing semantic memory.",
                    ),
                    idempotency_key=f"memory-reconcile:{receipt_id}:{decision['candidate_id']}:{index}:adopt",
                    stable_claim_key=f"legacy:{row['memory_id']}",
                )
                mutations.append(str(adopted.revision_id))
                retract = kind == "expire_all" or (kind == "keep_first_retract_rest" and row["memory_id"] != keeper)
                if retract:
                    await fact_service.retract(
                        fact_authority,
                        adopted.claim_id,
                        adopted.revision_id,
                        reason="Operator-reviewed reconciliation retired this prior current-state assertion.",
                        idempotency_key=f"memory-reconcile:{receipt_id}:{decision['candidate_id']}:{index}:retire",
                        expired=kind == "expire_all",
                    )
            applied.append({"candidate_id": decision["candidate_id"], "events": mutations})
    finally:
        await store.close()
    receipt: dict[str, Any] = {
        "kind": RECEIPT_KIND,
        "version": FORMAT_VERSION,
        "receipt_id": receipt_id,
        "audit_id": audit["audit_id"],
        "audit_sha256": audit["sha256"],
        "review_sha256": review["sha256"],
        "principal_id": audit["principal_id"],
        "runtime_profile_id": audit["runtime_profile_id"],
        "reviewer": review["reviewer"],
        "applied_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "decisions": review["decisions"],
        "applied": applied,
    }
    receipt["sha256"] = _document_digest(receipt)
    return receipt


def runtime_profiles_for_principal(connection: sqlite3.Connection, principal_id: str) -> tuple[str, ...]:
    rows = connection.execute(
        "SELECT runtime_profile_id FROM runtime_profile_owners WHERE principal_id = ? ORDER BY runtime_profile_id",
        (principal_id,),
    ).fetchall()
    return tuple(str(row[0]) for row in rows)


def write_audit(path: Path, audit: dict[str, Any]) -> None:
    _private_write(path, audit, immutable=True)


def render_audit_markdown(audit: dict[str, Any]) -> str:
    """Render the private JSON authority as a human-reviewable companion."""
    validate_audit(audit)
    lines = [
        "# Memory reconciliation audit",
        "",
        f"- Audit: `{audit['audit_id']}`",
        f"- Principal: `{audit['principal_id']}`",
        f"- Runtime: `{audit['runtime_profile_id']}`",
        f"- Corpus rows: {audit['corpus_count']}",
        f"- Review candidates: {audit['candidate_count']}",
        f"- Unchanged rejected/deferred candidates suppressed: {audit['suppressed_unchanged_count']}",
        "- Mutation status: **read only; no memory was changed**",
        "",
    ]
    for candidate in audit["candidates"]:
        lines.extend(
            [
                f"## {candidate['category']} — `{candidate['candidate_id']}`",
                "",
                f"Uncertainty: **{candidate['uncertainty']}**  ",
                f"Rationale: {candidate['rationale']}  ",
                f"Proposed lifecycle action: `{candidate['proposed_action']['kind']}`",
                "",
            ]
        )
        for row in candidate["evidence"]:
            provenance = (
                "/".join(
                    str(value)
                    for value in (row["backend"], row["provider"], row["model"], row["prompt_version"])
                    if value
                )
                or "unavailable"
            )
            lines.extend(
                [
                    f"### `{row['memory_id']}` ({row['kind']})",
                    "",
                    str(row["text"]),
                    "",
                    f"- Scope: `{row['scope']}`" + (f" / `{row['project_id']}`" if row["project_id"] else ""),
                    f"- Source: `{row['source'] or 'unavailable'}`",
                    f"- Confidence: `{row['confidence']}`",
                    f"- Model provenance: `{provenance}`",
                    f"- Temporal/provenance gaps: `{', '.join(row['migration_gaps']) or 'none'}`",
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def write_private_text(path: Path, content: str, *, immutable: bool = True) -> None:
    from kai.config import PROJECT_ROOT

    resolved = path.resolve(strict=False)
    if resolved == PROJECT_ROOT.resolve() or resolved.is_relative_to(PROJECT_ROOT.resolve()):
        raise MemoryReconciliationError(
            "Private memory-reconciliation artifacts cannot be written inside the source tree"
        )
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400 if immutable else 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    os.chmod(path, 0o400 if immutable else 0o600)


def write_review_template(path: Path, review: dict[str, Any]) -> None:
    _private_write(path, review, immutable=False)


def write_sealed_review(path: Path, review: dict[str, Any]) -> None:
    _private_write(path, review, immutable=True)


def write_receipt(path: Path, receipt: dict[str, Any]) -> None:
    _private_write(path, receipt, immutable=True)


def prior_dispositions(directory: Path) -> set[tuple[str, str]]:
    return _prior_dispositions(directory)
