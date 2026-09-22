"""Deterministic, non-mutating triage for legacy memory reconciliation.

The original reconciliation audit intentionally over-reports possible
problems.  Its candidates overlap, which makes it useful evidence but a poor
human work queue.  This module converts that evidence into an immutable plan
whose groups partition every legacy memory exactly once.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

from kai import memory_reconciliation

PLAN_KIND = "kai.memory_reconciliation.triage_plan"
PLAN_VERSION = 1
POLICY_VERSION = "legacy_triage_v1"

_TOKEN = re.compile(r"[a-z0-9]+")
_CURRENT_WORDS = frozenset({"currently", "current", "now", "today", "latest"})


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


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


class _Components:
    def __init__(self, values: set[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)

    def groups(self) -> list[list[str]]:
        grouped: dict[str, list[str]] = defaultdict(list)
        for value in sorted(self.parent):
            grouped[self.find(value)].append(value)
        return list(grouped.values())


def _group(
    classification: str,
    evidence: list[dict[str, Any]],
    *,
    resolution: str,
    rationale: str,
    action: dict[str, Any],
    deterministic: bool,
    bulk_eligible: bool,
    prior_review_evidence: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    ordered = sorted(evidence, key=lambda item: str(item["memory_id"]))
    state = {
        "classification": classification,
        "resolution": resolution,
        "rationale": rationale,
        "action": action,
        "evidence": ordered,
        "deterministic": deterministic,
        "bulk_eligible": bulk_eligible,
        "prior_review_evidence": sorted(
            prior_review_evidence or [], key=lambda item: (str(item["candidate_id"]), str(item["disposition"]))
        ),
    }
    state_sha256 = _digest(state)
    return {
        "group_id": f"mtg_{state_sha256[:32]}",
        "state_sha256": state_sha256,
        **state,
    }


def _near_duplicate(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left["scope"] != right["scope"] or left.get("project_id") != right.get("project_id"):
        return False
    left_tokens = _tokens(str(left["text"]))
    right_tokens = _tokens(str(right["text"]))
    if min(len(left_tokens), len(right_tokens)) < 4:
        return False
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens) >= 0.8


def build_triage_plan(
    audit: dict[str, Any],
    *,
    allowed_project_ids: frozenset[str] | None = None,
    prior_review_evidence: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build an immutable partitioned plan from one immutable audit."""
    memory_reconciliation.validate_audit(audit)
    evidence_by_id: dict[str, dict[str, Any]] = {}
    for candidate in audit["candidates"]:
        for item in candidate["evidence"]:
            memory_id = str(item["memory_id"])
            previous = evidence_by_id.setdefault(memory_id, item)
            if previous != item:
                raise memory_reconciliation.MemoryReconciliationError(
                    f"Audit contains conflicting snapshots for memory {memory_id}"
                )

    facts = {key: value for key, value in evidence_by_id.items() if value["kind"] == "fact"}
    episodes = {key: value for key, value in evidence_by_id.items() if value["kind"] == "episode"}
    components = _Components(set(facts))
    edge_types: dict[frozenset[str], set[str]] = defaultdict(set)

    exact: dict[tuple[str, str, str | None], list[str]] = defaultdict(list)
    for memory_id, item in facts.items():
        exact[(_normalized(str(item["text"])), str(item["scope"]), item.get("project_id"))].append(memory_id)
    for members in exact.values():
        for member in members[1:]:
            components.union(members[0], member)
            edge_types[frozenset({members[0], member})].add("exact")

    for candidate in audit["candidates"]:
        if candidate["category"] != "possible_contradiction":
            continue
        members = [str(item["memory_id"]) for item in candidate["evidence"] if str(item["memory_id"]) in facts]
        for member in members[1:]:
            components.union(members[0], member)
            edge_types[frozenset({members[0], member})].add("contradiction")

    fact_ids = sorted(facts)
    for index, left_id in enumerate(fact_ids):
        for right_id in fact_ids[index + 1 :]:
            if _normalized(str(facts[left_id]["text"])) == _normalized(str(facts[right_id]["text"])):
                continue
            if _near_duplicate(facts[left_id], facts[right_id]):
                components.union(left_id, right_id)
                edge_types[frozenset({left_id, right_id})].add("near_duplicate")

    observed_at = _timestamp(audit.get("generated_at")) or datetime.now(UTC)
    groups: list[dict[str, Any]] = []
    safe_adoptions: dict[tuple[str, str | None], list[dict[str, Any]]] = defaultdict(list)
    safe_expirations: dict[tuple[str, str | None], list[dict[str, Any]]] = defaultdict(list)

    for member_ids in components.groups():
        rows = [facts[memory_id] for memory_id in member_ids]
        pairs = [frozenset({left, right}) for index, left in enumerate(member_ids) for right in member_ids[index + 1 :]]
        types = set().union(*(edge_types.get(pair, set()) for pair in pairs)) if pairs else set()
        expired = [
            row for row in rows if (until := _timestamp(row.get("valid_until"))) is not None and until <= observed_at
        ]
        aging = [
            row
            for row in rows
            if _tokens(str(row["text"])) & _CURRENT_WORDS
            and (created := _timestamp(row.get("created_at"))) is not None
            and (observed_at - created).days >= 180
        ]
        uncertain_scope = [row for row in rows if "scope" in row.get("migration_gaps", ())]
        if uncertain_scope:
            groups.append(
                _group(
                    "uncertain_scope",
                    rows,
                    resolution="needs_review",
                    rationale="Legacy scope evidence is incomplete and cannot be inferred safely.",
                    action={"kind": "manual_edit_required"},
                    deterministic=False,
                    bulk_eligible=False,
                )
            )
        elif "contradiction" in types:
            groups.append(
                _group(
                    "contradiction",
                    rows,
                    resolution="needs_review",
                    rationale="The audit found statements with matching claims but opposite polarity.",
                    action={"kind": "manual_edit_required"},
                    deterministic=False,
                    bulk_eligible=False,
                )
            )
        elif len(expired) == len(rows):
            for row in rows:
                safe_expirations[(str(row["scope"]), row.get("project_id"))].append(row)
        elif expired:
            groups.append(
                _group(
                    "validity_conflict",
                    rows,
                    resolution="needs_review",
                    rationale="Related facts disagree about whether the claim is still valid.",
                    action={"kind": "manual_edit_required"},
                    deterministic=False,
                    bulk_eligible=False,
                )
            )
        elif aging:
            groups.append(
                _group(
                    "time_sensitive",
                    rows,
                    resolution="needs_review",
                    rationale="Time-sensitive wording may no longer describe current truth.",
                    action={"kind": "manual_edit_required"},
                    deterministic=False,
                    bulk_eligible=False,
                )
            )
        elif "exact" in types and len({_normalized(str(row["text"])) for row in rows}) == 1:
            keeper = min(rows, key=lambda item: (str(item.get("created_at") or ""), str(item["memory_id"])))
            groups.append(
                _group(
                    "exact_duplicates",
                    rows,
                    resolution="consolidate",
                    rationale="Normalized content and scope are identical; the earliest record is retained.",
                    action={
                        "kind": "keep_first_retract_rest",
                        "keeper_memory_id": keeper["memory_id"],
                        "migration_classification": "legacy_incomplete",
                    },
                    deterministic=True,
                    bulk_eligible=True,
                )
            )
        elif "near_duplicate" in types:
            groups.append(
                _group(
                    "near_duplicates",
                    rows,
                    resolution="needs_review",
                    rationale="The facts are lexically similar but not identical; consolidation requires judgment.",
                    action={"kind": "manual_edit_required"},
                    deterministic=False,
                    bulk_eligible=False,
                )
            )
        else:
            for row in rows:
                safe_adoptions[(str(row["scope"]), row.get("project_id"))].append(row)

    for (_scope, _project_id), rows in sorted(safe_adoptions.items(), key=lambda item: str(item[0])):
        groups.append(
            _group(
                "stable_non_conflicting",
                rows,
                resolution="adopt",
                rationale="No duplicate, contradiction, expiry, or time-sensitive marker was found.",
                action={"kind": "adopt_as_current", "migration_classification": "legacy_incomplete"},
                deterministic=True,
                bulk_eligible=True,
            )
        )
    for (_scope, _project_id), rows in sorted(safe_expirations.items(), key=lambda item: str(item[0])):
        groups.append(
            _group(
                "explicitly_expired",
                rows,
                resolution="obsolete",
                rationale="Every fact has an explicit validity end at or before the audit boundary.",
                action={"kind": "expire_all", "migration_classification": "legacy_incomplete"},
                deterministic=True,
                bulk_eligible=True,
            )
        )

    related_episode_members: list[set[str]] = []
    for candidate in audit["candidates"]:
        if candidate["category"] == "related_episodes":
            related_episode_members.append({str(item["memory_id"]) for item in candidate["evidence"]})
    assigned_episodes: set[str] = set()
    for members in related_episode_members:
        selected = sorted((members & set(episodes)) - assigned_episodes)
        if not selected:
            continue
        assigned_episodes.update(selected)
        groups.append(
            _group(
                "related_episode_history",
                [episodes[memory_id] for memory_id in selected],
                resolution="adopt",
                rationale="The episodes share source evidence and can be retained as an immutable sequence.",
                action={"kind": "record_episode_chain", "migration_classification": "legacy_incomplete"},
                deterministic=True,
                bulk_eligible=True,
            )
        )
    for memory_id in sorted(set(episodes) - assigned_episodes):
        groups.append(
            _group(
                "legacy_episode_history",
                [episodes[memory_id]],
                resolution="adopt",
                rationale="The historical episode is immutable and does not assert current truth.",
                action={"kind": "record_episode_chain", "migration_classification": "legacy_incomplete"},
                deterministic=True,
                bulk_eligible=True,
            )
        )

    prior = prior_review_evidence or []
    prior_by_memory: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for review in prior:
        memory_ids = review.get("memory_ids")
        if not isinstance(memory_ids, list):
            raise memory_reconciliation.MemoryReconciliationError("Prior review evidence is malformed")
        for memory_id in memory_ids:
            prior_by_memory[str(memory_id)].append(review)
    reviewed_groups: list[dict[str, Any]] = []
    split_safe = {"stable_non_conflicting", "explicitly_expired"}
    for group in groups:
        touched = {str(item["memory_id"]) for item in group["evidence"] if str(item["memory_id"]) in prior_by_memory}
        if not touched:
            reviewed_groups.append(group)
            continue
        relevant = {str(item["candidate_id"]): item for memory_id in touched for item in prior_by_memory[memory_id]}
        reviewed_rows = [item for item in group["evidence"] if str(item["memory_id"]) in touched]
        remaining_rows = [item for item in group["evidence"] if str(item["memory_id"]) not in touched]
        if group["classification"] in split_safe and remaining_rows:
            reviewed_groups.append(
                _group(
                    group["classification"],
                    remaining_rows,
                    resolution=group["resolution"],
                    rationale=group["rationale"],
                    action=group["action"],
                    deterministic=True,
                    bulk_eligible=True,
                )
            )
        else:
            reviewed_rows = list(group["evidence"])
        reviewed_groups.append(
            _group(
                "prior_review",
                reviewed_rows,
                resolution="needs_review",
                rationale="An earlier raw-audit decision exists and remains evidence for this grouped decision.",
                action={"kind": "manual_edit_required"},
                deterministic=False,
                bulk_eligible=False,
                prior_review_evidence=list(relevant.values()),
            )
        )

    authority_checked_groups: list[dict[str, Any]] = []
    for group in reviewed_groups:
        unavailable = {
            str(item["project_id"])
            for item in group["evidence"]
            if item.get("scope") == "project"
            and item.get("project_id")
            and allowed_project_ids is not None
            and str(item["project_id"]) not in allowed_project_ids
        }
        if not unavailable:
            authority_checked_groups.append(group)
            continue
        authority_checked_groups.append(
            _group(
                "missing_project_authority",
                list(group["evidence"]),
                resolution="needs_review",
                rationale="The current principal no longer has authority for this memory's project scope.",
                action={"kind": "manual_edit_required"},
                deterministic=False,
                bulk_eligible=False,
                prior_review_evidence=list(group["prior_review_evidence"]),
            )
        )

    ordered = sorted(
        authority_checked_groups,
        key=lambda item: (not item["deterministic"], item["classification"], item["group_id"]),
    )
    assigned = [str(item["memory_id"]) for group in ordered for item in group["evidence"]]
    if sorted(assigned) != sorted(evidence_by_id) or len(assigned) != len(set(assigned)):
        raise memory_reconciliation.MemoryReconciliationError(
            "Triage groups do not partition the audit memory corpus exactly once"
        )
    plan: dict[str, Any] = {
        "kind": PLAN_KIND,
        "version": PLAN_VERSION,
        "policy_version": POLICY_VERSION,
        "plan_id": f"mtp_{_digest({'audit': audit['sha256'], 'policy': POLICY_VERSION})[:32]}",
        "audit_id": audit["audit_id"],
        "audit_sha256": audit["sha256"],
        "corpus_sha256": audit["corpus_sha256"],
        "principal_id": audit["principal_id"],
        "runtime_profile_id": audit["runtime_profile_id"],
        "authorized_project_ids": None if allowed_project_ids is None else sorted(allowed_project_ids),
        "generated_at": audit["generated_at"],
        "group_count": len(ordered),
        "memory_count": len(evidence_by_id),
        "groups": ordered,
    }
    plan["sha256"] = _digest(plan)
    return plan


def validate_triage_plan(plan: dict[str, Any]) -> None:
    if plan.get("kind") != PLAN_KIND or plan.get("version") != PLAN_VERSION:
        raise memory_reconciliation.MemoryReconciliationError("Unsupported memory triage plan")
    expected = _digest({key: value for key, value in plan.items() if key != "sha256"})
    if plan.get("sha256") != expected:
        raise memory_reconciliation.MemoryReconciliationError("Memory triage plan digest does not match")
    groups = plan.get("groups")
    if not isinstance(groups, list) or len({item.get("group_id") for item in groups}) != len(groups):
        raise memory_reconciliation.MemoryReconciliationError("Memory triage groups are malformed")


__all__ = ["POLICY_VERSION", "build_triage_plan", "validate_triage_plan"]
