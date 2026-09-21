"""Deterministic admission, cadence, and fragmentation policy for memory extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass

POLICY_VERSION = "2"
MAX_FACTS_PER_EXCHANGE = 3
MAX_BATCH_EXCHANGES = 1

_HUMAN_CONVERSATION_SOURCES = frozenset({"telegram", "workshop_client"})
_ROUTINE_ACK_RE = re.compile(
    r"^(?:"
    r"(?:#\d+\s+)?(?:squash[- ]?)?merged"
    r"|installed(?:\s+and\s+confirmed)?"
    r"|confirmed"
    r"|done"
    r"|success(?:ful)?"
    r"|all\s+(?:tests?\s+)?(?:succeeded|successful)"
    r"|pings?\s+(?:were\s+)?(?:successful|succeeded|worked)"
    r"|proceed"
    r"|continue"
    r"|looks?\s+good"
    r")\.?$",
    re.IGNORECASE,
)
_QUALIFICATION_MARKER_RE = re.compile(r"^(?:Q\d+_[A-Z0-9_]+|PING|PONG)(?:[.!])?$", re.IGNORECASE)
_SLASH_COMMAND_RE = re.compile(r"^/[a-z][a-z0-9_-]*(?:\s|$)", re.IGNORECASE)
_STATUS_DUMP_RE = re.compile(
    r"(?:^|\n)(?:Service|Workshop [^:\n]+):\s+(?:active|loaded|complete|initialized|INCOMPLETE|DEGRADED)\b",
    re.MULTILINE,
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "in",
        "is",
        "it",
        "of",
        "on",
        "that",
        "the",
        "this",
        "to",
        "user",
        "with",
    }
)


@dataclass(frozen=True, slots=True)
class ExtractionAdmission:
    """One content-free pre-extraction decision."""

    admitted: bool
    reason: str
    cadence: str
    batch_size: int = 1
    batch_limit: int = MAX_BATCH_EXCHANGES

    def receipt_fields(self) -> tuple[tuple[str, str | int], ...]:
        return (
            ("admission", "admitted" if self.admitted else "suppressed"),
            ("admission_reason", self.reason),
            ("cadence", self.cadence),
            ("batch_size", self.batch_size),
            ("batch_limit", self.batch_limit),
        )


@dataclass(frozen=True, slots=True)
class FragmentationDecision:
    """Second-pass decision over already schema-validated fact candidates."""

    facts: tuple[dict, ...]
    outcome: str
    rejected_count: int

    def receipt_fields(self) -> tuple[tuple[str, str | int], ...]:
        return (
            ("fragmentation", self.outcome),
            ("fragmentation_rejected", self.rejected_count),
            ("fact_limit", MAX_FACTS_PER_EXCHANGE),
        )


def decide_extraction_admission(
    *,
    source_kind: str | None,
    run_kind: str | None,
    parent_run_id: str | None,
    user_text: str,
    assistant_text: str,
    canonical: bool,
) -> ExtractionAdmission:
    """Fail closed for known machine traffic while preserving real conversation.

    Compatibility/evaluation callers lack canonical run metadata and retain their
    historical behavior. Canonical production runs are admitted only from human
    conversation adapters. Routine commands, qualification markers, terse workflow
    acknowledgements, and pasted health summaries are suppressed before a model is
    started. Cross-run batching is intentionally capped at one exchange so every
    stored memory keeps exact message/run provenance.
    """

    if not canonical:
        return ExtractionAdmission(True, "compatibility_unclassified", "single_exchange")
    normalized_source = (source_kind or "").strip().casefold()
    normalized_kind = (run_kind or "respond").strip().casefold()
    if normalized_kind != "respond":
        return ExtractionAdmission(False, "non_response_run", "suppressed")
    if parent_run_id:
        return ExtractionAdmission(False, "delegated_child_run", "suppressed")
    if normalized_source not in _HUMAN_CONVERSATION_SOURCES:
        return ExtractionAdmission(False, "non_human_source", "suppressed")

    user = " ".join(user_text.split())
    assistant = " ".join(assistant_text.split())
    if _SLASH_COMMAND_RE.match(user):
        return ExtractionAdmission(False, "adapter_command", "suppressed")
    if _QUALIFICATION_MARKER_RE.fullmatch(user) or _QUALIFICATION_MARKER_RE.fullmatch(assistant):
        return ExtractionAdmission(False, "qualification_marker", "suppressed")
    if _ROUTINE_ACK_RE.fullmatch(user):
        return ExtractionAdmission(False, "routine_workflow_ack", "suppressed")
    if _STATUS_DUMP_RE.search(user_text):
        return ExtractionAdmission(False, "operational_status", "suppressed")
    return ExtractionAdmission(True, "human_conversation", "single_exchange")


def apply_fragmentation_policy(facts: list[dict]) -> FragmentationDecision:
    """Reject overlapping fragments and cap one exchange to three durable facts.

    The model is asked to merge facets before emission. This deterministic second
    pass cannot safely synthesize new wording, so it preserves the strongest member
    of an overlapping set and rejects the weaker fragments. Multiple facts remain
    valid when their normalized subjects are materially distinct.
    """

    if not facts:
        return FragmentationDecision((), "empty", 0)

    ranked = sorted(enumerate(facts), key=lambda item: _fact_priority(item[1], item[0]), reverse=True)
    kept: list[tuple[int, dict, frozenset[str]]] = []
    rejected = 0
    overlap_rejected = False
    cap_rejected = False
    for index, fact in ranked:
        tokens = _content_tokens(fact)
        if any(_facts_overlap(fact, tokens, prior, prior_tokens) for _, prior, prior_tokens in kept):
            rejected += 1
            overlap_rejected = True
            continue
        if len(kept) >= MAX_FACTS_PER_EXCHANGE:
            rejected += 1
            cap_rejected = True
            continue
        kept.append((index, fact, tokens))

    kept.sort(key=lambda item: item[0])
    if cap_rejected:
        outcome = "capped"
    elif overlap_rejected:
        outcome = "overlap_rejected"
    else:
        outcome = "accepted"
    return FragmentationDecision(tuple(item[1] for item in kept), outcome, rejected)


def _content_tokens(fact: dict) -> frozenset[str]:
    content = fact.get("content")
    if not isinstance(content, str):
        return frozenset()
    return frozenset(token for token in _TOKEN_RE.findall(content.casefold()) if token not in _STOPWORDS)


def _facts_overlap(
    fact: dict,
    tokens: frozenset[str],
    prior: dict,
    prior_tokens: frozenset[str],
) -> bool:
    if not tokens or not prior_tokens:
        return False
    if fact.get("intent") != prior.get("intent"):
        return False
    if fact.get("existing_id") != prior.get("existing_id"):
        return False
    if fact.get("scope_hint") != prior.get("scope_hint"):
        return False
    intersection = len(tokens & prior_tokens)
    union = len(tokens | prior_tokens)
    if tokens == prior_tokens:
        return True
    if min(len(tokens), len(prior_tokens)) >= 4 and (tokens <= prior_tokens or prior_tokens <= tokens):
        return True
    return intersection >= 4 and union > 0 and intersection / union >= 0.72


def _fact_priority(fact: dict, index: int) -> tuple[int, int, float, int, int]:
    tags = fact.get("tags")
    tag_set = set(tags) if isinstance(tags, list) else set()
    confidence = fact.get("confidence")
    numeric_confidence = float(confidence) if isinstance(confidence, (int, float)) else 0.0
    content = fact.get("content")
    return (
        1 if fact.get("intent") == "update_of" else 0,
        1 if "confirmed_action" in tag_set else 0,
        numeric_confidence,
        len(content) if isinstance(content, str) else 0,
        -index,
    )
