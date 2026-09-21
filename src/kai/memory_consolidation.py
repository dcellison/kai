"""Bounded candidate retrieval for semantic-memory consolidation.

The fact extractor can only consolidate against rows it is shown.  A single
semantic search over the assistant response misses common update shapes: the
user supplies the changed value, short acknowledgements carry little search
signal, and negations or renames are often not close paraphrases of the fact
they invalidate.  This module combines a small semantic query fan-out with a
bounded lexical scan, applies write-scope authority before ranking, and caps
the final prompt-facing set independently of the extractor's output schema.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from kai import memory
from kai.memory import MemoryResult

_MAX_SEMANTIC_QUERIES = 4
_MAX_SEMANTIC_QUERY_CHARS = 800
_MAX_CLAIM_CHARS = 400
_MAX_EXCHANGE_CHARS = 2500
_SEMANTIC_FETCH_MIN = 8
_SEMANTIC_FETCH_MAX = 32
_LEXICAL_SCAN_LIMIT = 1000

_CLAUSE_SPLIT_RE = re.compile(r"(?:\r?\n+|(?<=[.!?;])\s+)")
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/+-]*")

# These words describe the conversational or storage shape rather than the
# identity/value of a claim.  Letting one of them satisfy a one-token lexical
# match would make unrelated facts ("uses Python" vs "uses Redis") mutually
# authoritative.  Distinctive values such as ``redis``, ``celsius``, paths,
# model names, and resource names remain eligible anchors.
_WEAK_ANCHORS = frozenset(
    {
        "agent",
        "assistant",
        "backend",
        "change",
        "current",
        "fact",
        "memory",
        "model",
        "name",
        "new",
        "old",
        "prefer",
        "preference",
        "project",
        "repo",
        "repository",
        "set",
        "system",
        "use",
        "user",
        "value",
        "workspace",
    }
)
_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "his",
        "i",
        "in",
        "is",
        "it",
        "its",
        "me",
        "my",
        "no",
        "not",
        "of",
        "on",
        "or",
        "our",
        "she",
        "that",
        "the",
        "their",
        "them",
        "they",
        "this",
        "to",
        "we",
        "were",
        "with",
        "you",
        "your",
    }
)


@dataclass(frozen=True, slots=True)
class ConsolidationQuery:
    """One bounded semantic query and its evidence source."""

    text: str
    kind: str
    weight: float


@dataclass(frozen=True, slots=True)
class ConsolidationRetrieval:
    """Prompt-facing candidates plus non-content retrieval diagnostics."""

    candidates: tuple[MemoryResult, ...]
    query_kinds: tuple[str, ...]
    semantic_hits: int
    lexical_hits: int
    excluded_by_scope: int
    excluded_below_floor: int
    scanned_rows: int
    search_failures: int


@dataclass(slots=True)
class _RankedCandidate:
    row: MemoryResult
    semantic_score: float = 0.0
    reciprocal_rank: float = 0.0
    lexical_score: float = 0.0
    evidence: set[str] = field(default_factory=set)

    @property
    def rank(self) -> tuple[float, float, float, str]:
        # Lexical evidence leads because it is the contradiction/rename rescue
        # path.  Semantic score and reciprocal rank order paraphrase matches.
        combined = self.lexical_score * 2.0 + self.semantic_score + self.reciprocal_rank
        return (combined, self.lexical_score, self.semantic_score, self.row.id)


def scope_admitted(metadata: dict | None, allowed_project_id: str | None) -> bool:
    """Apply canonical read admission to a write-controlling candidate."""

    resolved = memory.resolve_memory_scope(metadata)
    return memory._scoped_memory_admission_reason(resolved, allowed_project_id=allowed_project_id) is None


def build_consolidation_queries(user_text: str, assistant_text: str) -> tuple[ConsolidationQuery, ...]:
    """Build at most four distinct queries from both sides of an exchange."""

    user_text = user_text[:_MAX_EXCHANGE_CHARS]
    assistant_text = assistant_text[:_MAX_EXCHANGE_CHARS]
    user = user_text[:_MAX_SEMANTIC_QUERY_CHARS].strip()
    assistant = assistant_text[:_MAX_SEMANTIC_QUERY_CHARS].strip()
    proposed: list[ConsolidationQuery] = []

    user_claim = _claim_fragment(user_text)
    if user_claim:
        proposed.append(ConsolidationQuery(user_claim, "user_claim", 1.4))
    if user:
        proposed.append(ConsolidationQuery(user, "user_exchange", 1.2))
    assistant_claim = _claim_fragment(assistant_text)
    if assistant_claim:
        proposed.append(ConsolidationQuery(assistant_claim, "assistant_claim", 1.0))
    if assistant:
        proposed.append(ConsolidationQuery(assistant, "assistant_exchange", 0.8))

    distinct: list[ConsolidationQuery] = []
    seen: set[str] = set()
    for query in proposed:
        normalized = " ".join(query.text.casefold().split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        distinct.append(query)
        if len(distinct) == _MAX_SEMANTIC_QUERIES:
            break
    return tuple(distinct)


def retrieve_consolidation_candidates(
    *,
    user_text: str,
    assistant_text: str,
    user_id: str,
    runtime_profile_id: str | None,
    allowed_project_id: str | None,
    limit: int,
    semantic_floor: float,
    search: Callable[..., list[MemoryResult] | None] | None = None,
    get_all: Callable[..., list[MemoryResult] | None] | None = None,
) -> ConsolidationRetrieval:
    """Return a bounded hybrid candidate set for one extraction attempt.

    Scope admission happens before a row can contribute ranking evidence.
    Foreign-project, task, unresolved legacy, and invalid rows therefore can
    neither enter the prompt nor become legal ``existing_id`` references.
    The final ``limit`` controls prompt size only; semantic fetch width and
    lexical scan width are separately bounded retrieval policies.
    """

    if limit <= 0:
        return ConsolidationRetrieval((), (), 0, 0, 0, 0, 0, 0)

    search_memory = search or memory.search
    list_memory = get_all or memory.get_all
    queries = build_consolidation_queries(user_text, assistant_text)
    fetch_limit = min(_SEMANTIC_FETCH_MAX, max(_SEMANTIC_FETCH_MIN, limit * 2))
    ranked: dict[str, _RankedCandidate] = {}
    excluded_scope_ids: set[str] = set()
    excluded_floor_ids: set[str] = set()
    semantic_ids: set[str] = set()
    lexical_ids: set[str] = set()
    search_failures = 0

    runtime_kwargs: dict[str, object] = {"user_id": user_id, "limit": fetch_limit}
    if runtime_profile_id is not None:
        runtime_kwargs["runtime_profile_id"] = runtime_profile_id

    for query in queries:
        try:
            rows = search_memory(query.text, **runtime_kwargs) or []
        except Exception:
            search_failures += 1
            continue
        for position, row in enumerate(rows[:fetch_limit]):
            if not _fact_candidate(row):
                continue
            if not scope_admitted(row.metadata, allowed_project_id):
                excluded_scope_ids.add(row.id)
                continue
            score = _finite_score(row.score)
            if score < semantic_floor:
                excluded_floor_ids.add(row.id)
                continue
            entry = ranked.setdefault(row.id, _RankedCandidate(row=row))
            if score > entry.semantic_score:
                entry.row = row
                entry.semantic_score = score
            entry.reciprocal_rank += query.weight / (position + 1)
            entry.evidence.add(query.kind)
            semantic_ids.add(row.id)

    # A bounded full-row scan supplies lexical neighbours whose vectors are
    # poor paraphrases precisely because the value was negated, replaced, or
    # renamed.  It is read once per extraction regardless of query count.
    list_kwargs: dict[str, object] = {"user_id": user_id, "limit": _LEXICAL_SCAN_LIMIT}
    if runtime_profile_id is not None:
        list_kwargs["runtime_profile_id"] = runtime_profile_id
    try:
        lexical_rows = list_memory(**list_kwargs) or []
    except Exception:
        lexical_rows = []
        search_failures += 1

    lexical_text = f"{user_text[:_MAX_EXCHANGE_CHARS]}\n{assistant_text[:_MAX_EXCHANGE_CHARS]}"
    for row in lexical_rows[:_LEXICAL_SCAN_LIMIT]:
        if not _fact_candidate(row):
            continue
        if not scope_admitted(row.metadata, allowed_project_id):
            excluded_scope_ids.add(row.id)
            continue
        lexical_score = _lexical_relatedness(lexical_text, row.text)
        if lexical_score <= 0.0:
            continue
        entry = ranked.setdefault(row.id, _RankedCandidate(row=row))
        if lexical_score > entry.lexical_score:
            entry.lexical_score = lexical_score
            # Preserve a real semantic score if this row was also found by
            # vector search; get_all rows conventionally carry score 0.
            if entry.semantic_score == 0.0:
                entry.row = row
        entry.evidence.add("lexical")
        lexical_ids.add(row.id)

    ordered = sorted(ranked.values(), key=lambda item: item.rank, reverse=True)
    candidates = tuple(item.row for item in ordered[:limit])
    return ConsolidationRetrieval(
        candidates=candidates,
        query_kinds=tuple(query.kind for query in queries),
        semantic_hits=len(semantic_ids),
        lexical_hits=len(lexical_ids),
        excluded_by_scope=len(excluded_scope_ids),
        excluded_below_floor=len(excluded_floor_ids),
        scanned_rows=min(len(lexical_rows), _LEXICAL_SCAN_LIMIT),
        search_failures=search_failures,
    )


def _claim_fragment(text: str) -> str:
    """Choose one bounded assertion-like clause from a message."""

    clauses: list[tuple[int, int, str]] = []
    change_markers = (" now ", " no longer ", " instead ", " renamed ", " changed ", " switched ", " moved ")
    for index, raw in enumerate(_CLAUSE_SPLIT_RE.split(text)):
        clause = " ".join(raw.strip().split())[:_MAX_CLAIM_CHARS]
        if not clause or clause.endswith("?") or len(_content_tokens(clause)) < 2:
            continue
        padded = f" {clause.casefold()} "
        change_weight = int(any(marker in padded for marker in change_markers))
        clauses.append((change_weight, len(_content_tokens(clause)), f"{index:04d}\0{clause}"))
    if not clauses:
        return ""
    _, _, encoded = max(clauses)
    return encoded.split("\0", 1)[1]


def _lexical_relatedness(exchange: str, candidate: str) -> float:
    query_tokens = _content_tokens(exchange)
    candidate_tokens = _content_tokens(candidate)
    if not query_tokens or not candidate_tokens:
        return 0.0
    overlap = query_tokens & candidate_tokens
    anchors = overlap - _WEAK_ANCHORS
    if len(anchors) >= 2:
        return min(1.0, 0.55 + 0.1 * len(anchors))
    if len(anchors) == 1:
        anchor = next(iter(anchors))
        if len(anchor) >= 4 or any(char.isdigit() for char in anchor) or any(char in anchor for char in "_./:+-"):
            return 0.6
    return 0.0


def _content_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for match in _TOKEN_RE.finditer(text):
        token = _stem(match.group(0).casefold().strip("._:/+-"))
        if len(token) >= 2 and token not in _STOP_WORDS:
            tokens.add(token)
    return tokens


def _stem(token: str) -> str:
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("ed"):
        base = token[:-2]
        return base + "e"
    if len(token) > 4 and token.endswith("s") and not token.endswith(("as", "is", "ss", "us")):
        return token[:-1]
    return token


def _fact_candidate(row: MemoryResult) -> bool:
    return row.memory_type != "episode" and (row.metadata or {}).get("source") != "episode"


def _finite_score(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    score = float(value)
    return score if score == score and abs(score) != float("inf") else 0.0


__all__ = [
    "ConsolidationQuery",
    "ConsolidationRetrieval",
    "build_consolidation_queries",
    "retrieve_consolidation_candidates",
    "scope_admitted",
]
