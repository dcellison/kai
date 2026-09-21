from __future__ import annotations

from collections.abc import Callable

import pytest

from kai.memory import MemoryResult
from kai.memory_consolidation import (
    build_consolidation_queries,
    retrieve_consolidation_candidates,
)

_GLOBAL = {"source": "extracted", "scope": "global", "scope_source": "classifier"}


def _row(
    memory_id: str,
    text: str,
    *,
    score: float = 0.0,
    metadata: dict | None = None,
) -> MemoryResult:
    return MemoryResult(
        id=memory_id,
        text=text,
        score=score,
        memory_type="fact",
        metadata=dict(_GLOBAL if metadata is None else metadata),
        created_at="2026-09-21T00:00:00Z",
    )


def _retrieve(
    user: str,
    assistant: str,
    *,
    semantic: Callable[..., list[MemoryResult] | None] | None = None,
    rows: list[MemoryResult] | None = None,
    allowed_project_id: str | None = None,
    limit: int = 8,
):
    return retrieve_consolidation_candidates(
        user_text=user,
        assistant_text=assistant,
        user_id="principal",
        runtime_profile_id="runtime",
        allowed_project_id=allowed_project_id,
        limit=limit,
        semantic_floor=0.3,
        search=semantic or (lambda *_args, **_kwargs: []),
        get_all=lambda **_kwargs: list(rows or []),
    )


def test_queries_use_both_sides_and_are_strictly_bounded():
    queries = build_consolidation_queries(
        "Background sentence. I no longer use Redis; I use Valkey instead.",
        "Understood. Valkey is now your cache for this project.",
    )

    assert 2 <= len(queries) <= 4
    assert any(query.kind.startswith("user_") for query in queries)
    assert any(query.kind.startswith("assistant_") for query in queries)
    assert all(len(query.text) <= 800 for query in queries)
    assert queries[0].kind == "user_claim"
    assert "no longer use Redis" in queries[0].text


def test_user_assertion_can_retrieve_candidate_absent_from_assistant_search():
    prior = _row("timezone", "User's timezone is Eastern.", score=0.82)
    seen: list[str] = []

    def semantic(query: str, **_kwargs):
        seen.append(query)
        return [prior] if "Pacific" in query and "timezone" in query else []

    result = _retrieve(
        "My timezone is now Pacific.",
        "Thanks, I have noted that.",
        semantic=semantic,
    )

    assert [candidate.id for candidate in result.candidates] == ["timezone"]
    assert any("My timezone is now Pacific" in query for query in seen)
    assert any("noted" in query for query in seen)
    assert result.semantic_hits == 1


@pytest.mark.parametrize(
    ("scenario", "user", "assistant", "prior_text"),
    [
        ("repeat", "I prefer Celsius.", "Celsius it is.", "User prefers Celsius."),
        (
            "refinement",
            "My timezone is America/Toronto, not just Eastern.",
            "I will use America/Toronto.",
            "User's timezone is Eastern.",
        ),
        (
            "preference_reversal",
            "I switched from Celsius to Fahrenheit.",
            "I will use Fahrenheit.",
            "User prefers Celsius.",
        ),
        (
            "renamed_resource",
            "I renamed project Atlas to Orion.",
            "The project is now Orion.",
            "The project is named Atlas.",
        ),
        (
            "negation",
            "I no longer use Redis; I use Valkey.",
            "Valkey has replaced Redis.",
            "User uses Redis.",
        ),
    ],
)
def test_reviewed_corpus_scenarios_have_lexical_rescue(
    scenario: str,
    user: str,
    assistant: str,
    prior_text: str,
):
    relevant = _row(scenario, prior_text)
    unrelated = _row("unrelated", "User writes documentation in Markdown.")

    result = _retrieve(user, assistant, rows=[relevant, unrelated])

    assert [candidate.id for candidate in result.candidates] == [scenario]
    assert result.lexical_hits == 1


def test_low_similarity_unrelated_semantic_hits_are_excluded():
    unrelated = _row("unrelated", "User writes documentation in Markdown.", score=0.12)

    result = _retrieve(
        "I now use Valkey.",
        "Valkey is configured.",
        semantic=lambda *_args, **_kwargs: [unrelated],
    )

    assert result.candidates == ()
    assert result.excluded_below_floor == 1


def test_foreign_scope_cannot_enter_through_semantic_or_lexical_retrieval():
    local = _row(
        "local",
        "Kai uses Valkey.",
        score=0.7,
        metadata={"source": "extracted", "scope": "project", "project_id": "kai", "scope_source": "classifier"},
    )
    foreign = _row(
        "foreign",
        "Anvil uses Redis.",
        score=0.99,
        metadata={"source": "extracted", "scope": "project", "project_id": "anvil", "scope_source": "classifier"},
    )

    result = _retrieve(
        "Kai no longer uses Redis; it uses Valkey.",
        "Kai now uses Valkey.",
        semantic=lambda *_args, **_kwargs: [foreign, local],
        rows=[foreign, local],
        allowed_project_id="kai",
    )

    assert [candidate.id for candidate in result.candidates] == ["local"]
    assert result.excluded_by_scope == 1


def test_candidate_budget_is_independent_and_caps_prompt_context():
    rows = [_row(f"candidate-{index}", f"User uses Redis cluster {index}.") for index in range(20)]

    result = _retrieve(
        "I no longer use Redis clusters.",
        "Redis was retired.",
        rows=rows,
        limit=3,
    )

    assert len(result.candidates) == 3
    assert result.lexical_hits == 20


def test_runtime_authority_is_forwarded_to_every_store_read():
    calls: list[tuple[str, dict[str, object]]] = []

    def semantic(query: str, **kwargs):
        calls.append((query, kwargs))
        return []

    def get_all(**kwargs):
        calls.append(("all", kwargs))
        return []

    retrieve_consolidation_candidates(
        user_text="I prefer Celsius.",
        assistant_text="Understood.",
        user_id="principal",
        runtime_profile_id="runtime",
        allowed_project_id=None,
        limit=8,
        semantic_floor=0.3,
        search=semantic,
        get_all=get_all,
    )

    assert calls
    assert all(kwargs["user_id"] == "principal" for _, kwargs in calls)
    assert all(kwargs["runtime_profile_id"] == "runtime" for _, kwargs in calls)
    assert calls[-1][1]["limit"] == 1000


def test_failures_degrade_to_other_bounded_retrieval_arm():
    lexical = _row("redis", "User uses Redis.")

    def broken(*_args, **_kwargs):
        raise RuntimeError("store unavailable")

    result = _retrieve(
        "I no longer use Redis.",
        "Understood.",
        semantic=broken,
        rows=[lexical],
    )

    assert [candidate.id for candidate in result.candidates] == ["redis"]
    assert result.search_failures == len(result.query_kinds)
