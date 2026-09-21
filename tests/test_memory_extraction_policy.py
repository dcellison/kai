"""Deterministic extraction admission, cadence, and fragmentation policy."""

from __future__ import annotations

from kai.memory_extraction_policy import (
    MAX_BATCH_EXCHANGES,
    MAX_FACTS_PER_EXCHANGE,
    apply_fragmentation_policy,
    decide_extraction_admission,
)


def _admission(user: str, assistant: str = "Acknowledged", **overrides):
    values = {
        "source_kind": "workshop_client",
        "run_kind": "respond",
        "parent_run_id": None,
        "user_text": user,
        "assistant_text": assistant,
        "canonical": True,
    }
    values.update(overrides)
    return decide_extraction_admission(**values)


def _fact(content: str, *, confidence: float = 0.9, intent: str = "new") -> dict:
    return {
        "content": content,
        "tags": ["fact"],
        "confidence": confidence,
        "intent": intent,
        "speaker": "user",
    }


def test_human_conversation_is_admitted_with_one_exchange_cadence():
    decision = _admission("I prefer Celsius for weather reports.")

    assert decision.admitted is True
    assert decision.reason == "human_conversation"
    assert decision.cadence == "single_exchange"
    assert decision.batch_size == decision.batch_limit == MAX_BATCH_EXCHANGES == 1


def test_compatibility_callers_preserve_existing_behavior():
    decision = _admission("Installed.", source_kind=None, canonical=False)

    assert decision.admitted is True
    assert decision.reason == "compatibility_unclassified"


def test_machine_and_delegated_runs_are_suppressed():
    assert _admission("Run report", source_kind="scheduled_job").reason == "non_human_source"
    assert _admission("Run report", run_kind="observe").reason == "non_response_run"
    assert _admission("Run report", parent_run_id="run_parent").reason == "delegated_child_run"


def test_routine_qualification_status_and_command_traffic_is_suppressed():
    assert _admission("PING", "PONG").reason == "qualification_marker"
    assert _admission("#1729 squash merged.").reason == "routine_workflow_ack"
    assert _admission("/stats").reason == "adapter_command"
    assert _admission("Service: com.syrinx.kai (loaded)\nWorkshop memory authority: active").reason == (
        "operational_status"
    )


def test_fragmentation_policy_keeps_distinct_atomic_facts():
    facts = [
        _fact("User lives in Toronto."),
        _fact("User prefers Celsius for temperatures."),
        _fact("User owns an M3 MacBook Pro."),
    ]

    decision = apply_fragmentation_policy(facts)

    assert decision.facts == tuple(facts)
    assert decision.outcome == "accepted"
    assert decision.rejected_count == 0


def test_fragmentation_policy_rejects_weaker_overlapping_facets():
    weaker = _fact("User prefers concise technical answers.", confidence=0.7)
    stronger = _fact("User strongly prefers concise technical answers in Kai.", confidence=0.95)

    decision = apply_fragmentation_policy([weaker, stronger])

    assert decision.facts == (stronger,)
    assert decision.outcome == "overlap_rejected"
    assert decision.rejected_count == 1


def test_fragmentation_policy_caps_materially_distinct_facts():
    facts = [_fact(f"User durable preference number {index} concerns topic {index}.") for index in range(5)]

    decision = apply_fragmentation_policy(facts)

    assert len(decision.facts) == MAX_FACTS_PER_EXCHANGE == 3
    assert decision.outcome == "capped"
    assert decision.rejected_count == 2
