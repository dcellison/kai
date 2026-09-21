from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from kai import memory_admin
from kai.config import Config
from kai.memory import MemoryResult
from kai.memory_model_evaluation import (
    FACT_ROLE,
    build_blind_review_template,
    build_plan_template,
    run_evaluation,
    score_evaluation,
    seal_evaluation_review,
    seal_plan,
    write_score,
)
from kai.memory_quality_corpus import (
    SCENARIO_TAGS,
    MemoryQualityCorpusError,
    ProductionReceipt,
    _document_digest,
    build_review_template,
    build_snapshot,
    seal_review,
)
from kai.workshop.domain import RuntimeProfileId
from kai.workshop.runtime_profiles import ProtectedRuntimeProfile, WorkshopRuntimeProfileRegistry

_PROFILE_ID = RuntimeProfileId("rtp_5d9af35df8d1c1330044993c2def7756")


def _receipt(index: int, *, role: str) -> ProductionReceipt:
    memory_id = f"mem_{index}"
    return ProductionReceipt(
        receipt_id=f"mer_{index}",
        principal_id="prn_owner",
        runtime_profile_id=str(_PROFILE_ID),
        run_id=f"run_{index}",
        source_message_id=f"msg_source_{index}",
        result_message_id=f"msg_result_{index}",
        extraction_role=role,
        backend="codex",
        provider="openai",
        model="gpt-5.6-luna",
        prompt_version="13" if role == FACT_ROLE else "2",
        schema_version="1",
        policy_version="1",
        status="completed",
        decision_outcome="stored",
        classifier_result=True if role == FACT_ROLE else None,
        proposed_intents=({"intent": "new"},),
        validation_outcome={"outcome": "accepted", "raw_count": 1, "accepted_count": 1},
        storage_outcome={
            "stored_count": 1,
            "replaced_count": 0,
            "skipped_count": 0,
            "decisions": [
                {
                    "index": 0,
                    "intent": "new" if role == FACT_ROLE else "store_episode",
                    "outcome": "stored",
                    "new_memory_id": memory_id,
                    "scope": "global",
                }
            ],
        },
        memory_scopes=({"scope": "global"},),
        candidate_ids=("existing_1",) if role == FACT_ROLE else (),
        created_at=f"2026-09-21T00:{index:02d}:00Z",
        completed_at=f"2026-09-21T00:{index:02d}:01Z",
        channel_id="chn_general",
        channel_kind="group",
        channel_name="General",
        agent_handle="kai",
        agent_display_name="Kai",
        source_body=f"Durable statement {index}",
        result_body=f"Acknowledged {index}",
        prior_pairs=(("Earlier question", "Earlier answer"),),
    )


def _memory(_owner: str, _profile: str, memory_id: str) -> MemoryResult:
    return MemoryResult(
        id=memory_id,
        text=f"Remembered {memory_id}",
        score=0.0,
        memory_type="fact",
        metadata={"source": "extracted", "scope": "global"},
        created_at="2026-09-21T00:00:00Z",
        updated_at="2026-09-21T00:00:00Z",
    )


def _reviewed_corpus() -> tuple[dict[str, object], dict[str, object]]:
    receipts = [_receipt(index, role=FACT_ROLE if index % 2 == 0 else "episode_generation") for index in range(30)]
    snapshot = build_snapshot(
        principal_id="prn_owner",
        receipts=receipts,
        memory_lookup=_memory,
        seed=1709,
    )
    review = build_review_template(snapshot)
    decisions = review["decisions"]
    assert isinstance(decisions, list)
    scenario_tags = sorted(SCENARIO_TAGS)
    cases = snapshot["cases"]
    assert isinstance(cases, list)
    roles = {str(case["case_id"]): str(case["extraction_role"]) for case in cases}
    for index, decision in enumerate(decisions):
        assert isinstance(decision, dict)
        role = roles[str(decision["case_id"])]
        decision.update(
            {
                "review_status": "complete",
                "scenario_tags": [scenario_tags[index]] if index < len(scenario_tags) else [],
                "case_labels": [],
                "expected_fact_count": 1 if role == FACT_ROLE else 0,
                "episode_expected": True,
                "update_expected": role == FACT_ROLE,
                "update_detected": role == FACT_ROLE,
            }
        )
        outputs = decision["outputs"]
        assert isinstance(outputs, list)
        for output in outputs:
            output.update(
                {
                    "verdict": "useful",
                    "labels": ["useful"],
                    "scope_correct": True,
                    "consolidation_correct": True if role == FACT_ROLE else None,
                }
            )
    return snapshot, seal_review(snapshot, review, reviewer="Daniel")


def _registry() -> WorkshopRuntimeProfileRegistry:
    profile = ProtectedRuntimeProfile(
        profile_id=_PROFILE_ID,
        display_name="Daniel",
        os_user="daniel",
        backend="codex",
        provider="openai",
        model="gpt-5.6-sol",
        timeout_seconds=600,
        allowed_services=(),
        home_workspace=Path("/tmp"),
        workspace_base=Path("/tmp"),
        allowed_workspaces=(Path("/tmp"),),
        allowed_models=("gpt-5.6-luna", "gpt-5.6-sol"),
    )
    return WorkshopRuntimeProfileRegistry((profile,))


def _plan(snapshot: dict[str, object], review: dict[str, object]) -> dict[str, object]:
    plan = build_plan_template(snapshot, review, runtime_profile_id=str(_PROFILE_ID))
    candidates = plan["candidates"]
    assert isinstance(candidates, list)
    candidates.extend(
        [
            {
                "role": "fact_extraction",
                "backend": "codex",
                "provider": "openai",
                "model": "gpt-5.6-sol",
                "baseline": False,
            },
            {
                "role": "episode_generation",
                "backend": "codex",
                "provider": "openai",
                "model": "gpt-5.6-sol",
                "baseline": False,
            },
        ]
    )
    return plan


def test_snapshot_v2_captures_replay_context():
    snapshot, _review = _reviewed_corpus()
    cases = snapshot["cases"]
    assert isinstance(cases, list)
    first = cases[0]
    assert isinstance(first, dict)
    conversation = first["conversation"]
    pipeline = first["pipeline"]
    assert isinstance(conversation, dict)
    assert isinstance(pipeline, dict)
    assert conversation["prior_pairs"] == [{"user": "Earlier question", "assistant": "Earlier answer"}]
    assert pipeline["candidate_context"]


def test_plan_rejects_model_outside_runtime_authority():
    snapshot, review = _reviewed_corpus()
    plan = _plan(snapshot, review)
    candidates = plan["candidates"]
    assert isinstance(candidates, list)
    candidates[-1]["model"] = "gpt-not-authorized"

    with pytest.raises(MemoryQualityCorpusError, match="outside runtime authority"):
        seal_plan(plan, snapshot, review, _registry())


@pytest.mark.asyncio
async def test_blind_review_scores_promotion_without_exposing_models(tmp_path):
    snapshot, review = _reviewed_corpus()
    sealed_plan, profile = seal_plan(_plan(snapshot, review), snapshot, review, _registry())

    async def fact_runner(case, candidate, principal_id, os_user, config):
        assert principal_id == "prn_owner"
        assert os_user == "daniel"
        assert candidate["backend"] == "codex"
        return {
            "outcome": "succeeded",
            "duration_ms": 100,
            "raw_fact_count": 1,
            "has_episode": True,
            "facts": [
                {
                    "content": f"Fact for {case['case_id']}",
                    "tags": ["fact"],
                    "confidence": 0.95,
                    "speaker": "user",
                    "intent": "new",
                    "scope_hint": "global",
                }
            ],
        }

    async def episode_runner(case, candidate, principal_id, os_user, config):
        return {
            "outcome": "succeeded",
            "duration_ms": 100,
            "episode": {
                "goal": f"Goal {case['case_id']}",
                "context": "Context",
                "approach": "Approach",
                "outcome": "Outcome",
                "outcome_quality": "success",
                "tags": ["test"],
                "actors": ["Daniel"],
            },
        }

    evaluation = await run_evaluation(
        snapshot,
        review,
        sealed_plan,
        profile,
        Config(telegram_bot_token="test", allowed_user_ids={1}),
        fact_runner=fact_runner,
        episode_runner=episode_runner,
    )
    blind = build_blind_review_template(snapshot, review, evaluation)
    assert "gpt-5.6" not in str(blind)
    decisions = blind["decisions"]
    assert isinstance(decisions, list)
    for decision in decisions:
        decision["review_status"] = "complete"
        for output in decision["outputs"]:
            output.update(
                {
                    "verdict": "useful",
                    "labels": ["useful"],
                    "scope_correct": True,
                    "consolidation_correct": True,
                }
            )
        if decision["episode"] is not None:
            decision["episode"].update({"verdict": "useful", "labels": ["useful"]})
    sealed_evaluation_review = seal_evaluation_review(blind, evaluation, reviewer="Daniel")
    score = score_evaluation(snapshot, review, sealed_plan, evaluation, sealed_evaluation_review)

    recommendations = score["recommendations"]
    assert isinstance(recommendations, list)
    assert len(recommendations) == 2
    assert all(item["promote"] is True for item in recommendations)
    assert {item["rollback_model"] for item in recommendations} == {"gpt-5.6-luna"}
    assert score["automatic_mutation"] is False

    score_path = tmp_path / "score.json"
    write_score(score_path, score)
    assert score_path.stat().st_mode & 0o777 == 0o400
    assert score_path.with_suffix(".md").stat().st_mode & 0o777 == 0o400

    mismatched_plan = deepcopy(sealed_plan)
    mismatched_plan["sealed_at"] = "2026-09-21T12:00:00Z"
    mismatched_plan["plan_sha256"] = _document_digest(mismatched_plan, "plan_sha256")
    with pytest.raises(MemoryQualityCorpusError, match="different sealed plan"):
        score_evaluation(snapshot, review, mismatched_plan, evaluation, sealed_evaluation_review)


@pytest.mark.asyncio
async def test_evaluation_review_detects_output_binding_drift():
    snapshot, review = _reviewed_corpus()
    sealed_plan, profile = seal_plan(_plan(snapshot, review), snapshot, review, _registry())

    async def no_outputs(case, candidate, principal_id, os_user, config):
        if candidate["role"] == FACT_ROLE:
            return {
                "outcome": "succeeded",
                "duration_ms": 1,
                "raw_fact_count": 0,
                "has_episode": True,
                "facts": [],
            }
        return {"outcome": "succeeded", "duration_ms": 1, "episode": None}

    evaluation = await run_evaluation(
        snapshot,
        review,
        sealed_plan,
        profile,
        Config(telegram_bot_token="test", allowed_user_ids={1}),
        fact_runner=no_outputs,
        episode_runner=no_outputs,
    )
    blind = build_blind_review_template(snapshot, review, evaluation)
    decisions = blind["decisions"]
    assert isinstance(decisions, list)
    for decision in decisions:
        decision["review_status"] = "complete"
        if decision["episode"] is not None:
            decision["episode"].update({"verdict": "not_applicable", "labels": []})

    first = decisions[0]
    first["outputs"].append(
        {
            "output_index": 0,
            "verdict": "not_applicable",
            "labels": [],
            "scope_correct": None,
            "consolidation_correct": None,
            "note": "",
        }
    )
    with pytest.raises(MemoryQualityCorpusError, match="do not match"):
        seal_evaluation_review(blind, evaluation, reviewer="Daniel")


def test_quality_corpus_parser_exposes_model_evaluation_stages():
    parser = memory_admin._build_parser()
    assert (
        parser.parse_args(
            ["quality-corpus", "evaluation-template", "snapshot.json", "review.json", str(_PROFILE_ID)]
        ).quality_command
        == "evaluation-template"
    )
    assert (
        parser.parse_args(["quality-corpus", "evaluate", "snapshot.json", "review.json", "plan.json"]).quality_command
        == "evaluate"
    )
    assert (
        parser.parse_args(
            [
                "quality-corpus",
                "score-evaluation",
                "snapshot.json",
                "review.json",
                "sealed-plan.json",
                "evaluation.json",
                "evaluation-review.json",
            ]
        ).quality_command
        == "score-evaluation"
    )
