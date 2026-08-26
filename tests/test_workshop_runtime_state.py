"""Tests for canonical profile-addressed runtime state."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from kai.workshop.domain import CanonicalMemoryProvenance, MessageId, RunId
from kai.workshop.runtime_state import WorkshopRuntimeStateWriter
from tests.workshop_profiles import profile_id


async def test_profile_state_ingests_memory_with_only_canonical_authority(monkeypatch):
    config = SimpleNamespace(episode_classifier_context_turns=4)
    profile = SimpleNamespace(
        profile_id=profile_id(101),
        os_user="daniel",
        backend="codex",
        provider="openai",
        allowed_triage_projects=("Kai",),
    )
    runtime_pool = SimpleNamespace(runtime_profile=Mock(return_value=profile))
    execution_state = SimpleNamespace(resolve_profile=Mock(return_value=SimpleNamespace(principal_id="prn_daniel")))
    ingest = AsyncMock()
    monkeypatch.setattr("kai.workshop.runtime_state.ingest_conversation_memory", ingest)

    state = WorkshopRuntimeStateWriter(config, runtime_pool, execution_state).for_profile(profile_id(101))
    provenance = CanonicalMemoryProvenance(RunId.new(), MessageId.new(), MessageId.new())
    assert not hasattr(state, "save_session")
    await state.ingest_memory(
        prompt="Hello",
        assistant_text="Hi",
        session_id="session-1",
        workspace="/workspace/project",
        canonical_provenance=provenance,
        canonical_prior_pairs=(("Earlier", "Exchange"),),
    )

    runtime_pool.runtime_profile.assert_called_once_with(profile_id(101))
    execution_state.resolve_profile.assert_called_once_with(profile_id(101))
    ingest.assert_awaited_once_with(
        prompt="Hello",
        assistant_text="Hi",
        chat_id=None,
        canonical_user_id="prn_daniel",
        runtime_profile_id=str(profile_id(101)),
        session_id="session-1",
        config=config,
        workspace="/workspace/project",
        user_log=None,
        assistant_log=None,
        canonical_provenance=provenance,
        canonical_prior_pairs=(("Earlier", "Exchange"),),
        effective_backend="codex",
        effective_provider="openai",
        os_user_override="daniel",
    )


def test_execution_triggers_do_not_own_common_post_run_effects():
    source_root = Path(__file__).parents[1] / "src" / "kai"
    for relative_path in (
        "bot.py",
        "workshop/client_commands.py",
        "workshop/scheduler.py",
    ):
        source = (source_root / relative_path).read_text(encoding="utf-8")
        assert "schedule_memory_ingestion" not in source
        assert "ingest_conversation_memory" not in source
        assert "save_session(" not in source
