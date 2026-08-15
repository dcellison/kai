"""Tests for profile-addressed compatibility-state writes."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from kai.conversation_compatibility import CanonicalMemoryProvenance
from kai.workshop.compatibility_state import WorkshopCompatibilityStateWriter
from kai.workshop.domain import MessageId, RunId
from tests.workshop_profiles import profile_id


async def test_profile_state_keeps_numeric_alias_private_without_jsonl(monkeypatch):
    config = SimpleNamespace(
        episode_classifier_context_turns=3,
        get_user_config=lambda runtime_config_id: (
            SimpleNamespace(os_user="daniel") if runtime_config_id == 101 else None
        ),
    )
    runtime_pool = SimpleNamespace(
        runtime_profile=Mock(
            return_value=SimpleNamespace(
                runtime_config_id=101,
                os_user="daniel",
                backend="codex",
            )
        )
    )
    save_session = AsyncMock()
    schedule = Mock()
    monkeypatch.setattr(
        "kai.workshop.compatibility_state.sessions.save_session",
        save_session,
    )
    monkeypatch.setattr(
        "kai.workshop.compatibility_state.schedule_memory_ingestion",
        schedule,
    )

    state = WorkshopCompatibilityStateWriter(config, runtime_pool).for_profile(profile_id(101))
    provenance = CanonicalMemoryProvenance(RunId.new(), MessageId.new(), MessageId.new())
    assert state.memory_context_turns == 3
    await state.save_session("session-1", "gpt-5.6-sol")
    state.schedule_memory_ingestion(
        prompt="Hello",
        assistant_text="Hi",
        session_id="session-1",
        workspace="/workspace/project",
        canonical_provenance=provenance,
        canonical_prior_pairs=(("Earlier", "Answer"),),
    )

    runtime_pool.runtime_profile.assert_called_once_with(profile_id(101))
    save_session.assert_awaited_once_with(101, "session-1", "gpt-5.6-sol")
    schedule.assert_called_once_with(
        prompt="Hello",
        assistant_text="Hi",
        chat_id=101,
        session_id="session-1",
        config=config,
        workspace="/workspace/project",
        user_log=None,
        assistant_log=None,
        canonical_provenance=provenance,
        canonical_prior_pairs=(("Earlier", "Answer"),),
        effective_backend="codex",
    )
