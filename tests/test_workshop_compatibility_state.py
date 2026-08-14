"""Tests for profile-addressed compatibility-state writes."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call

from kai.workshop.compatibility_state import WorkshopCompatibilityStateWriter
from tests.workshop_profiles import profile_id


async def test_profile_state_preserves_existing_storage_keys(monkeypatch):
    config = SimpleNamespace(
        get_user_config=lambda runtime_config_id: (
            SimpleNamespace(os_user="daniel") if runtime_config_id == 101 else None
        )
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
    log = Mock(side_effect=["user-log", "assistant-log"])
    save_session = AsyncMock()
    schedule = Mock()
    monkeypatch.setattr("kai.workshop.compatibility_state.log_message", log)
    monkeypatch.setattr(
        "kai.workshop.compatibility_state.sessions.save_session",
        save_session,
    )
    monkeypatch.setattr(
        "kai.workshop.compatibility_state.schedule_memory_ingestion",
        schedule,
    )

    state = WorkshopCompatibilityStateWriter(config, runtime_pool).for_profile(profile_id(101))
    user_log = state.append_history(direction="user", text="Hello")
    assistant_log = state.append_history(direction="assistant", text="Hi")
    await state.save_session("session-1", "gpt-5.6-sol")
    state.schedule_memory_ingestion(
        prompt="Hello",
        assistant_text="Hi",
        session_id="session-1",
        workspace="/workspace/project",
        user_log=user_log,
        assistant_log=assistant_log,
    )

    runtime_pool.runtime_profile.assert_called_once_with(profile_id(101))
    assert log.call_args_list == [
        call(
            direction="user",
            chat_id=101,
            text="Hello",
            reader_user="daniel",
        ),
        call(
            direction="assistant",
            chat_id=101,
            text="Hi",
            reader_user="daniel",
        ),
    ]
    save_session.assert_awaited_once_with(101, "session-1", "gpt-5.6-sol")
    schedule.assert_called_once_with(
        prompt="Hello",
        assistant_text="Hi",
        chat_id=101,
        session_id="session-1",
        config=config,
        workspace="/workspace/project",
        user_log="user-log",
        assistant_log="assistant-log",
        effective_backend="codex",
    )
