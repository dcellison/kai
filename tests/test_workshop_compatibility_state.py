"""Tests for profile-addressed compatibility-state writes."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from kai.workshop.compatibility_state import WorkshopCompatibilityStateWriter
from kai.workshop.domain import CanonicalMemoryProvenance, MessageId, RunId
from tests.workshop_profiles import profile_id


async def test_profile_state_retains_only_non_session_compatibility_boundaries(monkeypatch):
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
    ingest = AsyncMock()
    monkeypatch.setattr(
        "kai.workshop.compatibility_state.ingest_conversation_memory",
        ingest,
    )

    state = WorkshopCompatibilityStateWriter(config, runtime_pool).for_profile(profile_id(101))
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
    ingest.assert_awaited_once_with(
        prompt="Hello",
        assistant_text="Hi",
        chat_id=101,
        session_id="session-1",
        config=config,
        workspace="/workspace/project",
        user_log=None,
        assistant_log=None,
        canonical_provenance=provenance,
        canonical_prior_pairs=(("Earlier", "Exchange"),),
        effective_backend="codex",
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
