"""Protected profile-addressed Workshop runtime-pool contracts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from kai.backend import AgentResponse, StreamEvent
from kai.workshop.domain import RuntimeProfileId
from kai.workshop.runtime_pool import WorkshopRuntimePool
from kai.workshop.runtime_profiles import WorkshopRuntimeProfileError
from tests.workshop_profiles import profile_id, profile_registry


async def _events():
    yield StreamEvent(
        text_so_far="Done.",
        done=True,
        response=AgentResponse(text="Done.", success=True),
    )


async def test_profile_facade_owns_every_compatibility_pool_conversion():
    runtime = SimpleNamespace(selection="selection", workspace=Path("/srv/project"))
    pool = MagicMock()
    pool.prepare_execution = AsyncMock(return_value=runtime)
    pool.get_model.return_value = "gpt-5.6-sol"
    pool.send.return_value = _events()
    pool.get_effective_workspace = AsyncMock(return_value=Path("/srv/project"))
    profiles = WorkshopRuntimePool(pool, profile_registry(101))

    prepared = await profiles.prepare_execution(profile_id(101))
    model = profiles.get_model(profile_id(101))
    events = [event async for event in profiles.send("hello", runtime_profile_id=profile_id(101))]
    workspace = await profiles.get_effective_workspace(profile_id(101))

    assert prepared is runtime
    assert model == "gpt-5.6-sol"
    assert events[0].response is not None and events[0].response.text == "Done."
    assert workspace == Path("/srv/project")
    pool.prepare_execution.assert_awaited_once_with(101)
    pool.get_model.assert_called_once_with(101)
    pool.send.assert_called_once_with("hello", chat_id=101)
    pool.get_effective_workspace.assert_awaited_once_with(101)


async def test_unknown_profile_fails_before_touching_compatibility_pool():
    pool = MagicMock()
    profiles = WorkshopRuntimePool(pool, profile_registry(101))

    with pytest.raises(WorkshopRuntimeProfileError, match="protected operator policy"):
        await profiles.prepare_execution(RuntimeProfileId.new())

    pool.prepare_execution.assert_not_called()


def test_profile_facade_does_not_expose_transport_named_methods():
    profiles = WorkshopRuntimePool(MagicMock(), profile_registry(101))

    assert not hasattr(profiles, "chat_id")
    assert not hasattr(profiles, "telegram_user_id")
