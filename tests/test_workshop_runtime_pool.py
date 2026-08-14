"""Protected profile-addressed Workshop runtime-pool contracts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from kai.backend import AgentResponse, StreamEvent
from kai.config import Config, UserConfig
from kai.workshop.domain import RuntimeProfileId
from kai.workshop.runtime_pool import WorkshopRuntimePool
from kai.workshop.runtime_profiles import (
    ProtectedRuntimeProfile,
    WorkshopRuntimeProfileError,
    WorkshopRuntimeProfileRegistry,
)
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


def test_protected_profile_selects_backend_and_os_user_over_compatibility_config():
    from kai.pool import SubprocessPool

    config = Config(
        telegram_bot_token="test",
        allowed_user_ids={111},
        default_backend="claude",
        default_model="sonnet",
        user_configs={
            111: UserConfig(
                telegram_id=111,
                name="Alice",
                backend="claude",
                os_user="legacy-user",
            )
        },
    )
    profiles = WorkshopRuntimeProfileRegistry(
        (
            ProtectedRuntimeProfile(
                profile_id=RuntimeProfileId("rtp_11111111111111111111111111111111"),
                runtime_config_id=111,
                display_name="Protected coding runtime",
                os_user="protected-user",
                backend="codex",
                provider="openai",
            ),
        )
    )

    pool = SubprocessPool(config=config, services_info=[], runtime_profiles=profiles)
    instance = pool.get(111)

    assert instance.backend_name == "codex"
    assert instance.codex_user == "protected-user"


def test_profile_without_telegram_user_receives_runtime_credential(tmp_path, monkeypatch):
    from kai.pool import SubprocessPool

    runtime_config_id = 987654321012345
    monkeypatch.setattr("kai.backend.DATA_DIR", tmp_path)
    config = Config(
        telegram_bot_token="test",
        allowed_user_ids=set(),
        default_backend="claude",
        default_model="sonnet",
        user_configs={},
    )
    profiles = WorkshopRuntimeProfileRegistry(
        (
            ProtectedRuntimeProfile(
                profile_id=RuntimeProfileId("rtp_22222222222222222222222222222222"),
                runtime_config_id=runtime_config_id,
                display_name="Browser-only runtime",
                os_user=None,
                backend="codex",
                provider="openai",
            ),
        )
    )

    pool = SubprocessPool(config=config, services_info=[], runtime_profiles=profiles)
    instance = pool.get(runtime_config_id)
    principal = pool.internal_api_auth.authenticate(instance._api_context.webhook_secret)

    assert instance.backend_name == "codex"
    assert principal is not None
    assert principal.chat_id == runtime_config_id


def test_negative_group_key_retains_legacy_compatibility_runtime(tmp_path, monkeypatch):
    from kai.pool import SubprocessPool

    monkeypatch.setattr("kai.backend.DATA_DIR", tmp_path)
    config = Config(
        telegram_bot_token="test",
        allowed_user_ids={111},
        default_backend="codex",
        default_provider="openai",
        default_model="gpt-5.6-sol",
        user_configs={111: UserConfig(telegram_id=111, name="Alice")},
    )
    profiles = WorkshopRuntimeProfileRegistry(
        (
            ProtectedRuntimeProfile(
                profile_id=RuntimeProfileId("rtp_33333333333333333333333333333333"),
                runtime_config_id=111,
                display_name="Alice runtime",
                os_user=None,
                backend="codex",
                provider="openai",
            ),
        )
    )

    pool = SubprocessPool(config=config, services_info=[], runtime_profiles=profiles)
    instance = pool.get(-100999)

    assert instance.backend_name == "codex"
    principal = pool.internal_api_auth.authenticate(instance._api_context.webhook_secret)
    assert principal is not None
    assert principal.chat_id == -100999
    assert principal.allowed_services == frozenset()


def test_positive_configuration_key_without_profile_fails_closed(tmp_path, monkeypatch):
    from kai.pool import SubprocessPool

    monkeypatch.setattr("kai.backend.DATA_DIR", tmp_path)
    config = Config(
        telegram_bot_token="test",
        allowed_user_ids={111},
        default_backend="codex",
        default_provider="openai",
        default_model="gpt-5.6-sol",
        user_configs={111: UserConfig(telegram_id=111, name="Alice")},
    )
    profiles = WorkshopRuntimeProfileRegistry(
        (
            ProtectedRuntimeProfile(
                profile_id=RuntimeProfileId("rtp_44444444444444444444444444444444"),
                runtime_config_id=222,
                display_name="Different runtime",
                os_user=None,
                backend="codex",
                provider="openai",
            ),
        )
    )
    pool = SubprocessPool(config=config, services_info=[], runtime_profiles=profiles)

    with pytest.raises(WorkshopRuntimeProfileError, match="no protected runtime profile"):
        pool.get(111)
