"""Protected transport-neutral Workshop runtime-profile contracts."""

from __future__ import annotations

import pytest

from kai.config import Config, UserConfig
from kai.workshop.domain import RuntimeProfileId
from kai.workshop.runtime_profiles import (
    ProtectedRuntimeProfile,
    WorkshopRuntimeProfileError,
    WorkshopRuntimeProfileRegistry,
    runtime_profile_id_for_config_id,
)


def _config() -> Config:
    return Config(
        telegram_bot_token="test",
        allowed_user_ids={101, 202},
        default_backend="codex",
        default_provider="openai",
        default_model="gpt-5.6-sol",
        user_configs={
            101: UserConfig(
                telegram_id=101,
                name="Daniel",
                os_user="daniel",
                backend="codex",
            ),
            202: UserConfig(
                telegram_id=202,
                name="Scott",
                os_user="sellison",
                backend="claude",
            ),
        },
    )


def test_registry_exposes_opaque_stable_profiles_with_protected_policy():
    first = WorkshopRuntimeProfileRegistry.from_config(_config())
    second = WorkshopRuntimeProfileRegistry.from_config(_config())

    daniel = first.for_config_id(101)
    scott = first.for_config_id(202)

    assert isinstance(daniel.profile_id, RuntimeProfileId)
    assert daniel.profile_id == second.for_config_id(101).profile_id
    assert daniel.profile_id != scott.profile_id
    assert daniel.runtime_config_id == 101
    assert daniel.os_user == "daniel"
    assert daniel.backend == "codex"
    assert daniel.provider == "openai"
    assert scott.os_user == "sellison"
    assert scott.backend == "claude"
    assert scott.provider == "anthropic"


@pytest.mark.parametrize("value", (0, -1, True, "101"))
def test_profile_derivation_rejects_non_positive_integer_configuration_ids(value):
    with pytest.raises(WorkshopRuntimeProfileError, match="positive integer"):
        runtime_profile_id_for_config_id(value)  # type: ignore[arg-type]


def test_registry_rejects_unknown_profile_even_when_it_is_structurally_valid():
    registry = WorkshopRuntimeProfileRegistry.from_config(_config())

    with pytest.raises(WorkshopRuntimeProfileError, match="protected operator policy"):
        registry.resolve(RuntimeProfileId.new())


def test_registry_rejects_duplicate_configuration_authority():
    profile = ProtectedRuntimeProfile(
        runtime_profile_id_for_config_id(101),
        101,
        "Daniel",
        "daniel",
        "codex",
        "openai",
    )

    with pytest.raises(WorkshopRuntimeProfileError, match="Duplicate runtime profile ID"):
        WorkshopRuntimeProfileRegistry((profile, profile))
