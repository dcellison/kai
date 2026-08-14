"""Protected transport-neutral Workshop runtime-profile contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from kai.config import Config, UserConfig
from kai.workshop.domain import RuntimeProfileId
from kai.workshop.runtime_profiles import (
    ProtectedRuntimeProfile,
    WorkshopRuntimeProfileError,
    WorkshopRuntimeProfileRegistry,
    compatibility_runtime_config_id_for_profile_id,
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


def test_document_preserves_migrated_profile_identity_and_compatibility_key():
    profile_id = runtime_profile_id_for_config_id(101)
    registry = WorkshopRuntimeProfileRegistry.from_document(
        {
            "version": 1,
            "runtime_profiles": {
                str(profile_id): {
                    "display_name": "Daniel coding",
                    "compatibility_runtime_config_id": 101,
                    "os_user": "daniel",
                    "backend": "codex",
                    "provider": "openai",
                }
            },
        },
        backend_registry={"codex": object()},
    )

    profile = registry.resolve(profile_id)
    assert profile.profile_id == profile_id
    assert profile.runtime_config_id == 101
    assert profile.os_user == "daniel"
    assert profile.backend == "codex"
    assert profile.provider == "openai"


def test_non_telegram_profile_derives_stable_private_compatibility_key():
    profile_id = RuntimeProfileId("rtp_11111111111111111111111111111111")
    first = WorkshopRuntimeProfileRegistry.from_document(
        {
            "version": 1,
            "runtime_profiles": {
                str(profile_id): {
                    "display_name": "Browser-only coding",
                    "backend": "pi",
                    "provider": "openai-codex",
                }
            },
        },
        backend_registry={"pi": object()},
    )
    second = WorkshopRuntimeProfileRegistry.from_document(
        {
            "version": 1,
            "runtime_profiles": {
                str(profile_id): {
                    "display_name": "Browser-only coding",
                    "backend": "pi",
                    "provider": "openai-codex",
                }
            },
        },
        backend_registry={"pi": object()},
    )

    expected = compatibility_runtime_config_id_for_profile_id(profile_id)
    assert first.resolve(profile_id).runtime_config_id == expected
    assert second.resolve(profile_id).runtime_config_id == expected
    assert expected > 0


@pytest.mark.parametrize(
    ("backend", "provider"),
    (
        ("claude", "anthropic"),
        ("codex", "openai"),
        ("goose", "openai"),
        ("opencode", "anthropic"),
        ("pi", "openai-codex"),
    ),
)
def test_document_accepts_each_registered_backend_without_priority(backend, provider):
    profile_id = RuntimeProfileId("rtp_22222222222222222222222222222222")
    registry = WorkshopRuntimeProfileRegistry.from_document(
        {
            "version": 1,
            "runtime_profiles": {
                str(profile_id): {
                    "display_name": f"{backend} runtime",
                    "backend": backend,
                    "provider": provider,
                }
            },
        },
        backend_registry={backend: object()},
    )

    assert registry.resolve(profile_id).backend == backend


def test_document_rejects_backend_absent_from_protected_registry():
    profile_id = RuntimeProfileId("rtp_33333333333333333333333333333333")

    with pytest.raises(WorkshopRuntimeProfileError, match="not present in the backend registry"):
        WorkshopRuntimeProfileRegistry.from_document(
            {
                "version": 1,
                "runtime_profiles": {
                    str(profile_id): {
                        "display_name": "Unavailable runtime",
                        "backend": "codex",
                        "provider": "openai",
                    }
                },
            },
            backend_registry={"pi": object()},
        )


@pytest.mark.parametrize("backend", ("claude", "codex", "goose"))
def test_document_rejects_provider_not_supported_by_backend(backend):
    profile_id = RuntimeProfileId("rtp_34343434343434343434343434343434")

    with pytest.raises(WorkshopRuntimeProfileError, match=r"provider .* is not valid"):
        WorkshopRuntimeProfileRegistry.from_document(
            {
                "version": 1,
                "runtime_profiles": {
                    str(profile_id): {
                        "display_name": "Invalid provider",
                        "backend": backend,
                        "provider": "not-a-provider",
                    }
                },
            },
            backend_registry={backend: object()},
        )


@pytest.mark.parametrize(
    ("backend", "canonical_provider"),
    (("claude", "anthropic"), ("codex", "openai")),
)
def test_single_provider_backend_defaults_only_when_provider_is_omitted(backend, canonical_provider):
    profile_id = RuntimeProfileId("rtp_36363636363636363636363636363636")
    registry = WorkshopRuntimeProfileRegistry.from_document(
        {
            "version": 1,
            "runtime_profiles": {
                str(profile_id): {
                    "display_name": "Single-provider runtime",
                    "backend": backend,
                }
            },
        },
        backend_registry={backend: object()},
    )

    assert registry.resolve(profile_id).provider == canonical_provider


def test_document_rejects_invalid_os_user():
    profile_id = RuntimeProfileId("rtp_35353535353535353535353535353535")

    with pytest.raises(WorkshopRuntimeProfileError, match="os_user is invalid"):
        WorkshopRuntimeProfileRegistry.from_document(
            {
                "version": 1,
                "runtime_profiles": {
                    str(profile_id): {
                        "display_name": "Invalid OS user",
                        "os_user": "../../root",
                        "backend": "codex",
                        "provider": "openai",
                    }
                },
            },
            backend_registry={"codex": object()},
        )


@pytest.mark.parametrize(
    "document, message",
    (
        ({}, "version"),
        ({"version": 1, "runtime_profiles": {}}, "non-empty"),
        (
            {
                "version": 1,
                "runtime_profiles": {
                    "rtp_44444444444444444444444444444444": {
                        "display_name": "No provider",
                        "backend": "goose",
                    }
                },
            },
            "provider is required",
        ),
    ),
)
def test_document_fails_closed_on_incomplete_policy(document, message):
    with pytest.raises(WorkshopRuntimeProfileError, match=message):
        WorkshopRuntimeProfileRegistry.from_document(document, backend_registry={"goose": object()})


def test_explicit_policy_file_is_loaded_instead_of_config_projection(tmp_path, monkeypatch):
    profile_id = RuntimeProfileId("rtp_55555555555555555555555555555555")
    policy = tmp_path / "runtime-profiles.yaml"
    policy.write_text(
        """version: 1
runtime_profiles:
  rtp_55555555555555555555555555555555:
    display_name: Browser runtime
    backend: pi
    provider: openai-codex
"""
    )
    monkeypatch.setattr(
        "kai.workshop.runtime_profiles.load_backend_registry",
        lambda: {"pi": object()},
    )

    config = Config(
        telegram_bot_token="test",
        allowed_user_ids=set(),
        default_backend="codex",
        default_provider="openai",
        default_model="gpt-5.6-sol",
        user_configs={},
    )
    registry = WorkshopRuntimeProfileRegistry.load(config, path=policy)

    assert registry.resolve(profile_id).display_name == "Browser runtime"
    with pytest.raises(WorkshopRuntimeProfileError, match="protected operator policy"):
        registry.resolve(runtime_profile_id_for_config_id(101))


def test_loaded_policy_fails_when_migrated_execution_fields_drift(tmp_path, monkeypatch):
    profile_id = runtime_profile_id_for_config_id(101)
    policy = tmp_path / "runtime-profiles.yaml"
    policy.write_text(
        f"""version: 1
runtime_profiles:
  {profile_id}:
    display_name: Daniel
    compatibility_runtime_config_id: 101
    os_user: daniel
    backend: claude
    provider: anthropic
"""
    )
    monkeypatch.setattr(
        "kai.workshop.runtime_profiles.load_backend_registry",
        lambda: {"claude": object(), "codex": object()},
    )

    with pytest.raises(WorkshopRuntimeProfileError, match=r"conflicts with the migrated users\.yaml"):
        WorkshopRuntimeProfileRegistry.load(_config(), path=policy)


def test_uninstalled_development_without_policy_uses_compatibility_projection(monkeypatch):
    monkeypatch.delenv("KAI_INSTALL_DIR", raising=False)
    monkeypatch.delenv("KAI_RUNTIME_PROFILES_YAML", raising=False)
    monkeypatch.setattr(
        "kai.workshop.runtime_profiles.runtime_profiles_path",
        lambda: Path("/definitely/not/a/runtime-policy.yaml"),
    )

    registry = WorkshopRuntimeProfileRegistry.load(_config())

    assert registry.for_config_id(101).display_name == "Daniel"


def test_uninstalled_development_ignores_existing_canonical_policy(monkeypatch):
    monkeypatch.delenv("KAI_INSTALL_DIR", raising=False)
    monkeypatch.delenv("KAI_RUNTIME_PROFILES_YAML", raising=False)
    monkeypatch.setattr(
        "kai.workshop.runtime_profiles._policy_text",
        lambda _path: pytest.fail("development run must not read installed policy"),
    )

    registry = WorkshopRuntimeProfileRegistry.load(_config())

    assert registry.for_config_id(101).display_name == "Daniel"


def test_protected_startup_fails_closed_when_policy_is_unreadable(monkeypatch):
    monkeypatch.setenv("KAI_INSTALL_DIR", "/opt/kai")
    monkeypatch.delenv("KAI_RUNTIME_PROFILES_YAML", raising=False)
    monkeypatch.setattr("kai.workshop.runtime_profiles._read_protected_file", lambda _path: None)

    with pytest.raises(WorkshopRuntimeProfileError, match="missing or unreadable"):
        WorkshopRuntimeProfileRegistry.load(_config())
