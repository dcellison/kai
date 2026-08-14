"""Protected transport-neutral Workshop runtime-profile contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from kai.backend_registry import BackendRegistryEntry
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
    assert daniel.model == "gpt-5.6-sol"
    assert daniel.timeout_seconds == 120
    assert scott.os_user == "sellison"
    assert scott.backend == "claude"
    assert scott.provider == "anthropic"
    assert scott.model == "sonnet"
    assert scott.timeout_seconds == 120


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
        "gpt-5.6-sol",
        120,
        (),
        None,
        None,
        (),
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
                    "model": "gpt-5.6-sol",
                    "timeout_seconds": 240,
                    "allowed_services": ["perplexity", "weather"],
                    "allowed_workspaces": [],
                }
            },
        },
        backend_registry={"codex": {}},
    )

    profile = registry.resolve(profile_id)
    assert profile.profile_id == profile_id
    assert profile.runtime_config_id == 101
    assert profile.os_user == "daniel"
    assert profile.backend == "codex"
    assert profile.provider == "openai"
    assert profile.model == "gpt-5.6-sol"
    assert profile.timeout_seconds == 240
    assert profile.allowed_services == ("perplexity", "weather")


def test_document_owns_resolved_workspace_policy(tmp_path):
    home = tmp_path / "home"
    base = tmp_path / "projects"
    extra = tmp_path / "external"
    for path in (home, base, extra):
        path.mkdir()
    profile_id = runtime_profile_id_for_config_id(101)

    profile = WorkshopRuntimeProfileRegistry.from_document(
        {
            "version": 1,
            "runtime_profiles": {
                str(profile_id): {
                    "display_name": "Daniel coding",
                    "compatibility_runtime_config_id": 101,
                    "backend": "codex",
                    "provider": "openai",
                    "model": "gpt-5.6-sol",
                    "timeout_seconds": 120,
                    "allowed_services": [],
                    "home_workspace": str(home / "."),
                    "workspace_base": str(base / "."),
                    "allowed_workspaces": [str(extra / ".")],
                }
            },
        },
        backend_registry={"codex": {}},
    ).resolve(profile_id)

    assert profile.home_workspace == home.resolve()
    assert profile.workspace_base == base.resolve()
    assert profile.allowed_workspaces == (extra.resolve(),)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("home_workspace", "relative", "must be an absolute path"),
        ("workspace_base", "relative", "must be an absolute path"),
        ("allowed_workspaces", "not-a-list", "must be a list"),
    ),
)
def test_document_rejects_invalid_workspace_policy(field, value, message):
    profile_id = runtime_profile_id_for_config_id(101)
    entry = {
        "display_name": "Daniel coding",
        "compatibility_runtime_config_id": 101,
        "backend": "codex",
        "provider": "openai",
        "model": "gpt-5.6-sol",
        "timeout_seconds": 120,
        "allowed_services": [],
        "allowed_workspaces": [],
    }
    entry[field] = value

    with pytest.raises(WorkshopRuntimeProfileError, match=message):
        WorkshopRuntimeProfileRegistry.from_document(
            {
                "version": 1,
                "runtime_profiles": {str(profile_id): entry},
            },
            backend_registry={"codex": {}},
        )


def test_document_rejects_duplicate_resolved_allowed_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    profile_id = runtime_profile_id_for_config_id(101)

    with pytest.raises(WorkshopRuntimeProfileError, match="duplicated"):
        WorkshopRuntimeProfileRegistry.from_document(
            {
                "version": 1,
                "runtime_profiles": {
                    str(profile_id): {
                        "display_name": "Daniel coding",
                        "compatibility_runtime_config_id": 101,
                        "backend": "codex",
                        "provider": "openai",
                        "model": "gpt-5.6-sol",
                        "timeout_seconds": 120,
                        "allowed_services": [],
                        "allowed_workspaces": [str(workspace), str(workspace / ".")],
                    }
                },
            },
            backend_registry={"codex": {}},
        )


def test_unavailable_workspace_paths_remain_restrictive_and_do_not_create_false_drift(tmp_path):
    unavailable = (tmp_path / "later-mounted").resolve()
    profile_id = runtime_profile_id_for_config_id(101)
    registry = WorkshopRuntimeProfileRegistry.from_document(
        {
            "version": 1,
            "runtime_profiles": {
                str(profile_id): {
                    "display_name": "Daniel",
                    "compatibility_runtime_config_id": 101,
                    "os_user": "daniel",
                    "backend": "codex",
                    "provider": "openai",
                    "model": "gpt-5.6-sol",
                    "timeout_seconds": 120,
                    "allowed_services": [],
                    "home_workspace": str(unavailable),
                    "workspace_base": str(unavailable),
                    "allowed_workspaces": [str(unavailable)],
                }
            },
        },
        backend_registry={"codex": {}},
    )
    unavailable_profile = registry.resolve(profile_id)
    assert unavailable_profile.workspace_base == unavailable
    registry._validate_compatibility_projection(
        Config(
            telegram_bot_token="test",
            allowed_user_ids={101},
            default_backend="codex",
            default_provider="openai",
            default_model="gpt-5.6-sol",
            user_configs={
                101: UserConfig(
                    telegram_id=101,
                    name="Daniel",
                    os_user="daniel",
                    backend="codex",
                )
            },
        )
    )

    unavailable.mkdir()
    registry._validate_compatibility_projection(
        Config(
            telegram_bot_token="test",
            allowed_user_ids={101},
            default_backend="codex",
            default_provider="openai",
            default_model="gpt-5.6-sol",
            user_configs={
                101: UserConfig(
                    telegram_id=101,
                    name="Daniel",
                    os_user="daniel",
                    backend="codex",
                    home_workspace=unavailable,
                    workspace_base=unavailable,
                    allowed_workspaces=[unavailable],
                )
            },
        )
    )


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
                    "model": "openai-codex/gpt-5.5",
                    "timeout_seconds": 120,
                    "allowed_services": [],
                    "allowed_workspaces": [],
                }
            },
        },
        backend_registry={"pi": {}},
    )
    second = WorkshopRuntimeProfileRegistry.from_document(
        {
            "version": 1,
            "runtime_profiles": {
                str(profile_id): {
                    "display_name": "Browser-only coding",
                    "backend": "pi",
                    "provider": "openai-codex",
                    "model": "openai-codex/gpt-5.5",
                    "timeout_seconds": 120,
                    "allowed_services": [],
                    "allowed_workspaces": [],
                }
            },
        },
        backend_registry={"pi": {}},
    )

    expected = compatibility_runtime_config_id_for_profile_id(profile_id)
    assert first.resolve(profile_id).runtime_config_id == expected
    assert second.resolve(profile_id).runtime_config_id == expected
    assert expected > 0


@pytest.mark.parametrize(
    ("backend", "provider", "model"),
    (
        ("claude", "anthropic", "sonnet"),
        ("codex", "openai", "gpt-5.5"),
        ("goose", "openai", "gpt-5.5-pro"),
        ("opencode", "anthropic", "anthropic/claude-sonnet-4-6"),
        ("pi", "openai-codex", "openai-codex/gpt-5.5"),
    ),
)
def test_document_accepts_each_registered_backend_without_priority(backend, provider, model):
    profile_id = RuntimeProfileId("rtp_22222222222222222222222222222222")
    registry = WorkshopRuntimeProfileRegistry.from_document(
        {
            "version": 1,
            "runtime_profiles": {
                str(profile_id): {
                    "display_name": f"{backend} runtime",
                    "backend": backend,
                    "provider": provider,
                    "model": model,
                    "timeout_seconds": 120,
                    "allowed_services": [],
                    "allowed_workspaces": [],
                }
            },
        },
        backend_registry={backend: {}},
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
                        "model": "gpt-5.5",
                        "timeout_seconds": 120,
                        "allowed_services": [],
                        "allowed_workspaces": [],
                    }
                },
            },
            backend_registry={"pi": {}},
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
                        "model": "sonnet",
                        "timeout_seconds": 120,
                        "allowed_services": [],
                        "allowed_workspaces": [],
                    }
                },
            },
            backend_registry={backend: {}},
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
                    "model": "sonnet" if backend == "claude" else "gpt-5.5",
                    "timeout_seconds": 120,
                    "allowed_services": [],
                    "allowed_workspaces": [],
                }
            },
        },
        backend_registry={backend: {}},
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
                        "model": "gpt-5.5",
                        "timeout_seconds": 120,
                        "allowed_services": [],
                        "allowed_workspaces": [],
                    }
                },
            },
            backend_registry={"codex": {}},
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("model", None, "model is required"),
        ("model", "not-a-codex-model", "not valid"),
        ("timeout_seconds", None, "positive integer"),
        ("timeout_seconds", 0, "positive integer"),
        ("timeout_seconds", True, "positive integer"),
    ),
)
def test_document_fails_closed_on_invalid_model_or_timeout(field, value, message):
    profile_id = RuntimeProfileId("rtp_37373737373737373737373737373737")
    profile = {
        "display_name": "Protected runtime",
        "backend": "codex",
        "provider": "openai",
        "model": "gpt-5.5",
        "timeout_seconds": 120,
        "allowed_services": [],
        "allowed_workspaces": [],
    }
    if value is None:
        profile.pop(field)
    else:
        profile[field] = value

    with pytest.raises(WorkshopRuntimeProfileError, match=message):
        WorkshopRuntimeProfileRegistry.from_document(
            {"version": 1, "runtime_profiles": {str(profile_id): profile}},
            backend_registry={"codex": {}},
        )


@pytest.mark.parametrize(
    ("value", "message"),
    (
        (None, "is required"),
        ("perplexity", "must be a list"),
        ([1], "must be service names"),
        ([""], "is invalid"),
        (["*"], "is invalid"),
        (["service/path"], "is invalid"),
        (["perplexity", "perplexity"], "is duplicated"),
    ),
)
def test_document_fails_closed_on_invalid_service_scopes(value, message):
    profile_id = RuntimeProfileId("rtp_40404040404040404040404040404040")
    profile = {
        "display_name": "Protected runtime",
        "backend": "codex",
        "provider": "openai",
        "model": "gpt-5.5",
        "timeout_seconds": 120,
    }
    if value is not None:
        profile["allowed_services"] = value

    with pytest.raises(WorkshopRuntimeProfileError, match=message):
        WorkshopRuntimeProfileRegistry.from_document(
            {"version": 1, "runtime_profiles": {str(profile_id): profile}},
            backend_registry={"codex": {}},
        )


def test_document_enforces_loaded_backend_registry_model_ceiling():
    profile_id = RuntimeProfileId("rtp_38383838383838383838383838383838")
    backend = BackendRegistryEntry(
        id="codex",
        driver="codex",
        runtime="local_process",
        command="/usr/local/bin/codex",
        allowed_models=("gpt-5.4",),
    )

    with pytest.raises(WorkshopRuntimeProfileError, match="not valid"):
        WorkshopRuntimeProfileRegistry.from_document(
            {
                "version": 1,
                "runtime_profiles": {
                    str(profile_id): {
                        "display_name": "Disallowed model",
                        "backend": "codex",
                        "provider": "openai",
                        "model": "gpt-5.5",
                        "timeout_seconds": 120,
                        "allowed_services": [],
                        "allowed_workspaces": [],
                    }
                },
            },
            backend_registry={"codex": backend},
        )


def test_document_rejects_unknown_backend_registry_entry_type():
    profile_id = RuntimeProfileId("rtp_39393939393939393939393939393939")

    with pytest.raises(WorkshopRuntimeProfileError, match=r"registry entry.*invalid"):
        WorkshopRuntimeProfileRegistry.from_document(
            {
                "version": 1,
                "runtime_profiles": {
                    str(profile_id): {
                        "display_name": "Invalid registry entry",
                        "backend": "codex",
                        "provider": "openai",
                        "model": "gpt-5.5",
                        "timeout_seconds": 120,
                        "allowed_services": [],
                        "allowed_workspaces": [],
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
        WorkshopRuntimeProfileRegistry.from_document(document, backend_registry={"goose": {}})


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
    model: openai-codex/gpt-5.5
    timeout_seconds: 120
    allowed_services: []
    allowed_workspaces: []
"""
    )
    monkeypatch.setattr(
        "kai.workshop.runtime_profiles.load_backend_registry",
        lambda: {"pi": {}},
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
    model: sonnet
    timeout_seconds: 120
    allowed_services: []
    allowed_workspaces: []
"""
    )
    monkeypatch.setattr(
        "kai.workshop.runtime_profiles.load_backend_registry",
        lambda: {"claude": {}, "codex": {}},
    )

    with pytest.raises(WorkshopRuntimeProfileError, match=r"conflicts with the migrated users\.yaml"):
        WorkshopRuntimeProfileRegistry.load(_config(), path=policy)


def test_loaded_policy_fails_when_migrated_model_or_timeout_drifts(tmp_path, monkeypatch):
    profile_id = runtime_profile_id_for_config_id(101)
    policy = tmp_path / "runtime-profiles.yaml"
    policy.write_text(
        f"""version: 1
runtime_profiles:
  {profile_id}:
    display_name: Daniel
    compatibility_runtime_config_id: 101
    os_user: daniel
    backend: codex
    provider: openai
    model: gpt-5.5
    timeout_seconds: 999
    allowed_services: []
    allowed_workspaces: []
"""
    )
    monkeypatch.setattr(
        "kai.workshop.runtime_profiles.load_backend_registry",
        lambda: {"codex": {}, "claude": {}},
    )

    with pytest.raises(WorkshopRuntimeProfileError, match=r"conflicts with the migrated users\.yaml"):
        WorkshopRuntimeProfileRegistry.load(_config(), path=policy)


def test_loaded_policy_fails_when_migrated_service_scopes_drift(tmp_path, monkeypatch):
    profile_id = runtime_profile_id_for_config_id(101)
    policy = tmp_path / "runtime-profiles.yaml"
    policy.write_text(
        f"""version: 1
runtime_profiles:
  {profile_id}:
    display_name: Daniel
    compatibility_runtime_config_id: 101
    os_user: daniel
    backend: codex
    provider: openai
    model: gpt-5.6-sol
    timeout_seconds: 120
    allowed_services:
      - perplexity
    allowed_workspaces: []
"""
    )
    monkeypatch.setattr(
        "kai.workshop.runtime_profiles.load_backend_registry",
        lambda: {"codex": {}, "claude": {}},
    )

    with pytest.raises(WorkshopRuntimeProfileError, match=r"conflicts with the migrated users\.yaml"):
        WorkshopRuntimeProfileRegistry.load(_config(), path=policy)


def test_loaded_policy_accepts_migrated_service_scope_reordering(tmp_path, monkeypatch):
    config = _config()
    config.user_configs[101].allowed_services.extend(("perplexity", "weather"))
    profile_id = runtime_profile_id_for_config_id(101)
    policy = tmp_path / "runtime-profiles.yaml"
    policy.write_text(
        f"""version: 1
runtime_profiles:
  {profile_id}:
    display_name: Daniel
    compatibility_runtime_config_id: 101
    os_user: daniel
    backend: codex
    provider: openai
    model: gpt-5.6-sol
    timeout_seconds: 120
    allowed_services:
      - weather
      - perplexity
    allowed_workspaces: []
  {runtime_profile_id_for_config_id(202)}:
    display_name: Scott
    compatibility_runtime_config_id: 202
    os_user: sellison
    backend: claude
    provider: anthropic
    model: sonnet
    timeout_seconds: 120
    allowed_services: []
    allowed_workspaces: []
"""
    )
    monkeypatch.setattr(
        "kai.workshop.runtime_profiles.load_backend_registry",
        lambda: {"codex": {}, "claude": {}},
    )

    registry = WorkshopRuntimeProfileRegistry.load(config, path=policy)

    assert registry.resolve(profile_id).allowed_services == ("weather", "perplexity")


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
