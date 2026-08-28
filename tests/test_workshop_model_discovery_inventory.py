from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from kai.backend_registry import BackendRegistryEntry
from kai.workshop.domain import AgentId, ChannelId, PrincipalId, RuntimeAssignmentId, RuntimeProfileId
from kai.workshop.execution_state import (
    WorkshopExecutionStateNamespace,
    WorkshopExecutionStateRegistry,
)
from kai.workshop.model_discovery_inventory import (
    ModelDiscoveryAuthMode,
    ModelDiscoveryInventoryError,
    ModelDiscoveryReadiness,
    ModelDiscoverySelectionStatus,
    WorkshopModelDiscoveryInventoryService,
)
from kai.workshop.runtime_profiles import (
    ProtectedRuntimeBackend,
    ProtectedRuntimeProfile,
    WorkshopRuntimeProfileRegistry,
)


def _id(identifier_type, value: int):
    return identifier_type(f"{identifier_type.prefix}_{value:032x}")


def _executable(path: Path) -> Path:
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o700)
    return path


def _options() -> tuple[ProtectedRuntimeBackend, ...]:
    return (
        ProtectedRuntimeBackend("claude", "anthropic", "claude-default"),
        ProtectedRuntimeBackend("codex", "openai", "codex-default", ("codex-default",)),
        ProtectedRuntimeBackend("opencode", "openrouter", "opencode-default"),
        ProtectedRuntimeBackend("goose", "ollama", "goose-default"),
        ProtectedRuntimeBackend(
            "pi",
            "openai-codex",
            "pi-default",
            role_models=(("fast", "pi-fast"),),
        ),
    )


def _profile(value: int, *, os_user: str | None = "daniel") -> ProtectedRuntimeProfile:
    return ProtectedRuntimeProfile(
        profile_id=_id(RuntimeProfileId, value),
        display_name=f"Profile {value}",
        os_user=os_user,
        backend="claude",
        provider="anthropic",
        model="claude-default",
        timeout_seconds=300,
        allowed_services=(),
        home_workspace=None,
        workspace_base=None,
        allowed_workspaces=(),
        backend_options=_options(),
    )


def _namespace(value: int, principal: int | None = None) -> WorkshopExecutionStateNamespace:
    return WorkshopExecutionStateNamespace(
        principal_id=_id(PrincipalId, principal or value),
        channel_id=_id(ChannelId, value),
        agent_id=_id(AgentId, value),
        runtime_profile_id=_id(RuntimeProfileId, value),
        legacy_runtime_key=None,
    )


def _registry(tmp_path: Path) -> dict[str, BackendRegistryEntry]:
    return {
        backend: BackendRegistryEntry(
            id=backend,
            driver=backend,
            runtime="local_process",
            command=str(_executable(tmp_path / backend)),
        )
        for backend in ("claude", "codex", "opencode", "goose", "pi")
    }


def _service(
    tmp_path: Path,
    *,
    profiles: tuple[ProtectedRuntimeProfile, ...] | None = None,
    namespaces: tuple[WorkshopExecutionStateNamespace, ...] | None = None,
    selected: dict[RuntimeProfileId, tuple[str, str]] | None = None,
    environment: dict[str, str] | None = None,
    backend_registry: dict[str, BackendRegistryEntry] | None = None,
    codex_auth_mode: str = "subscription",
) -> WorkshopModelDiscoveryInventoryService:
    checked_profiles = profiles or (_profile(1),)
    checked_namespaces = namespaces or (_namespace(1),)
    selected_options = selected or {
        profile.profile_id: (profile.backend, profile.provider) for profile in checked_profiles
    }
    return WorkshopModelDiscoveryInventoryService(
        config=SimpleNamespace(codex_auth_mode=codex_auth_mode),  # type: ignore[arg-type]
        runtime_profiles=WorkshopRuntimeProfileRegistry(checked_profiles),
        execution_state=WorkshopExecutionStateRegistry(checked_namespaces),
        backend_registry=backend_registry if backend_registry is not None else _registry(tmp_path),
        selected_backend=lambda profile_id: selected_options[profile_id],
        service_os_user="kai",
        environment=environment or {},
    )


def test_inventory_resolves_all_five_backends_from_canonical_authorities(tmp_path: Path) -> None:
    service = _service(
        tmp_path,
        environment={"OPENROUTER_API_KEY": "opencode-secret"},
    )

    inventory = service.inventories[0]

    assert inventory.principal_id == _id(PrincipalId, 1)
    assert inventory.runtime_profile_id == _id(RuntimeProfileId, 1)
    assert inventory.runtime_assignment_id == RuntimeAssignmentId.derived(
        _id(ChannelId, 1),
        f"runtime-profile:{_id(AgentId, 1)}",
    )
    assert inventory.selected_option_id == "claude:anthropic"
    assert [option.option_id for option in inventory.backends] == [
        "claude:anthropic",
        "codex:openai",
        "goose:ollama",
        "opencode:openrouter",
        "pi:openai-codex",
    ]
    by_backend = {option.backend: option for option in inventory.backends}
    assert by_backend["claude"].status == ModelDiscoverySelectionStatus.SELECTED
    assert by_backend["claude"].auth.mode == ModelDiscoveryAuthMode.SUBSCRIPTION
    assert by_backend["codex"].auth.mode == ModelDiscoveryAuthMode.SUBSCRIPTION
    assert by_backend["opencode"].auth.mode == ModelDiscoveryAuthMode.API_KEY
    assert by_backend["opencode"].readiness == ModelDiscoveryReadiness.READY
    assert by_backend["goose"].auth.mode == ModelDiscoveryAuthMode.LOCAL
    assert by_backend["pi"].auth.mode == ModelDiscoveryAuthMode.SUBSCRIPTION
    assert by_backend["pi"].role_models == (("fast", "pi-fast"),)
    assert by_backend["pi"].cache_inputs.principal_id == _id(PrincipalId, 1)
    assert by_backend["pi"].cache_inputs.backend == "pi"
    assert by_backend["pi"].cache_inputs.provider == "openai-codex"
    assert service.operator_diagnostics.profiles == 1
    assert service.operator_diagnostics.options == 5
    assert service.operator_diagnostics.selected == 1
    assert "opencode-secret" not in repr(inventory)


def test_inventory_isolates_principals_and_supports_multiple_profiles(tmp_path: Path) -> None:
    profiles = (_profile(1), _profile(2), _profile(3))
    namespaces = (_namespace(1, 10), _namespace(2, 10), _namespace(3, 20))
    service = _service(tmp_path, profiles=profiles, namespaces=namespaces)

    first = service.for_principal(_id(PrincipalId, 10))

    assert [item.runtime_profile_id for item in first] == [
        _id(RuntimeProfileId, 1),
        _id(RuntimeProfileId, 2),
    ]
    assert service.for_principal(_id(PrincipalId, 20))[0].runtime_profile_id == _id(RuntimeProfileId, 3)
    assert service.for_principal("not-a-principal") == ()


def test_cache_key_tracks_lane_inputs_but_not_current_selection(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    profile = _profile(1)
    default = _service(tmp_path, backend_registry=registry)
    switched = _service(
        tmp_path,
        backend_registry=registry,
        selected={profile.profile_id: ("opencode", "openrouter")},
    )

    default_keys = {option.option_id: option.cache_key for option in default.inventories[0].backends}
    switched_keys = {option.option_id: option.cache_key for option in switched.inventories[0].backends}

    assert default_keys == switched_keys
    assert switched.inventories[0].selected_option_id == "opencode:openrouter"

    opencode_path = Path(registry["opencode"].command)
    opencode_path.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    modified_keys = {
        option.option_id: option.cache_key
        for option in _service(tmp_path, backend_registry=registry).inventories[0].backends
    }
    assert modified_keys["opencode:openrouter"] != default_keys["opencode:openrouter"]
    for option_id in default_keys.keys() - {"opencode:openrouter"}:
        assert modified_keys[option_id] == default_keys[option_id]


def test_cache_key_tracks_auth_mode_without_containing_secret(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    subscription = _service(tmp_path, backend_registry=registry)
    api_key = _service(
        tmp_path,
        backend_registry=registry,
        codex_auth_mode="api_key",
        environment={"OPENAI_API_KEY": "do-not-leak"},
    )
    subscription_codex = next(option for option in subscription.inventories[0].backends if option.backend == "codex")
    api_key_codex = next(option for option in api_key.inventories[0].backends if option.backend == "codex")

    assert subscription_codex.cache_key != api_key_codex.cache_key
    assert api_key_codex.auth.configured is True
    assert "do-not-leak" not in repr(api_key_codex)
    assert "do-not-leak" not in api_key_codex.cache_key


def test_inventory_classifies_missing_and_unsafe_executables(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    del registry["pi"]
    unsafe = Path(registry["goose"].command)
    unsafe.chmod(0o722)
    unavailable = Path(registry["opencode"].command)
    unavailable.unlink()

    inventory = _service(tmp_path, backend_registry=registry).inventories[0]
    by_backend = {option.backend: option for option in inventory.backends}

    assert by_backend["pi"].status == ModelDiscoverySelectionStatus.MISCONFIGURED
    assert by_backend["pi"].diagnostic == "Backend has no installed registry entry"
    assert by_backend["goose"].status == ModelDiscoverySelectionStatus.MISCONFIGURED
    assert by_backend["opencode"].status == ModelDiscoverySelectionStatus.UNAVAILABLE


def test_selected_but_unavailable_is_identified_as_selected(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    Path(registry["claude"].command).unlink()

    selected = _service(tmp_path, backend_registry=registry).inventories[0].backends[0]

    assert selected.backend == "claude"
    assert selected.selected is True
    assert selected.status == ModelDiscoverySelectionStatus.SELECTED
    assert selected.readiness == ModelDiscoveryReadiness.UNAVAILABLE


def test_api_key_auth_without_key_is_unavailable(tmp_path: Path) -> None:
    service = _service(tmp_path, codex_auth_mode="api_key")
    codex = next(option for option in service.inventories[0].backends if option.backend == "codex")

    assert codex.auth.mode == ModelDiscoveryAuthMode.API_KEY
    assert codex.auth.configured is False
    assert codex.readiness == ModelDiscoveryReadiness.UNAVAILABLE
    assert codex.status == ModelDiscoverySelectionStatus.UNAVAILABLE
    assert codex.diagnostic == "Required environment credential is not configured"


def test_inventory_rejects_selection_outside_protected_policy(tmp_path: Path) -> None:
    profile = _profile(1)
    service = _service(
        tmp_path,
        selected={profile.profile_id: ("goose", "openrouter")},
    )

    with pytest.raises(ModelDiscoveryInventoryError, match="outside protected policy"):
        _ = service.inventories
