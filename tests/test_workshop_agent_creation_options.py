from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

from kai.backend_registry import BackendRegistryEntry
from kai.workshop.agent_creation_options import WorkshopAgentCreationOptionsService
from kai.workshop.domain import AgentId, ChannelId, PrincipalId, RuntimeProfileId
from kai.workshop.execution_state import WorkshopExecutionStateNamespace, WorkshopExecutionStateRegistry
from kai.workshop.model_catalogue import ModelCatalogueRefreshStatus, WorkshopModelCatalogueService
from kai.workshop.model_discovery_inventory import WorkshopModelDiscoveryInventoryService
from kai.workshop.runtime_pool import WorkshopRuntimePool
from kai.workshop.runtime_profiles import (
    ProtectedRuntimeBackend,
    ProtectedRuntimeProfile,
    WorkshopRuntimeProfileRegistry,
)


def _id(identifier_type, value: int):
    return identifier_type(f"{identifier_type.prefix}_{value:032x}")


def _executable(path: Path) -> str:
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o700)
    return str(path)


def _profile(
    value: int,
    home: Path,
    allowed: Path,
    *,
    options: tuple[ProtectedRuntimeBackend, ...] | None = None,
) -> ProtectedRuntimeProfile:
    choices = options or (
        ProtectedRuntimeBackend("claude", "anthropic", "claude-default"),
        ProtectedRuntimeBackend("opencode", "openrouter", "open-default"),
    )
    return ProtectedRuntimeProfile(
        profile_id=_id(RuntimeProfileId, value),
        display_name=f"Profile {value}",
        os_user="daniel",
        backend="claude",
        provider="anthropic",
        model="claude-default",
        timeout_seconds=300,
        maximum_timeout_seconds=1800,
        allowed_services=(),
        home_workspace=home,
        workspace_base=allowed.parent,
        allowed_workspaces=(allowed,),
        backend_options=choices,
    )


def _namespace(value: int, principal: int) -> WorkshopExecutionStateNamespace:
    return WorkshopExecutionStateNamespace(
        _id(PrincipalId, principal),
        _id(ChannelId, value),
        _id(AgentId, value),
        _id(RuntimeProfileId, value),
        None,
    )


class _RuntimePool:
    def __init__(self, profiles: WorkshopRuntimeProfileRegistry) -> None:
        self._profiles = profiles

    def runtime_profile(self, profile_id: RuntimeProfileId) -> ProtectedRuntimeProfile:
        return self._profiles.resolve(profile_id)

    def get_home_workspace(self, profile_id: RuntimeProfileId) -> Path:
        profile = self._profiles.resolve(profile_id)
        assert profile.home_workspace is not None
        if not profile.home_workspace.is_dir():
            raise RuntimeError("unavailable")
        return profile.home_workspace

    async def resolve_workspace_access(self, profile_id: RuntimeProfileId) -> tuple[Path | None, list[Path]]:
        profile = self._profiles.resolve(profile_id)
        return profile.workspace_base, list(profile.allowed_workspaces)


async def _services(
    tmp_path: Path,
    *,
    profiles: tuple[ProtectedRuntimeProfile, ...],
    namespaces: tuple[WorkshopExecutionStateNamespace, ...],
) -> tuple[WorkshopAgentCreationOptionsService, WorkshopModelCatalogueService]:
    registry = WorkshopRuntimeProfileRegistry(profiles)
    execution = WorkshopExecutionStateRegistry(namespaces)
    inventory = WorkshopModelDiscoveryInventoryService(
        config=SimpleNamespace(codex_auth_mode="subscription"),  # type: ignore[arg-type]
        runtime_profiles=registry,
        execution_state=execution,
        backend_registry={
            backend: BackendRegistryEntry(
                id=backend,
                driver=backend,
                runtime="local_process",
                command=_executable(tmp_path / backend),
            )
            for backend in sorted({option.backend for profile in profiles for option in profile.backend_options})
        },
        selected_backend=lambda _profile_id: ("claude", "anthropic"),
        service_os_user="kai",
        environment={"OPENROUTER_API_KEY": "secret-not-for-output"},
    )

    async def selected_model(_profile_id: RuntimeProfileId) -> str:
        return "claude-selected"

    catalogue = await WorkshopModelCatalogueService.open(
        tmp_path / "kai.db",
        inventory,
        selected_model=selected_model,
        curated_models=(
            lambda lane: (
                {"claude-default": "Claude Default", "claude-new": "Claude New"} if lane.backend == "claude" else None
            )
        ),
    )
    return (
        WorkshopAgentCreationOptionsService(
            inventory,
            catalogue,
            cast(WorkshopRuntimePool, _RuntimePool(registry)),
        ),
        catalogue,
    )


async def test_projects_authorized_choices_without_mutation_and_isolates_principals(tmp_path: Path) -> None:
    alice_home = tmp_path / "alice-home"
    alice_repo = tmp_path / "repos" / "kai"
    bob_home = tmp_path / "bob-home"
    bob_repo = tmp_path / "repos" / "vox"
    for path in (alice_home, alice_repo, bob_home, bob_repo):
        path.mkdir(parents=True, exist_ok=True)
    profiles = (
        _profile(1, alice_home, alice_repo),
        _profile(2, bob_home, bob_repo),
    )
    service, catalogue = await _services(
        tmp_path,
        profiles=profiles,
        namespaces=(_namespace(1, 10), _namespace(2, 20)),
    )
    alice = _id(PrincipalId, 10)
    try:
        unsupported = await catalogue.refresh(
            catalogue.authority_for_principal(alice),
            _id(RuntimeProfileId, 1),
            "opencode:openrouter",
        )
        first = await service.inspect(alice)
        second = await service.inspect(alice)
    finally:
        await catalogue.close()

    assert unsupported.status == ModelCatalogueRefreshStatus.UNSUPPORTED
    assert first == second
    assert first.ready is True
    assert [runtime.runtime_profile_id for runtime in first.runtimes] == [_id(RuntimeProfileId, 1)]
    runtime = first.runtimes[0]
    assert runtime.current_backend_option_id == "claude:anthropic"
    assert runtime.default_workspace == str(alice_home)
    assert [(item.name, item.default, item.available) for item in runtime.workspaces] == [
        ("Home", True, True),
        ("kai", False, True),
    ]
    assert (runtime.minimum_timeout_seconds, runtime.default_timeout_seconds, runtime.maximum_timeout_seconds) == (
        1,
        300,
        1800,
    )
    by_backend = {item.option_id: item for item in runtime.backends}
    assert by_backend["claude:anthropic"].default_model == "claude-default"
    assert [item.model_id for item in by_backend["claude:anthropic"].models] == [
        "claude-default",
        "claude-new",
        "claude-selected",
    ]
    assert by_backend["claude:anthropic"].models[-1].retained is True
    assert by_backend["opencode:openrouter"].catalogue_status == "unsupported"
    assert by_backend["opencode:openrouter"].blockers == ()
    assert "secret-not-for-output" not in repr(first)


async def test_reports_actionable_readiness_without_creating_state(tmp_path: Path) -> None:
    missing_home = tmp_path / "missing-home"
    missing_repo = tmp_path / "missing-repo"
    profile = _profile(1, missing_home, missing_repo)
    service, catalogue = await _services(
        tmp_path,
        profiles=(profile,),
        namespaces=(_namespace(1, 10),),
    )
    try:
        snapshot = await service.inspect(_id(PrincipalId, 10))
        unknown = await service.inspect(_id(PrincipalId, 99))
    finally:
        await catalogue.close()

    assert snapshot.ready is False
    assert [item.code for item in snapshot.blockers] == ["no_ready_runtime_profiles"]
    assert [item.code for item in snapshot.runtimes[0].blockers] == ["default_workspace_unavailable"]
    assert unknown.ready is False
    assert unknown.runtimes == ()
    assert [item.code for item in unknown.blockers] == ["no_authorized_runtime_profiles"]


async def test_projects_every_supported_backend_family_in_stable_order(tmp_path: Path) -> None:
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    home.mkdir()
    repo.mkdir()
    options = (
        ProtectedRuntimeBackend("pi", "openai-codex", "pi-default"),
        ProtectedRuntimeBackend("goose", "ollama", "goose-default"),
        ProtectedRuntimeBackend("opencode", "openrouter", "opencode-default"),
        ProtectedRuntimeBackend("codex", "openai", "codex-default"),
        ProtectedRuntimeBackend("claude", "anthropic", "claude-default"),
    )
    profile = _profile(1, home, repo, options=options)
    service, catalogue = await _services(
        tmp_path,
        profiles=(profile,),
        namespaces=(_namespace(1, 10),),
    )
    try:
        snapshot = await service.inspect(_id(PrincipalId, 10))
    finally:
        await catalogue.close()

    assert [item.option_id for item in snapshot.runtimes[0].backends] == [
        "claude:anthropic",
        "codex:openai",
        "goose:ollama",
        "opencode:openrouter",
        "pi:openai-codex",
    ]
    assert all(item.default_model.endswith("-default") for item in snapshot.runtimes[0].backends)


async def test_unavailable_backends_block_readiness_without_hiding_choices(tmp_path: Path) -> None:
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    home.mkdir()
    repo.mkdir()
    profile = _profile(1, home, repo)
    service, catalogue = await _services(
        tmp_path,
        profiles=(profile,),
        namespaces=(_namespace(1, 10),),
    )
    (tmp_path / "claude").unlink()
    (tmp_path / "opencode").unlink()
    try:
        snapshot = await service.inspect(_id(PrincipalId, 10))
    finally:
        await catalogue.close()

    runtime = snapshot.runtimes[0]
    assert runtime.ready is False
    assert [item.code for item in runtime.blockers] == ["no_available_backend"]
    assert [item.code for item in snapshot.blockers] == ["no_ready_runtime_profiles"]
    assert [item.option_id for item in runtime.backends] == [
        "claude:anthropic",
        "opencode:openrouter",
    ]
    assert all(item.blockers[0].code == "backend_unavailable" for item in runtime.backends)
