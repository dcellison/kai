"""Transport-neutral settings/workspace service tests."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kai import sessions
from kai.config import Config, WorkspaceConfig
from kai.workshop.domain import AgentId, ChannelId, PrincipalId
from kai.workshop.execution_state import (
    WorkshopExecutionStateNamespace,
    WorkshopExecutionStateRegistry,
)
from kai.workshop.settings_workspaces import (
    EffectiveValue,
    WorkshopSettingsWorkspaceAccessDenied,
    WorkshopSettingsWorkspaceConflict,
    WorkshopSettingsWorkspaceService,
    WorkshopSettingsWorkspaceValidationError,
)
from tests.workshop_profiles import profile_id


class _RuntimePool:
    def __init__(self, home: Path, allowed: Path) -> None:
        self.home = home
        self.allowed = allowed
        self.model = "gpt-5.6-sol"
        self.timeout = 120
        self.workspace = home
        self.events: list[str] = []
        self.profile = SimpleNamespace(
            profile_id=profile_id(101),
            legacy_runtime_key=101,
            backend="codex",
            provider="openai",
            model="gpt-5.6-sol",
            timeout_seconds=120,
            maximum_timeout_seconds=600,
            allowed_models=None,
        )

    def runtime_profile(self, _profile_id):
        return self.profile

    def is_running(self, _profile_id) -> bool:
        return True

    async def get_effective_workspace(self, _profile_id) -> Path:
        return self.workspace

    def get_home_workspace(self, _profile_id) -> Path:
        return self.home

    async def resolve_workspace_access(self, _profile_id):
        return self.allowed.parent, [self.allowed]

    def set_model(self, _profile_id, model: str) -> None:
        self.model = model
        self.events.append(f"model:{model}")

    def set_model_if_running(self, _profile_id, model: str) -> None:
        self.model = model
        self.events.append(f"running-model:{model}")

    def set_timeout_if_running(self, _profile_id, timeout: int) -> None:
        self.timeout = timeout
        self.events.append(f"timeout:{timeout}")

    async def restart(self, _profile_id) -> None:
        self.events.append("restart")

    async def change_workspace(self, _profile_id, workspace: Path, *, workspace_config) -> None:
        self.workspace = workspace
        self.events.append(f"workspace:{workspace}")

    async def apply_workspace_config_if_running(
        self,
        _profile_id,
        workspace: Path,
        *,
        workspace_config,
        model: str,
        timeout_seconds: int,
    ) -> None:
        self.workspace = workspace
        self.model = model
        self.timeout = timeout_seconds
        self.events.append(f"workspace-config:{workspace}")


def _service(tmp_path: Path, *, config: Config | None = None):
    home = tmp_path / "home"
    allowed = tmp_path / "projects" / "kai"
    home.mkdir()
    allowed.mkdir(parents=True)
    principal_id = PrincipalId("prn_" + "1" * 32)
    channel_id = ChannelId("chn_" + "2" * 32)
    namespace = WorkshopExecutionStateNamespace(
        principal_id=principal_id,
        channel_id=channel_id,
        agent_id=AgentId("agt_" + "3" * 32),
        runtime_profile_id=profile_id(101),
        legacy_runtime_key=101,
    )
    pool = _RuntimePool(home, allowed)
    config = config or Config(
        telegram_bot_token="unused",
        allowed_user_ids=set(),
        default_backend="codex",
        default_model="gpt-5.6-sol",
    )
    service = WorkshopSettingsWorkspaceService(
        config,
        pool,  # type: ignore[arg-type]
        WorkshopExecutionStateRegistry((namespace,)),
    )
    authority = service.authority_for_principal_channel(
        principal_id,
        channel_id,
    )
    return service, pool, authority, principal_id, channel_id


async def test_exact_canonical_authority_fails_closed(tmp_path: Path) -> None:
    service, _, authority, principal_id, channel_id = _service(tmp_path)

    assert service.authority_for_principal_channel(principal_id, channel_id) == authority
    assert not hasattr(authority, "runtime_config_id")
    with pytest.raises(WorkshopSettingsWorkspaceAccessDenied):
        service.authority_for_principal_channel(
            PrincipalId("prn_" + "4" * 32),
            channel_id,
        )
    with pytest.raises(WorkshopSettingsWorkspaceAccessDenied):
        service.authority_for_principal_channel(
            principal_id,
            ChannelId("chn_" + "5" * 32),
        )


async def test_model_change_uses_one_ordered_core_transaction(
    tmp_path: Path,
    monkeypatch,
) -> None:
    service, pool, authority, _, _ = _service(tmp_path)
    calls: list[tuple[object, ...]] = []

    async def get_settings(_namespace):
        return {"model": "gpt-5.6-sol", "timeout": "120"}

    async def history(_namespace):
        return []

    async def workspace_settings(_namespace, _workspace: str):
        return {}

    async def record(name: str, *args):
        calls.append((name, *args))

    monkeypatch.setattr(sessions, "get_canonical_execution_settings", get_settings)
    monkeypatch.setattr(sessions, "get_canonical_workspace_history", history)
    monkeypatch.setattr(
        sessions,
        "get_canonical_workspace_config_settings",
        workspace_settings,
    )

    async def replace_state(namespace, execution, *, workspace_path=None, workspace_settings=None):
        calls.append(("replace", namespace, execution, workspace_path, workspace_settings))

    monkeypatch.setattr(sessions, "replace_canonical_settings_state", replace_state)
    monkeypatch.setattr(
        sessions,
        "clear_canonical_runtime_session",
        lambda *args: record("clear-session", *args),
    )

    snapshot = await service.set_model(authority, "gpt-5.6-terra")

    assert pool.events == [f"workspace-config:{pool.home}"]
    assert calls == [
        (
            "replace",
            calls[0][1],
            {"model": "gpt-5.6-terra", "timeout": "120"},
            str(pool.home),
            {},
        ),
        ("clear-session", calls[0][1]),
    ]
    assert snapshot.backend == "codex"


async def test_inspection_reports_workspace_precedence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    service, pool, authority, _, _ = _service(tmp_path)

    async def get_settings(_namespace):
        return {"model": "gpt-5.6-sol", "timeout": "120"}

    async def workspace_settings(_namespace, _workspace: str):
        return {"model": "gpt-5.6-terra", "timeout": "240"}

    async def history(_namespace):
        return []

    monkeypatch.setattr(sessions, "get_canonical_execution_settings", get_settings)
    monkeypatch.setattr(
        sessions,
        "get_canonical_workspace_config_settings",
        workspace_settings,
    )
    monkeypatch.setattr(sessions, "get_canonical_workspace_history", history)

    snapshot = await service.inspect(authority)

    assert snapshot.model.value == "gpt-5.6-terra"
    assert snapshot.model.source == "workspace override"
    assert snapshot.timeout_seconds.value == 240
    assert snapshot.timeout_seconds.source == "workspace override"
    assert snapshot.workspaces[0].path == str(pool.home.resolve())
    assert snapshot.workspaces[0].name == "Home"
    assert snapshot.workspaces[0].home is True
    assert pool.events == []


async def test_workspace_policy_precedes_runtime_override_with_default_attribution(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = Config(
        telegram_bot_token="unused",
        allowed_user_ids=set(),
        default_backend="codex",
        default_model="gpt-5.6-sol",
    )
    service, pool, authority, _, _ = _service(tmp_path, config=config)
    config.workspace_configs[pool.home.resolve()] = WorkspaceConfig(
        path=pool.home.resolve(),
        model="gpt-5.6-terra",
        timeout=240,
    )

    async def get_settings(_namespace):
        return {"model": "gpt-5.5", "timeout": "180"}

    async def workspace_settings(_namespace, _workspace):
        return {}

    async def history(_namespace):
        return []

    monkeypatch.setattr(sessions, "get_canonical_execution_settings", get_settings)
    monkeypatch.setattr(sessions, "get_canonical_workspace_config_settings", workspace_settings)
    monkeypatch.setattr(sessions, "get_canonical_workspace_history", history)

    snapshot = await service.inspect(authority)

    assert snapshot.model == EffectiveValue(
        "gpt-5.6-terra",
        "workspace policy",
        "gpt-5.6-sol",
    )
    assert snapshot.timeout_seconds == EffectiveValue(240, "workspace policy", 120)


async def test_model_persistence_failure_does_not_mutate_live_runtime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    service, pool, authority, _, _ = _service(tmp_path)

    async def fail_persistence(*_args, **_kwargs):
        raise OSError("database unavailable")

    async def get_settings(_namespace):
        return {}

    async def workspace_settings(_namespace, _workspace):
        return {}

    async def history(_namespace):
        return []

    monkeypatch.setattr(sessions, "replace_canonical_settings_state", fail_persistence)
    monkeypatch.setattr(sessions, "get_canonical_execution_settings", get_settings)
    monkeypatch.setattr(sessions, "get_canonical_workspace_config_settings", workspace_settings)
    monkeypatch.setattr(sessions, "get_canonical_workspace_history", history)

    with pytest.raises(OSError, match="database unavailable"):
        await service.set_model(authority, "gpt-5.6-terra")

    assert pool.events == []
    assert pool.model == "gpt-5.6-sol"


async def test_workspace_switch_rejects_paths_outside_runtime_grants(
    tmp_path: Path,
) -> None:
    service, pool, authority, _, _ = _service(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()

    with pytest.raises(WorkshopSettingsWorkspaceAccessDenied):
        await service.switch_workspace(authority, str(outside))

    assert pool.events == []


async def test_timeout_validation_precedes_persistence(tmp_path: Path) -> None:
    service, pool, authority, _, _ = _service(tmp_path)

    with pytest.raises(WorkshopSettingsWorkspaceValidationError):
        await service.set_timeout(authority, 601)

    assert pool.events == []


async def test_workspace_config_mutation_applies_core_precedence_without_exposing_env_values(
    tmp_path: Path,
    monkeypatch,
) -> None:
    service, pool, authority, _, _ = _service(tmp_path)
    stored: dict[str, str] = {}

    async def get_settings(_namespace):
        return {}

    async def get_workspace_settings(
        _namespace,
        _workspace: str,
    ):
        return dict(stored)

    async def replace_state(
        _namespace,
        execution,
        *,
        workspace_path=None,
        workspace_settings=None,
    ) -> None:
        stored.clear()
        stored.update(workspace_settings or {})

    async def build_config(*_args):
        return None

    async def clear_session(_namespace) -> None:
        return None

    monkeypatch.setattr(sessions, "get_canonical_execution_settings", get_settings)
    monkeypatch.setattr(
        sessions,
        "get_canonical_workspace_config_settings",
        get_workspace_settings,
    )
    monkeypatch.setattr(
        sessions,
        "replace_canonical_settings_state",
        replace_state,
    )
    monkeypatch.setattr(sessions, "build_canonical_workspace_config", build_config)
    monkeypatch.setattr(sessions, "clear_canonical_runtime_session", clear_session)

    model_snapshot = await service.set_workspace_config(
        authority,
        field="model",
        value="gpt-5.6-terra",
    )
    env_snapshot = await service.set_workspace_environment_variable(
        authority,
        key="WORKSHOP_SECRET",
        value="not-for-read-responses",
    )

    assert model_snapshot.model == EffectiveValue(
        "gpt-5.6-terra",
        "workspace override",
        "gpt-5.6-sol",
    )
    assert env_snapshot.environment_keys == ("WORKSHOP_SECRET",)
    assert "not-for-read-responses" not in repr(env_snapshot)
    assert pool.events == [
        f"workspace-config:{pool.home}",
        f"workspace-config:{pool.home}",
    ]


def _canonical_state(monkeypatch, *, execution=None, workspace=None):
    stored_execution = dict(execution or {})
    stored_workspace = dict(workspace or {})
    clear_calls: list[str] = []
    replace_calls: list[tuple[dict[str, str], dict[str, str] | None]] = []

    async def get_execution(_namespace):
        return dict(stored_execution)

    async def get_workspace(_namespace, _path):
        return dict(stored_workspace)

    async def get_history(_namespace):
        return []

    async def replace_state(
        _namespace,
        new_execution,
        *,
        workspace_path=None,
        workspace_settings=None,
    ):
        stored_execution.clear()
        stored_execution.update(new_execution)
        if workspace_settings is not None:
            stored_workspace.clear()
            stored_workspace.update(workspace_settings)
        replace_calls.append((dict(new_execution), None if workspace_settings is None else dict(workspace_settings)))

    async def clear_session(_namespace):
        clear_calls.append("clear")

    monkeypatch.setattr(sessions, "get_canonical_execution_settings", get_execution)
    monkeypatch.setattr(sessions, "get_canonical_workspace_config_settings", get_workspace)
    monkeypatch.setattr(sessions, "get_canonical_workspace_history", get_history)
    monkeypatch.setattr(sessions, "replace_canonical_settings_state", replace_state)
    monkeypatch.setattr(sessions, "clear_canonical_runtime_session", clear_session)
    monkeypatch.setattr(sessions, "build_canonical_workspace_config", AsyncMock(return_value=None))
    return stored_execution, stored_workspace, replace_calls, clear_calls


async def test_capability_catalog_and_model_validation_share_protected_policy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    service, pool, authority, _, _ = _service(tmp_path)
    pool.profile.allowed_models = ("gpt-5.6-sol",)
    _, _, replace_calls, _ = _canonical_state(monkeypatch)

    snapshot = await service.inspect(authority)

    assert snapshot.model.default_value == "gpt-5.6-sol"
    assert snapshot.timeout_seconds.default_value == 120
    assert snapshot.model_options is not None
    assert tuple(option.model_id for option in snapshot.model_options) == ("gpt-5.6-sol",)
    assert [(item.scope, item.field) for item in snapshot.capabilities] == [
        ("runtime", "model"),
        ("runtime", "timeout"),
        ("runtime", "workspace"),
    ]
    timeout = snapshot.capabilities[1]
    assert (timeout.minimum, timeout.maximum) == (1, 600)

    with pytest.raises(WorkshopSettingsWorkspaceValidationError, match="not allowed"):
        await service.set_model(
            authority,
            "gpt-5.6-terra",
            expected_revision=snapshot.revision,
        )
    assert replace_calls == []
    assert pool.events == []


async def test_timeout_capability_and_validation_follow_protected_runtime_policy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    service, pool, authority, _, _ = _service(tmp_path)
    pool.profile.timeout_seconds = 1800
    pool.profile.maximum_timeout_seconds = 1800
    execution, _, replace_calls, _ = _canonical_state(monkeypatch)

    initial = await service.inspect(authority)

    assert initial.timeout_seconds == EffectiveValue(1800, "runtime policy", 1800)
    assert initial.capabilities[1].maximum == 1800
    workspace = await service.workspace_config(authority)
    assert workspace.capabilities[1].maximum == 1800

    changed = await service.set_timeout(
        authority,
        1200,
        expected_revision=initial.revision,
    )
    assert changed.timeout_seconds == EffectiveValue(1200, "runtime override", 1800)
    assert execution == {"timeout": "1200"}

    reset = await service.reset_settings(
        authority,
        "timeout",
        expected_revision=changed.revision,
    )
    assert reset.timeout_seconds == EffectiveValue(1800, "runtime policy", 1800)
    assert execution == {}

    with pytest.raises(WorkshopSettingsWorkspaceValidationError, match="1800"):
        await service.set_timeout(authority, 1801)
    assert len(replace_calls) == 2


async def test_stale_revision_fails_before_persistent_or_live_mutation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    service, pool, authority, _, _ = _service(tmp_path)
    _, _, replace_calls, clear_calls = _canonical_state(monkeypatch)

    with pytest.raises(WorkshopSettingsWorkspaceConflict, match="reload"):
        await service.set_timeout(
            authority,
            180,
            expected_revision="sws_stale",
        )

    assert replace_calls == []
    assert clear_calls == []
    assert pool.events == []


async def test_session_invalidation_failure_restores_persistence_and_live_runtime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    service, pool, authority, _, _ = _service(tmp_path)
    execution, workspace, replace_calls, _ = _canonical_state(
        monkeypatch,
        execution={"model": "gpt-5.6-sol", "timeout": "120"},
    )
    clear_attempts = 0

    async def fail_clear_once(_namespace):
        nonlocal clear_attempts
        clear_attempts += 1
        raise OSError("session store unavailable")

    monkeypatch.setattr(sessions, "clear_canonical_runtime_session", fail_clear_once)
    snapshot = await service.inspect(authority)

    with pytest.raises(OSError, match="session store unavailable"):
        await service.set_timeout(
            authority,
            180,
            expected_revision=snapshot.revision,
        )

    assert execution == {"model": "gpt-5.6-sol", "timeout": "120"}
    assert workspace == {}
    assert replace_calls[-1] == ({"model": "gpt-5.6-sol", "timeout": "120"}, {})
    assert pool.model == "gpt-5.6-sol"
    assert pool.timeout == 120
    assert pool.events == [
        f"workspace-config:{pool.home}",
        f"workspace-config:{pool.home}",
    ]


async def test_successful_mutation_reports_restart_and_session_invalidation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    service, pool, authority, _, _ = _service(tmp_path)
    execution, _, _, clear_calls = _canonical_state(monkeypatch)
    initial = await service.inspect(authority)

    changed = await service.set_timeout(
        authority,
        180,
        expected_revision=initial.revision,
    )

    assert execution == {"timeout": "180"}
    assert changed.timeout_seconds.value == 180
    assert changed.timeout_seconds.source == "runtime override"
    assert changed.mutation is not None
    assert changed.mutation.changed is True
    assert changed.mutation.runtime_action == "restarted"
    assert changed.mutation.provider_session_invalidated is True
    assert changed.revision != initial.revision
    assert clear_calls == ["clear"]
    assert pool.events == [f"workspace-config:{pool.home}"]


async def test_self_service_workspace_surface_rejects_environment_and_invalid_prompt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    service, _, authority, _, _ = _service(tmp_path)
    _, _, replace_calls, _ = _canonical_state(monkeypatch)
    current = await service.workspace_config(authority)

    with pytest.raises(WorkshopSettingsWorkspaceValidationError, match="Unsupported self-service"):
        await service.set_self_service_workspace_config(
            authority,
            field="env",
            value='{"SECRET":"value"}',
            expected_revision=current.revision,
        )
    with pytest.raises(WorkshopSettingsWorkspaceValidationError, match="no null bytes"):
        await service.set_self_service_workspace_config(
            authority,
            field="prompt",
            value="bad\x00prompt",
            expected_revision=current.revision,
        )
    assert replace_calls == []
