"""Transport-neutral settings/workspace service tests."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from kai import sessions
from kai.config import Config
from kai.workshop.domain import AgentId, ChannelId, PrincipalId
from kai.workshop.execution_state import (
    WorkshopExecutionStateNamespace,
    WorkshopExecutionStateRegistry,
)
from kai.workshop.settings_workspaces import (
    EffectiveValue,
    WorkshopSettingsWorkspaceAccessDenied,
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
            runtime_config_id=101,
            backend="codex",
            provider="openai",
            model="gpt-5.6-sol",
            timeout_seconds=120,
        )

    def runtime_profile(self, _profile_id):
        return self.profile

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


def _service(tmp_path: Path):
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
        runtime_config_id=101,
    )
    pool = _RuntimePool(home, allowed)
    service = WorkshopSettingsWorkspaceService(
        Config(
            telegram_bot_token="unused",
            allowed_user_ids=set(),
            default_backend="codex",
            default_model="gpt-5.6-sol",
        ),
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
    monkeypatch.setattr(
        sessions,
        "set_canonical_execution_setting",
        lambda *args: record("set", *args),
    )
    monkeypatch.setattr(
        sessions,
        "delete_canonical_workspace_config_setting",
        lambda *args: record("clear-workspace-model", *args),
    )
    monkeypatch.setattr(
        sessions,
        "clear_canonical_runtime_session",
        lambda *args: record("clear-session", *args),
    )

    snapshot = await service.set_model(authority, "gpt-5.6-terra")

    assert pool.events[:2] == ["running-model:gpt-5.6-terra", "restart"]
    assert calls == [
        ("set", calls[0][1], "model", "gpt-5.6-terra"),
        ("clear-workspace-model", calls[0][1], str(pool.home), "model"),
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


async def test_model_persistence_failure_does_not_mutate_live_runtime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    service, pool, authority, _, _ = _service(tmp_path)

    async def fail_persistence(*_args):
        raise OSError("database unavailable")

    async def get_settings(_namespace):
        return {}

    monkeypatch.setattr(sessions, "set_canonical_execution_setting", fail_persistence)
    monkeypatch.setattr(sessions, "get_canonical_execution_settings", get_settings)

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

    async def set_workspace_setting(
        _namespace,
        _workspace: str,
        field: str,
        value: str,
    ) -> None:
        stored[field] = value

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
        "set_canonical_workspace_config_setting",
        set_workspace_setting,
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
    )
    assert env_snapshot.environment_keys == ("WORKSHOP_SECRET",)
    assert "not-for-read-responses" not in repr(env_snapshot)
    assert pool.events == [
        f"workspace-config:{pool.home}",
        f"workspace-config:{pool.home}",
    ]
