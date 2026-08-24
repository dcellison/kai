"""Transport-neutral settings and workspace authority for Workshop runtimes."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

from kai import sessions
from kai.config import (
    Config,
    WorkspaceConfig,
    canonicalize_model_for_backend,
    models_for_backend,
    validate_model_for_backend,
)
from kai.workshop.domain import AgentId, ChannelId, PrincipalId, RuntimeProfileId
from kai.workshop.execution_state import (
    WorkshopExecutionStateNamespace,
    WorkshopExecutionStateRegistry,
)
from kai.workshop.runtime_pool import WorkshopRuntimePool
from kai.workspace_utils import is_workspace_allowed


class WorkshopSettingsWorkspaceError(RuntimeError):
    """Base failure for a settings/workspace operation."""


class WorkshopSettingsWorkspaceAccessDenied(WorkshopSettingsWorkspaceError):
    """The canonical caller does not own the requested execution lane."""


class WorkshopSettingsWorkspaceValidationError(WorkshopSettingsWorkspaceError):
    """A requested setting or workspace is invalid."""


@dataclass(frozen=True, slots=True)
class EffectiveValue:
    value: str | int
    source: str


@dataclass(frozen=True, slots=True)
class ModelOption:
    model_id: str
    display_name: str


@dataclass(frozen=True, slots=True)
class WorkspaceOption:
    path: str
    name: str
    current: bool
    home: bool


@dataclass(frozen=True, slots=True)
class SettingsWorkspaceSnapshot:
    principal_id: PrincipalId
    channel_id: ChannelId
    runtime_profile_id: RuntimeProfileId
    backend: str
    provider: str
    model: EffectiveValue
    timeout_seconds: EffectiveValue
    workspace: str
    model_options: tuple[ModelOption, ...] | None
    workspaces: tuple[WorkspaceOption, ...]


@dataclass(frozen=True, slots=True)
class WorkspaceConfigSnapshot:
    workspace: str
    model: EffectiveValue
    timeout_seconds: EffectiveValue
    environment_keys: tuple[str, ...]
    prompt: str | None
    has_prompt: bool
    prompt_source: str | None
    override_fields: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SettingsWorkspaceAuthority:
    """Canonical authority for one channel/agent runtime assignment."""

    principal_id: PrincipalId
    channel_id: ChannelId
    agent_id: AgentId
    runtime_profile_id: RuntimeProfileId


class WorkshopSettingsWorkspaceService:
    """Own settings/workspace mutations above all client adapters.

    Public methods and persistence are addressed only by canonical authority;
    transport identities never enter this service.
    """

    def __init__(
        self,
        config: Config,
        runtime_pool: WorkshopRuntimePool,
        execution_state: WorkshopExecutionStateRegistry,
    ) -> None:
        self._config = config
        self._runtime_pool = runtime_pool
        self._execution_state = execution_state
        self._locks: dict[RuntimeProfileId, asyncio.Lock] = {}

    def authority_for_principal_channel(
        self,
        principal_id: str | PrincipalId,
        channel_id: str | ChannelId,
    ) -> SettingsWorkspaceAuthority:
        namespace = self._execution_state.maybe_for_principal_channel(
            principal_id,
            channel_id,
        )
        if namespace is None:
            raise WorkshopSettingsWorkspaceAccessDenied("The principal does not own settings for this channel")
        return self._canonical_authority(namespace)

    def authority_for_principal_profile(
        self,
        principal_id: str | PrincipalId,
        runtime_profile_id: str | RuntimeProfileId,
    ) -> SettingsWorkspaceAuthority:
        namespace = self._execution_state.maybe_for_runtime_profile_id(runtime_profile_id)
        try:
            canonical_principal = principal_id if isinstance(principal_id, PrincipalId) else PrincipalId(principal_id)
        except (TypeError, ValueError) as exc:
            raise WorkshopSettingsWorkspaceAccessDenied("The principal does not own this runtime profile") from exc
        if namespace is None or namespace.principal_id != canonical_principal:
            raise WorkshopSettingsWorkspaceAccessDenied("The principal does not own this runtime profile")
        return self._canonical_authority(namespace)

    def _namespace(
        self,
        authority: SettingsWorkspaceAuthority,
    ) -> WorkshopExecutionStateNamespace:
        namespace = self._execution_state.maybe_for_runtime_profile_id(authority.runtime_profile_id)
        if namespace is None:
            raise WorkshopSettingsWorkspaceAccessDenied("Runtime profile has no canonical execution state")
        if (
            namespace.principal_id != authority.principal_id
            or namespace.channel_id != authority.channel_id
            or namespace.agent_id != authority.agent_id
        ):
            raise WorkshopSettingsWorkspaceAccessDenied("Runtime profile authority changed")
        return namespace

    def _lock(self, authority: SettingsWorkspaceAuthority) -> asyncio.Lock:
        return self._locks.setdefault(authority.runtime_profile_id, asyncio.Lock())

    async def inspect(
        self,
        authority: SettingsWorkspaceAuthority,
    ) -> SettingsWorkspaceSnapshot:
        profile = self._runtime_pool.runtime_profile(authority.runtime_profile_id)
        namespace = self._namespace(authority)
        workspace = await self._runtime_pool.get_effective_workspace(authority.runtime_profile_id)
        model, timeout_value = await self._effective_values(
            authority,
            workspace,
        )
        home = self._runtime_pool.get_home_workspace(authority.runtime_profile_id)
        home_resolved = home.resolve()
        base, allowed = await self._runtime_pool.resolve_workspace_access(authority.runtime_profile_id)
        history = await sessions.get_canonical_workspace_history(namespace)
        candidates: list[Path] = [home, workspace, *allowed]
        candidates.extend(Path(str(item["path"])) for item in history)
        seen: set[Path] = set()
        workspace_options: list[WorkspaceOption] = []
        for path in candidates:
            resolved = path.expanduser().resolve()
            if resolved in seen or not resolved.is_dir():
                continue
            if resolved != home_resolved and not is_workspace_allowed(
                resolved,
                base,
                allowed,
            ):
                continue
            seen.add(resolved)
            workspace_options.append(
                WorkspaceOption(
                    path=str(resolved),
                    name=self._workspace_name(resolved, base, home_resolved),
                    current=resolved == workspace.resolve(),
                    home=resolved == home_resolved,
                )
            )

        curated_models = models_for_backend(profile.backend, profile.provider)
        model_options = (
            tuple(ModelOption(model_id, display_name) for model_id, display_name in curated_models.items())
            if curated_models is not None
            else None
        )
        return SettingsWorkspaceSnapshot(
            principal_id=authority.principal_id,
            channel_id=authority.channel_id,
            runtime_profile_id=authority.runtime_profile_id,
            backend=profile.backend,
            provider=profile.provider,
            model=model,
            timeout_seconds=timeout_value,
            workspace=str(workspace.resolve()),
            model_options=model_options,
            workspaces=tuple(workspace_options),
        )

    async def set_model(
        self,
        authority: SettingsWorkspaceAuthority,
        model: str,
    ) -> SettingsWorkspaceSnapshot:
        profile = self._runtime_pool.runtime_profile(authority.runtime_profile_id)
        normalized = canonicalize_model_for_backend(model.strip(), profile.backend)
        if not normalized or not validate_model_for_backend(
            normalized,
            profile.backend,
            profile.provider,
        ):
            raise WorkshopSettingsWorkspaceValidationError("The model is not allowed by this runtime profile")
        async with self._lock(authority):
            namespace = self._namespace(authority)
            workspace = str(await self._runtime_pool.get_effective_workspace(authority.runtime_profile_id))
            prior_user = await sessions.get_canonical_execution_settings(namespace)
            await sessions.set_canonical_execution_setting(namespace, "model", normalized)
            try:
                await sessions.delete_canonical_workspace_config_setting(
                    namespace,
                    workspace,
                    "model",
                )
            except BaseException:
                if "model" in prior_user:
                    await sessions.set_canonical_execution_setting(
                        namespace,
                        "model",
                        prior_user["model"],
                    )
                else:
                    await sessions.delete_canonical_execution_setting(
                        namespace,
                        "model",
                    )
                raise
            self._runtime_pool.set_model_if_running(
                authority.runtime_profile_id,
                normalized,
            )
            await self._runtime_pool.restart(authority.runtime_profile_id)
            await sessions.clear_canonical_runtime_session(namespace)
        return await self.inspect(authority)

    async def set_timeout(
        self,
        authority: SettingsWorkspaceAuthority,
        timeout_seconds: int,
    ) -> SettingsWorkspaceSnapshot:
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int):
            raise WorkshopSettingsWorkspaceValidationError("Timeout must be a whole number of seconds")
        if timeout_seconds <= 0 or timeout_seconds > 600:
            raise WorkshopSettingsWorkspaceValidationError("Timeout must be between 1 and 600 seconds")
        async with self._lock(authority):
            namespace = self._namespace(authority)
            await sessions.set_canonical_execution_setting(
                namespace,
                "timeout",
                str(timeout_seconds),
            )
            self._runtime_pool.set_timeout_if_running(
                authority.runtime_profile_id,
                timeout_seconds,
            )
        return await self.inspect(authority)

    async def reset_settings(
        self,
        authority: SettingsWorkspaceAuthority,
        field: str | None = None,
    ) -> SettingsWorkspaceSnapshot:
        if field not in {None, "model", "timeout"}:
            raise WorkshopSettingsWorkspaceValidationError("Only model and timeout settings can be reset")
        async with self._lock(authority):
            namespace = self._namespace(authority)
            if field is None:
                await sessions.delete_all_canonical_user_settings(namespace)
            else:
                await sessions.delete_canonical_execution_setting(namespace, field)
            workspace = await self._runtime_pool.get_effective_workspace(authority.runtime_profile_id)
            model, timeout = await self._effective_values(authority, workspace)
            self._runtime_pool.set_model_if_running(
                authority.runtime_profile_id,
                str(model.value),
            )
            self._runtime_pool.set_timeout_if_running(
                authority.runtime_profile_id,
                int(timeout.value),
            )
            await self._runtime_pool.restart(authority.runtime_profile_id)
            await sessions.clear_canonical_runtime_session(namespace)
        return await self.inspect(authority)

    async def switch_workspace(
        self,
        authority: SettingsWorkspaceAuthority,
        workspace_path: str,
    ) -> SettingsWorkspaceSnapshot:
        if not workspace_path or not Path(workspace_path).is_absolute():
            raise WorkshopSettingsWorkspaceValidationError("Workspace path must be absolute")
        requested = Path(workspace_path).expanduser().resolve()
        async with self._lock(authority):
            home = self._runtime_pool.get_home_workspace(authority.runtime_profile_id).resolve()
            base, allowed = await self._runtime_pool.resolve_workspace_access(authority.runtime_profile_id)
            if not requested.is_dir():
                raise WorkshopSettingsWorkspaceValidationError("Workspace directory is unavailable")
            if requested != home and not is_workspace_allowed(
                requested,
                base,
                allowed,
            ):
                raise WorkshopSettingsWorkspaceAccessDenied("Workspace is outside this runtime profile's grants")
            namespace = self._namespace(authority)
            yaml_config = self._config.get_workspace_config(requested)
            workspace_config = await sessions.build_canonical_workspace_config(
                yaml_config,
                requested,
                namespace,
            )
            prior_settings = await sessions.get_canonical_execution_settings(namespace)
            prior_workspace = prior_settings.get("workspace")
            if requested == home:
                await sessions.delete_canonical_execution_setting(namespace, "workspace")
            else:
                await sessions.set_canonical_execution_setting(
                    namespace,
                    "workspace",
                    str(requested),
                )
            try:
                await self._runtime_pool.change_workspace(
                    authority.runtime_profile_id,
                    requested,
                    workspace_config=workspace_config,
                )
            except BaseException:
                if prior_workspace is None:
                    await sessions.delete_canonical_execution_setting(namespace, "workspace")
                else:
                    await sessions.set_canonical_execution_setting(
                        namespace,
                        "workspace",
                        prior_workspace,
                    )
                raise
            await sessions.clear_canonical_runtime_session(namespace)
            if requested != home:
                await sessions.upsert_canonical_workspace_history(
                    namespace,
                    str(requested),
                )
        return await self.inspect(authority)

    async def workspace_config(
        self,
        authority: SettingsWorkspaceAuthority,
        workspace_path: str | None = None,
    ) -> WorkspaceConfigSnapshot:
        namespace = self._namespace(authority)
        workspace = await self._authorized_workspace(authority, workspace_path)
        yaml_config = self._config.get_workspace_config(workspace)
        overrides = await sessions.get_canonical_workspace_config_settings(
            namespace,
            str(workspace),
        )
        model, timeout = await self._effective_values(authority, workspace)
        env_keys: set[str] = set(yaml_config.env if yaml_config and yaml_config.env else ())
        if raw_env := overrides.get("env"):
            try:
                decoded = json.loads(raw_env)
                if isinstance(decoded, dict):
                    env_keys.update(str(key) for key in decoded)
            except json.JSONDecodeError:
                pass
        prompt: str | None = None
        prompt_source: str | None = None
        has_prompt = False
        if "prompt" in overrides:
            prompt = overrides["prompt"]
            prompt_source = "workspace override"
            has_prompt = True
        elif yaml_config is not None:
            if yaml_config.system_prompt:
                prompt = yaml_config.system_prompt
                prompt_source = "workspaces.yaml"
                has_prompt = True
            elif yaml_config.system_prompt_file:
                prompt_source = "workspaces.yaml file"
                has_prompt = True
        return WorkspaceConfigSnapshot(
            workspace=str(workspace),
            model=model,
            timeout_seconds=timeout,
            environment_keys=tuple(sorted(env_keys)),
            prompt=prompt,
            has_prompt=has_prompt,
            prompt_source=prompt_source,
            override_fields=tuple(sorted(overrides)),
        )

    async def set_workspace_config(
        self,
        authority: SettingsWorkspaceAuthority,
        *,
        field: str,
        value: str,
        workspace_path: str | None = None,
    ) -> WorkspaceConfigSnapshot:
        if field not in {"model", "timeout", "env", "prompt"}:
            raise WorkshopSettingsWorkspaceValidationError("Unsupported workspace setting")
        profile = self._runtime_pool.runtime_profile(authority.runtime_profile_id)
        if field == "model":
            value = canonicalize_model_for_backend(value.strip(), profile.backend)
            if not validate_model_for_backend(value, profile.backend, profile.provider):
                raise WorkshopSettingsWorkspaceValidationError("The model is not allowed by this runtime profile")
        elif field == "timeout":
            try:
                parsed_timeout = int(value)
            except ValueError as exc:
                raise WorkshopSettingsWorkspaceValidationError("Timeout must be a whole number of seconds") from exc
            if parsed_timeout <= 0 or parsed_timeout > 600:
                raise WorkshopSettingsWorkspaceValidationError("Timeout must be between 1 and 600 seconds")
            value = str(parsed_timeout)
        elif field == "env":
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError as exc:
                raise WorkshopSettingsWorkspaceValidationError("Environment override must be a JSON object") from exc
            if not isinstance(decoded, dict) or not all(
                isinstance(key, str) and isinstance(item, str) for key, item in decoded.items()
            ):
                raise WorkshopSettingsWorkspaceValidationError(
                    "Environment override must contain string keys and values"
                )
            value = json.dumps(decoded, sort_keys=True)
        async with self._lock(authority):
            workspace = await self._authorized_workspace(
                authority,
                workspace_path,
            )
            await self._set_workspace_config_locked(
                authority,
                workspace,
                field=field,
                value=value,
            )
        return await self.workspace_config(authority, str(workspace))

    async def reset_workspace_config(
        self,
        authority: SettingsWorkspaceAuthority,
        *,
        field: str | None = None,
        workspace_path: str | None = None,
    ) -> WorkspaceConfigSnapshot:
        if field not in {None, "model", "timeout", "env", "prompt"}:
            raise WorkshopSettingsWorkspaceValidationError("Unsupported workspace setting")
        async with self._lock(authority):
            workspace = await self._authorized_workspace(
                authority,
                workspace_path,
            )
            await self._reset_workspace_config_locked(
                authority,
                workspace,
                field=field,
            )
        return await self.workspace_config(authority, str(workspace))

    async def set_workspace_environment_variable(
        self,
        authority: SettingsWorkspaceAuthority,
        *,
        key: str,
        value: str,
        workspace_path: str | None = None,
    ) -> WorkspaceConfigSnapshot:
        if not key:
            raise WorkshopSettingsWorkspaceValidationError("Environment key cannot be empty")
        async with self._lock(authority):
            workspace = await self._authorized_workspace(
                authority,
                workspace_path,
            )
            settings = await sessions.get_canonical_workspace_config_settings(
                self._namespace(authority),
                str(workspace),
            )
            environment = self._decode_environment(settings.get("env"))
            environment[key] = value
            await self._set_workspace_config_locked(
                authority,
                workspace,
                field="env",
                value=json.dumps(environment, sort_keys=True),
            )
        return await self.workspace_config(authority, str(workspace))

    async def delete_workspace_environment_variable(
        self,
        authority: SettingsWorkspaceAuthority,
        *,
        key: str,
        workspace_path: str | None = None,
    ) -> tuple[WorkspaceConfigSnapshot, bool]:
        async with self._lock(authority):
            workspace = await self._authorized_workspace(
                authority,
                workspace_path,
            )
            settings = await sessions.get_canonical_workspace_config_settings(
                self._namespace(authority),
                str(workspace),
            )
            environment = self._decode_environment(settings.get("env"))
            if key not in environment:
                changed = False
            else:
                changed = True
                del environment[key]
                if environment:
                    await self._set_workspace_config_locked(
                        authority,
                        workspace,
                        field="env",
                        value=json.dumps(environment, sort_keys=True),
                    )
                else:
                    await self._reset_workspace_config_locked(
                        authority,
                        workspace,
                        field="env",
                    )
        return await self.workspace_config(authority, str(workspace)), changed

    async def _set_workspace_config_locked(
        self,
        authority: SettingsWorkspaceAuthority,
        workspace: Path,
        *,
        field: str,
        value: str,
    ) -> None:
        namespace = self._namespace(authority)
        prior = await sessions.get_canonical_workspace_config_settings(
            namespace,
            str(workspace),
        )
        await sessions.set_canonical_workspace_config_setting(
            namespace,
            str(workspace),
            field,
            value,
        )
        try:
            await self._apply_workspace_config(authority, workspace)
        except BaseException:
            if field in prior:
                await sessions.set_canonical_workspace_config_setting(
                    namespace,
                    str(workspace),
                    field,
                    prior[field],
                )
            else:
                await sessions.delete_canonical_workspace_config_setting(
                    namespace,
                    str(workspace),
                    field,
                )
            raise

    async def _reset_workspace_config_locked(
        self,
        authority: SettingsWorkspaceAuthority,
        workspace: Path,
        *,
        field: str | None,
    ) -> None:
        namespace = self._namespace(authority)
        prior = await sessions.get_canonical_workspace_config_settings(
            namespace,
            str(workspace),
        )
        if field is None:
            await sessions.delete_all_canonical_workspace_config(
                namespace,
                str(workspace),
            )
        else:
            await sessions.delete_canonical_workspace_config_setting(
                namespace,
                str(workspace),
                field,
            )
        try:
            await self._apply_workspace_config(authority, workspace)
        except BaseException:
            await sessions.delete_all_canonical_workspace_config(
                namespace,
                str(workspace),
            )
            for prior_field, prior_value in prior.items():
                await sessions.set_canonical_workspace_config_setting(
                    namespace,
                    str(workspace),
                    prior_field,
                    prior_value,
                )
            raise

    async def _authorized_workspace(
        self,
        authority: SettingsWorkspaceAuthority,
        workspace_path: str | None,
    ) -> Path:
        workspace = (
            Path(workspace_path).expanduser().resolve()
            if workspace_path is not None
            else await self._runtime_pool.get_effective_workspace(authority.runtime_profile_id)
        )
        home = self._runtime_pool.get_home_workspace(authority.runtime_profile_id).resolve()
        base, allowed = await self._runtime_pool.resolve_workspace_access(authority.runtime_profile_id)
        if not workspace.is_dir():
            raise WorkshopSettingsWorkspaceValidationError("Workspace directory is unavailable")
        if workspace != home and not is_workspace_allowed(workspace, base, allowed):
            raise WorkshopSettingsWorkspaceAccessDenied("Workspace is outside this runtime profile's grants")
        return workspace

    async def _apply_workspace_config(
        self,
        authority: SettingsWorkspaceAuthority,
        workspace: Path,
    ) -> None:
        namespace = self._namespace(authority)
        yaml_config: WorkspaceConfig | None = self._config.get_workspace_config(workspace)
        config = await sessions.build_canonical_workspace_config(
            yaml_config,
            workspace,
            namespace,
        )
        model, timeout = await self._effective_values(authority, workspace)
        await self._runtime_pool.apply_workspace_config_if_running(
            authority.runtime_profile_id,
            workspace,
            workspace_config=config,
            model=str(model.value),
            timeout_seconds=int(timeout.value),
        )
        await sessions.clear_canonical_runtime_session(namespace)

    async def _effective_values(
        self,
        authority: SettingsWorkspaceAuthority,
        workspace: Path,
    ) -> tuple[EffectiveValue, EffectiveValue]:
        profile = self._runtime_pool.runtime_profile(authority.runtime_profile_id)
        namespace = self._namespace(authority)
        user = await sessions.get_canonical_execution_settings(namespace)
        workspace_overrides = await sessions.get_canonical_workspace_config_settings(
            namespace,
            str(workspace),
        )
        yaml_config = self._config.get_workspace_config(workspace)

        model = EffectiveValue(profile.model, "runtime policy")
        model_candidates = (
            (workspace_overrides.get("model"), "workspace override"),
            (yaml_config.model if yaml_config else None, "workspaces.yaml"),
            (user.get("model"), "user override"),
        )
        for raw_model, source in model_candidates:
            if not raw_model:
                continue
            candidate = canonicalize_model_for_backend(
                raw_model,
                profile.backend,
            )
            if validate_model_for_backend(
                candidate,
                profile.backend,
                profile.provider,
            ):
                model = EffectiveValue(candidate, source)
                break

        timeout = EffectiveValue(profile.timeout_seconds, "runtime policy")
        timeout_candidates = (
            (workspace_overrides.get("timeout"), "workspace override"),
            (yaml_config.timeout if yaml_config else None, "workspaces.yaml"),
            (user.get("timeout"), "user override"),
        )
        for raw_timeout, source in timeout_candidates:
            if raw_timeout is None:
                continue
            try:
                parsed_timeout = int(raw_timeout)
            except (TypeError, ValueError):
                continue
            if 0 < parsed_timeout <= 600:
                timeout = EffectiveValue(parsed_timeout, source)
                break
        return model, timeout

    @staticmethod
    def _decode_environment(raw: str | None) -> dict[str, str]:
        if raw is None:
            return {}
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        if not isinstance(decoded, dict):
            return {}
        return {
            str(key): str(value) for key, value in decoded.items() if isinstance(key, str) and isinstance(value, str)
        }

    @staticmethod
    def _canonical_authority(
        namespace: WorkshopExecutionStateNamespace,
    ) -> SettingsWorkspaceAuthority:
        return SettingsWorkspaceAuthority(
            principal_id=namespace.principal_id,
            channel_id=namespace.channel_id,
            agent_id=namespace.agent_id,
            runtime_profile_id=namespace.runtime_profile_id,
        )

    @staticmethod
    def _workspace_name(path: Path, base: Path | None, home: Path) -> str:
        if path == home:
            return "Home"
        if base is not None:
            try:
                return str(path.relative_to(base.resolve())) or path.name
            except ValueError:
                pass
        return path.name
