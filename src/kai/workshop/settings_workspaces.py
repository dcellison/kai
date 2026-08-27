"""Transport-neutral settings and workspace authority for Workshop runtimes."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from kai import sessions
from kai.config import (
    Config,
    WorkspaceConfig,
    canonicalize_model_for_backend,
    models_for_backend_policy,
    validate_model_for_backend_policy,
)
from kai.workshop.domain import AgentId, ChannelId, PrincipalId, RuntimeProfileId
from kai.workshop.execution_state import (
    WorkshopExecutionStateNamespace,
    WorkshopExecutionStateRegistry,
)
from kai.workshop.runtime_pool import WorkshopRuntimePool
from kai.workshop.runtime_profiles import WorkshopRuntimeProfileError
from kai.workspace_utils import is_workspace_allowed


class WorkshopSettingsWorkspaceError(RuntimeError):
    """Base failure for a settings/workspace operation."""


class WorkshopSettingsWorkspaceAccessDenied(WorkshopSettingsWorkspaceError):
    """The canonical caller does not own the requested execution lane."""


class WorkshopSettingsWorkspaceValidationError(WorkshopSettingsWorkspaceError):
    """A requested setting or workspace is invalid."""


class WorkshopSettingsWorkspaceConflict(WorkshopSettingsWorkspaceError):
    """The caller based a mutation on stale settings state."""


class WorkshopSettingsWorkspaceBusy(WorkshopSettingsWorkspaceConflict):
    """The principal-local runtime is actively executing and cannot cut over."""


class WorkshopSettingsWorkspaceConsistencyError(WorkshopSettingsWorkspaceError):
    """A failed mutation could not restore persistent and live state."""


MIN_SELF_SERVICE_TIMEOUT_SECONDS = 1
MAX_SELF_SERVICE_PROMPT_CHARACTERS = 32_000


@dataclass(frozen=True, slots=True)
class EffectiveValue:
    value: str | int
    source: str
    default_value: str | int


@dataclass(frozen=True, slots=True)
class EditableCapability:
    field: str
    scope: str
    value_type: str
    resettable: bool
    choices: tuple[str, ...] | None = None
    minimum: int | None = None
    maximum: int | None = None


@dataclass(frozen=True, slots=True)
class SettingsMutationOutcome:
    operation: str
    changed: bool
    runtime_action: str
    provider_session_invalidated: bool


@dataclass(frozen=True, slots=True)
class ModelOption:
    model_id: str
    display_name: str


@dataclass(frozen=True, slots=True)
class BackendOption:
    option_id: str
    backend: str
    provider: str
    current: bool


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
    backend_option_id: str
    backend: str
    provider: str
    model: EffectiveValue
    timeout_seconds: EffectiveValue
    workspace: str
    model_options: tuple[ModelOption, ...] | None
    workspaces: tuple[WorkspaceOption, ...]
    revision: str
    capabilities: tuple[EditableCapability, ...]
    backend_options: tuple[BackendOption, ...] = ()
    mutation: SettingsMutationOutcome | None = None


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
    revision: str
    capabilities: tuple[EditableCapability, ...]
    mutation: SettingsMutationOutcome | None = None


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
        async with self._lock(authority):
            return await self._inspect_locked(authority)

    async def _inspect_locked(
        self,
        authority: SettingsWorkspaceAuthority,
        *,
        mutation: SettingsMutationOutcome | None = None,
    ) -> SettingsWorkspaceSnapshot:
        profile = self._runtime_pool.runtime_profile(authority.runtime_profile_id)
        effective_backend, effective_provider = self._runtime_pool.get_backend_provider(authority.runtime_profile_id)
        effective_option = profile.backend_option(f"{effective_backend}:{effective_provider}")
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

        curated_models = models_for_backend_policy(
            effective_option.backend,
            effective_option.provider,
            allowed_models=effective_option.allowed_models,
        )
        model_options = (
            tuple(ModelOption(model_id, display_name) for model_id, display_name in curated_models.items())
            if curated_models is not None
            else None
        )
        revision = await self._revision(authority, workspace)
        return SettingsWorkspaceSnapshot(
            principal_id=authority.principal_id,
            channel_id=authority.channel_id,
            runtime_profile_id=authority.runtime_profile_id,
            backend_option_id=effective_option.option_id,
            backend=effective_backend,
            provider=effective_provider,
            backend_options=tuple(
                BackendOption(
                    option.option_id,
                    option.backend,
                    option.provider,
                    option.option_id == effective_option.option_id,
                )
                for option in profile.backend_options
            ),
            model=model,
            timeout_seconds=timeout_value,
            workspace=str(workspace.resolve()),
            model_options=model_options,
            workspaces=tuple(workspace_options),
            revision=revision,
            capabilities=self._capabilities(
                model_options,
                maximum_timeout_seconds=profile.maximum_timeout_seconds,
                backend_choices=tuple(option.option_id for option in profile.backend_options),
            ),
            mutation=mutation,
        )

    async def set_model(
        self,
        authority: SettingsWorkspaceAuthority,
        model: str,
        *,
        expected_revision: str | None = None,
        clear_workspace_override: bool = True,
    ) -> SettingsWorkspaceSnapshot:
        profile = self._runtime_pool.runtime_profile(authority.runtime_profile_id)
        backend, _provider = self._runtime_pool.get_backend_provider(authority.runtime_profile_id)
        option = profile.backend_option(f"{backend}:{_provider}")
        normalized = canonicalize_model_for_backend(model.strip(), option.backend)
        if not normalized or not validate_model_for_backend_policy(
            normalized,
            option.backend,
            option.provider,
            allowed_models=option.allowed_models,
        ):
            raise WorkshopSettingsWorkspaceValidationError("The model is not allowed by this runtime profile")
        async with self._lock(authority):
            return await self._mutate_runtime_settings_locked(
                authority,
                operation="set_runtime_model",
                expected_revision=expected_revision,
                updates={"model": normalized},
                clear_workspace_field="model" if clear_workspace_override else None,
            )

    async def set_backend(
        self,
        authority: SettingsWorkspaceAuthority,
        backend_option_id: str,
        *,
        expected_revision: str | None = None,
    ) -> SettingsWorkspaceSnapshot:
        requested = backend_option_id.strip().lower()
        profile = self._runtime_pool.runtime_profile(authority.runtime_profile_id)
        try:
            requested_option = profile.backend_option(requested)
        except WorkshopRuntimeProfileError as exc:
            raise WorkshopSettingsWorkspaceValidationError(
                "The backend is not allowed by this runtime profile"
            ) from exc
        async with self._lock(authority):
            current = await self._inspect_locked(authority)
            self._check_revision(current.revision, expected_revision)
            if requested_option.option_id == current.backend_option_id:
                return await self._inspect_locked(
                    authority,
                    mutation=SettingsMutationOutcome(
                        operation="set_runtime_backend",
                        changed=False,
                        runtime_action="unchanged",
                        provider_session_invalidated=False,
                    ),
                )
            if self._runtime_pool.is_in_flight(authority.runtime_profile_id):
                raise WorkshopSettingsWorkspaceBusy(
                    "The backend cannot be changed while this runtime has an active run"
                )
            namespace = self._namespace(authority)
            prior_settings = await sessions.get_canonical_execution_settings(namespace)
            desired_settings = dict(prior_settings)
            desired_settings["backend"] = requested_option.option_id
            # Model overrides remain canonical state but are applied only when
            # valid for the selected backend.  This neither carries an
            # incompatible model into the new process nor discards a useful
            # choice when the principal later switches back.
            was_running = self._runtime_pool.is_running(authority.runtime_profile_id)

            async def commit_selection() -> None:
                await sessions.replace_canonical_settings_state(namespace, desired_settings)
                await sessions.clear_canonical_runtime_session(namespace)

            try:
                selected = await self._runtime_pool.select_backend(
                    authority.runtime_profile_id,
                    requested_option.option_id,
                    commit_selection=commit_selection,
                )
            except BaseException:
                try:
                    await sessions.replace_canonical_settings_state(namespace, prior_settings)
                    active_backend, active_provider = self._runtime_pool.get_backend_provider(
                        authority.runtime_profile_id
                    )
                    if f"{active_backend}:{active_provider}" != current.backend_option_id:
                        await self._runtime_pool.select_backend(
                            authority.runtime_profile_id,
                            current.backend_option_id,
                        )
                except BaseException as rollback_exc:
                    raise WorkshopSettingsWorkspaceConsistencyError(
                        "Backend switch failed and the prior selection could not be restored"
                    ) from rollback_exc
                raise
            if not selected:
                raise WorkshopSettingsWorkspaceBusy(
                    "The backend cannot be changed while this runtime has an active run"
                )
            return await self._inspect_locked(
                authority,
                mutation=SettingsMutationOutcome(
                    operation="set_runtime_backend",
                    changed=True,
                    runtime_action="restarted" if was_running else "deferred_until_next_run",
                    provider_session_invalidated=True,
                ),
            )

    async def set_timeout(
        self,
        authority: SettingsWorkspaceAuthority,
        timeout_seconds: int,
        *,
        expected_revision: str | None = None,
    ) -> SettingsWorkspaceSnapshot:
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int):
            raise WorkshopSettingsWorkspaceValidationError("Timeout must be a whole number of seconds")
        profile = self._runtime_pool.runtime_profile(authority.runtime_profile_id)
        maximum_timeout = profile.maximum_timeout_seconds
        if not MIN_SELF_SERVICE_TIMEOUT_SECONDS <= timeout_seconds <= maximum_timeout:
            raise WorkshopSettingsWorkspaceValidationError(
                f"Timeout must be between {MIN_SELF_SERVICE_TIMEOUT_SECONDS} and {maximum_timeout} seconds"
            )
        async with self._lock(authority):
            return await self._mutate_runtime_settings_locked(
                authority,
                operation="set_runtime_timeout",
                expected_revision=expected_revision,
                updates={"timeout": str(timeout_seconds)},
            )

    async def reset_settings(
        self,
        authority: SettingsWorkspaceAuthority,
        field: str | None = None,
        *,
        expected_revision: str | None = None,
    ) -> SettingsWorkspaceSnapshot:
        if field not in {None, "model", "timeout"}:
            raise WorkshopSettingsWorkspaceValidationError("Only model and timeout settings can be reset")
        async with self._lock(authority):
            return await self._mutate_runtime_settings_locked(
                authority,
                operation="reset_runtime_settings" if field is None else f"reset_runtime_{field}",
                expected_revision=expected_revision,
                removals={"model", "timeout"} if field is None else {field},
            )

    async def switch_workspace(
        self,
        authority: SettingsWorkspaceAuthority,
        workspace_path: str,
        *,
        expected_revision: str | None = None,
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
            current = await self._inspect_locked(authority)
            self._check_revision(current.revision, expected_revision)
            was_running = self._runtime_pool.is_running(authority.runtime_profile_id)
            namespace = self._namespace(authority)
            prior_settings = await sessions.get_canonical_execution_settings(namespace)
            prior_workspace = await self._runtime_pool.get_effective_workspace(authority.runtime_profile_id)
            desired_settings = dict(prior_settings)
            if requested == home:
                desired_settings.pop("workspace", None)
            else:
                desired_settings["workspace"] = str(requested)
            changed = requested != prior_workspace.resolve()
            if not changed:
                return await self._inspect_locked(
                    authority,
                    mutation=SettingsMutationOutcome(
                        operation="switch_workspace",
                        changed=False,
                        runtime_action="unchanged",
                        provider_session_invalidated=False,
                    ),
                )
            await sessions.replace_canonical_settings_state(namespace, desired_settings)
            try:
                await self._apply_runtime_state(authority, requested)
                if requested != home:
                    await sessions.upsert_canonical_workspace_history(
                        namespace,
                        str(requested),
                    )
                await sessions.clear_canonical_runtime_session(namespace)
            except BaseException:
                await self._restore_after_failure(
                    authority,
                    namespace,
                    execution_settings=prior_settings,
                    workspace=prior_workspace,
                )
                raise
            return await self._inspect_locked(
                authority,
                mutation=SettingsMutationOutcome(
                    operation="switch_workspace",
                    changed=True,
                    runtime_action="restarted" if was_running else "deferred_until_next_run",
                    provider_session_invalidated=True,
                ),
            )

    async def workspace_config(
        self,
        authority: SettingsWorkspaceAuthority,
        workspace_path: str | None = None,
    ) -> WorkspaceConfigSnapshot:
        async with self._lock(authority):
            return await self._workspace_config_locked(authority, workspace_path)

    async def _workspace_config_locked(
        self,
        authority: SettingsWorkspaceAuthority,
        workspace_path: str | None = None,
        *,
        mutation: SettingsMutationOutcome | None = None,
    ) -> WorkspaceConfigSnapshot:
        namespace = self._namespace(authority)
        profile = self._runtime_pool.runtime_profile(authority.runtime_profile_id)
        backend, _provider = self._runtime_pool.get_backend_provider(authority.runtime_profile_id)
        backend_option = profile.backend_option(f"{backend}:{_provider}")
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
            revision=self._state_revision(
                authority,
                workspace,
                await sessions.get_canonical_execution_settings(namespace),
                overrides,
            ),
            capabilities=self._workspace_capabilities(
                models_for_backend_policy(
                    backend_option.backend,
                    backend_option.provider,
                    allowed_models=backend_option.allowed_models,
                ),
                maximum_timeout_seconds=profile.maximum_timeout_seconds,
            ),
            mutation=mutation,
        )

    async def set_workspace_config(
        self,
        authority: SettingsWorkspaceAuthority,
        *,
        field: str,
        value: str,
        workspace_path: str | None = None,
        expected_revision: str | None = None,
    ) -> WorkspaceConfigSnapshot:
        if field not in {"model", "timeout", "env", "prompt"}:
            raise WorkshopSettingsWorkspaceValidationError("Unsupported workspace setting")
        profile = self._runtime_pool.runtime_profile(authority.runtime_profile_id)
        backend, _provider = self._runtime_pool.get_backend_provider(authority.runtime_profile_id)
        backend_option = profile.backend_option(f"{backend}:{_provider}")
        if field == "model":
            value = canonicalize_model_for_backend(value.strip(), backend_option.backend)
            if not validate_model_for_backend_policy(
                value,
                backend_option.backend,
                backend_option.provider,
                allowed_models=backend_option.allowed_models,
            ):
                raise WorkshopSettingsWorkspaceValidationError("The model is not allowed by this runtime profile")
        elif field == "timeout":
            try:
                parsed_timeout = int(value)
            except ValueError as exc:
                raise WorkshopSettingsWorkspaceValidationError("Timeout must be a whole number of seconds") from exc
            if not MIN_SELF_SERVICE_TIMEOUT_SECONDS <= parsed_timeout <= profile.maximum_timeout_seconds:
                raise WorkshopSettingsWorkspaceValidationError(
                    f"Timeout must be between {MIN_SELF_SERVICE_TIMEOUT_SECONDS} "
                    f"and {profile.maximum_timeout_seconds} seconds"
                )
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
        elif "\x00" in value or len(value) > MAX_SELF_SERVICE_PROMPT_CHARACTERS:
            raise WorkshopSettingsWorkspaceValidationError(
                f"Prompt must contain at most {MAX_SELF_SERVICE_PROMPT_CHARACTERS} characters and no null bytes"
            )
        async with self._lock(authority):
            workspace = await self._authorized_workspace(
                authority,
                workspace_path,
            )
            current = await self._workspace_config_locked(authority, str(workspace))
            self._check_revision(current.revision, expected_revision)
            prior = await sessions.get_canonical_workspace_config_settings(
                self._namespace(authority),
                str(workspace),
            )
            was_running = self._runtime_pool.is_running(authority.runtime_profile_id)
            if prior.get(field) == value:
                return await self._workspace_config_locked(
                    authority,
                    str(workspace),
                    mutation=SettingsMutationOutcome(
                        operation=f"set_workspace_{field}",
                        changed=False,
                        runtime_action="unchanged",
                        provider_session_invalidated=False,
                    ),
                )
            applied = await self._set_workspace_config_locked(
                authority,
                workspace,
                field=field,
                value=value,
            )
            return await self._workspace_config_locked(
                authority,
                str(workspace),
                mutation=SettingsMutationOutcome(
                    operation=f"set_workspace_{field}",
                    changed=True,
                    runtime_action=("restarted" if applied and was_running else "deferred_until_next_run"),
                    provider_session_invalidated=applied,
                ),
            )

    async def reset_workspace_config(
        self,
        authority: SettingsWorkspaceAuthority,
        *,
        field: str | None = None,
        workspace_path: str | None = None,
        expected_revision: str | None = None,
    ) -> WorkspaceConfigSnapshot:
        if field not in {None, "model", "timeout", "env", "prompt"}:
            raise WorkshopSettingsWorkspaceValidationError("Unsupported workspace setting")
        async with self._lock(authority):
            workspace = await self._authorized_workspace(
                authority,
                workspace_path,
            )
            current = await self._workspace_config_locked(authority, str(workspace))
            self._check_revision(current.revision, expected_revision)
            prior = await sessions.get_canonical_workspace_config_settings(
                self._namespace(authority),
                str(workspace),
            )
            was_running = self._runtime_pool.is_running(authority.runtime_profile_id)
            reset_fields = set(prior) if field is None else {field} & set(prior)
            if not reset_fields:
                return await self._workspace_config_locked(
                    authority,
                    str(workspace),
                    mutation=SettingsMutationOutcome(
                        operation="reset_workspace_all" if field is None else f"reset_workspace_{field}",
                        changed=False,
                        runtime_action="unchanged",
                        provider_session_invalidated=False,
                    ),
                )
            applied = await self._reset_workspace_config_locked(
                authority,
                workspace,
                field=field,
            )
            return await self._workspace_config_locked(
                authority,
                str(workspace),
                mutation=SettingsMutationOutcome(
                    operation="reset_workspace_all" if field is None else f"reset_workspace_{field}",
                    changed=True,
                    runtime_action=("restarted" if applied and was_running else "deferred_until_next_run"),
                    provider_session_invalidated=applied,
                ),
            )

    async def set_self_service_workspace_config(
        self,
        authority: SettingsWorkspaceAuthority,
        *,
        field: str,
        value: str,
        workspace_path: str | None = None,
        expected_revision: str,
    ) -> WorkspaceConfigSnapshot:
        if field not in {"model", "timeout", "prompt"}:
            raise WorkshopSettingsWorkspaceValidationError("Unsupported self-service workspace setting")
        return await self.set_workspace_config(
            authority,
            field=field,
            value=value,
            workspace_path=workspace_path,
            expected_revision=expected_revision,
        )

    async def reset_self_service_workspace_config(
        self,
        authority: SettingsWorkspaceAuthority,
        *,
        field: str | None = None,
        workspace_path: str | None = None,
        expected_revision: str,
    ) -> WorkspaceConfigSnapshot:
        if field not in {None, "model", "timeout", "prompt"}:
            raise WorkshopSettingsWorkspaceValidationError("Unsupported self-service workspace setting")
        if field is None:
            # "All" on the self-service surface deliberately preserves the
            # operator/Telegram environment override field.
            snapshot = await self._reset_self_service_workspace_fields(
                authority,
                workspace_path=workspace_path,
                expected_revision=expected_revision,
            )
            return snapshot
        return await self.reset_workspace_config(
            authority,
            field=field,
            workspace_path=workspace_path,
            expected_revision=expected_revision,
        )

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
    ) -> bool:
        namespace = self._namespace(authority)
        prior = await sessions.get_canonical_workspace_config_settings(
            namespace,
            str(workspace),
        )
        prior_execution = await sessions.get_canonical_execution_settings(namespace)
        active = await self._runtime_pool.get_effective_workspace(authority.runtime_profile_id)
        desired = dict(prior)
        desired[field] = value
        await sessions.replace_canonical_settings_state(
            namespace,
            prior_execution,
            workspace_path=str(workspace),
            workspace_settings=desired,
        )
        if workspace.resolve() != active.resolve():
            return False
        try:
            await self._apply_runtime_state(authority, workspace)
            await sessions.clear_canonical_runtime_session(namespace)
        except BaseException:
            await self._restore_after_failure(
                authority,
                namespace,
                execution_settings=prior_execution,
                workspace=workspace,
                workspace_settings=prior,
            )
            raise
        return True

    async def _reset_workspace_config_locked(
        self,
        authority: SettingsWorkspaceAuthority,
        workspace: Path,
        *,
        field: str | None,
    ) -> bool:
        namespace = self._namespace(authority)
        prior = await sessions.get_canonical_workspace_config_settings(
            namespace,
            str(workspace),
        )
        prior_execution = await sessions.get_canonical_execution_settings(namespace)
        active = await self._runtime_pool.get_effective_workspace(authority.runtime_profile_id)
        desired = {} if field is None else {key: value for key, value in prior.items() if key != field}
        await sessions.replace_canonical_settings_state(
            namespace,
            prior_execution,
            workspace_path=str(workspace),
            workspace_settings=desired,
        )
        if workspace.resolve() != active.resolve():
            return False
        try:
            await self._apply_runtime_state(authority, workspace)
            await sessions.clear_canonical_runtime_session(namespace)
        except BaseException:
            await self._restore_after_failure(
                authority,
                namespace,
                execution_settings=prior_execution,
                workspace=workspace,
                workspace_settings=prior,
            )
            raise
        return True

    async def _reset_self_service_workspace_fields(
        self,
        authority: SettingsWorkspaceAuthority,
        *,
        workspace_path: str | None,
        expected_revision: str,
    ) -> WorkspaceConfigSnapshot:
        async with self._lock(authority):
            workspace = await self._authorized_workspace(authority, workspace_path)
            current = await self._workspace_config_locked(authority, str(workspace))
            self._check_revision(current.revision, expected_revision)
            namespace = self._namespace(authority)
            prior = await sessions.get_canonical_workspace_config_settings(namespace, str(workspace))
            was_running = self._runtime_pool.is_running(authority.runtime_profile_id)
            desired = {field: value for field, value in prior.items() if field not in {"model", "timeout", "prompt"}}
            if desired == prior:
                return await self._workspace_config_locked(
                    authority,
                    str(workspace),
                    mutation=SettingsMutationOutcome(
                        operation="reset_workspace_all",
                        changed=False,
                        runtime_action="unchanged",
                        provider_session_invalidated=False,
                    ),
                )
            prior_execution = await sessions.get_canonical_execution_settings(namespace)
            active = await self._runtime_pool.get_effective_workspace(authority.runtime_profile_id)
            await sessions.replace_canonical_settings_state(
                namespace,
                prior_execution,
                workspace_path=str(workspace),
                workspace_settings=desired,
            )
            applied = workspace.resolve() == active.resolve()
            if applied:
                try:
                    await self._apply_runtime_state(authority, workspace)
                    await sessions.clear_canonical_runtime_session(namespace)
                except BaseException:
                    await self._restore_after_failure(
                        authority,
                        namespace,
                        execution_settings=prior_execution,
                        workspace=workspace,
                        workspace_settings=prior,
                    )
                    raise
            return await self._workspace_config_locked(
                authority,
                str(workspace),
                mutation=SettingsMutationOutcome(
                    operation="reset_workspace_all",
                    changed=True,
                    runtime_action=("restarted" if applied and was_running else "deferred_until_next_run"),
                    provider_session_invalidated=applied,
                ),
            )

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

    async def _mutate_runtime_settings_locked(
        self,
        authority: SettingsWorkspaceAuthority,
        *,
        operation: str,
        expected_revision: str | None,
        updates: dict[str, str] | None = None,
        removals: set[str] | None = None,
        clear_workspace_field: str | None = None,
    ) -> SettingsWorkspaceSnapshot:
        current = await self._inspect_locked(authority)
        self._check_revision(current.revision, expected_revision)
        was_running = self._runtime_pool.is_running(authority.runtime_profile_id)
        namespace = self._namespace(authority)
        workspace = await self._runtime_pool.get_effective_workspace(authority.runtime_profile_id)
        prior_execution = await sessions.get_canonical_execution_settings(namespace)
        prior_workspace = await sessions.get_canonical_workspace_config_settings(namespace, str(workspace))
        desired_execution = dict(prior_execution)
        for field in removals or ():
            desired_execution.pop(field, None)
        desired_execution.update(updates or {})
        desired_workspace = dict(prior_workspace)
        if clear_workspace_field is not None:
            desired_workspace.pop(clear_workspace_field, None)
        if desired_execution == prior_execution and desired_workspace == prior_workspace:
            return await self._inspect_locked(
                authority,
                mutation=SettingsMutationOutcome(
                    operation=operation,
                    changed=False,
                    runtime_action="unchanged",
                    provider_session_invalidated=False,
                ),
            )
        await sessions.replace_canonical_settings_state(
            namespace,
            desired_execution,
            workspace_path=str(workspace),
            workspace_settings=desired_workspace,
        )
        try:
            await self._apply_runtime_state(authority, workspace)
            await sessions.clear_canonical_runtime_session(namespace)
        except BaseException:
            await self._restore_after_failure(
                authority,
                namespace,
                execution_settings=prior_execution,
                workspace=workspace,
                workspace_settings=prior_workspace,
            )
            raise
        return await self._inspect_locked(
            authority,
            mutation=SettingsMutationOutcome(
                operation=operation,
                changed=True,
                runtime_action="restarted" if was_running else "deferred_until_next_run",
                provider_session_invalidated=True,
            ),
        )

    async def _apply_runtime_state(
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

    async def _restore_after_failure(
        self,
        authority: SettingsWorkspaceAuthority,
        namespace: WorkshopExecutionStateNamespace,
        *,
        execution_settings: dict[str, str],
        workspace: Path,
        workspace_settings: dict[str, str] | None = None,
    ) -> None:
        try:
            await sessions.replace_canonical_settings_state(
                namespace,
                execution_settings,
                workspace_path=str(workspace) if workspace_settings is not None else None,
                workspace_settings=workspace_settings,
            )
            await self._apply_runtime_state(authority, workspace)
        except BaseException as rollback_exc:
            raise WorkshopSettingsWorkspaceConsistencyError(
                "Settings mutation failed and prior runtime state could not be restored"
            ) from rollback_exc

    async def _revision(
        self,
        authority: SettingsWorkspaceAuthority,
        workspace: Path,
    ) -> str:
        namespace = self._namespace(authority)
        execution = await sessions.get_canonical_execution_settings(namespace)
        workspace_settings = await sessions.get_canonical_workspace_config_settings(namespace, str(workspace))
        return self._state_revision(authority, workspace, execution, workspace_settings)

    @staticmethod
    def _state_revision(
        authority: SettingsWorkspaceAuthority,
        workspace: Path,
        execution_settings: dict[str, str],
        workspace_settings: dict[str, str],
    ) -> str:
        revision_workspace_settings = dict(workspace_settings)
        if raw_environment := revision_workspace_settings.get("env"):
            environment_keys = sorted(WorkshopSettingsWorkspaceService._decode_environment(raw_environment))
            revision_workspace_settings["env"] = json.dumps(
                {"keys": environment_keys},
                separators=(",", ":"),
            )
        encoded = json.dumps(
            {
                "runtime_profile_id": str(authority.runtime_profile_id),
                "channel_id": str(authority.channel_id),
                "agent_id": str(authority.agent_id),
                "workspace": str(workspace.resolve()),
                "execution": execution_settings,
                # Environment values are outside the browser self-service
                # surface and must not become susceptible to offline guessing
                # through a client-visible deterministic digest.
                "workspace_settings": revision_workspace_settings,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return "sws_" + hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _check_revision(current: str, expected: str | None) -> None:
        if expected is not None and expected != current:
            raise WorkshopSettingsWorkspaceConflict("Settings changed since they were loaded; reload and try again")

    @staticmethod
    def _capabilities(
        model_options: tuple[ModelOption, ...] | None,
        *,
        maximum_timeout_seconds: int,
        backend_choices: tuple[str, ...],
    ) -> tuple[EditableCapability, ...]:
        return (
            EditableCapability(
                field="backend",
                scope="runtime",
                value_type="backend_id",
                resettable=False,
                choices=backend_choices,
            ),
            EditableCapability(
                field="model",
                scope="runtime",
                value_type="model_id",
                resettable=True,
                choices=tuple(option.model_id for option in model_options) if model_options is not None else None,
            ),
            EditableCapability(
                field="timeout",
                scope="runtime",
                value_type="integer_seconds",
                resettable=True,
                minimum=MIN_SELF_SERVICE_TIMEOUT_SECONDS,
                maximum=maximum_timeout_seconds,
            ),
            EditableCapability(
                field="workspace",
                scope="runtime",
                value_type="authorized_workspace",
                resettable=False,
            ),
        )

    @staticmethod
    def _workspace_capabilities(
        model_options: dict[str, str] | None,
        *,
        maximum_timeout_seconds: int,
    ) -> tuple[EditableCapability, ...]:
        return (
            EditableCapability(
                "model",
                "workspace",
                "model_id",
                True,
                choices=tuple(model_options) if model_options is not None else None,
            ),
            EditableCapability(
                "timeout",
                "workspace",
                "integer_seconds",
                True,
                minimum=MIN_SELF_SERVICE_TIMEOUT_SECONDS,
                maximum=maximum_timeout_seconds,
            ),
            EditableCapability(
                "prompt",
                "workspace",
                "text",
                True,
                maximum=MAX_SELF_SERVICE_PROMPT_CHARACTERS,
            ),
        )

    async def _effective_values(
        self,
        authority: SettingsWorkspaceAuthority,
        workspace: Path,
    ) -> tuple[EffectiveValue, EffectiveValue]:
        profile = self._runtime_pool.runtime_profile(authority.runtime_profile_id)
        backend, _provider = self._runtime_pool.get_backend_provider(authority.runtime_profile_id)
        backend_option = profile.backend_option(f"{backend}:{_provider}")
        namespace = self._namespace(authority)
        user = await sessions.get_canonical_execution_settings(namespace)
        workspace_overrides = await sessions.get_canonical_workspace_config_settings(
            namespace,
            str(workspace),
        )
        yaml_config = self._config.get_workspace_config(workspace)

        model = EffectiveValue(backend_option.model, "runtime policy", backend_option.model)
        model_candidates = (
            (workspace_overrides.get("model"), "workspace override"),
            (yaml_config.model if yaml_config else None, "workspace policy"),
            (user.get("model"), "runtime override"),
        )
        for raw_model, source in model_candidates:
            if not raw_model:
                continue
            candidate = canonicalize_model_for_backend(
                raw_model,
                backend_option.backend,
            )
            if validate_model_for_backend_policy(
                candidate,
                backend_option.backend,
                backend_option.provider,
                allowed_models=backend_option.allowed_models,
            ):
                model = EffectiveValue(candidate, source, backend_option.model)
                break

        timeout = EffectiveValue(
            profile.timeout_seconds,
            "runtime policy",
            profile.timeout_seconds,
        )
        timeout_candidates = (
            (workspace_overrides.get("timeout"), "workspace override"),
            (yaml_config.timeout if yaml_config else None, "workspace policy"),
            (user.get("timeout"), "runtime override"),
        )
        for raw_timeout, source in timeout_candidates:
            if raw_timeout is None:
                continue
            try:
                parsed_timeout = int(raw_timeout)
            except (TypeError, ValueError):
                continue
            if MIN_SELF_SERVICE_TIMEOUT_SECONDS <= parsed_timeout <= profile.maximum_timeout_seconds:
                timeout = EffectiveValue(
                    parsed_timeout,
                    source,
                    profile.timeout_seconds,
                )
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
