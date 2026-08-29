"""
Canonical agent-runtime subprocess pool with lazy creation and idle eviction.

Provides functionality to:
1. Manage protected AgentBackend instances by canonical runtime-profile ID
2. Create instances lazily on first message with per-runtime configuration
3. Route prompts to the correct runtime subprocess
4. Evict idle subprocesses to reclaim memory on resource-constrained machines
5. Restore per-user saved workspaces on first interaction

Each runtime gets its own backend instance with full conversation isolation,
independent lifecycle, and OS-level enforcement via sudo -u when an OS user is
selected by protected runtime policy (or compatibility config outside it).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from kai import sessions
from kai.backend import AgentBackend, StreamEvent, require_backend_name, resolve_home_workspace
from kai.claude import ClaudeCodeBackend
from kai.config import (
    OPEN_ENDED_PROVIDERS,
    VALID_BACKENDS,
    Config,
    ModelRole,
    WorkspaceConfig,
    canonicalize_model_for_backend,
    get_default_model_for_backend,
    get_effective_provider,
    get_model_for,
    get_user_backend_and_provider,
    resolve_user_model,
    validate_model_for_backend,
    validate_model_for_backend_policy,
)
from kai.goose import GooseBackend
from kai.internal_api_auth import InternalAPIAuth
from kai.workshop.domain import RuntimeProfileId
from kai.workspace_utils import is_workspace_allowed

if TYPE_CHECKING:
    from kai.workshop.execution_state import WorkshopExecutionStateNamespace
    from kai.workshop.internal_api_contexts import WorkshopInternalAPIContextRegistry
    from kai.workshop.runtime_profiles import ProtectedRuntimeProfile, WorkshopRuntimeProfileRegistry

log = logging.getLogger(__name__)

type RuntimeSelector = int | RuntimeProfileId
type RuntimePoolKey = int | RuntimeProfileId


@dataclass(frozen=True, slots=True)
class _RoutedRuntimeKey:
    runtime_profile_id: RuntimeProfileId
    backend_option_id: str


type RuntimeInstanceKey = RuntimePoolKey | _RoutedRuntimeKey


def _required_legacy_key(value: int | None) -> int:
    """Narrow an unprotected adapter key after canonical routing is excluded."""
    if value is None:
        raise RuntimeError("Unprotected runtime is missing its adapter key")
    return value


# How often the eviction loop checks for idle subprocesses (seconds).
_EVICTION_CHECK_INTERVAL = 60

# Maximum time to wait for shutdown() in force_kill before falling
# back to raw SIGKILL (seconds).
_FORCE_KILL_TIMEOUT = 5


@dataclass(frozen=True, slots=True)
class PreparedBackendSelection:
    """Effective backend-neutral selection bound to one prepared runtime."""

    backend: str
    provider: str
    model: str


@dataclass(frozen=True, slots=True)
class _PreparedRuntimeFingerprint:
    selection: PreparedBackendSelection
    workspace: Path
    timeout_seconds: int


class PreparedBackendExecution:
    """One-shot handle that can dispatch only through its exact prepared runtime."""

    __slots__ = (
        "_consumed",
        "_fingerprint",
        "_instance",
        "_instance_key",
        "_pool",
        "_runtime_selector",
    )

    def __init__(
        self,
        pool: SubprocessPool,
        runtime_selector: RuntimePoolKey,
        instance_key: RuntimeInstanceKey,
        instance: AgentBackend,
    ) -> None:
        self._pool = pool
        self._runtime_selector = runtime_selector
        self._instance_key = instance_key
        self._instance = instance
        self._fingerprint = _runtime_fingerprint(instance)
        self._consumed = False

    @property
    def selection(self) -> PreparedBackendSelection:
        return self._fingerprint.selection

    @property
    def workspace(self) -> Path:
        return self._fingerprint.workspace

    def stage_canonical_history(self, history: str) -> None:
        """Stage restart context on this exact protected runtime."""
        self._pool._validate_prepared(self)
        self._instance.stage_canonical_history(history)

    def stage_canonical_agent_context(self, context: str) -> None:
        """Stage the run-bound agent revision on this exact runtime."""
        self._pool._validate_prepared(self)
        self._instance.stage_canonical_agent_context(context)

    def validate_current(self) -> None:
        """Fail before dispatch if the protected runtime selection drifted."""
        self._pool._validate_prepared(self)

    async def cancel(self) -> None:
        """Stop only the exact runtime bound to this protected handle."""
        await self._pool._cancel_prepared(self)

    async def stream(self, prompt: str | list) -> AsyncGenerator[StreamEvent]:
        if self._consumed:
            raise RuntimeError("Prepared backend execution is one-shot")
        self._consumed = True
        async for event in self._pool._send_prepared(self, prompt):
            yield event


def _runtime_fingerprint(instance: AgentBackend) -> _PreparedRuntimeFingerprint:
    return _PreparedRuntimeFingerprint(
        selection=PreparedBackendSelection(
            backend=require_backend_name(instance),
            provider=instance.provider,
            model=instance.model,
        ),
        workspace=instance.workspace,
        timeout_seconds=instance.timeout_seconds,
    )


class SubprocessPool:
    """
    Agent subprocess pool with canonical protected-runtime lifecycle ownership.

    Each protected RuntimeProfileId gets an independent AgentBackend instance
    running as its configured OS user. Compatibility integer selectors are
    normalized at entry and never become protected lifecycle keys. Negative
    Telegram group IDs and unprotected development runtimes remain explicit
    compatibility keys until their adapter paths are retired.

    Thread safety: send() for a given runtime is serialized by the
    per-channel execution lane. A short per-runtime transition lock also
    serializes execution preparation with backend replacement. The lock is
    not held while an agent streams, and unrelated runtimes remain independent.
    """

    def __init__(
        self,
        *,
        config: Config,
        services_info: list[dict],
        runtime_profiles: WorkshopRuntimeProfileRegistry | None = None,
        internal_api_contexts: WorkshopInternalAPIContextRegistry | None = None,
    ):
        self._config = config
        self._runtime_profiles = runtime_profiles
        available_service_names = {
            service["name"] for service in services_info if isinstance(service.get("name"), str) and service["name"]
        }
        allowed_services_by_profile = {}
        self._services_info_by_runtime: dict[RuntimePoolKey, list[dict]] = {}
        runtime_config_ids = config.allowed_user_ids
        protected_profiles_by_config_id = (
            {
                legacy_key: runtime_profiles.resolve(profile_id)
                for profile_id, legacy_key in runtime_profiles.legacy_runtime_archive
            }
            if runtime_profiles is not None
            else {}
        )
        runtime_policies: list[tuple[RuntimePoolKey, frozenset[str]]] = []
        if runtime_profiles is not None:
            runtime_policies.extend(
                (profile.profile_id, frozenset(profile.allowed_services)) for profile in runtime_profiles.profiles
            )
        else:
            runtime_policies.extend(
                (
                    chat_id,
                    frozenset(
                        config.get_user_config(chat_id).allowed_services
                        if config.get_user_config(chat_id) is not None
                        else ()
                    ),
                )
                for chat_id in runtime_config_ids
            )
        for runtime_key, requested in runtime_policies:
            unavailable = requested - available_service_names
            if unavailable:
                log.warning(
                    "Runtime %s has unavailable allowed_services: %s; denying them",
                    runtime_key,
                    ", ".join(sorted(unavailable)),
                )
            allowed = requested & available_service_names
            if runtime_profiles is not None:
                service_profile_id = runtime_key
            else:
                from kai.workshop.runtime_profiles import runtime_profile_id_for_config_id

                assert isinstance(runtime_key, int)
                service_profile_id = runtime_profile_id_for_config_id(runtime_key)
            allowed_services_by_profile[service_profile_id] = allowed
            self._services_info_by_runtime[runtime_key] = [
                service for service in services_info if service.get("name") in allowed
            ]
        if available_service_names and not any(allowed_services_by_profile.values()):
            log.warning(
                "External services are loaded but no user has an allowed_services grant; "
                "the agent service proxy is disabled for every user"
            )
        # Tokens are random and process-local. The pool owns the store because
        # it creates every persistent backend; webhook.start() receives this
        # same object through bot_data so credential resolution cannot drift.
        if internal_api_contexts is None:
            if config.protected_install:
                raise RuntimeError("Protected runtime requires canonical internal API execution contexts")
            from kai.workshop.internal_api_contexts import WorkshopInternalAPIExecutionContext
            from kai.workshop.runtime_profiles import runtime_profile_id_for_config_id

            contexts_by_runtime = {
                (
                    protected_profiles_by_config_id[runtime_config_id].profile_id
                    if runtime_config_id in protected_profiles_by_config_id
                    else runtime_config_id
                ): WorkshopInternalAPIExecutionContext.for_unprotected_runtime(
                    runtime_config_id,
                    (
                        protected_profiles_by_config_id[runtime_config_id].profile_id
                        if runtime_config_id in protected_profiles_by_config_id
                        else runtime_profile_id_for_config_id(runtime_config_id)
                    ),
                )
                for runtime_config_id in runtime_config_ids
            }
            contexts = tuple(contexts_by_runtime.values())
        else:
            contexts = internal_api_contexts.contexts
            if runtime_profiles is None:
                from kai.workshop.runtime_profiles import runtime_profile_id_for_config_id

                contexts_by_runtime = {
                    runtime_config_id: internal_api_contexts.for_runtime_profile(
                        runtime_profile_id_for_config_id(runtime_config_id)
                    )
                    for runtime_config_id in runtime_config_ids
                }
            else:
                contexts_by_runtime = {
                    profile.profile_id: internal_api_contexts.for_runtime_profile(profile.profile_id)
                    for profile in runtime_profiles.profiles
                }
        self._internal_api_contexts = internal_api_contexts
        self._protected_internal_api_contexts = config.protected_install
        self._contexts_by_runtime = contexts_by_runtime
        self._internal_api_auth = InternalAPIAuth.for_execution_contexts(
            contexts,
            allowed_services_by_profile=allowed_services_by_profile,
        )
        self._pool: dict[RuntimeInstanceKey, AgentBackend] = {}
        self._last_activity: dict[RuntimeInstanceKey, float] = {}
        # Pending one-time setup for a freshly-created backend instance.
        # Split into two flags so a command-path workspace resolver call
        # can finish the workspace half without suppressing the DB-
        # settings half that only the first send() applies. A conflated
        # marker would let a /memory or /workspace read clear the
        # pending state before send() ever had a chance to push the
        # user's stored model/timeout into the new subprocess.
        self._pending_workspace_restore: set[RuntimePoolKey] = set()
        self._pending_settings_restore: set[RuntimePoolKey] = set()
        self._in_flight: set[RuntimeInstanceKey] = set()
        # Canonical backend selection is hydrated from execution state at
        # startup and updated only by the settings authority.  The protected
        # profile remains the policy boundary; this cache merely makes the
        # async persisted choice available to synchronous construction and
        # role-model resolution paths.
        self._selected_backends: dict[RuntimePoolKey, str] = {}
        # Serialize the brief prepare/dispatch boundary with targeted backend
        # replacement.  The lock is released as soon as a send is marked
        # in-flight, so unrelated runtimes and the remainder of the active
        # stream are never held behind a settings mutation.
        self._backend_transition_locks: dict[RuntimePoolKey, asyncio.Lock] = {}
        self._eviction_task: asyncio.Task | None = None

    async def hydrate_backend_selections(self) -> None:
        """Load every protected profile's canonical backend selection."""
        if self._runtime_profiles is None:
            return
        from kai.workshop.runtime_profiles import WorkshopRuntimeProfileError

        for profile in self._runtime_profiles.profiles:
            namespace = self._canonical_namespace(profile.profile_id)
            if namespace is None:
                raise RuntimeError("Protected runtime has no canonical execution namespace")
            selected = (
                (
                    await sessions.read_canonical_execution_settings(
                        Path(self._config.session_db_path),
                        namespace,
                    )
                )
                .get("backend", "")
                .strip()
            )
            if selected:
                try:
                    selected = profile.backend_option(selected).option_id
                except WorkshopRuntimeProfileError:
                    log.error(
                        "Ignoring unauthorized canonical backend %r for runtime profile %s",
                        selected,
                        profile.profile_id,
                    )
                    selected = ""
            self._selected_backends[profile.profile_id] = selected or profile.default_backend_option.option_id

    def _backend_option(self, runtime: RuntimeSelector):
        runtime_key, _legacy_key, profile = self._resolve_runtime(runtime)
        if profile is None:
            return None
        return profile.backend_option(
            self._selected_backends.get(runtime_key, profile.default_backend_option.option_id)
        )

    def _backend_transition_lock(self, runtime_key: RuntimePoolKey) -> asyncio.Lock:
        return self._backend_transition_locks.setdefault(runtime_key, asyncio.Lock())

    def _resolve_runtime(
        self,
        runtime: RuntimeSelector,
    ) -> tuple[RuntimePoolKey, int | None, ProtectedRuntimeProfile | None]:
        """Resolve a caller selector to its canonical pool key and legacy state key."""
        from kai.workshop.runtime_profiles import WorkshopRuntimeProfileError

        if isinstance(runtime, bool):
            raise TypeError("Runtime selector must be a runtime profile ID or integer")
        if isinstance(runtime, RuntimeProfileId):
            if self._runtime_profiles is None:
                raise RuntimeError("Runtime profile selector requires protected runtime policy")
            profile = self._runtime_profiles.resolve(runtime)
            legacy_key = self._runtime_profiles.legacy_runtime_key(profile.profile_id)
            return profile.profile_id, legacy_key, profile
        if not isinstance(runtime, int):
            raise TypeError("Runtime selector must be a runtime profile ID or integer")
        if self._runtime_profiles is None:
            return runtime, runtime, None
        try:
            profile = self._runtime_profiles.profile_for_legacy_runtime_key(runtime)
        except WorkshopRuntimeProfileError:
            if runtime >= 0:
                raise
            return runtime, runtime, None
        return profile.profile_id, runtime, profile

    def _protected_profile(self, runtime: RuntimeSelector):
        """Resolve protected policy while preserving the negative-group bridge."""
        # Imported at call time: kai.workshop's package initialization
        # reaches back into this module through kai.sessions, so a
        # module-level import of any workshop submodule from here is
        # circular for some first-import orders.
        return self._resolve_runtime(runtime)[2]

    def _canonical_namespace(
        self,
        runtime: RuntimeSelector,
    ) -> WorkshopExecutionStateNamespace | None:
        """Build the exact canonical state owner for a protected runtime."""
        runtime_key, _legacy_key, profile = self._resolve_runtime(runtime)
        if profile is None:
            return None
        context = self._contexts_by_runtime.get(runtime_key)
        if context is None:
            raise RuntimeError("Protected runtime has no canonical execution context")
        from kai.workshop.execution_state import WorkshopExecutionStateNamespace

        return WorkshopExecutionStateNamespace(
            principal_id=context.principal_id,
            channel_id=context.channel_id,
            agent_id=context.agent_id,
            runtime_profile_id=context.runtime_profile_id,
            legacy_runtime_key=self._runtime_profiles.legacy_runtime_key(profile.profile_id),
        )

    def get_runtime_profile(self, runtime: RuntimeSelector) -> ProtectedRuntimeProfile | None:
        """Return protected policy for one runtime, when that boundary applies."""
        return self._protected_profile(runtime)

    def get_backend_provider(self, runtime: RuntimeSelector) -> tuple[str, str]:
        """Resolve the active route without creating a backend process."""
        _, runtime_config_id, profile = self._resolve_runtime(runtime)
        if profile is not None:
            option = self._backend_option(runtime)
            assert option is not None
            return option.backend, option.provider
        assert runtime_config_id is not None
        instance = self.get_if_exists(runtime)
        if instance is not None:
            return require_backend_name(instance), instance.provider
        return get_user_backend_and_provider(
            self._config.get_user_config(runtime_config_id),
            self._config,
        )

    def get_os_user(self, runtime: RuntimeSelector) -> str | None:
        """Resolve the OS execution identity from protected policy when present."""
        _, runtime_config_id, profile = self._resolve_runtime(runtime)
        if profile is not None:
            return profile.os_user
        assert runtime_config_id is not None
        user = self._config.get_user_config(runtime_config_id)
        return user.os_user if user is not None else None

    def get_role_model(self, runtime: RuntimeSelector, role: ModelRole) -> str:
        """Resolve an operator role model against the policy-selected route."""
        _, runtime_config_id, profile = self._resolve_runtime(runtime)
        backend, provider = self.get_backend_provider(runtime)
        if profile is not None:
            option = self._backend_option(runtime)
            assert option is not None
            profile_models = dict(option.role_models)
            raw_model = profile_models.get(role.value) or self._config.default_models.get(role.value)
            model = canonicalize_model_for_backend(
                raw_model or get_model_for(role, backend, provider),
                backend,
            )
            if validate_model_for_backend_policy(
                model,
                backend,
                provider,
                allowed_models=option.allowed_models,
            ):
                return model
            fallback = get_model_for(role, backend, provider)
            log.warning(
                "Ignoring %s model %r for runtime %s because it is invalid for %s/%s; using %r",
                role.value,
                model,
                profile.profile_id,
                backend,
                provider,
                fallback,
            )
            return fallback
        assert runtime_config_id is not None
        model = canonicalize_model_for_backend(
            resolve_user_model(
                role,
                self._config.get_user_config(runtime_config_id),
                self._config,
                backend=backend,
                provider=provider,
            ),
            backend,
        )
        if validate_model_for_backend(model, backend, provider):
            return model
        fallback = get_model_for(role, backend, provider)
        log.warning(
            "Ignoring %s model %r for runtime %d because it is invalid for %s/%s; using %r",
            role.value,
            model,
            runtime_config_id,
            backend,
            provider,
            fallback,
        )
        return fallback

    def get_home_workspace(self, runtime: RuntimeSelector) -> Path:
        """Resolve home through protected profile policy when available."""
        _, runtime_config_id, profile = self._resolve_runtime(runtime)
        if profile is not None:
            if profile.home_workspace is not None:
                if not profile.home_workspace.is_dir():
                    raise RuntimeError(
                        f"Protected runtime {profile.profile_id} home workspace is unavailable: "
                        f"{profile.home_workspace}. Mount or create it, or update the protected runtime profile."
                    )
                return profile.home_workspace
            context = self._contexts_by_runtime.get(profile.profile_id)
            if context is None:
                raise RuntimeError("Protected runtime has no canonical home owner")
            path = self._config.session_db_path.parent / "home" / str(context.principal_id)
            if not path.is_dir():
                if self._config.protected_install:
                    raise RuntimeError(
                        f"Protected runtime {profile.profile_id} home is not provisioned: {path}. Run `make install`."
                    )
                path.mkdir(parents=True, exist_ok=True)
            return path
        assert runtime_config_id is not None
        return resolve_home_workspace(runtime_config_id, self._config)

    def get_static_workspace_policy(
        self,
        runtime: RuntimeSelector,
    ) -> tuple[Path | None, tuple[Path, ...], bool]:
        """Return per-runtime base/list policy and whether it is protected."""
        _, runtime_config_id, profile = self._resolve_runtime(runtime)
        if profile is not None:
            return profile.workspace_base, profile.allowed_workspaces, True
        assert runtime_config_id is not None
        user = self._config.get_user_config(runtime_config_id)
        if user is None:
            return None, (), False
        return user.workspace_base, tuple(user.allowed_workspaces), False

    async def resolve_workspace_access(self, runtime: RuntimeSelector) -> tuple[Path | None, list[Path]]:
        """Resolve mutable and static workspace access through runtime policy."""
        _, runtime_config_id, profile = self._resolve_runtime(runtime)
        base, allowed, protected = self.get_static_workspace_policy(runtime)
        if profile is not None:
            namespace = self._canonical_namespace(runtime)
            assert namespace is not None
            return await sessions.resolve_canonical_workspace_access(
                namespace,
                self._config,
                workspace_base=base,
                static_allowed_workspaces=allowed,
            )
        assert runtime_config_id is not None
        return await sessions.resolve_workspace_access(
            runtime_config_id,
            self._config,
            use_user_config=not protected,
            workspace_base=base,
            static_allowed_workspaces=allowed if protected else (),
        )

    @property
    def internal_api_auth(self) -> InternalAPIAuth:
        """Return the credential store shared with the loopback HTTP server."""
        return self._internal_api_auth

    # ── Instance management ─────────────────────────────────────────

    def get(self, runtime: RuntimeSelector) -> AgentBackend:
        """
        Get or create a backend instance for the given user.

        Creates lazily on first access. The subprocess itself starts even
        later (on first send()), not here - __init__ is cheap;
        _ensure_started() is where the process spawns.
        """
        runtime_key, _, _ = self._resolve_runtime(runtime)
        if runtime_key not in self._pool:
            instance = self._create_instance(runtime)
            require_backend_name(instance)
            self._pool[runtime_key] = instance
            # Mark both one-time setup steps as pending. The split
            # matters at command-path callers that finish the workspace
            # half via get_effective_workspace; settings still need to
            # land on the first ordinary send().
            self._pending_workspace_restore.add(runtime_key)
            self._pending_settings_restore.add(runtime_key)
        self._last_activity[runtime_key] = time.monotonic()
        return self._pool[runtime_key]

    def _create_instance(
        self,
        runtime: RuntimeSelector,
        *,
        backend_option_id: str | None = None,
        model_override: str | None = None,
    ) -> AgentBackend:
        """
        Create an AgentBackend for a specific user.

        Protected profiles own execution policy. Compatibility callers use
        the existing UserConfig/global cascade.

        Protected profile fields take precedence over compatibility config.

        Per-user DB overrides (set via /settings or /model) are applied
        in _apply_user_db_settings() since they require async DB access.
        """
        runtime_key, chat_id, protected_profile = self._resolve_runtime(runtime)
        user = self._config.get_user_config(chat_id) if chat_id is not None else None

        # Resolve through the single backend.resolve_home_workspace
        # helper: users.yaml override first, else DATA_DIR/home/<principal_id>/.
        # The old global home field on Config was removed by #353 because
        # it pointed every unconfigured user at a shared directory.
        workspace = self.get_home_workspace(runtime)

        ws_config = self._config.get_workspace_config(workspace)

        # Per-user backend and provider, falling back to global config.
        # Routed through the canonical get_user_backend_and_provider
        # resolver so single-provider backends report their canonical
        # provider even when the provider is unset, matching the same
        # cascade bot.py and the install-time validator use. Without
        # this, a per-user backend differing from the global backend can
        # inherit an incompatible global model.
        if protected_profile is not None:
            option = (
                protected_profile.backend_option(backend_option_id)
                if backend_option_id is not None
                else self._backend_option(runtime)
            )
            assert option is not None
            backend = option.backend
            effective_provider = option.provider
        else:
            backend, effective_provider = get_user_backend_and_provider(user, self._config)
        if backend not in VALID_BACKENDS:
            raise RuntimeError(f"Cannot create backend instance for unsupported backend {backend!r}")
        # get_effective_provider hardcodes the backend->provider rule for
        # claude (anthropic) and codex (openai) so global_provider lines
        # up with effective_provider whenever the user has not overridden
        # the backend; no codex-specific patch needed here.
        global_provider = get_effective_provider(self._config.default_backend, self._config.default_provider)

        # Protected Workshop policy owns the effective conversational model.
        # The compatibility cascade remains only for Telegram groups and
        # uninstalled/dev runs that do not resolve a protected profile.
        if protected_profile is not None:
            option = (
                protected_profile.backend_option(backend_option_id)
                if backend_option_id is not None
                else self._backend_option(runtime)
            )
            assert option is not None
            model = model_override or option.model
        else:
            # Per-user model. When the user's effective backend differs
            # from the global one, the global default_model may not be
            # valid; fall back to the per-backend default instead.
            if user and user.model:
                model = user.model
            elif backend == self._config.default_backend and effective_provider == global_provider:
                model = self._config.default_model
                # Catch the case where the global backend itself is an open-ended
                # provider and DEFAULT_MODEL is a backend-specific value
                # that may be invalid for the open-ended provider.
                # Startup validation passes because open-ended providers accept
                # any model string, but the provider API will reject it.
                if effective_provider in OPEN_ENDED_PROVIDERS:
                    log.warning(
                        "No model configured for open-ended provider '%s' (runtime %s); "
                        "using global default '%s' which may not be valid for this provider",
                        effective_provider,
                        chat_id,
                        model,
                    )
            else:
                model = get_default_model_for_backend(backend, effective_provider)

        # Canonicalize retired backend-specific spellings after the
        # complete precedence cascade and before constructing a backend.
        # This keeps direct Config objects and persisted users.yaml values
        # from reaching the Codex app-server with the rejected gpt-5.6
        # family shorthand.
        model = canonicalize_model_for_backend(model, backend)

        if protected_profile is not None:
            timeout = protected_profile.timeout_seconds
        elif user and user.timeout is not None:
            timeout = user.timeout
        else:
            timeout = self._config.default_timeout
        # home_ws is what the backend treats as "home" for the foreign-
        # workspace reminder. Same resolution as the workspace above so
        # the two cannot drift; pre-#353 this took a different path that
        # could disagree with the active workspace for unconfigured users.
        #
        # This is a deliberate second call to resolve_home_workspace and
        # NOT an alias for `workspace`: workspace is the INITIAL landing
        # directory and can be overridden (DB /workspace switch, a
        # per-user workspace_config pinning a non-home path, etc.) but
        # home_ws is always the canonical per-user home the backend
        # uses to detect "foreign workspace" and inject the identity
        # reminder. If we wrote `home_ws = workspace` the reminder
        # injection would silently break the moment a user pins a
        # non-home default. The redundant stat/mkdir inside
        # ensure_user_home is cheap (idempotent mkdir + chmod); the
        # clarity of two separate resolution calls is worth it.
        home_ws = self.get_home_workspace(runtime)

        # os_user for sudo -u isolation. None = run as bot user.
        # Resolved here (rather than inside each branch) because all
        # conversational backends consume it for per-user subprocess isolation.
        # Each agent's auth state is per-OS-user and operator-
        # provisioned under the target user's home (claude OAuth
        # credentials, codex auth.json, goose keychain or env keys,
        # opencode auth.json via `opencode auth login` run as that
        # user); the `sudo -H` wrap each backend applies is what
        # points the subprocess at the right home.
        os_user = protected_profile.os_user if protected_profile is not None else user.os_user if user else None

        # Backend selection is explicit. No branch may substitute a
        # different backend for a missing or unknown identifier.
        # Internal API availability is independent of public webhook ingress.
        # Agents receive only their random principal credential, never an
        # external webhook signing secret.
        internal_api_context = self._contexts_by_runtime.get(runtime_key)
        if internal_api_context is None:
            if self._protected_internal_api_contexts:
                raise RuntimeError("Runtime has no canonical internal API execution context")
            from kai.workshop.internal_api_contexts import WorkshopInternalAPIExecutionContext
            from kai.workshop.runtime_profiles import runtime_profile_id_for_config_id

            assert chat_id is not None
            internal_api_context = WorkshopInternalAPIExecutionContext.for_unprotected_runtime(
                chat_id,
                runtime_profile_id_for_config_id(abs(chat_id)),
            )
            self._contexts_by_runtime[runtime_key] = internal_api_context
        internal_api_credential = self._internal_api_auth.agent_credential_for(internal_api_context)
        services_info = self._services_info_by_runtime.get(runtime_key, [])

        if backend == "goose":
            return GooseBackend(
                model=model,
                workspace=workspace,
                home_workspace=home_ws,
                webhook_port=self._config.webhook_port,
                webhook_secret=internal_api_credential,
                timeout_seconds=timeout,
                services_info=services_info,
                workspace_config=ws_config,
                provider=effective_provider,
                memory_enabled=self._config.memory_enabled,
                os_user=os_user,
                max_session_hours=self._config.agent_max_session_hours,
                turn_deadline_seconds=self._config.turn_deadline_seconds,
                defer_user_file_reads=getattr(self._config, "protected_install", False) is True,
            )

        if backend == "opencode":
            # Import locally so opencode.py is only imported on an
            # opencode-active install. Mirrors the codex pattern below
            # rather than goose's module-top import.
            from kai.opencode import OpenCodeBackend

            return OpenCodeBackend(
                model=model,
                workspace=workspace,
                home_workspace=home_ws,
                webhook_port=self._config.webhook_port,
                webhook_secret=internal_api_credential,
                timeout_seconds=timeout,
                services_info=services_info,
                workspace_config=ws_config,
                provider=effective_provider,
                memory_enabled=self._config.memory_enabled,
                os_user=os_user,
                max_session_hours=self._config.agent_max_session_hours,
                turn_deadline_seconds=self._config.turn_deadline_seconds,
                defer_user_file_reads=getattr(self._config, "protected_install", False) is True,
            )

        if backend == "codex":
            # Import locally so codex.py is only imported on a
            # codex-active install. Mirrors the goose pattern above
            # (imported at module top) but keeps the import optional
            # for installs where codex CLI is not available.
            from kai.codex import CodexBackend

            return CodexBackend(
                model=model,
                workspace=workspace,
                home_workspace=home_ws,
                webhook_port=self._config.webhook_port,
                webhook_secret=internal_api_credential,
                timeout_seconds=timeout,
                services_info=services_info,
                workspace_config=ws_config,
                provider="openai",
                codex_user=os_user,
                memory_enabled=self._config.memory_enabled,
                max_session_hours=self._config.agent_max_session_hours,
                codex_effort_level=self._config.codex_effort_level,
                codex_turn_deadline_seconds=self._config.codex_turn_deadline_seconds,
                defer_user_file_reads=getattr(self._config, "protected_install", False) is True,
            )

        if backend == "pi":
            from kai.pi import PiBackend

            return PiBackend(
                model=model,
                workspace=workspace,
                home_workspace=home_ws,
                webhook_port=self._config.webhook_port,
                webhook_secret=internal_api_credential,
                timeout_seconds=timeout,
                services_info=services_info,
                workspace_config=ws_config,
                provider=effective_provider,
                memory_enabled=self._config.memory_enabled,
                os_user=os_user,
                max_session_hours=self._config.agent_max_session_hours,
                turn_deadline_seconds=self._config.turn_deadline_seconds,
                defer_user_file_reads=getattr(self._config, "protected_install", False) is True,
            )

        if backend == "claude":
            return ClaudeCodeBackend(
                model=model,
                workspace=workspace,
                home_workspace=home_ws,
                webhook_port=self._config.webhook_port,
                webhook_secret=internal_api_credential,
                timeout_seconds=timeout,
                services_info=services_info,
                claude_user=os_user,
                max_session_hours=self._config.agent_max_session_hours,
                workspace_config=ws_config,
                autocompact_pct=self._config.claude_autocompact_pct,
                claude_effort_level=self._config.claude_effort_level,
                turn_deadline_seconds=self._config.turn_deadline_seconds,
                memory_enabled=self._config.memory_enabled,
                defer_user_file_reads=getattr(self._config, "protected_install", False) is True,
            )

        raise RuntimeError(f"Cannot create backend instance for unsupported backend {backend!r}")

    # ── Prompt routing ──────────────────────────────────────────────

    async def _prepare_instance(self, runtime: RuntimeSelector) -> AgentBackend:
        """Apply every pending effective setting before returning one runtime."""
        runtime_key, _, _ = self._resolve_runtime(runtime)
        instance = self.get(runtime)
        if runtime_key in self._pending_workspace_restore:
            await self._attempt_workspace_restore(runtime, instance)
        if runtime_key in self._pending_settings_restore:
            await self._apply_user_db_settings(runtime, instance)
            self._pending_settings_restore.discard(runtime_key)
        return instance

    async def prepare_execution(self, runtime: RuntimeSelector) -> PreparedBackendExecution:
        """Bind the effective selection to the exact runtime that may dispatch later."""
        runtime_key, _, _ = self._resolve_runtime(runtime)
        async with self._backend_transition_lock(runtime_key):
            prepared = PreparedBackendExecution(
                self,
                runtime_key,
                runtime_key,
                await self._prepare_instance(runtime),
            )
            # Preparation is already part of an accepted canonical run.  Hold
            # the runtime's active marker across the short stage-before-send
            # window so a concurrent settings request cannot invalidate the
            # prepared handle or interrupt that run.
            self._in_flight.add(runtime_key)
            return prepared

    async def prepare_routed_execution(
        self,
        runtime: RuntimeProfileId,
        backend_option_id: str,
        model: str,
    ) -> PreparedBackendExecution:
        """Bind an alternate authorized option without changing the selected route."""
        runtime_key, _, profile = self._resolve_runtime(runtime)
        if profile is None or not isinstance(runtime_key, RuntimeProfileId):
            raise RuntimeError("Explicit task routing requires a protected runtime profile")
        option = profile.backend_option(backend_option_id)
        selected = self._backend_option(runtime)
        assert selected is not None
        if option.option_id == selected.option_id:
            return await self.prepare_execution(runtime)
        if not validate_model_for_backend_policy(
            model,
            option.backend,
            option.provider,
            allowed_models=option.allowed_models,
        ):
            raise RuntimeError("Routed model is not allowed by the protected backend option")

        instance_key = _RoutedRuntimeKey(runtime_key, option.option_id)
        async with self._backend_transition_lock(runtime_key):
            workspace = await self.get_effective_workspace(runtime)
            instance = self._pool.get(instance_key)
            if instance is not None:
                fingerprint = _runtime_fingerprint(instance)
                expected = PreparedBackendSelection(option.backend, option.provider, model)
                if fingerprint.selection != expected or fingerprint.workspace.resolve() != workspace.resolve():
                    await self._shutdown_instance(instance_key, instance, reason="routed runtime changed")
                    instance = None
            if instance is None:
                instance = self._create_instance(
                    runtime,
                    backend_option_id=option.option_id,
                    model_override=model,
                )
                if instance.workspace.resolve() != workspace.resolve():
                    namespace = self._canonical_namespace(runtime)
                    assert namespace is not None
                    workspace_config = await sessions.build_canonical_workspace_config(
                        self._config.get_workspace_config(workspace),
                        workspace,
                        namespace,
                    )
                    await instance.change_workspace(workspace, workspace_config=workspace_config)
                # The durable route decision owns the exact model. A workspace
                # override may have changed it while applying the workspace.
                instance.model = model
                namespace = self._canonical_namespace(runtime)
                assert namespace is not None
                settings = await sessions.get_canonical_execution_settings(namespace)
                workspace_timeout = instance.workspace_config.timeout if instance.workspace_config else None
                if workspace_timeout is None and "timeout" in settings:
                    instance.timeout_seconds = int(settings["timeout"])
                self._pool[instance_key] = instance
            self._last_activity[instance_key] = time.monotonic()
            prepared = PreparedBackendExecution(
                self,
                runtime_key,
                instance_key,
                instance,
            )
            self._in_flight.add(instance_key)
            return prepared

    async def send(
        self,
        prompt: str | list,
        *,
        runtime: RuntimeSelector | None = None,
        chat_id: int | None = None,
    ) -> AsyncGenerator[StreamEvent]:
        """
        Route a prompt to the user's subprocess.

        On the first call for a newly created instance, restores the
        user's saved workspace and applies DB-stored model/timeout
        before sending. Each piece runs only if the corresponding
        pending flag is set, so a command-path resolver call that
        already handled the workspace half does not duplicate work
        here and a workspace-handled-but-settings-pending instance
        still gets its DB settings on first send.

        Marks the user as in-flight to prevent eviction mid-stream.
        """
        if (runtime is None) == (chat_id is None):
            raise TypeError("Exactly one of runtime or chat_id must be provided")
        if runtime is not None:
            selector: RuntimeSelector = runtime
        else:
            assert chat_id is not None
            selector = chat_id
        runtime_key, runtime_config_id, profile = self._resolve_runtime(selector)
        async with self._backend_transition_lock(runtime_key):
            instance = await self._prepare_instance(selector)
            self._last_activity[runtime_key] = time.monotonic()
            self._in_flight.add(runtime_key)
        try:
            if profile is not None:
                runtime_identity = self._contexts_by_runtime.get(runtime_key)
                if runtime_identity is None:
                    raise RuntimeError("Protected runtime has no canonical backend identity")
                stream = instance.send(prompt, runtime_identity=runtime_identity)
            else:
                assert runtime_config_id is not None
                stream = instance.send(prompt, chat_id=runtime_config_id)
            async for event in stream:
                yield event
        finally:
            instance.discard_canonical_history()
            instance.discard_canonical_agent_context()
            self._in_flight.discard(runtime_key)
            self._last_activity[runtime_key] = time.monotonic()

    async def _send_prepared(
        self,
        prepared: PreparedBackendExecution,
        prompt: str | list,
    ) -> AsyncGenerator[StreamEvent]:
        """Dispatch through a prepared handle only while its runtime remains exact."""
        self._validate_prepared(prepared)
        runtime_key = prepared._runtime_selector
        instance_key = prepared._instance_key
        _, runtime_config_id, profile = self._resolve_runtime(runtime_key)
        instance = prepared._instance
        self._last_activity[instance_key] = time.monotonic()
        self._in_flight.add(instance_key)
        try:
            if profile is not None:
                runtime_identity = self._contexts_by_runtime.get(runtime_key)
                if runtime_identity is None:
                    raise RuntimeError("Protected runtime has no canonical backend identity")
                stream = instance.send(prompt, runtime_identity=runtime_identity)
            else:
                assert runtime_config_id is not None
                stream = instance.send(prompt, chat_id=runtime_config_id)
            async for event in stream:
                yield event
        finally:
            instance.discard_canonical_history()
            instance.discard_canonical_agent_context()
            self._in_flight.discard(instance_key)
            self._last_activity[instance_key] = time.monotonic()

    def _validate_prepared(self, prepared: PreparedBackendExecution) -> None:
        instance_key = prepared._instance_key
        instance = self._pool.get(instance_key)
        if instance is not prepared._instance:
            raise RuntimeError("Prepared backend runtime is no longer current")
        if instance_key in self._pending_workspace_restore or instance_key in self._pending_settings_restore:
            raise RuntimeError("Prepared backend runtime has pending effective settings")
        if _runtime_fingerprint(instance) != prepared._fingerprint:
            raise RuntimeError("Prepared backend runtime changed before dispatch")

    async def _cancel_prepared(self, prepared: PreparedBackendExecution) -> None:
        self._validate_prepared(prepared)
        instance_key = prepared._instance_key
        instance = prepared._instance
        try:
            await asyncio.wait_for(instance.shutdown(), timeout=_FORCE_KILL_TIMEOUT)
        except Exception:
            instance.force_kill()
            log.warning("prepared cancellation: shutdown failed for runtime %s, sent SIGKILL", instance_key)
        finally:
            # Never remove or stop a replacement installed while the exact
            # prepared runtime was shutting down.
            if self._pool.get(instance_key) is instance:
                self._pool.pop(instance_key, None)
                self._last_activity.pop(instance_key, None)
            self._in_flight.discard(instance_key)

    async def _attempt_workspace_restore(self, runtime: RuntimeSelector, instance: AgentBackend) -> Path:
        """Restore the saved workspace into a live instance.

        Reads `settings.workspace:<chat_id>`, validates path existence
        and effective runtime workspace access, and switches
        the instance into the saved path when valid. Stale rows (path
        gone or no longer allowed) are deleted so an admin removing a
        path does not have users silently bypass the restriction.

        Always clears the workspace-restore flag for `chat_id` before
        returning, regardless of whether the saved row was valid,
        stale, or absent. This is the invariant the command-path
        resolver relies on: a single decision finalizes the effective
        workspace for the live instance.

        Returns the effective workspace path (saved when valid,
        otherwise the user's home workspace).
        """
        runtime_key, chat_id, profile = self._resolve_runtime(runtime)
        namespace = self._canonical_namespace(runtime)
        if profile is not None:
            assert namespace is not None
            saved = (await sessions.get_canonical_execution_settings(namespace)).get("workspace")
        else:
            assert chat_id is not None
            saved = await sessions.get_active_workspace(chat_id)
        effective: Path | None = None
        if saved:
            ws_path = Path(saved)
            if not ws_path.is_dir():
                log.warning(
                    "Saved workspace for runtime %s no longer exists: %s",
                    runtime_key,
                    saved,
                )
                if namespace is not None:
                    await sessions.delete_canonical_execution_setting(namespace, "workspace")
                else:
                    assert chat_id is not None
                    await sessions.delete_active_workspace(chat_id)
            else:
                base, allowed = await self.resolve_workspace_access(runtime)
                if not is_workspace_allowed(ws_path, base, allowed):
                    log.warning(
                        "Saved workspace for runtime %s is no longer allowed: %s",
                        runtime_key,
                        saved,
                    )
                    if namespace is not None:
                        await sessions.delete_canonical_execution_setting(namespace, "workspace")
                    else:
                        assert chat_id is not None
                        await sessions.delete_active_workspace(chat_id)
                else:
                    # Layer DB overrides on top of YAML baseline so the
                    # user's per-workspace config (set via /workspace
                    # config) is applied on startup, not just after
                    # explicit switches.
                    yaml_config = self._config.get_workspace_config(ws_path)
                    ws_config = (
                        await sessions.build_canonical_workspace_config(yaml_config, ws_path, namespace)
                        if namespace is not None
                        else await sessions.build_workspace_config(yaml_config, ws_path, _required_legacy_key(chat_id))
                    )
                    await instance.change_workspace(ws_path, workspace_config=ws_config)
                    log.info("Restored workspace for runtime %s: %s", runtime_key, ws_path)
                    effective = ws_path

        self._pending_workspace_restore.discard(runtime_key)
        if effective is not None:
            return effective
        return self.get_home_workspace(runtime)

    async def _apply_user_db_settings(self, runtime: RuntimeSelector, instance: AgentBackend) -> None:
        """Apply per-user DB-stored settings to a live instance.

        Covers the model and timeout overrides set via /settings or
        /model. _create_instance() already applied the protected policy or
        compatibility baseline, so this layer only handles the DB tier. Workspace
        config takes precedence over user settings (more specific
        wins).

        Called from send() on the first ordinary text turn after
        instance creation. The command-path resolver never invokes
        this helper: settings need a subprocess restart to take
        effect when the model changes, which would interrupt any UI
        operation that just wanted to read the effective workspace.
        """
        _, chat_id, _profile = self._resolve_runtime(runtime)
        namespace = self._canonical_namespace(runtime)
        db_settings = (
            await sessions.get_canonical_execution_settings(namespace)
            if namespace is not None
            else await sessions.get_user_settings(_required_legacy_key(chat_id))
        )
        if not db_settings:
            return

        # Track whether any flag-level setting changed (requires restart
        # because the value is baked into the CLI command at startup)
        needs_restart = False

        # Validate the instance identity before applying backend-specific
        # model policy. Missing or unknown identities are invariant
        # violations and must not inherit another backend's rules.
        instance_backend = require_backend_name(instance)

        # Model: only apply if workspace config has a VALID model for
        # this backend. A workspaces.yaml entry like
        # `model: gpt-5.4-nano` applied to a codex instance is rejected
        # by apply_workspace_model() at backend __init__ time (the
        # helper returns the current model and logs a warning), but
        # the original WorkspaceConfig still carrying the invalid
        # model field remains stored on instance.workspace_config.
        # Treating any non-empty workspace_config.model as
        # precedence-bearing therefore blocks a valid per-user DB
        # model (e.g. gpt-5.4-mini) even though the workspace override
        # was never applied. Re-run the same validation here so the
        # precedence guard matches what was actually applied to
        # instance.model.
        ws_model_raw = instance.workspace_config.model if instance.workspace_config else None
        ws_model = canonicalize_model_for_backend(ws_model_raw, instance_backend) if ws_model_raw is not None else None
        ws_model_applied = bool(ws_model and validate_model_for_backend(ws_model, instance_backend, instance.provider))
        if not ws_model_applied and "model" in db_settings:
            stored_model_raw = db_settings["model"]
            stored_model = canonicalize_model_for_backend(stored_model_raw, instance_backend)
            option = self._backend_option(runtime)
            model_is_valid = (
                validate_model_for_backend_policy(
                    stored_model,
                    instance_backend,
                    instance.provider,
                    allowed_models=option.allowed_models,
                )
                if option is not None
                else validate_model_for_backend(
                    stored_model,
                    instance_backend,
                    instance.provider,
                )
            )
            if model_is_valid:
                if stored_model != stored_model_raw:
                    if namespace is not None:
                        await sessions.set_canonical_execution_setting(namespace, "model", stored_model)
                    else:
                        await sessions.set_user_setting(_required_legacy_key(chat_id), "model", stored_model)
                    log.info(
                        "Migrated stored model for runtime %s from '%s' to '%s'",
                        runtime,
                        stored_model_raw,
                        stored_model,
                    )
                if stored_model != instance.model:
                    instance.model = stored_model
                    needs_restart = True
            else:
                log.warning(
                    "Ignoring stored model '%s' for runtime %s (invalid for backend '%s'/provider '%s')",
                    stored_model,
                    chat_id,
                    instance_backend,
                    instance.provider,
                )

        # Timeout: workspace config timeout overrides user default
        ws_timeout = instance.workspace_config.timeout if instance.workspace_config else None
        if ws_timeout is None and "timeout" in db_settings:
            try:
                instance.timeout_seconds = int(db_settings["timeout"])
            except (ValueError, TypeError):
                log.warning("Corrupt timeout in DB for runtime %s", runtime)

        # restart() kills the subprocess and spawns a new one, but the
        # backend *object* is preserved. Mutations made above (timeout,
        # model) survive the restart because the new subprocess reads
        # from self.* attrs.
        if needs_restart:
            log.info("Restarting process for runtime %s: per-runtime DB overrides differ", runtime)
            await instance.restart()

    # ── Per-user actions ────────────────────────────────────────────

    def get_if_exists(self, runtime: RuntimeSelector) -> AgentBackend | None:
        """
        Look up a user's subprocess without creating one.

        Use this for operations that should be no-ops when no subprocess
        exists (e.g., /stop on an idle user). Contrast with get(), which
        creates on first access. Does NOT update last_activity to avoid
        side effects (e.g., force_kill refreshing the timestamp of a
        process it's about to kill).
        """
        runtime_key, _, _ = self._resolve_runtime(runtime)
        return self._pool.get(runtime_key)

    async def _shutdown_instance(
        self,
        instance_key: RuntimeInstanceKey,
        instance: AgentBackend,
        *,
        reason: str,
    ) -> None:
        try:
            await asyncio.wait_for(instance.shutdown(), timeout=_FORCE_KILL_TIMEOUT)
        except Exception:
            instance.force_kill()
            log.warning("%s: shutdown failed for runtime %s, sent SIGKILL", reason, instance_key)
        finally:
            if self._pool.get(instance_key) is instance:
                self._pool.pop(instance_key, None)
                self._last_activity.pop(instance_key, None)
            self._in_flight.discard(instance_key)

    async def force_kill(self, runtime: RuntimeSelector) -> None:
        """
        Kill a specific user's subprocess and remove it from the pool.

        Uses shutdown() with a short timeout for clean process reaping
        and stderr task cancellation. Falls back to raw SIGKILL on any
        non-cancellation failure (timeout, OSError, etc.). Cleanup
        (pool removal) runs unconditionally via finally.

        The instance is kept in the pool during shutdown so it remains
        tracked. It is only removed after the subprocess is confirmed
        dead (either via clean shutdown or SIGKILL fallback).
        """
        runtime_key, _, _ = self._resolve_runtime(runtime)
        instance = self._pool.get(runtime_key)
        if not instance:
            # No instance to kill; clean up any orphaned tracking entry
            self._last_activity.pop(runtime_key, None)
            return
        try:
            await asyncio.wait_for(instance.shutdown(), timeout=_FORCE_KILL_TIMEOUT)
        except Exception:
            # Any failure (timeout, OSError, etc.) - fall back to raw
            # SIGKILL. instance.force_kill() is effectively infallible
            # (catches its own OSError).
            instance.force_kill()
            log.warning("force_kill: shutdown failed for runtime %s, sent SIGKILL", runtime_key)
        finally:
            # Remove from tracking regardless of how shutdown ended.
            # The finally block ensures cleanup even if CancelledError
            # (a BaseException, not caught by except Exception) propagates.
            self._pool.pop(runtime_key, None)
            self._last_activity.pop(runtime_key, None)
            self._pending_workspace_restore.discard(runtime_key)
            self._pending_settings_restore.discard(runtime_key)

    def is_in_flight(self, runtime: RuntimeSelector) -> bool:
        """Return whether this one runtime currently owns an active send."""
        runtime_key, _, _ = self._resolve_runtime(runtime)
        return self._profile_has_in_flight(runtime_key)

    def _profile_has_in_flight(self, runtime_key: RuntimePoolKey) -> bool:
        return any(
            key == runtime_key or (isinstance(key, _RoutedRuntimeKey) and key.runtime_profile_id == runtime_key)
            for key in self._in_flight
        )

    async def select_backend(
        self,
        runtime: RuntimeSelector,
        backend_option_id: str,
        *,
        commit_selection: Callable[[], Awaitable[None]] | None = None,
    ) -> bool:
        """Atomically replace one idle runtime's backend.

        Return false without running ``commit_selection`` when a canonical run
        already owns the runtime.  The callback lets the settings authority
        persist the selected backend and invalidate its provider session under
        the same transition lock that guards execution preparation.
        """
        runtime_key, _, profile = self._resolve_runtime(runtime)
        if profile is None:
            raise RuntimeError("Backend selection requires a protected runtime profile")
        option = profile.backend_option(backend_option_id)
        async with self._backend_transition_lock(runtime_key):
            if self._profile_has_in_flight(runtime_key):
                return False
            await self.force_kill(runtime)
            if commit_selection is not None:
                await commit_selection()
            self._selected_backends[runtime_key] = option.option_id
            return True

    async def change_workspace(
        self,
        runtime: RuntimeSelector,
        new_workspace: Path,
        workspace_config: WorkspaceConfig | None = None,
    ) -> None:
        """Switch a specific user's workspace."""
        runtime_key, _, _ = self._resolve_runtime(runtime)
        instance = self.get(runtime)
        # Explicit workspace change supersedes any pending restore.
        # Without this, the next send() would restore the old saved
        # workspace over the one just set. Settings-restore stays
        # pending: an explicit /workspace switch carries no signal
        # about user-level model or timeout preferences, and pre-split
        # behavior accidentally suppressed the DB-settings job here.
        self._pending_workspace_restore.discard(runtime_key)
        await instance.change_workspace(new_workspace, workspace_config=workspace_config)

    async def restart(self, runtime: RuntimeSelector) -> None:
        """Restart a specific user's subprocess."""
        instance = self.get_if_exists(runtime)
        if instance:
            await instance.restart()

    # ── Per-user property accessors ─────────────────────────────────

    def get_model(self, runtime: RuntimeSelector) -> str:
        """Get the active model without bypassing protected runtime policy."""
        instance = self.get_if_exists(runtime)
        if instance is not None:
            return instance.model
        profile = self._protected_profile(runtime)
        if profile is not None:
            option = self._backend_option(runtime)
            assert option is not None
            return option.model
        return self._config.default_model

    async def get_effective_model(self, runtime: RuntimeSelector) -> str:
        """Get the effective model, checking persisted settings if no instance exists.

        Protected runtimes use the user settings DB over their protected
        profile baseline. Compatibility runtimes retain the existing
        users.yaml/global cascade through ``resolve_user_defaults``.

        Use this in command handlers that display the current model
        (like /models) where accuracy matters more than speed.
        """
        _, chat_id, profile = self._resolve_runtime(runtime)
        instance = self.get_if_exists(runtime)
        if instance:
            return instance.model
        if profile is None:
            defaults = await sessions.resolve_user_defaults(chat_id, self._config)
            return defaults["model"]
        namespace = self._canonical_namespace(runtime)
        db_settings = (
            await sessions.get_canonical_execution_settings(namespace)
            if namespace is not None
            else await sessions.get_user_settings(chat_id)
        )
        option = self._backend_option(runtime)
        assert option is not None
        raw_model = db_settings.get("model", "").strip()
        if not raw_model:
            return option.model
        model = canonicalize_model_for_backend(raw_model, option.backend)
        if validate_model_for_backend_policy(
            model,
            option.backend,
            option.provider,
            allowed_models=option.allowed_models,
        ):
            return model
        return option.model

    def set_model(self, runtime: RuntimeSelector, model: str) -> None:
        """Set the model for a user's subprocess."""
        instance = self.get(runtime)
        instance.model = canonicalize_model_for_backend(model, require_backend_name(instance))

    async def get_effective_workspace(self, runtime: RuntimeSelector) -> Path:
        """
        Get the effective workspace for a user, eagerly restoring the
        saved workspace into the live instance when one exists.

        All user-visible workspace decisions (command displays,
        equality checks, scope detection, file confinement) flow
        through this method so command-path callers cannot observe
        the home default while `settings.workspace:<chat_id>` points
        at a valid saved path. The matching send-path restore stays
        as a backstop.

        Resolution order:

        1. If a live instance exists AND has no pending workspace
           restore: return its current workspace.
        2. Else read `settings.workspace:<chat_id>`. If unset: return
           the user's home workspace. If an instance exists, discard
           the workspace-restore flag (the effective path IS home).
        3. If set, validate path existence and per-user workspace
           access. If invalid: delete the stale row, return home,
           and discard the workspace-restore flag on any pre-existing
           instance.
        4. If valid: create or reuse the instance, then delegate to
           `_attempt_workspace_restore` which performs the switch,
           clears the flag, and returns the saved path.

        The resolver only touches the workspace-restore flag.
        `_pending_settings_restore` is the send path's responsibility,
        so user-level model/timeout overrides still land on the first
        ordinary text turn even if a command-path resolver call
        already finalized the workspace half.
        """
        runtime_key, chat_id, _profile = self._resolve_runtime(runtime)
        namespace = self._canonical_namespace(runtime)
        instance = self._pool.get(runtime_key)
        if instance is not None and runtime_key not in self._pending_workspace_restore:
            return instance.workspace

        saved = (
            (await sessions.get_canonical_execution_settings(namespace)).get("workspace")
            if namespace is not None
            else await sessions.get_active_workspace(chat_id)
        )
        if not saved:
            if instance is not None:
                self._pending_workspace_restore.discard(runtime_key)
            return self.get_home_workspace(runtime)

        ws_path = Path(saved)
        base, allowed = await self.resolve_workspace_access(runtime)
        if not ws_path.is_dir() or not is_workspace_allowed(ws_path, base, allowed):
            log.warning(
                "Saved workspace for runtime %s invalid; clearing: %s",
                runtime_key,
                saved,
            )
            if namespace is not None:
                await sessions.delete_canonical_execution_setting(namespace, "workspace")
            else:
                await sessions.delete_active_workspace(chat_id)
            if instance is not None:
                self._pending_workspace_restore.discard(runtime_key)
            return self.get_home_workspace(runtime)

        # Only now do we materialize an instance. Avoiding instance
        # creation in the no-saved-row and stale-row branches keeps
        # the resolver cheap for cold reads that just want to confirm
        # "home is the right answer."
        instance = self.get(runtime)
        # Re-validate inside _attempt_workspace_restore via its own
        # DB read so a concurrent writer that cleared the setting
        # between this check and the helper call is still handled
        # correctly (it returns home instead of trying to switch to a
        # since-deleted target).
        return await self._attempt_workspace_restore(runtime, instance)

    def is_alive(self, runtime: RuntimeSelector) -> bool:
        """True if this user's subprocess is running."""
        instance = self.get_if_exists(runtime)
        return instance.is_alive if instance else False

    def get_session_id(self, runtime: RuntimeSelector) -> str | None:
        """Get the session ID for a user's subprocess."""
        instance = self.get_if_exists(runtime)
        return instance.session_id if instance else None

    # ── Idle eviction ───────────────────────────────────────────────

    def start(self) -> None:
        """Start the eviction background task (if eviction is enabled)."""
        if self._config.agent_idle_timeout > 0:
            self._eviction_task = asyncio.create_task(self._eviction_loop())

    async def _eviction_loop(self) -> None:
        """Periodically kill idle subprocesses to free memory."""
        idle_timeout = self._config.agent_idle_timeout
        while True:
            await asyncio.sleep(_EVICTION_CHECK_INTERVAL)
            now = time.monotonic()
            to_evict = [
                runtime_key
                for runtime_key, last in self._last_activity.items()
                if now - last > idle_timeout and runtime_key in self._pool and runtime_key not in self._in_flight
            ]
            for runtime_key in to_evict:
                # Re-check all three snapshot conditions before evicting.
                # Between the snapshot and this iteration (and between
                # iterations), await points in shutdown() yield control.
                # Other coroutines can: refresh activity timestamps,
                # enter send() (adding to _in_flight), or call
                # force_kill() (removing from _pool). All three must
                # be re-verified to avoid evicting active conversations.
                # Pool membership first: if force_kill() already removed the
                # instance, clean up the orphaned _last_activity entry and
                # skip. This must be checked before the timestamp/in-flight
                # guards so the cleanup always fires when the instance is gone.
                if runtime_key not in self._pool:
                    self._last_activity.pop(runtime_key, None)
                    continue
                if self._last_activity.get(runtime_key, 0) > now:
                    continue
                if runtime_key in self._in_flight:
                    continue
                instance = self._pool.get(runtime_key)
                try:
                    if instance and instance.is_alive:
                        try:
                            log.info("Evicting idle subprocess for runtime %s", runtime_key)
                            await instance.shutdown()
                        except Exception:
                            # Graceful shutdown failed. Fall back to raw SIGKILL
                            # so the process doesn't become an orphan. force_kill()
                            # is effectively infallible (catches its own OSError).
                            log.exception(
                                "Error evicting subprocess for runtime %s, sending SIGKILL",
                                runtime_key,
                            )
                            instance.force_kill()
                finally:
                    # Remove from tracking after shutdown (alive instances) or
                    # unconditionally (dead instances). The finally block ensures
                    # cleanup even if CancelledError propagates from shutdown().
                    self._pool.pop(runtime_key, None)
                    self._last_activity.pop(runtime_key, None)

    async def shutdown(self) -> None:
        """Shut down all subprocesses and stop the eviction task."""
        if self._eviction_task:
            self._eviction_task.cancel()
            try:
                await self._eviction_task
            except asyncio.CancelledError:
                pass
            self._eviction_task = None
        for runtime_key, instance in self._pool.items():
            try:
                log.info("Shutting down subprocess for runtime %s", runtime_key)
                await instance.shutdown()
            except Exception:
                log.exception("Error shutting down subprocess for runtime %s", runtime_key)
        self._pool.clear()
        self._last_activity.clear()
