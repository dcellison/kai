"""
Per-user agent subprocess pool with lazy creation and idle eviction.

Provides functionality to:
1. Manage a dict of AgentBackend instances keyed by chat_id
2. Create instances lazily on first message with per-user configuration
3. Route prompts to the correct user's subprocess
4. Evict idle subprocesses to reclaim memory on resource-constrained machines
5. Restore per-user saved workspaces on first interaction

Each user gets their own backend instance with full conversation isolation,
independent lifecycle, and OS-level enforcement via sudo -u when os_user is
configured in users.yaml.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from pathlib import Path

from kai import sessions
from kai.backend import AgentBackend, StreamEvent, require_backend_name, resolve_home_workspace
from kai.claude import ClaudeCodeBackend
from kai.config import (
    OPEN_ENDED_PROVIDERS,
    VALID_BACKENDS,
    Config,
    WorkspaceConfig,
    canonicalize_model_for_backend,
    get_default_model_for_backend,
    get_effective_provider,
    get_user_backend_and_provider,
    validate_model_for_backend,
)
from kai.goose import GooseBackend
from kai.internal_api_auth import InternalAPIAuth
from kai.workspace_utils import is_workspace_allowed

log = logging.getLogger(__name__)


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

    __slots__ = ("_chat_id", "_consumed", "_fingerprint", "_instance", "_pool")

    def __init__(self, pool: SubprocessPool, chat_id: int, instance: AgentBackend) -> None:
        self._pool = pool
        self._chat_id = chat_id
        self._instance = instance
        self._fingerprint = _runtime_fingerprint(instance)
        self._consumed = False

    @property
    def selection(self) -> PreparedBackendSelection:
        return self._fingerprint.selection

    @property
    def workspace(self) -> Path:
        return self._fingerprint.workspace

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
    Per-user agent subprocess pool with lazy creation and idle eviction.

    Each user gets an independent AgentBackend instance running as their
    OS user. Instances are created on first message and evicted after idle
    timeout to manage memory on resource-constrained machines.

    Thread safety: send() for a given chat_id is serialized by the
    per-chat lock in bot.py/cron.py. The pool does not add its own
    locking because the callers already guarantee single-writer-per-user.
    If a future caller bypasses the per-chat lock, add an asyncio.Lock
    per chat_id here.
    """

    def __init__(
        self,
        *,
        config: Config,
        services_info: list[dict],
    ):
        self._config = config
        available_service_names = {
            service["name"] for service in services_info if isinstance(service.get("name"), str) and service["name"]
        }
        allowed_services_by_user: dict[int, frozenset[str]] = {}
        self._services_info_by_user: dict[int, list[dict]] = {}
        for chat_id in config.allowed_user_ids:
            user = config.get_user_config(chat_id)
            requested = frozenset(user.allowed_services if user else [])
            unavailable = requested - available_service_names
            if unavailable:
                log.warning(
                    "users.yaml: user %d has unavailable allowed_services: %s; denying them",
                    chat_id,
                    ", ".join(sorted(unavailable)),
                )
            allowed = requested & available_service_names
            allowed_services_by_user[chat_id] = allowed
            self._services_info_by_user[chat_id] = [
                service for service in services_info if service.get("name") in allowed
            ]
        if available_service_names and not any(allowed_services_by_user.values()):
            log.warning(
                "External services are loaded but no user has an allowed_services grant; "
                "the agent service proxy is disabled for every user"
            )
        # Tokens are random and process-local. The pool owns the store because
        # it creates every persistent backend; webhook.start() receives this
        # same object through bot_data so credential resolution cannot drift.
        self._internal_api_auth = InternalAPIAuth.for_users(
            config.allowed_user_ids,
            allowed_services_by_user=allowed_services_by_user,
        )
        self._pool: dict[int, AgentBackend] = {}
        self._last_activity: dict[int, float] = {}
        # Pending one-time setup for a freshly-created backend instance.
        # Split into two flags so a command-path workspace resolver call
        # can finish the workspace half without suppressing the DB-
        # settings half that only the first send() applies. A conflated
        # marker would let a /memory or /workspace read clear the
        # pending state before send() ever had a chance to push the
        # user's stored model/timeout into the new subprocess.
        self._pending_workspace_restore: set[int] = set()
        self._pending_settings_restore: set[int] = set()
        self._in_flight: set[int] = set()  # chat_ids with active send()
        self._eviction_task: asyncio.Task | None = None

    @property
    def internal_api_auth(self) -> InternalAPIAuth:
        """Return the credential store shared with the loopback HTTP server."""
        return self._internal_api_auth

    # ── Instance management ─────────────────────────────────────────

    def get(self, chat_id: int) -> AgentBackend:
        """
        Get or create a backend instance for the given user.

        Creates lazily on first access. The subprocess itself starts even
        later (on first send()), not here - __init__ is cheap;
        _ensure_started() is where the process spawns.
        """
        if chat_id not in self._pool:
            instance = self._create_instance(chat_id)
            require_backend_name(instance)
            self._pool[chat_id] = instance
            # Mark both one-time setup steps as pending. The split
            # matters at command-path callers that finish the workspace
            # half via get_effective_workspace; settings still need to
            # land on the first ordinary send().
            self._pending_workspace_restore.add(chat_id)
            self._pending_settings_restore.add(chat_id)
        self._last_activity[chat_id] = time.monotonic()
        return self._pool[chat_id]

    def _create_instance(self, chat_id: int) -> AgentBackend:
        """
        Create an AgentBackend for a specific user.

        Backend selection: per-user backend (from users.yaml)
        overrides the global config.default_backend. Both share the same
        ABC interface; pool.py is backend-agnostic after this point.

        Resolution order for each setting:
        1. UserConfig from users.yaml (os_user, home_workspace, model,
           timeout, backend, provider)
        2. Global config defaults (from .env)

        Per-user DB overrides (set via /settings or /model) are applied
        in _apply_user_db_settings() since they require async DB access.
        """
        user = self._config.get_user_config(chat_id)

        # Resolve through the single backend.resolve_home_workspace
        # helper: users.yaml override first, else DATA_DIR/home/<principal_id>/.
        # The old global home field on Config was removed by #353 because
        # it pointed every unconfigured user at a shared directory.
        workspace = resolve_home_workspace(chat_id, self._config)

        ws_config = self._config.get_workspace_config(workspace)

        # Per-user backend and provider, falling back to global config.
        # Routed through the canonical get_user_backend_and_provider
        # resolver so single-provider backends report their canonical
        # provider even when the provider is unset, matching the same
        # cascade bot.py and the install-time validator use. Without
        # this, a per-user backend differing from the global backend can
        # inherit an incompatible global model.
        backend, effective_provider = get_user_backend_and_provider(user, self._config)
        if backend not in VALID_BACKENDS:
            raise RuntimeError(f"Cannot create backend instance for unsupported backend {backend!r}")
        # get_effective_provider hardcodes the backend->provider rule for
        # claude (anthropic) and codex (openai) so global_provider lines
        # up with effective_provider whenever the user has not overridden
        # the backend; no codex-specific patch needed here.
        global_provider = get_effective_provider(self._config.default_backend, self._config.default_provider)

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
                    "No model configured for open-ended provider '%s' (chat %d); "
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

        timeout = user.timeout if user and user.timeout is not None else self._config.default_timeout
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
        home_ws = resolve_home_workspace(chat_id, self._config)

        # os_user for sudo -u isolation. None = run as bot user.
        # Resolved here (rather than inside each branch) because all
        # conversational backends consume it for per-user subprocess isolation.
        # Each agent's auth state is per-OS-user and operator-
        # provisioned under the target user's home (claude OAuth
        # credentials, codex auth.json, goose keychain or env keys,
        # opencode auth.json via `opencode auth login` run as that
        # user); the `sudo -H` wrap each backend applies is what
        # points the subprocess at the right home.
        os_user = user.os_user if user else None

        # Backend selection is explicit. No branch may substitute a
        # different backend for a missing or unknown identifier.
        # Internal API availability is independent of public webhook ingress.
        # Agents receive only their random principal credential, never an
        # external webhook signing secret.
        internal_api_credential = self._internal_api_auth.agent_credential_for(chat_id)
        services_info = self._services_info_by_user.get(chat_id, [])

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
                memory_enabled=self._config.memory_enabled,
                defer_user_file_reads=getattr(self._config, "protected_install", False) is True,
            )

        raise RuntimeError(f"Cannot create backend instance for unsupported backend {backend!r}")

    # ── Prompt routing ──────────────────────────────────────────────

    async def _prepare_instance(self, chat_id: int) -> AgentBackend:
        """Apply every pending effective setting before returning one runtime."""
        instance = self.get(chat_id)
        if chat_id in self._pending_workspace_restore:
            await self._attempt_workspace_restore(chat_id, instance)
        if chat_id in self._pending_settings_restore:
            await self._apply_user_db_settings(chat_id, instance)
            self._pending_settings_restore.discard(chat_id)
        return instance

    async def prepare_execution(self, chat_id: int) -> PreparedBackendExecution:
        """Bind the effective selection to the exact runtime that may dispatch later."""
        return PreparedBackendExecution(self, chat_id, await self._prepare_instance(chat_id))

    async def send(self, prompt: str | list, *, chat_id: int) -> AsyncGenerator[StreamEvent]:
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
        instance = await self._prepare_instance(chat_id)
        self._last_activity[chat_id] = time.monotonic()
        self._in_flight.add(chat_id)
        try:
            async for event in instance.send(prompt, chat_id=chat_id):
                yield event
        finally:
            self._in_flight.discard(chat_id)
            self._last_activity[chat_id] = time.monotonic()

    async def _send_prepared(
        self,
        prepared: PreparedBackendExecution,
        prompt: str | list,
    ) -> AsyncGenerator[StreamEvent]:
        """Dispatch through a prepared handle only while its runtime remains exact."""
        self._validate_prepared(prepared)
        chat_id = prepared._chat_id
        instance = prepared._instance
        self._last_activity[chat_id] = time.monotonic()
        self._in_flight.add(chat_id)
        try:
            async for event in instance.send(prompt, chat_id=chat_id):
                yield event
        finally:
            self._in_flight.discard(chat_id)
            self._last_activity[chat_id] = time.monotonic()

    def _validate_prepared(self, prepared: PreparedBackendExecution) -> None:
        chat_id = prepared._chat_id
        instance = self.get_if_exists(chat_id)
        if instance is not prepared._instance:
            raise RuntimeError("Prepared backend runtime is no longer current")
        if chat_id in self._pending_workspace_restore or chat_id in self._pending_settings_restore:
            raise RuntimeError("Prepared backend runtime has pending effective settings")
        if _runtime_fingerprint(instance) != prepared._fingerprint:
            raise RuntimeError("Prepared backend runtime changed before dispatch")

    async def _cancel_prepared(self, prepared: PreparedBackendExecution) -> None:
        self._validate_prepared(prepared)
        chat_id = prepared._chat_id
        instance = prepared._instance
        try:
            await asyncio.wait_for(instance.shutdown(), timeout=_FORCE_KILL_TIMEOUT)
        except Exception:
            instance.force_kill()
            log.warning("prepared cancellation: shutdown failed for user %d, sent SIGKILL", chat_id)
        finally:
            # Never remove or stop a replacement installed while the exact
            # prepared runtime was shutting down.
            if self._pool.get(chat_id) is instance:
                self._pool.pop(chat_id, None)
                self._last_activity.pop(chat_id, None)

    async def _attempt_workspace_restore(self, chat_id: int, instance: AgentBackend) -> Path:
        """Restore the saved workspace into a live instance.

        Reads `settings.workspace:<chat_id>`, validates path existence
        and per-user workspace access (workspace_base from users.yaml,
        allowed list from DB + global ALLOWED_WORKSPACES), and switches
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
        saved = await sessions.get_setting(f"workspace:{chat_id}")
        effective: Path | None = None
        if saved:
            ws_path = Path(saved)
            if not ws_path.is_dir():
                log.warning(
                    "Saved workspace for user %d no longer exists: %s",
                    chat_id,
                    saved,
                )
                await sessions.delete_setting(f"workspace:{chat_id}")
            else:
                base, allowed = await sessions.resolve_workspace_access(chat_id, self._config)
                if not is_workspace_allowed(ws_path, base, allowed):
                    log.warning(
                        "Saved workspace for user %d is no longer allowed: %s",
                        chat_id,
                        saved,
                    )
                    await sessions.delete_setting(f"workspace:{chat_id}")
                else:
                    # Layer DB overrides on top of YAML baseline so the
                    # user's per-workspace config (set via /workspace
                    # config) is applied on startup, not just after
                    # explicit switches.
                    yaml_config = self._config.get_workspace_config(ws_path)
                    ws_config = await sessions.build_workspace_config(yaml_config, ws_path, chat_id)
                    await instance.change_workspace(ws_path, workspace_config=ws_config)
                    log.info("Restored workspace for user %d: %s", chat_id, ws_path)
                    effective = ws_path

        self._pending_workspace_restore.discard(chat_id)
        if effective is not None:
            return effective
        return resolve_home_workspace(chat_id, self._config)

    async def _apply_user_db_settings(self, chat_id: int, instance: AgentBackend) -> None:
        """Apply per-user DB-stored settings to a live instance.

        Covers the model and timeout overrides set via /settings or
        /model. _create_instance() already applied the users.yaml
        baselines, so this layer only handles the DB tier. Workspace
        config takes precedence over user settings (more specific
        wins).

        Called from send() on the first ordinary text turn after
        instance creation. The command-path resolver never invokes
        this helper: settings need a subprocess restart to take
        effect when the model changes, which would interrupt any UI
        operation that just wanted to read the effective workspace.
        """
        db_settings = await sessions.get_user_settings(chat_id)
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
            if validate_model_for_backend(stored_model, instance_backend, instance.provider):
                if stored_model != stored_model_raw:
                    await sessions.set_user_setting(chat_id, "model", stored_model)
                    log.info(
                        "Migrated stored model for user %d from '%s' to '%s'",
                        chat_id,
                        stored_model_raw,
                        stored_model,
                    )
                if stored_model != instance.model:
                    instance.model = stored_model
                    needs_restart = True
            else:
                log.warning(
                    "Ignoring stored model '%s' for user %d (invalid for backend '%s'/provider '%s')",
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
                log.warning("Corrupt timeout in DB for user %d", chat_id)

        # restart() kills the subprocess and spawns a new one, but the
        # backend *object* is preserved. Mutations made above (timeout,
        # model) survive the restart because the new subprocess reads
        # from self.* attrs.
        if needs_restart:
            log.info("Restarting process for user %d: per-user DB overrides differ", chat_id)
            await instance.restart()

    # ── Per-user actions ────────────────────────────────────────────

    def get_if_exists(self, chat_id: int) -> AgentBackend | None:
        """
        Look up a user's subprocess without creating one.

        Use this for operations that should be no-ops when no subprocess
        exists (e.g., /stop on an idle user). Contrast with get(), which
        creates on first access. Does NOT update last_activity to avoid
        side effects (e.g., force_kill refreshing the timestamp of a
        process it's about to kill).
        """
        return self._pool.get(chat_id)

    async def force_kill(self, chat_id: int) -> None:
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
        instance = self._pool.get(chat_id)
        if not instance:
            # No instance to kill; clean up any orphaned tracking entry
            self._last_activity.pop(chat_id, None)
            return
        try:
            await asyncio.wait_for(instance.shutdown(), timeout=_FORCE_KILL_TIMEOUT)
        except Exception:
            # Any failure (timeout, OSError, etc.) - fall back to raw
            # SIGKILL. instance.force_kill() is effectively infallible
            # (catches its own OSError).
            instance.force_kill()
            log.warning("force_kill: shutdown failed for user %d, sent SIGKILL", chat_id)
        finally:
            # Remove from tracking regardless of how shutdown ended.
            # The finally block ensures cleanup even if CancelledError
            # (a BaseException, not caught by except Exception) propagates.
            self._pool.pop(chat_id, None)
            self._last_activity.pop(chat_id, None)

    async def change_workspace(
        self,
        chat_id: int,
        new_workspace: Path,
        workspace_config: WorkspaceConfig | None = None,
    ) -> None:
        """Switch a specific user's workspace."""
        instance = self.get(chat_id)
        # Explicit workspace change supersedes any pending restore.
        # Without this, the next send() would restore the old saved
        # workspace over the one just set. Settings-restore stays
        # pending: an explicit /workspace switch carries no signal
        # about user-level model or timeout preferences, and pre-split
        # behavior accidentally suppressed the DB-settings job here.
        self._pending_workspace_restore.discard(chat_id)
        await instance.change_workspace(new_workspace, workspace_config=workspace_config)

    async def restart(self, chat_id: int) -> None:
        """Restart a specific user's subprocess."""
        instance = self.get_if_exists(chat_id)
        if instance:
            await instance.restart()

    # ── Per-user property accessors ─────────────────────────────────

    def get_model(self, chat_id: int) -> str:
        """Get the active model for a user (or global default if no instance)."""
        instance = self.get_if_exists(chat_id)
        return instance.model if instance else self._config.default_model

    async def get_effective_model(self, chat_id: int) -> str:
        """Get the effective model, checking persisted settings if no instance exists.

        Unlike get_model() which returns the global default when no
        subprocess instance exists (e.g., after a service restart before
        the first message), this method resolves the model from the DB
        using the same precedence chain as resolve_user_defaults():
        user settings DB > users.yaml > global default.

        Use this in command handlers that display the current model
        (like /models) where accuracy matters more than speed.
        """
        instance = self.get_if_exists(chat_id)
        if instance:
            return instance.model
        defaults = await sessions.resolve_user_defaults(chat_id, self._config)
        return defaults["model"]

    def set_model(self, chat_id: int, model: str) -> None:
        """Set the model for a user's subprocess."""
        instance = self.get(chat_id)
        instance.model = canonicalize_model_for_backend(model, require_backend_name(instance))

    async def get_effective_workspace(self, chat_id: int) -> Path:
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
        instance = self._pool.get(chat_id)
        if instance is not None and chat_id not in self._pending_workspace_restore:
            return instance.workspace

        saved = await sessions.get_setting(f"workspace:{chat_id}")
        if not saved:
            if instance is not None:
                self._pending_workspace_restore.discard(chat_id)
            return resolve_home_workspace(chat_id, self._config)

        ws_path = Path(saved)
        base, allowed = await sessions.resolve_workspace_access(chat_id, self._config)
        if not ws_path.is_dir() or not is_workspace_allowed(ws_path, base, allowed):
            log.warning(
                "Saved workspace for user %d invalid; clearing: %s",
                chat_id,
                saved,
            )
            await sessions.delete_setting(f"workspace:{chat_id}")
            if instance is not None:
                self._pending_workspace_restore.discard(chat_id)
            return resolve_home_workspace(chat_id, self._config)

        # Only now do we materialize an instance. Avoiding instance
        # creation in the no-saved-row and stale-row branches keeps
        # the resolver cheap for cold reads that just want to confirm
        # "home is the right answer."
        instance = self.get(chat_id)
        # Re-validate inside _attempt_workspace_restore via its own
        # DB read so a concurrent writer that cleared the setting
        # between this check and the helper call is still handled
        # correctly (it returns home instead of trying to switch to a
        # since-deleted target).
        return await self._attempt_workspace_restore(chat_id, instance)

    def is_alive(self, chat_id: int) -> bool:
        """True if this user's subprocess is running."""
        instance = self.get_if_exists(chat_id)
        return instance.is_alive if instance else False

    def get_session_id(self, chat_id: int) -> str | None:
        """Get the session ID for a user's subprocess."""
        instance = self.get_if_exists(chat_id)
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
                chat_id
                for chat_id, last in self._last_activity.items()
                if now - last > idle_timeout and chat_id in self._pool and chat_id not in self._in_flight
            ]
            for chat_id in to_evict:
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
                if chat_id not in self._pool:
                    self._last_activity.pop(chat_id, None)
                    continue
                if self._last_activity.get(chat_id, 0) > now:
                    continue
                if chat_id in self._in_flight:
                    continue
                instance = self._pool.get(chat_id)
                try:
                    if instance and instance.is_alive:
                        try:
                            log.info("Evicting idle subprocess for user %d", chat_id)
                            await instance.shutdown()
                        except Exception:
                            # Graceful shutdown failed. Fall back to raw SIGKILL
                            # so the process doesn't become an orphan. force_kill()
                            # is effectively infallible (catches its own OSError).
                            log.exception("Error evicting subprocess for user %d, sending SIGKILL", chat_id)
                            instance.force_kill()
                finally:
                    # Remove from tracking after shutdown (alive instances) or
                    # unconditionally (dead instances). The finally block ensures
                    # cleanup even if CancelledError propagates from shutdown().
                    self._pool.pop(chat_id, None)
                    self._last_activity.pop(chat_id, None)

    async def shutdown(self) -> None:
        """Shut down all subprocesses and stop the eviction task."""
        if self._eviction_task:
            self._eviction_task.cancel()
            try:
                await self._eviction_task
            except asyncio.CancelledError:
                pass
            self._eviction_task = None
        for chat_id, instance in self._pool.items():
            try:
                log.info("Shutting down subprocess for user %d", chat_id)
                await instance.shutdown()
            except Exception:
                log.exception("Error shutting down subprocess for user %d", chat_id)
        self._pool.clear()
        self._last_activity.clear()
