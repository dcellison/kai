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

import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from pathlib import Path

from kai import sessions
from kai.backend import AgentBackend, StreamEvent, resolve_home_workspace
from kai.claude import ClaudeCodeBackend
from kai.config import (
    CODEX_DEFAULT_MODEL,
    OPEN_ENDED_PROVIDERS,
    PROVIDER_DEFAULTS,
    Config,
    WorkspaceConfig,
    get_effective_provider,
    get_user_backend_and_provider,
    validate_model_for_backend,
)
from kai.goose import GooseBackend
from kai.workspace_utils import is_workspace_allowed

log = logging.getLogger(__name__)


# How often the eviction loop checks for idle subprocesses (seconds).
_EVICTION_CHECK_INTERVAL = 60

# Maximum time to wait for shutdown() in force_kill before falling
# back to raw SIGKILL (seconds).
_FORCE_KILL_TIMEOUT = 5


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
        self._services_info = services_info
        self._pool: dict[int, AgentBackend] = {}
        self._last_activity: dict[int, float] = {}
        self._needs_workspace_restore: set[int] = set()
        self._in_flight: set[int] = set()  # chat_ids with active send()
        self._eviction_task: asyncio.Task | None = None

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
            self._pool[chat_id] = instance
            self._needs_workspace_restore.add(chat_id)
        self._last_activity[chat_id] = time.monotonic()
        return self._pool[chat_id]

    def _create_instance(self, chat_id: int) -> AgentBackend:
        """
        Create an AgentBackend for a specific user.

        Backend selection: per-user agent_backend (from users.yaml)
        overrides the global config.agent_backend. Both share the same
        ABC interface; pool.py is backend-agnostic after this point.

        Resolution order for each setting:
        1. UserConfig from users.yaml (os_user, home_workspace, model,
           budget, timeout, context_window, agent_backend, llm_provider)
        2. Global config defaults (from .env)

        Per-user DB overrides (set via /settings or /model) are applied
        in _restore_workspace() since they require async DB access.
        """
        user = self._config.get_user_config(chat_id)

        # Resolve through the single backend.resolve_home_workspace
        # helper: users.yaml override first, else DATA_DIR/home/<chat_id>/.
        # The old global home field on Config was removed by #353 because
        # it pointed every unconfigured user at a shared directory.
        workspace = resolve_home_workspace(chat_id, self._config)

        ws_config = self._config.get_workspace_config(workspace)

        # Per-user backend and provider, falling back to global config.
        # Routed through the canonical get_user_backend_and_provider
        # resolver so codex always reports provider="openai" even when
        # llm_provider is unset, matching the same cascade bot.py and
        # the install-time validator use. Without this, a user with
        # `agent_backend: codex` on a globally-claude install ended up
        # with effective_provider="" and the fallback below dispatched
        # the global default_model ("sonnet"), which codex CLI rejects.
        backend, effective_provider = get_user_backend_and_provider(user, self._config)
        # get_effective_provider hardcodes the backend->provider rule for
        # claude (anthropic) and codex (openai) so global_provider lines
        # up with effective_provider whenever the user has not overridden
        # the backend; no codex-specific patch needed here.
        global_provider = get_effective_provider(self._config.agent_backend, self._config.llm_provider)

        # Per-user model. When the user's effective backend differs
        # from the global one, the global default_model may not be
        # valid; fall back to the per-backend default instead.
        if user and user.model:
            model = user.model
        elif backend == self._config.agent_backend and effective_provider == global_provider:
            model = self._config.default_model
            # Catch the case where the global backend itself is an open-ended
            # provider and DEFAULT_MODEL is something generic like "sonnet".
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
        elif backend == "codex":
            # Per-user codex override on a non-codex global install.
            # Use codex's own default (gpt-5.5) - not PROVIDER_DEFAULTS["openai"]
            # which goose-on-openai still consults and which would
            # bypass the codex/goose surface separation.
            model = CODEX_DEFAULT_MODEL
        elif backend == "opencode":
            # Per-user opencode override on a non-opencode global install
            # with no user.model. There is no safe per-provider default
            # we can guess (opencode model strings are full provider/model
            # IDs and we do not know which providers the operator has
            # authenticated). Pass empty so OpenCodeBackend.build_env
            # skips OPENCODE_CONFIG_CONTENT and OpenCode falls back to
            # its own config files. Warn so the operator notices.
            log.warning(
                "No model configured for opencode user %d; OpenCode will use "
                "its own config defaults (set DEFAULT_MODEL globally or per-user "
                "in users.yaml to override)",
                chat_id,
            )
            model = ""
        else:
            model = PROVIDER_DEFAULTS.get(effective_provider, "")
            if not model:
                # Open-ended provider (openrouter, ollama) with no model configured.
                # Use the global default as a last resort but warn - it's almost
                # certainly wrong (e.g., "sonnet" sent to ollama).
                log.warning(
                    "No model configured for provider '%s' (chat %d); "
                    "falling back to global default '%s' which may not be valid",
                    effective_provider,
                    chat_id,
                    self._config.default_model,
                )
                model = self._config.default_model

        # budget_ceiling doubles as the fallback default for unconfigured users
        budget = user.max_budget if user and user.max_budget is not None else self._config.budget_ceiling
        timeout = user.timeout if user and user.timeout is not None else self._config.claude_timeout_seconds
        context_window = (
            user.context_window if user and user.context_window is not None else self._config.claude_max_context_window
        )
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

        # Backend selection: "goose" uses Goose ACP, "opencode" uses
        # OpenCode ACP, "codex" uses OpenAI Codex CLI's app-server
        # JSON-RPC protocol, anything else (including the default
        # "claude") uses Claude Code CLI.
        if backend == "goose":
            return GooseBackend(
                model=model,
                workspace=workspace,
                home_workspace=home_ws,
                webhook_port=self._config.webhook_port,
                webhook_secret=self._config.webhook_secret,
                max_budget_usd=budget,
                timeout_seconds=timeout,
                services_info=self._services_info,
                workspace_config=ws_config,
                max_context_window=context_window,
                provider=effective_provider,
                memory_enabled=self._config.memory_enabled,
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
                webhook_secret=self._config.webhook_secret,
                max_budget_usd=budget,
                timeout_seconds=timeout,
                services_info=self._services_info,
                workspace_config=ws_config,
                max_context_window=context_window,
                provider=effective_provider,
                memory_enabled=self._config.memory_enabled,
            )

        # os_user for sudo -u isolation. None = run as bot user.
        # Resolved here (rather than inside each branch) because both
        # ClaudeCodeBackend and CodexBackend consume it for per-user
        # subprocess isolation. Goose and OpenCode do not: both run as
        # the service user. Goose bills against a single GOOSE_PROVIDER
        # auth set in /etc/kai/env; OpenCode reads its auth from
        # ~/.local/share/opencode/auth.json which is per-OS-user but
        # operator-managed via `opencode auth login` outside the wizard.
        os_user = user.os_user if user else None

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
                webhook_secret=self._config.webhook_secret,
                max_budget_usd=budget,
                timeout_seconds=timeout,
                services_info=self._services_info,
                workspace_config=ws_config,
                max_context_window=context_window,
                provider="openai",
                codex_user=os_user,
                memory_enabled=self._config.memory_enabled,
            )

        return ClaudeCodeBackend(
            model=model,
            workspace=workspace,
            home_workspace=home_ws,
            webhook_port=self._config.webhook_port,
            webhook_secret=self._config.webhook_secret,
            max_budget_usd=budget,
            timeout_seconds=timeout,
            services_info=self._services_info,
            claude_user=os_user,
            max_session_hours=self._config.claude_max_session_hours,
            workspace_config=ws_config,
            max_context_window=context_window,
            autocompact_pct=self._config.claude_autocompact_pct,
            claude_effort_level=self._config.claude_effort_level,
            memory_enabled=self._config.memory_enabled,
        )

    # ── Prompt routing ──────────────────────────────────────────────

    async def send(self, prompt: str | list, *, chat_id: int) -> AsyncGenerator[StreamEvent]:
        """
        Route a prompt to the user's subprocess.

        On the first call for a newly created instance, restores the
        user's saved workspace from the database before sending.
        Marks the user as in-flight to prevent eviction mid-stream.
        """
        instance = self.get(chat_id)
        if chat_id in self._needs_workspace_restore:
            await self._restore_workspace(chat_id, instance)
            self._needs_workspace_restore.discard(chat_id)
        self._last_activity[chat_id] = time.monotonic()
        self._in_flight.add(chat_id)
        try:
            async for event in instance.send(prompt, chat_id=chat_id):
                yield event
        finally:
            self._in_flight.discard(chat_id)
            self._last_activity[chat_id] = time.monotonic()

    async def _restore_workspace(self, chat_id: int, instance: AgentBackend) -> None:
        """Restore a user's saved workspace from the database.

        Validates that the saved workspace is still an allowed path
        using per-user workspace access (workspace_base from users.yaml,
        allowed list from DB + global ALLOWED_WORKSPACES). An admin who
        removes a path should not have users silently bypass the
        restriction on their next message.
        """
        saved = await sessions.get_setting(f"workspace:{chat_id}")
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
                # Resolve per-user workspace access for the allowed check
                base, allowed = await sessions.resolve_workspace_access(chat_id, self._config)
                if not is_workspace_allowed(ws_path, base, allowed):
                    log.warning(
                        "Saved workspace for user %d is no longer allowed: %s",
                        chat_id,
                        saved,
                    )
                    await sessions.delete_setting(f"workspace:{chat_id}")
                else:
                    # Layer DB overrides on top of YAML baseline so user's
                    # per-workspace config (set via /workspace config) is
                    # applied on startup, not just after explicit switches.
                    yaml_config = self._config.get_workspace_config(ws_path)
                    ws_config = await sessions.build_workspace_config(yaml_config, ws_path, chat_id)
                    await instance.change_workspace(ws_path, workspace_config=ws_config)
                    log.info("Restored workspace for user %d: %s", chat_id, ws_path)

        # Apply per-user DB overrides (set via /settings or /model).
        # _create_instance() already applied users.yaml baselines, so we
        # only need the DB layer here. Workspace config takes precedence
        # over user settings (more specific wins).
        db_settings = await sessions.get_user_settings(chat_id)
        if db_settings:
            # Track whether any flag-level setting changed (requires restart
            # because the value is baked into the CLI command at startup)
            needs_restart = False

            # Read the backend identifier off the instance. Every
            # concrete backend sets `backend_name` as a class attribute
            # (claude_code / goose / codex / opencode). The ABC default
            # is the empty string, so a test double or legacy stub that
            # never overrides falls through to "claude" with a warning;
            # this preserves the historical default for unknown shapes
            # while keeping real backends out of the class-name if-chain
            # that previously had to be extended for every new backend.
            # Hoisted out of the DB-model branch because the ws_model
            # guard below also needs it.
            instance_backend = instance.backend_name
            if not instance_backend:
                log.warning(
                    "Instance %s has empty backend_name; falling back to 'claude'. "
                    "Concrete backends must set the backend_name class attribute.",
                    type(instance).__name__,
                )
                instance_backend = "claude"

            # Model: only apply if workspace config has a VALID model
            # for this backend. A workspaces.yaml entry like
            # `model: gpt-5.4-nano` applied to a codex instance is
            # rejected by apply_workspace_model() at backend __init__
            # time (the helper returns the current model and logs a
            # warning), but the original WorkspaceConfig - still
            # carrying the invalid model field - remains stored on
            # instance.workspace_config. Treating any non-empty
            # workspace_config.model as precedence-bearing therefore
            # blocks a valid per-user DB model (e.g. gpt-5.4-mini)
            # even though the workspace override was never applied.
            # Re-run the same validation here so the precedence guard
            # matches what was actually applied to instance.model.
            ws_model_raw = instance.workspace_config.model if instance.workspace_config else None
            ws_model_applied = bool(
                ws_model_raw and validate_model_for_backend(ws_model_raw, instance_backend, instance.provider)
            )
            if not ws_model_applied and "model" in db_settings and db_settings["model"] != instance.model:
                stored_model = db_settings["model"]
                if validate_model_for_backend(stored_model, instance_backend, instance.provider):
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

            # Budget: workspace config budget overrides user default
            ws_budget = instance.workspace_config.budget if instance.workspace_config else None
            if ws_budget is None and "budget" in db_settings:
                try:
                    instance.max_budget_usd = float(db_settings["budget"])
                except (ValueError, TypeError):
                    log.warning("Corrupt budget in DB for user %d", chat_id)

            # Timeout: workspace config timeout overrides user default
            ws_timeout = instance.workspace_config.timeout if instance.workspace_config else None
            if ws_timeout is None and "timeout" in db_settings:
                try:
                    instance.timeout_seconds = int(db_settings["timeout"])
                except (ValueError, TypeError):
                    log.warning("Corrupt timeout in DB for user %d", chat_id)

            # Context window: CLI flag (--settings), requires restart if changed.
            # No workspace-config guard here because WorkspaceConfig doesn't
            # have a context_window field. If one is added, guard this block
            # the same way model/budget/timeout are guarded above.
            if "context_window" in db_settings:
                try:
                    new_ctx = int(db_settings["context_window"])
                    if new_ctx != instance.max_context_window:
                        instance.max_context_window = new_ctx
                        needs_restart = True
                except (ValueError, TypeError):
                    log.warning("Corrupt context_window in DB for user %d", chat_id)

            # restart() kills the subprocess and spawns a new one, but
            # the backend *object* is preserved. Mutations made
            # above (budget, timeout, model, context_window) survive the
            # restart because the new subprocess reads from self.* attrs.
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
        # workspace over the one just set.
        self._needs_workspace_restore.discard(chat_id)
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
        instance.model = model

    def get_workspace(self, chat_id: int) -> Path:
        """
        Get the active workspace for a user.

        When no backend instance exists yet (user has not sent their
        first message in this process lifetime) we resolve the per-user
        home directory rather than the removed global default. This
        matches what _build_backend would do on first send().
        """
        instance = self.get_if_exists(chat_id)
        return instance.workspace if instance else resolve_home_workspace(chat_id, self._config)

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
        if self._config.claude_idle_timeout > 0:
            self._eviction_task = asyncio.create_task(self._eviction_loop())

    async def _eviction_loop(self) -> None:
        """Periodically kill idle subprocesses to free memory."""
        idle_timeout = self._config.claude_idle_timeout
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
