"""
Per-user Claude subprocess pool with lazy creation and idle eviction.

Provides functionality to:
1. Manage a dict of PersistentClaude instances keyed by chat_id
2. Create instances lazily on first message with per-user configuration
3. Route prompts to the correct user's subprocess
4. Evict idle subprocesses to reclaim memory on resource-constrained machines
5. Restore per-user saved workspaces on first interaction

This replaces the single shared PersistentClaude instance from Phases 1-2.
Each user gets their own Claude subprocess with full conversation isolation,
independent lifecycle, and OS-level enforcement via sudo -u when os_user is
configured in users.yaml.
"""

import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from pathlib import Path

from kai import sessions
from kai.claude import PersistentClaude, StreamEvent
from kai.config import Config, WorkspaceConfig

log = logging.getLogger(__name__)


def _is_workspace_allowed(path: Path, config: Config) -> bool:
    """Return True if path is covered by a configured workspace source.

    Accepts paths under WORKSPACE_BASE or in ALLOWED_WORKSPACES. If
    neither is configured, all paths are accepted (permissive mode).
    Duplicated from bot.py to avoid circular import (bot imports pool).
    """
    base = config.workspace_base
    if not base and not config.allowed_workspaces:
        return True
    resolved = path.resolve()
    resolved_base = base.resolve() if base else None
    in_base = resolved_base and (str(resolved).startswith(str(resolved_base) + "/") or resolved == resolved_base)
    in_allowed = resolved in config.allowed_workspaces
    return bool(in_base or in_allowed)


# How often the eviction loop checks for idle subprocesses (seconds).
_EVICTION_CHECK_INTERVAL = 60

# Maximum time to wait for shutdown() in force_kill before falling
# back to raw SIGKILL (seconds).
_FORCE_KILL_TIMEOUT = 5


class SubprocessPool:
    """
    Per-user Claude subprocess pool with lazy creation and idle eviction.

    Each user gets an independent PersistentClaude instance running as their
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
        self._pool: dict[int, PersistentClaude] = {}
        self._last_activity: dict[int, float] = {}
        self._needs_workspace_restore: set[int] = set()
        self._in_flight: set[int] = set()  # chat_ids with active send()
        self._eviction_task: asyncio.Task | None = None

    # ── Instance management ─────────────────────────────────────────

    def get(self, chat_id: int) -> PersistentClaude:
        """
        Get or create a PersistentClaude for the given user.

        Creates lazily on first access. The subprocess itself starts even
        later (on first send()), not here - PersistentClaude.__init__ is
        cheap; _ensure_started() is where the process spawns.
        """
        if chat_id not in self._pool:
            instance = self._create_instance(chat_id)
            self._pool[chat_id] = instance
            self._needs_workspace_restore.add(chat_id)
        self._last_activity[chat_id] = time.monotonic()
        return self._pool[chat_id]

    def _create_instance(self, chat_id: int) -> PersistentClaude:
        """
        Create a PersistentClaude for a specific user.

        Resolution order for each setting:
        1. UserConfig from users.yaml (os_user, home_workspace)
        2. Global config defaults (model, budget, timeout)

        Saved workspace from the database is deferred to the first
        send() call (async operation, can't run in sync get()).
        """
        user = self._config.get_user_config(chat_id)

        # Determine the user's home workspace
        workspace = self._config.claude_workspace
        if user and user.home_workspace:
            workspace = user.home_workspace

        # os_user for sudo -u isolation. None = run as bot user.
        os_user = user.os_user if user else self._config.claude_user

        ws_config = self._config.get_workspace_config(workspace)

        return PersistentClaude(
            model=self._config.claude_model,
            workspace=workspace,
            home_workspace=user.home_workspace if user else self._config.claude_workspace,
            webhook_port=self._config.webhook_port,
            webhook_secret=self._config.webhook_secret,
            max_budget_usd=self._config.claude_max_budget_usd,
            timeout_seconds=self._config.claude_timeout_seconds,
            services_info=self._services_info,
            claude_user=os_user,
            max_session_hours=self._config.claude_max_session_hours,
            workspace_config=ws_config,
            max_context_window=self._config.claude_max_context_window,
            autocompact_pct=self._config.claude_autocompact_pct,
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

    async def _restore_workspace(self, chat_id: int, instance: PersistentClaude) -> None:
        """Restore a user's saved workspace from the database.

        Validates that the saved workspace is still an allowed path
        (under WORKSPACE_BASE or in ALLOWED_WORKSPACES). An admin who
        removes a path from the allowed set should not have users
        silently bypass the restriction on their next message.
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
            elif not _is_workspace_allowed(ws_path, self._config):
                log.warning(
                    "Saved workspace for user %d is no longer allowed: %s",
                    chat_id,
                    saved,
                )
                await sessions.delete_setting(f"workspace:{chat_id}")
            else:
                ws_config = self._config.get_workspace_config(ws_path)
                await instance.change_workspace(ws_path, workspace_config=ws_config)
                log.info("Restored workspace for user %d: %s", chat_id, ws_path)

    # ── Per-user actions ────────────────────────────────────────────

    def get_if_exists(self, chat_id: int) -> PersistentClaude | None:
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
        return instance.model if instance else self._config.claude_model

    def set_model(self, chat_id: int, model: str) -> None:
        """Set the model for a user's subprocess."""
        instance = self.get(chat_id)
        instance.model = model

    def get_workspace(self, chat_id: int) -> Path:
        """Get the active workspace for a user."""
        instance = self.get_if_exists(chat_id)
        return instance.workspace if instance else self._config.claude_workspace

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
