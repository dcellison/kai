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
from collections.abc import AsyncIterator
from pathlib import Path

from kai import sessions
from kai.claude import PersistentClaude, StreamEvent
from kai.config import Config, WorkspaceConfig

log = logging.getLogger(__name__)

# How often the eviction loop checks for idle subprocesses (seconds).
_EVICTION_CHECK_INTERVAL = 60

# Default idle timeout if not configured via CLAUDE_IDLE_TIMEOUT (seconds).
# 30 minutes balances memory reclamation with avoiding unnecessary restarts.
_DEFAULT_IDLE_TIMEOUT = 1800


class SubprocessPool:
    """
    Per-user Claude subprocess pool with lazy creation and idle eviction.

    Each user gets an independent PersistentClaude instance running as their
    OS user. Instances are created on first message and evicted after idle
    timeout to manage memory on resource-constrained machines.
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
        )

    # ── Prompt routing ──────────────────────────────────────────────

    async def send(self, prompt: str | list, *, chat_id: int) -> AsyncIterator[StreamEvent]:
        """
        Route a prompt to the user's subprocess.

        On the first call for a newly created instance, restores the
        user's saved workspace from the database before sending.
        """
        instance = self.get(chat_id)
        if chat_id in self._needs_workspace_restore:
            await self._restore_workspace(chat_id, instance)
            self._needs_workspace_restore.discard(chat_id)
        self._last_activity[chat_id] = time.monotonic()
        async for event in instance.send(prompt, chat_id=chat_id):
            yield event
        self._last_activity[chat_id] = time.monotonic()

    async def _restore_workspace(self, chat_id: int, instance: PersistentClaude) -> None:
        """Restore a user's saved workspace from the database."""
        saved = await sessions.get_setting(f"workspace:{chat_id}")
        if saved:
            ws_path = Path(saved)
            if ws_path.is_dir():
                ws_config = self._config.get_workspace_config(ws_path)
                await instance.change_workspace(ws_path, workspace_config=ws_config)
                log.info("Restored workspace for user %d: %s", chat_id, ws_path)
            else:
                log.warning(
                    "Saved workspace for user %d no longer exists: %s",
                    chat_id,
                    saved,
                )
                await sessions.delete_setting(f"workspace:{chat_id}")

    # ── Per-user actions ────────────────────────────────────────────

    def force_kill(self, chat_id: int) -> None:
        """Kill a specific user's subprocess immediately."""
        instance = self._pool.get(chat_id)
        if instance:
            instance.force_kill()

    async def change_workspace(
        self,
        chat_id: int,
        new_workspace: Path,
        workspace_config: WorkspaceConfig | None = None,
    ) -> None:
        """Switch a specific user's workspace."""
        instance = self.get(chat_id)
        await instance.change_workspace(new_workspace, workspace_config=workspace_config)

    async def restart(self, chat_id: int) -> None:
        """Restart a specific user's subprocess."""
        instance = self._pool.get(chat_id)
        if instance:
            await instance.restart()

    # ── Per-user property accessors ─────────────────────────────────

    def get_model(self, chat_id: int) -> str:
        """Get the active model for a user (or global default if no instance)."""
        instance = self._pool.get(chat_id)
        return instance.model if instance else self._config.claude_model

    def set_model(self, chat_id: int, model: str) -> None:
        """Set the model for a user's subprocess."""
        instance = self.get(chat_id)
        instance.model = model

    def get_workspace(self, chat_id: int) -> Path:
        """Get the active workspace for a user."""
        instance = self._pool.get(chat_id)
        return instance.workspace if instance else self._config.claude_workspace

    def is_alive(self, chat_id: int) -> bool:
        """True if this user's subprocess is running."""
        instance = self._pool.get(chat_id)
        return instance.is_alive if instance else False

    def get_session_id(self, chat_id: int) -> str | None:
        """Get the session ID for a user's subprocess."""
        instance = self._pool.get(chat_id)
        return instance.session_id if instance else None

    # ── Idle eviction ───────────────────────────────────────────────

    def start(self) -> None:
        """Start the eviction background task."""
        self._eviction_task = asyncio.create_task(self._eviction_loop())

    async def _eviction_loop(self) -> None:
        """Periodically kill idle subprocesses to free memory."""
        while True:
            await asyncio.sleep(_EVICTION_CHECK_INTERVAL)
            now = time.monotonic()
            idle_timeout = self._config.claude_idle_timeout or _DEFAULT_IDLE_TIMEOUT
            to_evict = [
                chat_id
                for chat_id, last in self._last_activity.items()
                if now - last > idle_timeout and chat_id in self._pool
            ]
            for chat_id in to_evict:
                instance = self._pool.pop(chat_id, None)
                if instance and instance.is_alive:
                    log.info("Evicting idle subprocess for user %d", chat_id)
                    await instance.shutdown()
                self._last_activity.pop(chat_id, None)

    async def shutdown(self) -> None:
        """Shut down all subprocesses and stop the eviction task."""
        if self._eviction_task:
            self._eviction_task.cancel()
            self._eviction_task = None
        for chat_id, instance in self._pool.items():
            log.info("Shutting down subprocess for user %d", chat_id)
            await instance.shutdown()
        self._pool.clear()
        self._last_activity.clear()
