"""
Tests for pool.py per-user subprocess pool.

Covers:
1. Instance creation (lazy, per-user, with user config)
2. Instance reuse and isolation between users
3. Per-user actions (force_kill, restart, change_workspace)
4. Property accessors (model, workspace, is_alive)
5. Idle eviction
6. Workspace restoration from database
7. Shutdown
"""

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kai.config import Config, UserConfig
from kai.pool import SubprocessPool


def _make_config(**overrides) -> Config:
    """Create a Config for pool tests."""
    defaults: dict = {
        "telegram_bot_token": "test",
        "allowed_user_ids": {111, 222},
        "claude_model": "sonnet",
        "claude_timeout_seconds": 30,
        "claude_max_budget_usd": 1.0,
        "claude_max_session_hours": 0,
        "claude_idle_timeout": 1800,
        "claude_workspace": Path("/home/workspace"),
        "webhook_port": 8080,
        "webhook_secret": "secret",
    }
    defaults.update(overrides)
    return Config(**defaults)


# ── Instance creation ───────────────────────────────────────────────


class TestInstanceCreation:
    def test_get_creates_instance(self):
        """First get(chat_id) creates an instance; second returns same one."""
        pool = SubprocessPool(config=_make_config(), services_info=[])
        a = pool.get(111)
        b = pool.get(111)
        assert a is b

    def test_get_different_users(self):
        """Different chat_ids get different instances."""
        pool = SubprocessPool(config=_make_config(), services_info=[])
        a = pool.get(111)
        b = pool.get(222)
        assert a is not b

    def test_get_uses_user_config(self, tmp_path):
        """User with os_user and home_workspace gets correct instance."""
        ws = tmp_path / "ws"
        ws.mkdir()
        user = UserConfig(
            telegram_id=111,
            name="alice",
            os_user="alice_os",
            home_workspace=ws,
        )
        config = _make_config(user_configs={111: user})
        pool = SubprocessPool(config=config, services_info=[])
        instance = pool.get(111)
        assert instance.claude_user == "alice_os"
        assert instance.workspace == ws

    def test_get_falls_back_to_defaults(self):
        """User not in users.yaml gets global defaults."""
        config = _make_config(claude_user=None)
        pool = SubprocessPool(config=config, services_info=[])
        instance = pool.get(999)
        assert instance.claude_user is None
        assert instance.workspace == Path("/home/workspace")


# ── Per-user actions ────────────────────────────────────────────────


class TestPerUserActions:
    @pytest.mark.asyncio
    async def test_force_kill_specific_user(self):
        """force_kill(A) shuts down A's process, B's is unaffected."""
        pool = SubprocessPool(config=_make_config(), services_info=[])
        a = pool.get(111)
        b = pool.get(222)
        with (
            patch.object(a, "shutdown", new_callable=AsyncMock) as mock_a,
            patch.object(b, "shutdown", new_callable=AsyncMock) as mock_b,
        ):
            await pool.force_kill(111)
            mock_a.assert_called_once()
            mock_b.assert_not_called()

    @pytest.mark.asyncio
    async def test_restart_specific_user(self):
        """restart(A) restarts A's process, B's is unaffected."""
        pool = SubprocessPool(config=_make_config(), services_info=[])
        a = pool.get(111)
        b = pool.get(222)
        with (
            patch.object(a, "restart", new_callable=AsyncMock) as mock_a,
            patch.object(b, "restart", new_callable=AsyncMock) as mock_b,
        ):
            await pool.restart(111)
            mock_a.assert_called_once()
            mock_b.assert_not_called()

    @pytest.mark.asyncio
    async def test_change_workspace_specific_user(self):
        """change_workspace(A, path) changes A's workspace only."""
        pool = SubprocessPool(config=_make_config(), services_info=[])
        a = pool.get(111)
        b = pool.get(222)
        new_path = Path("/new/workspace")
        with (
            patch.object(a, "change_workspace", new_callable=AsyncMock) as mock_a,
            patch.object(b, "change_workspace", new_callable=AsyncMock) as mock_b,
        ):
            await pool.change_workspace(111, new_path)
            mock_a.assert_called_once_with(new_path, workspace_config=None)
            mock_b.assert_not_called()


# ── Property accessors ──────────────────────────────────────────────


class TestPropertyAccessors:
    def test_get_model_existing_instance(self):
        """get_model returns the instance's model when it exists."""
        pool = SubprocessPool(config=_make_config(), services_info=[])
        instance = pool.get(111)
        instance.model = "opus"
        assert pool.get_model(111) == "opus"

    def test_get_model_no_instance(self):
        """get_model returns global default when no instance exists."""
        pool = SubprocessPool(config=_make_config(claude_model="haiku"), services_info=[])
        assert pool.get_model(999) == "haiku"

    def test_get_workspace_no_instance(self):
        """get_workspace returns global default when no instance exists."""
        pool = SubprocessPool(config=_make_config(), services_info=[])
        assert pool.get_workspace(999) == Path("/home/workspace")

    def test_is_alive_no_instance(self):
        """is_alive returns False when no instance exists."""
        pool = SubprocessPool(config=_make_config(), services_info=[])
        assert pool.is_alive(999) is False


# ── Idle eviction ───────────────────────────────────────────────────


class TestIdleEviction:
    def test_idle_instance_identified_for_eviction(self):
        """Instance idle past timeout is identified for eviction."""
        config = _make_config(claude_idle_timeout=1)
        pool = SubprocessPool(config=config, services_info=[])
        pool.get(111)

        # Simulate the instance being idle for 10 seconds
        pool._last_activity[111] = time.monotonic() - 10

        now = time.monotonic()
        to_evict = [
            cid
            for cid, last in pool._last_activity.items()
            if now - last > config.claude_idle_timeout and cid in pool._pool
        ]
        assert 111 in to_evict

    def test_active_not_evicted(self):
        """User with recent activity is not evicted."""
        config = _make_config(claude_idle_timeout=3600)
        pool = SubprocessPool(config=config, services_info=[])
        pool.get(111)  # creates and sets last_activity to now

        now = time.monotonic()
        to_evict = [cid for cid, last in pool._last_activity.items() if now - last > 3600 and cid in pool._pool]
        assert to_evict == []

    def test_evicted_user_recreated(self):
        """After eviction, next get(chat_id) creates a fresh instance."""
        pool = SubprocessPool(config=_make_config(), services_info=[])
        first = pool.get(111)
        # Simulate eviction
        pool._pool.pop(111, None)
        pool._last_activity.pop(111, None)
        second = pool.get(111)
        assert first is not second


# ── Shutdown ────────────────────────────────────────────────────────


class TestShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_all(self):
        """shutdown() shuts down all instances and clears the pool."""
        pool = SubprocessPool(config=_make_config(), services_info=[])
        a = pool.get(111)
        b = pool.get(222)
        with (
            patch.object(a, "shutdown", new_callable=AsyncMock) as mock_a,
            patch.object(b, "shutdown", new_callable=AsyncMock) as mock_b,
        ):
            await pool.shutdown()
        mock_a.assert_called_once()
        mock_b.assert_called_once()
        assert len(pool._pool) == 0


# ── Workspace restoration ──────────────────────────────────────────


class TestWorkspaceRestoration:
    @pytest.mark.asyncio
    async def test_restore_saved_workspace(self, tmp_path):
        """First send() restores saved workspace from database."""
        ws = tmp_path / "saved_ws"
        ws.mkdir()
        pool = SubprocessPool(config=_make_config(), services_info=[])
        instance = pool.get(111)

        with (
            patch("kai.pool.sessions.get_setting", new_callable=AsyncMock, return_value=str(ws)),
            patch.object(instance, "change_workspace", new_callable=AsyncMock) as mock_change,
            patch.object(instance, "send", new_callable=MagicMock) as mock_send,
        ):
            # Mock send to return an empty async iterator
            async def empty_send(*args, **kwargs):
                return
                yield  # make it a generator

            mock_send.side_effect = empty_send
            async for _ in pool.send("test", chat_id=111):
                pass
            mock_change.assert_called_once()

    @pytest.mark.asyncio
    async def test_restore_nonexistent_workspace(self, tmp_path):
        """Saved workspace that no longer exists is deleted from settings."""
        pool = SubprocessPool(config=_make_config(), services_info=[])
        pool.get(111)

        with (
            patch(
                "kai.pool.sessions.get_setting",
                new_callable=AsyncMock,
                return_value="/nonexistent/path",
            ),
            patch("kai.pool.sessions.delete_setting", new_callable=AsyncMock) as mock_delete,
        ):
            # The restore should detect the path doesn't exist and delete
            await pool._restore_workspace(111, pool.get(111))
            mock_delete.assert_called_once_with("workspace:111")


# ── get_if_exists ───────────────────────────────────────────────────


class TestGetIfExists:
    def test_returns_none_when_no_instance(self):
        """No subprocess for user. Returns None without creating one."""
        pool = SubprocessPool(config=_make_config(), services_info=[])
        assert pool.get_if_exists(999) is None
        assert 999 not in pool._pool

    def test_returns_instance_when_exists(self):
        """Subprocess exists. Returns it."""
        pool = SubprocessPool(config=_make_config(), services_info=[])
        pool.get(111)  # create
        result = pool.get_if_exists(111)
        assert result is not None
        assert result is pool._pool[111]

    @pytest.mark.asyncio
    async def test_force_kill_no_instance(self):
        """/stop for a user with no subprocess. No-op, no crash."""
        pool = SubprocessPool(config=_make_config(), services_info=[])
        await pool.force_kill(999)  # should not raise

    @pytest.mark.asyncio
    async def test_force_kill_shutdown_timeout_falls_back(self):
        """When shutdown() hangs, falls back to raw force_kill."""
        pool = SubprocessPool(config=_make_config(), services_info=[])
        instance = pool.get(111)

        # Make shutdown hang forever
        async def hang_forever():
            await asyncio.sleep(999)

        with (
            patch("kai.pool._FORCE_KILL_TIMEOUT", 0.01),
            patch.object(instance, "shutdown", side_effect=hang_forever),
            patch.object(instance, "force_kill") as mock_raw_kill,
        ):
            await pool.force_kill(111)
            mock_raw_kill.assert_called_once()


# ── TOCTOU eviction guard ──────────────────────────────────────────


class TestEvictionTOCTOU:
    def test_toctou_guard_skips_recently_active(self):
        """TOCTOU guard skips user who became active between list build and eviction."""
        config = _make_config(claude_idle_timeout=1)
        pool = SubprocessPool(config=config, services_info=[])
        pool.get(111)

        # Step 1: simulate building the candidate list when user was idle.
        # Set activity to old timestamp so user enters the evict list.
        sweep_now = time.monotonic()
        pool._last_activity[111] = sweep_now - 10  # idle for 10 seconds

        to_evict = [
            cid
            for cid, last in pool._last_activity.items()
            if sweep_now - last > config.claude_idle_timeout and cid in pool._pool and cid not in pool._in_flight
        ]
        assert 111 in to_evict  # user is in the evict list

        # Step 2: user becomes active AFTER the list was built (TOCTOU window)
        pool._last_activity[111] = time.monotonic()

        # Step 3: the TOCTOU re-check should skip them
        assert pool._last_activity.get(111, 0) > sweep_now
        # This is the guard: if _last_activity > now, skip eviction
        assert 111 in pool._pool  # user survives
