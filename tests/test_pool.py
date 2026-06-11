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
from kai.goose import GooseBackend
from kai.pool import SubprocessPool


def _make_config(**overrides) -> Config:
    """
    Create a Config for pool tests.

    `claude_workspace=` is accepted as a back-compat alias for pre-#353
    tests that wanted a specific home directory for the default test
    chat. It translates to a UserConfig home_workspace override for
    chat IDs 111 and 222 (the test fixture's two default users), since
    Config no longer has a global claude_workspace field. Tests that
    pass their own `user_configs` win over the back-compat translation.
    """
    legacy_home = overrides.pop("claude_workspace", None)
    if legacy_home is not None and "user_configs" not in overrides:
        overrides["user_configs"] = {
            cid: UserConfig(telegram_id=cid, name=f"u{cid}", home_workspace=legacy_home) for cid in (111, 222)
        }
    defaults: dict = {
        "telegram_bot_token": "test",
        "allowed_user_ids": {111, 222},
        "default_model": "sonnet",
        "claude_timeout_seconds": 30,
        "claude_max_session_hours": 0,
        "claude_idle_timeout": 1800,
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

    def test_get_falls_back_to_defaults(self, tmp_path, monkeypatch):
        """
        User in users.yaml with no per-user overrides gets global
        defaults plus the per-user DATA_DIR/home/<chat_id>/ landing
        directory. The shared global home was removed; the new
        fallback is keyed by chat_id.
        """
        # Point the resolver at a tmp DATA_DIR so the test does not
        # touch the host's real /var/lib/kai or PROJECT_ROOT tree.
        monkeypatch.setattr("kai.backend.DATA_DIR", tmp_path)
        config = _make_config(
            user_configs={999: UserConfig(telegram_id=999, name="bob")},
        )
        pool = SubprocessPool(config=config, services_info=[])
        instance = pool.get(999)
        assert instance.claude_user is None
        # Per-user landing directory under the patched DATA_DIR.
        assert instance.workspace == tmp_path / "home" / "999"
        # ensure_user_home is idempotent and creates the dir on resolve.
        assert (tmp_path / "home" / "999").is_dir()

    def test_create_uses_user_config_settings(self, tmp_path):
        """User with model/timeout in users.yaml gets those values."""
        ws = tmp_path / "ws"
        ws.mkdir()
        user = UserConfig(
            telegram_id=111,
            name="alice",
            home_workspace=ws,
            model="opus",
            timeout=300,
        )
        config = _make_config(user_configs={111: user})
        pool = SubprocessPool(config=config, services_info=[])
        instance = pool.get(111)
        assert instance.model == "opus"
        assert instance.timeout_seconds == 300

    def test_create_falls_back_to_global_for_missing_user_fields(self):
        """User with no model/timeout gets global defaults."""
        user = UserConfig(telegram_id=111, name="alice")
        config = _make_config(
            user_configs={111: user},
            default_model="haiku",
            claude_timeout_seconds=60,
        )
        pool = SubprocessPool(config=config, services_info=[])
        instance = pool.get(111)
        assert instance.model == "haiku"
        assert instance.timeout_seconds == 60


# ── Per-user backend/provider routing ──────────────────────────────


class TestPerUserBackendRouting:
    def test_user_gets_goose_when_global_is_claude(self):
        """User with agent_backend=goose gets GooseBackend even when global is claude."""
        user = UserConfig(
            telegram_id=111,
            name="alice",
            agent_backend="goose",
            llm_provider="openai",
            model="gpt-5.4",
        )
        config = _make_config(user_configs={111: user})
        pool = SubprocessPool(config=config, services_info=[])
        instance = pool.get(111)
        assert isinstance(instance, GooseBackend)
        assert instance.provider == "openai"
        assert instance.model == "gpt-5.4"

    def test_user_without_backend_gets_global(self):
        """User with no agent_backend gets the global backend (claude)."""
        user = UserConfig(telegram_id=111, name="alice")
        config = _make_config(user_configs={111: user})
        pool = SubprocessPool(config=config, services_info=[])
        instance = pool.get(111)
        # Default global backend is claude -> ClaudeCodeBackend
        assert not isinstance(instance, GooseBackend)
        assert instance.provider == "anthropic"

    def test_user_provider_overrides_global(self):
        """User with llm_provider gets GooseBackend with that provider."""
        user = UserConfig(
            telegram_id=111,
            name="alice",
            agent_backend="goose",
            llm_provider="google",
            model="gemini-3-flash",
        )
        config = _make_config(
            agent_backend="goose",
            llm_provider="openai",
            default_model="gpt-5.4",
            user_configs={111: user},
        )
        pool = SubprocessPool(config=config, services_info=[])
        instance = pool.get(111)
        assert isinstance(instance, GooseBackend)
        # User's provider overrides global
        assert instance.provider == "google"
        assert instance.model == "gemini-3-flash"

    def test_model_default_cascade_cross_provider(self):
        """User on different provider than global gets provider-specific default."""
        user = UserConfig(
            telegram_id=111,
            name="alice",
            agent_backend="goose",
            llm_provider="openai",
            # No model set - should get openai default, not global "sonnet"
        )
        config = _make_config(user_configs={111: user})
        pool = SubprocessPool(config=config, services_info=[])
        instance = pool.get(111)
        # PROVIDER_DEFAULTS["openai"] = "gpt-5.5-pro" (the strongest
        # current OpenAI text model; per the agent-role-strongest rule).
        assert instance.model == "gpt-5.5-pro"

    @pytest.mark.asyncio
    async def test_invalid_stored_model_on_provider_mismatch(self):
        """DB has 'opus' but user is now on openai - falls back to provider default."""
        user = UserConfig(
            telegram_id=111,
            name="alice",
            agent_backend="goose",
            llm_provider="openai",
            model="gpt-5.4",
        )
        config = _make_config(user_configs={111: user})
        pool = SubprocessPool(config=config, services_info=[])
        instance = pool.get(111)
        assert instance.model == "gpt-5.4"

        # Simulate DB override with a model from a different provider
        db_settings = {"model": "opus"}
        with (
            patch("kai.pool.sessions.get_setting", new_callable=AsyncMock, return_value=None),
            patch("kai.pool.sessions.get_user_settings", new_callable=AsyncMock, return_value=db_settings),
            patch.object(instance, "restart", new_callable=AsyncMock),
        ):
            await pool._restore_workspace(111, instance)
            # "opus" is invalid for openai - should be ignored, model stays at gpt-5.4
            assert instance.model == "gpt-5.4"

    def test_opencode_user_on_claude_global_install(self):
        """
        Per-user `agent_backend: opencode` on a globally-claude install
        with a free-text model gets an OpenCodeBackend with the supplied
        model intact. OpenCode model strings are full provider/model
        IDs; no per-provider default substitution applies.
        """
        from kai.opencode import OpenCodeBackend

        user = UserConfig(
            telegram_id=111,
            name="alice",
            agent_backend="opencode",
            model="anthropic/claude-sonnet-4-6",
        )
        config = _make_config(
            agent_backend="claude",
            llm_provider="anthropic",
            default_model="sonnet",
            user_configs={111: user},
        )
        pool = SubprocessPool(config=config, services_info=[])
        instance = pool.get(111)
        assert isinstance(instance, OpenCodeBackend)
        assert instance.model == "anthropic/claude-sonnet-4-6"

    def test_opencode_user_without_model_falls_back_to_empty(self, caplog):
        """
        Per-user `agent_backend: opencode` with no model and global
        backend != opencode: OpenCodeBackend gets model="" so OPENCODE_
        CONFIG_CONTENT is omitted and OpenCode uses its own config
        files. A warning is logged so the operator notices the gap.
        """
        import logging

        from kai.opencode import OpenCodeBackend

        user = UserConfig(
            telegram_id=111,
            name="alice",
            agent_backend="opencode",
            # NO model: tests the bare opencode-override case.
        )
        config = _make_config(
            agent_backend="claude",
            llm_provider="anthropic",
            default_model="sonnet",
            user_configs={111: user},
        )
        pool = SubprocessPool(config=config, services_info=[])
        with caplog.at_level(logging.WARNING, logger="kai.pool"):
            instance = pool.get(111)
        assert isinstance(instance, OpenCodeBackend)
        assert instance.model == ""
        assert any("No model configured for opencode" in rec.message for rec in caplog.records)

    def test_codex_user_on_claude_global_install_gets_codex_default(self):
        """
        Per-user `agent_backend: codex` on a globally-claude install
        must NOT fall through to the global default_model ("sonnet"),
        which codex CLI rejects. Without the
        get_user_backend_and_provider routing in _create_instance,
        effective_provider was "" for these users, the cascade
        skipped the per-backend default branch, and the instance
        ended up running codex with sonnet.
        """
        from kai.codex import CodexBackend

        user = UserConfig(
            telegram_id=111,
            name="alice",
            agent_backend="codex",
            # NO llm_provider, NO model - tests the bare codex-override case.
        )
        # Globally-claude install (the failure surface from PR #489 review).
        config = _make_config(
            agent_backend="claude",
            llm_provider="anthropic",
            default_model="sonnet",
            user_configs={111: user},
        )
        pool = SubprocessPool(config=config, services_info=[])
        instance = pool.get(111)
        # Routes to codex, not claude.
        assert isinstance(instance, CodexBackend)
        # And uses codex's own default, not the global "sonnet".
        assert instance.model == "gpt-5.5"
        assert instance.provider == "openai"

    def test_goose_user_receives_os_user(self):
        """A goose user's os_user reaches GooseBackend, which wires
        the per-user sudo isolation in the shared ACP layer (same
        contract claude_user / codex_user carry on their backends)."""
        user = UserConfig(
            telegram_id=111,
            name="alice",
            agent_backend="goose",
            llm_provider="anthropic",
            model="sonnet",
            os_user="alice-os",
        )
        config = _make_config(user_configs={111: user})
        pool = SubprocessPool(config=config, services_info=[])
        instance = pool.get(111)
        assert isinstance(instance, GooseBackend)
        assert instance.os_user == "alice-os"

    def test_opencode_user_does_not_receive_os_user(self):
        """OpenCode chat stays on the service user even when the
        users.yaml entry carries an os_user: its auth file is
        per-OS-user and operator-provisioned, so the pool does not
        wire the isolation for opencode until that provisioning
        story exists."""
        from kai.opencode import OpenCodeBackend

        user = UserConfig(
            telegram_id=111,
            name="alice",
            agent_backend="opencode",
            model="anthropic/claude-sonnet-4-6",
            os_user="alice-os",
        )
        config = _make_config(user_configs={111: user})
        pool = SubprocessPool(config=config, services_info=[])
        instance = pool.get(111)
        assert isinstance(instance, OpenCodeBackend)
        assert instance.os_user is None


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
        pool = SubprocessPool(config=_make_config(default_model="haiku"), services_info=[])
        assert pool.get_model(999) == "haiku"

    @pytest.mark.asyncio
    async def test_get_effective_model_existing_instance(self):
        """get_effective_model returns the instance's model when it exists."""
        pool = SubprocessPool(config=_make_config(), services_info=[])
        instance = pool.get(111)
        instance.model = "opus"
        assert await pool.get_effective_model(111) == "opus"

    @pytest.mark.asyncio
    async def test_get_effective_model_no_instance(self):
        """get_effective_model falls back to DB settings when no instance exists."""
        pool = SubprocessPool(config=_make_config(default_model="sonnet"), services_info=[])
        # Simulate a user who set opus via /model (persisted in user settings DB).
        # With no instance, get_model() would return "sonnet" (global default),
        # but get_effective_model() should resolve from the DB and return "opus".
        with patch(
            "kai.sessions.resolve_user_defaults",
            new_callable=AsyncMock,
            return_value={
                "model": "opus",
                "timeout": 120,
            },
        ):
            assert await pool.get_effective_model(999) == "opus"

    def test_get_workspace_no_instance(self, tmp_path, monkeypatch):
        """
        get_workspace returns the per-user default when no instance exists.

        Post-#353 the fallback resolves through resolve_home_workspace
        instead of the removed global Config field, landing the user in
        DATA_DIR/home/<chat_id>/.
        """
        monkeypatch.setattr("kai.backend.DATA_DIR", tmp_path)
        pool = SubprocessPool(config=_make_config(), services_info=[])
        assert pool.get_workspace(999) == tmp_path / "home" / "999"

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
            # resolve_workspace_access returns permissive (None, []) so the
            # workspace passes _is_workspace_allowed.
            patch("kai.pool.sessions.resolve_workspace_access", new_callable=AsyncMock, return_value=(None, [])),
            patch("kai.pool.sessions.build_workspace_config", new_callable=AsyncMock, return_value=None),
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
    async def test_restore_applies_db_user_settings(self, tmp_path):
        """DB per-user settings override users.yaml baseline on restore."""
        ws = tmp_path / "saved_ws"
        ws.mkdir()
        # Users.yaml gives alice model=sonnet (via global default)
        user = UserConfig(telegram_id=111, name="alice")
        config = _make_config(user_configs={111: user})
        pool = SubprocessPool(config=config, services_info=[])
        instance = pool.get(111)
        # Instance starts with global model "sonnet"
        assert instance.model == "sonnet"

        # Simulate DB overrides: user set model=opus
        db_settings = {"model": "opus"}
        with (
            patch("kai.pool.sessions.get_setting", new_callable=AsyncMock, return_value=None),
            patch("kai.pool.sessions.get_user_settings", new_callable=AsyncMock, return_value=db_settings),
            patch.object(instance, "restart", new_callable=AsyncMock) as mock_restart,
        ):
            await pool._restore_workspace(111, instance)
            # Model is a CLI flag, so changing it triggers restart
            assert instance.model == "opus"
            mock_restart.assert_called_once()

    @pytest.mark.asyncio
    async def test_restore_no_db_settings_no_restart(self, tmp_path):
        """No DB per-user settings means no unnecessary restart."""
        pool = SubprocessPool(config=_make_config(), services_info=[])
        instance = pool.get(111)

        with (
            patch("kai.pool.sessions.get_setting", new_callable=AsyncMock, return_value=None),
            patch("kai.pool.sessions.get_user_settings", new_callable=AsyncMock, return_value={}),
            patch.object(instance, "restart", new_callable=AsyncMock) as mock_restart,
        ):
            await pool._restore_workspace(111, instance)
            mock_restart.assert_not_called()

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

    @pytest.mark.asyncio
    async def test_restore_workspace_in_user_allowed_list(self, tmp_path):
        """Saved workspace in per-user allowed list is restored."""
        ws = tmp_path / "user-ws"
        ws.mkdir()
        pool = SubprocessPool(config=_make_config(), services_info=[])
        instance = pool.get(111)

        with (
            patch("kai.pool.sessions.get_setting", new_callable=AsyncMock, return_value=str(ws)),
            # Per-user resolve returns the workspace in the allowed list
            patch(
                "kai.pool.sessions.resolve_workspace_access",
                new_callable=AsyncMock,
                return_value=(None, [ws]),
            ),
            patch("kai.pool.sessions.build_workspace_config", new_callable=AsyncMock, return_value=None),
            patch("kai.pool.sessions.get_user_settings", new_callable=AsyncMock, return_value={}),
            patch.object(instance, "change_workspace", new_callable=AsyncMock) as mock_change,
        ):
            await pool._restore_workspace(111, instance)
            mock_change.assert_called_once()

    @pytest.mark.asyncio
    async def test_restore_workspace_no_longer_allowed(self, tmp_path):
        """Saved workspace not in per-user allowed list is deleted."""
        ws = tmp_path / "revoked-ws"
        ws.mkdir()
        other_base = tmp_path / "base"
        other_base.mkdir()
        pool = SubprocessPool(config=_make_config(), services_info=[])
        pool.get(111)

        with (
            patch("kai.pool.sessions.get_setting", new_callable=AsyncMock, return_value=str(ws)),
            # Per-user resolve returns base that doesn't cover the workspace
            patch(
                "kai.pool.sessions.resolve_workspace_access",
                new_callable=AsyncMock,
                return_value=(other_base, []),
            ),
            patch("kai.pool.sessions.delete_setting", new_callable=AsyncMock) as mock_delete,
            patch("kai.pool.sessions.get_user_settings", new_callable=AsyncMock, return_value={}),
        ):
            await pool._restore_workspace(111, pool.get(111))
            mock_delete.assert_called_once_with("workspace:111")

    @pytest.mark.asyncio
    async def test_rejected_workspace_model_does_not_block_db_model(self, tmp_path):
        """
        A workspaces.yaml override invalid for the active backend (e.g.
        codex workspace pinning model=gpt-5.4-nano, which codex CLI
        rejects) is dropped by apply_workspace_model() but the original
        WorkspaceConfig object is still stored on instance.workspace_config
        with the invalid model field. The precedence guard in
        _restore_workspace must NOT treat that stored field as
        "workspace model applied"; otherwise a perfectly valid per-user
        DB model (gpt-5.4-mini) gets silently suppressed by a never-
        applied workspace override. Re-validate ws_model against the
        active backend before treating it as precedence-bearing.
        """
        from kai.codex import CodexBackend
        from kai.config import WorkspaceConfig

        # Workspace config pins gpt-5.4-nano, which is in PROVIDER_MODELS
        # ["openai"] but NOT in CODEX_MODELS - codex CLI rejects it.
        ws = (tmp_path / "ws").resolve()
        ws.mkdir()
        ws_config = WorkspaceConfig(path=ws, model="gpt-5.4-nano")
        user = UserConfig(
            telegram_id=111,
            name="alice",
            agent_backend="codex",
            home_workspace=ws,
        )
        config = _make_config(
            agent_backend="codex",
            llm_provider="openai",
            default_model="gpt-5.5",
            user_configs={111: user},
        )
        # get_workspace_config resolves the lookup key, so the stored key
        # must match what `.resolve()` returns.
        config.workspace_configs[ws] = ws_config

        pool = SubprocessPool(config=config, services_info=[])
        instance = pool.get(111)
        assert isinstance(instance, CodexBackend)
        # apply_workspace_model rejected the nano override at __init__,
        # so the instance kept the codex default. The WorkspaceConfig
        # object still has model="gpt-5.4-nano" on it.
        assert instance.model == "gpt-5.5"
        assert instance.workspace_config is not None
        assert instance.workspace_config.model == "gpt-5.4-nano"

        # DB has a valid per-user model override: gpt-5.4-mini.
        db_settings = {"model": "gpt-5.4-mini"}
        with (
            patch("kai.pool.sessions.get_setting", new_callable=AsyncMock, return_value=None),
            patch("kai.pool.sessions.get_user_settings", new_callable=AsyncMock, return_value=db_settings),
            patch.object(instance, "restart", new_callable=AsyncMock) as mock_restart,
        ):
            await pool._restore_workspace(111, instance)
            # The DB model wins because the workspace model was never
            # actually applied (rejected by apply_workspace_model).
            assert instance.model == "gpt-5.4-mini"
            mock_restart.assert_called_once()

    @pytest.mark.asyncio
    async def test_restore_workspace_under_user_base(self, tmp_path):
        """Saved workspace under per-user workspace_base is restored."""
        base = tmp_path / "base"
        ws = base / "project"
        ws.mkdir(parents=True)
        pool = SubprocessPool(config=_make_config(), services_info=[])
        instance = pool.get(111)

        with (
            patch("kai.pool.sessions.get_setting", new_callable=AsyncMock, return_value=str(ws)),
            # Per-user resolve returns a base that covers the workspace
            patch(
                "kai.pool.sessions.resolve_workspace_access",
                new_callable=AsyncMock,
                return_value=(base, []),
            ),
            patch("kai.pool.sessions.build_workspace_config", new_callable=AsyncMock, return_value=None),
            patch("kai.pool.sessions.get_user_settings", new_callable=AsyncMock, return_value={}),
            patch.object(instance, "change_workspace", new_callable=AsyncMock) as mock_change,
        ):
            await pool._restore_workspace(111, instance)
            mock_change.assert_called_once()


# ── Backend-name dispatch in restore path ──────────────────────────


class TestRestoreBackendNameDispatch:
    """
    _restore_workspace reads the backend identifier off `instance.backend_name`
    instead of inspecting the concrete class name. Pin each real backend's
    value plus the falsy fallback so an OpenCode (or any future) backend
    cannot regress the dispatch by setting backend_name incorrectly.
    """

    @pytest.mark.asyncio
    async def test_uses_instance_backend_name_for_validation(self, tmp_path):
        """The restore path validates the DB model against instance.backend_name.

        Spec PR 1 step 3: read backend_name off the instance, do not
        special-case CodexBackend / GooseBackend class names.
        """
        pool = SubprocessPool(config=_make_config(), services_info=[])
        instance = pool.get(111)
        # Force a non-default backend_name on this instance to prove
        # the dispatch reads off the attribute, not the class.
        instance.backend_name = "goose"
        instance.provider = "openai"

        db_settings = {"model": "gpt-5.4-mini"}
        with (
            patch("kai.pool.sessions.get_setting", new_callable=AsyncMock, return_value=None),
            patch("kai.pool.sessions.get_user_settings", new_callable=AsyncMock, return_value=db_settings),
            patch("kai.pool.validate_model_for_backend", return_value=True) as mock_validate,
            patch.object(instance, "restart", new_callable=AsyncMock),
        ):
            await pool._restore_workspace(111, instance)
            # Validator received the instance's backend_name, not "claude".
            # validate_model_for_backend may also be called for ws_model_raw,
            # but ws_model_raw is None here so the only invocation is for
            # the DB model.
            mock_validate.assert_called_once_with("gpt-5.4-mini", "goose", "openai")

    @pytest.mark.asyncio
    async def test_empty_backend_name_falls_back_to_claude_with_warning(self, tmp_path, caplog):
        """A test double or legacy stub with empty backend_name routes to "claude" with a warning."""
        import logging

        pool = SubprocessPool(config=_make_config(), services_info=[])
        instance = pool.get(111)
        # Simulate a stub backend that never overrode the ABC default.
        instance.backend_name = ""
        instance.provider = "anthropic"
        # Instance default model is "sonnet"; DB model must differ to
        # actually reach the validate_model_for_backend call in the
        # restore path (the guard short-circuits when they match).
        db_settings = {"model": "opus"}
        with (
            caplog.at_level(logging.WARNING, logger="kai.pool"),
            patch("kai.pool.sessions.get_setting", new_callable=AsyncMock, return_value=None),
            patch("kai.pool.sessions.get_user_settings", new_callable=AsyncMock, return_value=db_settings),
            patch("kai.pool.validate_model_for_backend", return_value=True) as mock_validate,
            patch.object(instance, "restart", new_callable=AsyncMock),
        ):
            await pool._restore_workspace(111, instance)
            # Fallback string is "claude".
            mock_validate.assert_called_once_with("opus", "claude", "anthropic")
        # Warning was logged about the empty backend_name.
        assert any("empty backend_name" in rec.message for rec in caplog.records)


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

        # Instance should be removed from pool after SIGKILL fallback
        assert 111 not in pool._pool

    @pytest.mark.asyncio
    async def test_force_kill_catches_non_timeout_exceptions(self):
        """force_kill catches any exception from shutdown, not just TimeoutError."""
        pool = SubprocessPool(config=_make_config(), services_info=[])
        instance = pool.get(111)

        with (
            patch.object(instance, "shutdown", side_effect=RuntimeError("unexpected")),
            patch.object(instance, "force_kill") as mock_raw_kill,
        ):
            # Should not propagate the exception to the caller
            await pool.force_kill(111)
            mock_raw_kill.assert_called_once()

        # Instance removed from pool after fallback
        assert 111 not in pool._pool
        assert 111 not in pool._last_activity

    @pytest.mark.asyncio
    async def test_force_kill_pops_after_successful_shutdown(self):
        """Instance is removed from pool only after shutdown succeeds."""
        pool = SubprocessPool(config=_make_config(), services_info=[])
        instance = pool.get(111)

        popped_during_shutdown = []

        async def check_pool_during_shutdown():
            # During shutdown, instance should still be in the pool
            popped_during_shutdown.append(111 in pool._pool)

        with patch.object(instance, "shutdown", side_effect=check_pool_during_shutdown):
            await pool.force_kill(111)

        # Instance was in pool during shutdown
        assert popped_during_shutdown == [True]
        # Instance removed after shutdown completed
        assert 111 not in pool._pool


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

    @pytest.mark.asyncio
    async def test_toctou_guard_skips_in_flight(self):
        """TOCTOU guard skips user who entered send() between snapshot and eviction."""
        config = _make_config(claude_idle_timeout=1)
        pool = SubprocessPool(config=config, services_info=[])

        # Two idle users: A's shutdown is the yield point, B gets the TOCTOU change.
        # get() order determines to_evict order (dict insertion order); 111 must
        # be processed first so its shutdown side effect modifies 222's state.
        a = pool.get(111)
        pool.get(222)
        pool._last_activity[111] = time.monotonic() - 10
        pool._last_activity[222] = time.monotonic() - 10

        async def a_shutdown_adds_b_in_flight():
            # Simulate user 222 entering send() during A's shutdown
            pool._in_flight.add(222)

        sleep_count = 0

        async def mock_sleep(_duration):
            nonlocal sleep_count
            sleep_count += 1
            if sleep_count > 1:
                raise asyncio.CancelledError

        # Set _proc so is_alive returns True (it checks _proc.returncode)
        mock_proc = MagicMock()
        mock_proc.returncode = None
        a._proc = mock_proc

        with (
            patch.object(a, "shutdown", side_effect=a_shutdown_adds_b_in_flight),
            patch("kai.pool.asyncio.sleep", side_effect=mock_sleep),
        ):
            try:
                await pool._eviction_loop()
            except asyncio.CancelledError:
                pass

        # A was evicted (first in the loop, before the TOCTOU change)
        assert 111 not in pool._pool
        # B survived (in-flight re-check caught the change)
        assert 222 in pool._pool

    @pytest.mark.asyncio
    async def test_toctou_guard_skips_removed_from_pool(self):
        """TOCTOU guard skips user removed from pool between snapshot and eviction."""
        config = _make_config(claude_idle_timeout=1)
        pool = SubprocessPool(config=config, services_info=[])

        # get() order determines to_evict order (dict insertion order); 111 must
        # be processed first so its shutdown side effect modifies 222's state.
        a = pool.get(111)
        pool.get(222)
        pool._last_activity[111] = time.monotonic() - 10
        pool._last_activity[222] = time.monotonic() - 10

        async def a_shutdown_removes_b():
            # Simulate force_kill removing user 222 during A's shutdown
            pool._pool.pop(222, None)

        sleep_count = 0

        async def mock_sleep(_duration):
            nonlocal sleep_count
            sleep_count += 1
            if sleep_count > 1:
                raise asyncio.CancelledError

        mock_proc = MagicMock()
        mock_proc.returncode = None
        a._proc = mock_proc

        with (
            patch.object(a, "shutdown", side_effect=a_shutdown_removes_b),
            patch("kai.pool.asyncio.sleep", side_effect=mock_sleep),
        ):
            try:
                await pool._eviction_loop()
            except asyncio.CancelledError:
                pass

        # A was evicted
        assert 111 not in pool._pool
        # B's _last_activity was cleaned up by the pool-membership guard
        assert 222 not in pool._last_activity

    @pytest.mark.asyncio
    async def test_eviction_proceeds_when_all_checks_pass(self):
        """User passing all three re-checks is evicted normally."""
        config = _make_config(claude_idle_timeout=1)
        pool = SubprocessPool(config=config, services_info=[])

        instance = pool.get(111)
        pool._last_activity[111] = time.monotonic() - 10

        sleep_count = 0

        async def mock_sleep(_duration):
            nonlocal sleep_count
            sleep_count += 1
            if sleep_count > 1:
                raise asyncio.CancelledError

        mock_proc = MagicMock()
        mock_proc.returncode = None
        instance._proc = mock_proc

        with (
            patch.object(instance, "shutdown", new_callable=AsyncMock),
            patch("kai.pool.asyncio.sleep", side_effect=mock_sleep),
        ):
            try:
                await pool._eviction_loop()
            except asyncio.CancelledError:
                pass

        # User was evicted: removed from pool and last_activity
        assert 111 not in pool._pool
        assert 111 not in pool._last_activity

    @pytest.mark.asyncio
    async def test_eviction_failure_triggers_sigkill_fallback(self):
        """Failed shutdown() in eviction loop triggers force_kill fallback."""
        config = _make_config(claude_idle_timeout=1)
        pool = SubprocessPool(config=config, services_info=[])

        instance = pool.get(111)
        pool._last_activity[111] = time.monotonic() - 10

        mock_proc = MagicMock()
        mock_proc.returncode = None
        instance._proc = mock_proc

        sleep_count = 0

        async def mock_sleep(_duration):
            nonlocal sleep_count
            sleep_count += 1
            if sleep_count > 1:
                raise asyncio.CancelledError

        with (
            patch.object(instance, "shutdown", side_effect=RuntimeError("crash")),
            patch.object(instance, "force_kill") as mock_raw_kill,
            patch("kai.pool.asyncio.sleep", side_effect=mock_sleep),
        ):
            try:
                await pool._eviction_loop()
            except asyncio.CancelledError:
                pass

        # SIGKILL fallback was called
        mock_raw_kill.assert_called_once()
        # Instance was removed from pool after fallback (not orphaned)
        assert 111 not in pool._pool
        assert 111 not in pool._last_activity

    @pytest.mark.asyncio
    async def test_eviction_pops_after_shutdown(self):
        """Instance stays in pool during shutdown, removed only after success."""
        config = _make_config(claude_idle_timeout=1)
        pool = SubprocessPool(config=config, services_info=[])

        instance = pool.get(111)
        pool._last_activity[111] = time.monotonic() - 10

        mock_proc = MagicMock()
        mock_proc.returncode = None
        instance._proc = mock_proc

        in_pool_during_shutdown = []

        async def check_pool():
            in_pool_during_shutdown.append(111 in pool._pool)

        sleep_count = 0

        async def mock_sleep(_duration):
            nonlocal sleep_count
            sleep_count += 1
            if sleep_count > 1:
                raise asyncio.CancelledError

        with (
            patch.object(instance, "shutdown", side_effect=check_pool),
            patch("kai.pool.asyncio.sleep", side_effect=mock_sleep),
        ):
            try:
                await pool._eviction_loop()
            except asyncio.CancelledError:
                pass

        # Instance was in pool during shutdown
        assert in_pool_during_shutdown == [True]
        # Removed after shutdown completed
        assert 111 not in pool._pool
