"""
Tests for self-service GitHub repo subscriptions (issue #220, PR 2 of 2).

Covers the bot command handlers (/github token, /github add, /github remove),
the webhook URL derivation helper, the github_api.py HTTP client functions,
and the updated _show_github display with source attribution and token status.

All GitHub API calls are mocked - no real HTTP requests are made.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from kai import github_api
from kai.bot import (
    _derive_webhook_url,
    _github_api_ensure_webhook,
    _github_api_remove_webhook,
    _handle_github_add,
    _handle_github_remove,
    _handle_github_token,
    _show_github,
    handle_github,
)
from kai.config import Config, UserConfig
from kai.github_api import GitHubAPIError

# ── Test helpers ────────────────────────────────────────────────────


def _make_config(**overrides) -> Config:
    """Create a Config for tests with sensible defaults."""
    defaults: dict = {
        "telegram_bot_token": "test-token",
        "allowed_user_ids": {1},
        "telegram_webhook_url": "https://api.example.com/webhook/telegram",
        "github_webhook_secret": "test-secret",
        "webhook_port": 8080,
    }
    defaults.update(overrides)
    return Config(**defaults)


def _make_update(text="hello", chat_id=12345, user_id=1):
    """Create a mock Telegram Update for handler tests."""
    update = MagicMock()
    update.message.text = text
    update.message.reply_text = AsyncMock()
    update.effective_chat.id = chat_id
    update.effective_user.id = user_id
    return update


def _make_context(config=None, args=None):
    """Create a mock PTB context with bot_data and args."""
    ctx = MagicMock()
    pool = MagicMock()
    ctx.bot_data = {
        "config": config or _make_config(),
        "pool": pool,
    }
    ctx.args = args or []
    return ctx


# ── _derive_webhook_url ────────────────────────────────────────────


class TestDeriveWebhookUrl:
    """Tests for the URL derivation helper that maps a Telegram webhook
    URL to the corresponding GitHub webhook endpoint."""

    def test_basic_derivation(self):
        """Replaces the Telegram path with /webhook/github."""
        result = _derive_webhook_url("https://api.syrinx.net/webhook/telegram")
        assert result == "https://api.syrinx.net/webhook/github"

    def test_preserves_scheme_and_host(self):
        """Scheme and authority are preserved, only path changes."""
        result = _derive_webhook_url("http://localhost:8080/anything/here")
        assert result == "http://localhost:8080/webhook/github"

    def test_preserves_query_and_fragment(self):
        """Query and fragment are preserved (urlunparse only replaces path).

        Not expected in practice - Telegram webhook URLs don't have query
        strings. This documents the actual behavior of _replace(path=...).
        """
        result = _derive_webhook_url("https://example.com/foo?bar=1#baz")
        assert result == "https://example.com/webhook/github?bar=1#baz"


# ── _handle_github_token ───────────────────────────────────────────


class TestHandleGithubToken:
    """Tests for /github token store and clear."""

    @pytest.mark.asyncio
    async def test_store_token(self):
        """/github token <pat> stores the token without echoing it."""
        update = _make_update()
        mock_set = AsyncMock()

        with patch("kai.bot.sessions.set_setting", mock_set):
            await _handle_github_token(update, 12345, ["ghp_abc123"])

        mock_set.assert_called_once_with("github_token:12345", "ghp_abc123")
        reply = update.message.reply_text.call_args[0][0]
        assert "stored" in reply.lower()
        # Token value must never appear in the reply
        assert "ghp_abc123" not in reply

    @pytest.mark.asyncio
    async def test_clear_token(self):
        """/github token clear removes the stored token."""
        update = _make_update()
        mock_delete = AsyncMock()

        with patch("kai.bot.sessions.delete_setting", mock_delete):
            await _handle_github_token(update, 12345, ["clear"])

        mock_delete.assert_called_once_with("github_token:12345")
        reply = update.message.reply_text.call_args[0][0]
        assert "removed" in reply.lower()

    @pytest.mark.asyncio
    async def test_clear_case_insensitive(self):
        """/github token CLEAR works regardless of case."""
        update = _make_update()
        mock_delete = AsyncMock()

        with patch("kai.bot.sessions.delete_setting", mock_delete):
            await _handle_github_token(update, 12345, ["CLEAR"])

        mock_delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_args_shows_usage(self):
        """/github token with no arguments shows usage text."""
        update = _make_update()
        await _handle_github_token(update, 12345, [])

        reply = update.message.reply_text.call_args[0][0]
        assert "usage" in reply.lower()


# ── _handle_github_add ─────────────────────────────────────────────


class TestHandleGithubAdd:
    """Tests for /github add <owner/repo>."""

    def _patch_sessions(
        self,
        effective=None,
        added=None,
        removed=None,
        token=None,
    ):
        """Build a dict of patches for the sessions functions called by _handle_github_add."""
        return {
            "get_effective_repos": AsyncMock(return_value=effective or []),
            "get_github_added_repos": AsyncMock(return_value=list(added or [])),
            "get_github_removed_repos": AsyncMock(return_value=list(removed or [])),
            "set_github_added_repos": AsyncMock(),
            "set_github_removed_repos": AsyncMock(),
            "get_setting": AsyncMock(return_value=token),
        }

    @pytest.mark.asyncio
    async def test_no_args_shows_usage(self):
        """/github add with no repo shows usage."""
        update = _make_update()
        config = _make_config()
        await _handle_github_add(update, 12345, [], config)

        reply = update.message.reply_text.call_args[0][0]
        assert "usage" in reply.lower()

    @pytest.mark.asyncio
    async def test_invalid_format_rejected(self):
        """Repo name that doesn't match owner/repo format is rejected."""
        update = _make_update()
        config = _make_config()
        await _handle_github_add(update, 12345, ["not-a-repo"], config)

        reply = update.message.reply_text.call_args[0][0]
        assert "invalid" in reply.lower()

    @pytest.mark.asyncio
    async def test_already_subscribed(self):
        """Adding a repo the user is already subscribed to shows a message."""
        update = _make_update()
        config = _make_config()
        patches = self._patch_sessions(effective=["alice/repo"])

        with patch("kai.bot.sessions", MagicMock(**patches)):
            await _handle_github_add(update, 12345, ["Alice/Repo"], config)

        reply = update.message.reply_text.call_args[0][0]
        assert "already subscribed" in reply.lower()

    @pytest.mark.asyncio
    async def test_add_no_token_shows_manual_fallback(self):
        """Adding a repo without a stored token shows manual webhook instructions."""
        update = _make_update()
        config = _make_config()
        patches = self._patch_sessions(effective=[], added=[], token=None)

        with patch("kai.bot.sessions", MagicMock(**patches)):
            await _handle_github_add(update, 12345, ["alice/repo"], config)

        reply = update.message.reply_text.call_args[0][0]
        assert "subscribed" in reply.lower()
        assert "admin:repo_hook" in reply
        assert "webhook" in reply.lower()

    @pytest.mark.asyncio
    async def test_add_with_token_success(self):
        """Adding a repo with a valid token auto-registers the webhook."""
        update = _make_update()
        config = _make_config()
        patches = self._patch_sessions(effective=[], added=[], token="ghp_test")

        with (
            patch("kai.bot.sessions", MagicMock(**patches)),
            patch("kai.bot._github_api_ensure_webhook", new_callable=AsyncMock) as mock_ensure,
        ):
            await _handle_github_add(update, 12345, ["alice/repo"], config)

        mock_ensure.assert_called_once_with("alice/repo", "ghp_test", config)
        reply = update.message.reply_text.call_args[0][0]
        assert "subscribed" in reply.lower()
        assert "webhook registered" in reply.lower()

    @pytest.mark.asyncio
    async def test_add_with_token_403_shows_manual(self):
        """403 from GitHub (no admin access) still subscribes, shows manual fallback."""
        update = _make_update()
        config = _make_config()
        patches = self._patch_sessions(effective=[], added=[], token="ghp_test")

        with (
            patch("kai.bot.sessions", MagicMock(**patches)),
            patch(
                "kai.bot._github_api_ensure_webhook",
                new_callable=AsyncMock,
                side_effect=GitHubAPIError(403, "Forbidden"),
            ),
        ):
            await _handle_github_add(update, 12345, ["alice/repo"], config)

        reply = update.message.reply_text.call_args[0][0]
        assert "subscribed" in reply.lower()
        assert "admin:repo_hook" in reply

    @pytest.mark.asyncio
    async def test_add_with_token_404_aborts(self):
        """404 from GitHub rolls back the subscription."""
        update = _make_update()
        config = _make_config()
        mock_sessions = MagicMock(**self._patch_sessions(effective=[], added=[], token="ghp_test"))

        with (
            patch("kai.bot.sessions", mock_sessions),
            patch(
                "kai.bot._github_api_ensure_webhook",
                new_callable=AsyncMock,
                side_effect=GitHubAPIError(404, "Not Found"),
            ),
        ):
            await _handle_github_add(update, 12345, ["alice/repo"], config)

        reply = update.message.reply_text.call_args[0][0]
        assert "not found" in reply.lower()
        # Subscription should have been rolled back
        mock_sessions.set_github_added_repos.assert_called()

    @pytest.mark.asyncio
    async def test_add_with_token_401_warns(self):
        """401 from GitHub (bad token) subscribes but warns about token."""
        update = _make_update()
        config = _make_config()
        patches = self._patch_sessions(effective=[], added=[], token="ghp_expired")

        with (
            patch("kai.bot.sessions", MagicMock(**patches)),
            patch(
                "kai.bot._github_api_ensure_webhook",
                new_callable=AsyncMock,
                side_effect=GitHubAPIError(401, "Unauthorized"),
            ),
        ):
            await _handle_github_add(update, 12345, ["alice/repo"], config)

        reply = update.message.reply_text.call_args[0][0]
        assert "subscribed" in reply.lower()
        assert "invalid or expired" in reply.lower()

    @pytest.mark.asyncio
    async def test_add_network_error_subscribes_and_warns(self):
        """Network errors still complete the subscription with a retry hint."""
        update = _make_update()
        config = _make_config()
        patches = self._patch_sessions(effective=[], added=[], token="ghp_test")

        with (
            patch("kai.bot.sessions", MagicMock(**patches)),
            patch(
                "kai.bot._github_api_ensure_webhook",
                new_callable=AsyncMock,
                side_effect=GitHubAPIError(0, "timeout"),
            ),
        ):
            await _handle_github_add(update, 12345, ["alice/repo"], config)

        reply = update.message.reply_text.call_args[0][0]
        assert "subscribed" in reply.lower()
        assert "network error" in reply.lower()

    @pytest.mark.asyncio
    async def test_readd_after_remove(self):
        """Re-adding a previously removed repo cancels the removal."""
        update = _make_update()
        config = _make_config()
        # Repo is in removed list (not in effective), no token
        patches = self._patch_sessions(
            effective=[],
            added=[],
            removed=["alice/repo"],
            token=None,
        )

        with patch("kai.bot.sessions", MagicMock(**patches)) as mock_sessions:
            await _handle_github_add(update, 12345, ["alice/repo"], config)

        reply = update.message.reply_text.call_args[0][0]
        assert "re-subscribed" in reply.lower()
        # Should have cleared alice/repo from the removed list
        mock_sessions.set_github_removed_repos.assert_called_once()
        call_args = mock_sessions.set_github_removed_repos.call_args[0]
        assert "alice/repo" not in call_args[1]

    @pytest.mark.asyncio
    async def test_readd_after_remove_404_rolls_back(self):
        """Re-add that gets a 404 restores the removal (puts repo back in removed list)."""
        update = _make_update()
        config = _make_config()
        # Repo was previously removed, user tries to re-add it
        patches = self._patch_sessions(
            effective=[],
            added=[],
            removed=["alice/repo"],
            token="ghp_test",
        )

        with (
            patch("kai.bot.sessions", MagicMock(**patches)) as mock_sessions,
            patch(
                "kai.bot._github_api_ensure_webhook",
                new_callable=AsyncMock,
                side_effect=GitHubAPIError(404, "Not Found"),
            ),
        ):
            await _handle_github_add(update, 12345, ["alice/repo"], config)

        reply = update.message.reply_text.call_args[0][0]
        assert "not found" in reply.lower()
        # The removal should have been restored (repo back in removed list)
        mock_sessions.set_github_removed_repos.assert_called()
        # First call clears it from removed (the re-add), second call
        # restores it (the 404 rollback)
        calls = mock_sessions.set_github_removed_repos.call_args_list
        assert len(calls) == 2
        # Second call should put alice/repo back
        assert "alice/repo" in calls[1][0][1]

    @pytest.mark.asyncio
    async def test_polling_mode_shows_manual_fallback(self):
        """In polling mode (no webhook URL), always shows manual fallback."""
        update = _make_update()
        # Polling mode: no telegram_webhook_url
        config = _make_config(telegram_webhook_url=None)
        patches = self._patch_sessions(effective=[], added=[], token="ghp_test")

        with (
            patch("kai.bot.sessions", MagicMock(**patches)),
            patch(
                "kai.bot._github_api_ensure_webhook",
                new_callable=AsyncMock,
                side_effect=GitHubAPIError(0, "No webhook URL"),
            ),
        ):
            await _handle_github_add(update, 12345, ["alice/repo"], config)

        reply = update.message.reply_text.call_args[0][0]
        assert "subscribed" in reply.lower()


# ── _handle_github_remove ──────────────────────────────────────────


class TestHandleGithubRemove:
    """Tests for /github remove <owner/repo>."""

    def _patch_sessions(
        self,
        effective=None,
        added=None,
        removed=None,
        token=None,
    ):
        """Build a dict of patches for sessions functions called by _handle_github_remove."""
        return {
            "get_effective_repos": AsyncMock(return_value=effective or []),
            "get_github_added_repos": AsyncMock(return_value=list(added or [])),
            "get_github_removed_repos": AsyncMock(return_value=list(removed or [])),
            "set_github_added_repos": AsyncMock(),
            "set_github_removed_repos": AsyncMock(),
            "get_setting": AsyncMock(return_value=token),
        }

    @pytest.mark.asyncio
    async def test_no_args_shows_usage(self):
        """/github remove with no repo shows usage."""
        update = _make_update()
        config = _make_config()
        await _handle_github_remove(update, 12345, [], config)

        reply = update.message.reply_text.call_args[0][0]
        assert "usage" in reply.lower()

    @pytest.mark.asyncio
    async def test_invalid_format_rejected(self):
        """Repo name that doesn't match owner/repo format is rejected."""
        update = _make_update()
        config = _make_config()
        await _handle_github_remove(update, 12345, ["bad"], config)

        reply = update.message.reply_text.call_args[0][0]
        assert "invalid" in reply.lower()

    @pytest.mark.asyncio
    async def test_not_subscribed(self):
        """Removing a repo the user is not subscribed to shows a message."""
        update = _make_update()
        config = _make_config()
        patches = self._patch_sessions(effective=[])

        with patch("kai.bot.sessions", MagicMock(**patches)):
            await _handle_github_remove(update, 12345, ["alice/repo"], config)

        reply = update.message.reply_text.call_args[0][0]
        assert "not subscribed" in reply.lower()

    @pytest.mark.asyncio
    async def test_last_subscriber_with_token(self):
        """Last subscriber with a token removes the webhook automatically."""
        update = _make_update()
        # No other users configured
        config = _make_config(user_configs={12345: UserConfig(telegram_id=12345, name="alice")})
        patches = self._patch_sessions(
            effective=["alice/repo"],
            added=["alice/repo"],
            token="ghp_test",
        )

        with (
            patch("kai.bot.sessions", MagicMock(**patches)),
            patch("kai.bot._github_api_remove_webhook", new_callable=AsyncMock) as mock_remove,
        ):
            await _handle_github_remove(update, 12345, ["alice/repo"], config)

        mock_remove.assert_called_once_with("alice/repo", "ghp_test", config)
        reply = update.message.reply_text.call_args[0][0]
        assert "unsubscribed" in reply.lower()
        assert "webhook removed" in reply.lower()

    @pytest.mark.asyncio
    async def test_last_subscriber_without_token(self):
        """Last subscriber without a token gets manual removal instructions."""
        update = _make_update()
        config = _make_config(user_configs={12345: UserConfig(telegram_id=12345, name="alice")})
        patches = self._patch_sessions(
            effective=["alice/repo"],
            added=["alice/repo"],
            token=None,
        )

        with patch("kai.bot.sessions", MagicMock(**patches)):
            await _handle_github_remove(update, 12345, ["alice/repo"], config)

        reply = update.message.reply_text.call_args[0][0]
        assert "unsubscribed" in reply.lower()
        assert "settings > webhooks" in reply.lower()

    @pytest.mark.asyncio
    async def test_other_subscribers_remain(self):
        """When other subscribers exist, the webhook is kept."""
        update = _make_update()
        # Two users, both subscribed
        bob = UserConfig(telegram_id=99999, name="bob", github_repos=["alice/repo"])
        config = _make_config(
            user_configs={
                12345: UserConfig(telegram_id=12345, name="alice"),
                99999: bob,
            }
        )
        # Alice is subscribed (effective includes the repo), Bob too
        patches = self._patch_sessions(
            effective=["alice/repo"],
            added=["alice/repo"],
            token="ghp_test",
        )

        # get_effective_repos is called twice:
        #   1. For Alice (chat_id=12345) to check she's subscribed
        #   2. For Bob (chat_id=99999) in the cross-user subscriber check
        # Both calls need to return the repo so Alice passes the initial
        # check and Bob keeps the webhook alive.
        async def _mock_effective(chat_id, yaml_repos):
            return ["alice/repo"]  # Both users are subscribed

        mock_sessions = MagicMock(**patches)
        mock_sessions.get_effective_repos = AsyncMock(side_effect=_mock_effective)

        with patch("kai.bot.sessions", mock_sessions):
            await _handle_github_remove(update, 12345, ["alice/repo"], config)

        reply = update.message.reply_text.call_args[0][0]
        assert "unsubscribed" in reply.lower()
        assert "webhook kept" in reply.lower()

    @pytest.mark.asyncio
    async def test_remove_from_yaml_repo(self):
        """Removing a yaml-only repo adds it to the removed list (not added list)."""
        update = _make_update()
        config = _make_config(
            user_configs={12345: UserConfig(telegram_id=12345, name="alice", github_repos=["alice/repo"])}
        )
        # Repo is in effective (from yaml), not in DB-added
        patches = self._patch_sessions(
            effective=["alice/repo"],
            added=[],  # Not in added list (it's from yaml)
            token=None,
        )

        with patch("kai.bot.sessions", MagicMock(**patches)) as mock_sessions:
            await _handle_github_remove(update, 12345, ["alice/repo"], config)

        # Should have added to removed list (not tried to remove from added)
        mock_sessions.set_github_removed_repos.assert_called_once()
        call_args = mock_sessions.set_github_removed_repos.call_args[0]
        assert "alice/repo" in call_args[1]

    @pytest.mark.asyncio
    async def test_remove_webhook_failure_still_unsubscribes(self):
        """Webhook deregistration failure still completes the unsubscription."""
        update = _make_update()
        config = _make_config(user_configs={12345: UserConfig(telegram_id=12345, name="alice")})
        patches = self._patch_sessions(
            effective=["alice/repo"],
            added=["alice/repo"],
            token="ghp_test",
        )

        with (
            patch("kai.bot.sessions", MagicMock(**patches)),
            patch(
                "kai.bot._github_api_remove_webhook",
                new_callable=AsyncMock,
                side_effect=GitHubAPIError(500, "Server error"),
            ),
        ):
            await _handle_github_remove(update, 12345, ["alice/repo"], config)

        reply = update.message.reply_text.call_args[0][0]
        assert "unsubscribed" in reply.lower()
        assert "could not remove" in reply.lower()


# ── _github_api_ensure_webhook ─────────────────────────────────────


class TestEnsureWebhook:
    """Tests for the ensure_webhook bridge function in bot.py."""

    @pytest.mark.asyncio
    async def test_polling_mode_raises(self):
        """Raises GitHubAPIError(0) when telegram_webhook_url is None."""
        config = _make_config(telegram_webhook_url=None)
        with pytest.raises(GitHubAPIError, match="polling mode"):
            await _github_api_ensure_webhook("alice/repo", "ghp_test", config)

    @pytest.mark.asyncio
    async def test_missing_github_secret_raises(self):
        """GitHub registration cannot borrow the generic secret."""
        config = _make_config(
            github_webhook_secret="",
            generic_webhook_secret="generic-secret",
        )

        with pytest.raises(GitHubAPIError, match="GITHUB_WEBHOOK_SECRET"):
            await _github_api_ensure_webhook("alice/repo", "ghp_test", config)

    @pytest.mark.asyncio
    async def test_already_exists_stores_hook_id(self):
        """When the hook already exists, stores its ID and returns."""
        config = _make_config()

        with (
            patch("kai.bot.github_api.check_webhook_exists", new_callable=AsyncMock, return_value=(True, 42)),
            patch("kai.bot.github_api.register_webhook", new_callable=AsyncMock) as mock_register,
            patch("kai.bot.sessions.set_setting", new_callable=AsyncMock) as mock_set,
        ):
            await _github_api_ensure_webhook("alice/repo", "ghp_test", config)

        mock_register.assert_not_called()
        mock_set.assert_called_once_with("github_hook_id:alice/repo", "42")

    @pytest.mark.asyncio
    async def test_registers_and_stores_hook_id(self):
        """When no hook exists, registers one and stores the ID."""
        config = _make_config()

        with (
            patch("kai.bot.github_api.check_webhook_exists", new_callable=AsyncMock, return_value=(False, None)),
            patch("kai.bot.github_api.register_webhook", new_callable=AsyncMock, return_value=99) as mock_register,
            patch("kai.bot.sessions.set_setting", new_callable=AsyncMock) as mock_set,
        ):
            await _github_api_ensure_webhook("alice/repo", "ghp_test", config)

        mock_register.assert_called_once_with(
            "alice",
            "repo",
            "ghp_test",
            "https://api.example.com/webhook/github",
            "test-secret",
        )
        mock_set.assert_called_once_with("github_hook_id:alice/repo", "99")


# ── _github_api_remove_webhook ─────────────────────────────────────


class TestRemoveWebhook:
    """Tests for the remove_webhook bridge function in bot.py."""

    @pytest.mark.asyncio
    async def test_polling_mode_noop(self):
        """No-op when telegram_webhook_url is None."""
        config = _make_config(telegram_webhook_url=None)
        # Should not raise or call any API
        await _github_api_remove_webhook("alice/repo", "ghp_test", config)

    @pytest.mark.asyncio
    async def test_uses_stored_hook_id(self):
        """Uses the stored hook ID to deregister, then deletes the setting."""
        config = _make_config()

        with (
            patch("kai.bot.sessions.get_setting", new_callable=AsyncMock, return_value="42"),
            patch("kai.bot.sessions.delete_setting", new_callable=AsyncMock) as mock_delete,
            patch("kai.bot.github_api.deregister_webhook", new_callable=AsyncMock) as mock_dereg,
        ):
            await _github_api_remove_webhook("alice/repo", "ghp_test", config)

        mock_dereg.assert_called_once_with("alice", "repo", 42, "ghp_test")
        mock_delete.assert_called_once_with("github_hook_id:alice/repo")

    @pytest.mark.asyncio
    async def test_falls_back_to_check(self):
        """When no stored hook ID, queries GitHub to find the hook."""
        config = _make_config()

        with (
            patch("kai.bot.sessions.get_setting", new_callable=AsyncMock, return_value=None),
            patch("kai.bot.github_api.check_webhook_exists", new_callable=AsyncMock, return_value=(True, 77)),
            patch("kai.bot.github_api.deregister_webhook", new_callable=AsyncMock) as mock_dereg,
        ):
            await _github_api_remove_webhook("alice/repo", "ghp_test", config)

        mock_dereg.assert_called_once_with("alice", "repo", 77, "ghp_test")

    @pytest.mark.asyncio
    async def test_already_gone_noop(self):
        """When the hook doesn't exist (not stored, not found), does nothing."""
        config = _make_config()

        with (
            patch("kai.bot.sessions.get_setting", new_callable=AsyncMock, return_value=None),
            patch("kai.bot.github_api.check_webhook_exists", new_callable=AsyncMock, return_value=(False, None)),
            patch("kai.bot.github_api.deregister_webhook", new_callable=AsyncMock) as mock_dereg,
        ):
            await _github_api_remove_webhook("alice/repo", "ghp_test", config)

        mock_dereg.assert_not_called()


# ── _show_github with repos source attribution ─────────────────────


class TestShowGithubRepos:
    """Tests for the updated _show_github display with repo source
    attribution and token status."""

    @pytest.mark.asyncio
    async def test_yaml_repo_shows_source(self):
        """Repos from users.yaml are labeled as such."""
        update = _make_update()
        user = UserConfig(
            telegram_id=12345,
            name="alice",
            github_repos=["alice/repo"],
        )
        config = _make_config(user_configs={12345: user})
        settings = {
            "repos": ["alice/repo"],
            "notify_chat_id": 12345,
            "pr_review": False,
            "issue_triage": False,
        }

        with (
            patch("kai.bot.sessions.resolve_github_settings", new_callable=AsyncMock, return_value=settings),
            patch("kai.bot.sessions.get_github_db_settings", new_callable=AsyncMock, return_value={}),
            patch("kai.bot.sessions.get_github_added_repos", new_callable=AsyncMock, return_value=[]),
            patch("kai.bot.sessions.get_setting", new_callable=AsyncMock, return_value=None),
        ):
            await _show_github(update, 12345, config)

        reply = update.message.reply_text.call_args[0][0]
        assert "alice/repo" in reply
        assert "(users.yaml)" in reply
        assert "GitHub token: not set" in reply

    @pytest.mark.asyncio
    async def test_added_repo_shows_source(self):
        """Repos added via /github add are labeled."""
        update = _make_update()
        config = _make_config()
        settings = {
            "repos": ["alice/repo"],
            "notify_chat_id": 12345,
            "pr_review": False,
            "issue_triage": False,
        }

        with (
            patch("kai.bot.sessions.resolve_github_settings", new_callable=AsyncMock, return_value=settings),
            patch("kai.bot.sessions.get_github_db_settings", new_callable=AsyncMock, return_value={}),
            patch("kai.bot.sessions.get_github_added_repos", new_callable=AsyncMock, return_value=["alice/repo"]),
            patch("kai.bot.sessions.get_setting", new_callable=AsyncMock, return_value="ghp_stored"),
        ):
            await _show_github(update, 12345, config)

        reply = update.message.reply_text.call_args[0][0]
        assert "alice/repo" in reply
        assert "(added via /github add)" in reply
        assert "GitHub token: stored" in reply

    @pytest.mark.asyncio
    async def test_mixed_repos_show_correct_sources(self):
        """Mixed yaml and DB-added repos each show their correct source."""
        update = _make_update()
        user = UserConfig(
            telegram_id=12345,
            name="alice",
            github_repos=["alice/yaml-repo"],
        )
        config = _make_config(user_configs={12345: user})
        settings = {
            "repos": ["alice/db-repo", "alice/yaml-repo"],
            "notify_chat_id": 12345,
            "pr_review": False,
            "issue_triage": False,
        }

        with (
            patch("kai.bot.sessions.resolve_github_settings", new_callable=AsyncMock, return_value=settings),
            patch("kai.bot.sessions.get_github_db_settings", new_callable=AsyncMock, return_value={}),
            patch("kai.bot.sessions.get_github_added_repos", new_callable=AsyncMock, return_value=["alice/db-repo"]),
            patch("kai.bot.sessions.get_setting", new_callable=AsyncMock, return_value=None),
        ):
            await _show_github(update, 12345, config)

        reply = update.message.reply_text.call_args[0][0]
        assert "alice/yaml-repo" in reply
        assert "(users.yaml)" in reply
        assert "alice/db-repo" in reply
        assert "(added via /github add)" in reply


# ── github_api module ──────────────────────────────────────────────


class TestGitHubAPI:
    """Tests for the low-level GitHub API client functions.

    All HTTP calls are mocked via aiohttp.ClientSession patches.
    """

    @pytest.mark.asyncio
    async def test_check_webhook_exists_found(self):
        """Finds a matching webhook by URL comparison."""
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(
            return_value=[
                {"id": 42, "config": {"url": "https://api.example.com/webhook/github"}},
                {"id": 99, "config": {"url": "https://other.com/hook"}},
            ]
        )

        with patch("kai.github_api.aiohttp.ClientSession") as mock_session_cls:
            session_instance = MagicMock()
            mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=session_instance)
            mock_session_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            session_instance.get.return_value.__aenter__ = AsyncMock(return_value=mock_resp)
            session_instance.get.return_value.__aexit__ = AsyncMock(return_value=False)

            exists, hook_id = await github_api.check_webhook_exists(
                "alice",
                "repo",
                "ghp_test",
                "https://api.example.com/webhook/github",
            )

        assert exists is True
        assert hook_id == 42

    @pytest.mark.asyncio
    async def test_check_webhook_exists_not_found(self):
        """Returns (False, None) when no matching webhook exists."""
        mock_resp = AsyncMock()
        mock_resp.status = 200
        # Empty list = no hooks at all
        mock_resp.json = AsyncMock(return_value=[])

        with patch("kai.github_api.aiohttp.ClientSession") as mock_session_cls:
            session_instance = MagicMock()
            mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=session_instance)
            mock_session_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            session_instance.get.return_value.__aenter__ = AsyncMock(return_value=mock_resp)
            session_instance.get.return_value.__aexit__ = AsyncMock(return_value=False)

            exists, hook_id = await github_api.check_webhook_exists(
                "alice",
                "repo",
                "ghp_test",
                "https://api.example.com/webhook/github",
            )

        assert exists is False
        assert hook_id is None

    @pytest.mark.asyncio
    async def test_check_webhook_exists_http_error(self):
        """Raises GitHubAPIError on non-200 response."""
        mock_resp = AsyncMock()
        mock_resp.status = 401
        mock_resp.text = AsyncMock(return_value="Unauthorized")

        with patch("kai.github_api.aiohttp.ClientSession") as mock_session_cls:
            session_instance = MagicMock()
            mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=session_instance)
            mock_session_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            session_instance.get.return_value.__aenter__ = AsyncMock(return_value=mock_resp)
            session_instance.get.return_value.__aexit__ = AsyncMock(return_value=False)

            with pytest.raises(GitHubAPIError) as exc_info:
                await github_api.check_webhook_exists(
                    "alice",
                    "repo",
                    "ghp_test",
                    "https://example.com/hook",
                )
            assert exc_info.value.status == 401

    @pytest.mark.asyncio
    async def test_register_webhook_success(self):
        """Returns hook ID on 201 Created."""
        mock_resp = AsyncMock()
        mock_resp.status = 201
        mock_resp.json = AsyncMock(return_value={"id": 55})

        with patch("kai.github_api.aiohttp.ClientSession") as mock_session_cls:
            session_instance = MagicMock()
            mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=session_instance)
            mock_session_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            session_instance.post.return_value.__aenter__ = AsyncMock(return_value=mock_resp)
            session_instance.post.return_value.__aexit__ = AsyncMock(return_value=False)

            hook_id = await github_api.register_webhook(
                "alice",
                "repo",
                "ghp_test",
                "https://api.example.com/webhook/github",
                "secret",
            )

        assert hook_id == 55

    @pytest.mark.asyncio
    async def test_register_webhook_403(self):
        """Raises GitHubAPIError(403) when user lacks admin access."""
        mock_resp = AsyncMock()
        mock_resp.status = 403
        mock_resp.text = AsyncMock(return_value="Forbidden")

        with patch("kai.github_api.aiohttp.ClientSession") as mock_session_cls:
            session_instance = MagicMock()
            mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=session_instance)
            mock_session_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            session_instance.post.return_value.__aenter__ = AsyncMock(return_value=mock_resp)
            session_instance.post.return_value.__aexit__ = AsyncMock(return_value=False)

            with pytest.raises(GitHubAPIError) as exc_info:
                await github_api.register_webhook(
                    "alice",
                    "repo",
                    "ghp_test",
                    "https://example.com/hook",
                    "secret",
                )
            assert exc_info.value.status == 403

    @pytest.mark.asyncio
    async def test_deregister_webhook_success(self):
        """204 No Content means successful deletion."""
        mock_resp = AsyncMock()
        mock_resp.status = 204

        with patch("kai.github_api.aiohttp.ClientSession") as mock_session_cls:
            session_instance = MagicMock()
            mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=session_instance)
            mock_session_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            session_instance.delete.return_value.__aenter__ = AsyncMock(return_value=mock_resp)
            session_instance.delete.return_value.__aexit__ = AsyncMock(return_value=False)

            # Should not raise
            await github_api.deregister_webhook("alice", "repo", 42, "ghp_test")

    @pytest.mark.asyncio
    async def test_deregister_webhook_404_is_success(self):
        """404 means already gone, treated as success."""
        mock_resp = AsyncMock()
        mock_resp.status = 404

        with patch("kai.github_api.aiohttp.ClientSession") as mock_session_cls:
            session_instance = MagicMock()
            mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=session_instance)
            mock_session_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            session_instance.delete.return_value.__aenter__ = AsyncMock(return_value=mock_resp)
            session_instance.delete.return_value.__aexit__ = AsyncMock(return_value=False)

            # Should not raise (404 = already gone = success)
            await github_api.deregister_webhook("alice", "repo", 42, "ghp_test")

    @pytest.mark.asyncio
    async def test_deregister_webhook_error(self):
        """Non-204/404 raises GitHubAPIError."""
        mock_resp = AsyncMock()
        mock_resp.status = 500
        mock_resp.text = AsyncMock(return_value="Server Error")

        with patch("kai.github_api.aiohttp.ClientSession") as mock_session_cls:
            session_instance = MagicMock()
            mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=session_instance)
            mock_session_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            session_instance.delete.return_value.__aenter__ = AsyncMock(return_value=mock_resp)
            session_instance.delete.return_value.__aexit__ = AsyncMock(return_value=False)

            with pytest.raises(GitHubAPIError) as exc_info:
                await github_api.deregister_webhook("alice", "repo", 42, "ghp_test")
            assert exc_info.value.status == 500

    @pytest.mark.asyncio
    async def test_network_error_raises(self):
        """aiohttp.ClientError is wrapped in GitHubAPIError(0)."""
        with patch("kai.github_api.aiohttp.ClientSession") as mock_session_cls:
            session_instance = MagicMock()
            mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=session_instance)
            mock_session_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            session_instance.get.return_value.__aenter__ = AsyncMock(
                side_effect=aiohttp.ClientError("connection failed"),
            )

            with pytest.raises(GitHubAPIError) as exc_info:
                await github_api.check_webhook_exists(
                    "alice",
                    "repo",
                    "ghp_test",
                    "https://example.com/hook",
                )
            assert exc_info.value.status == 0


# ── handle_github dispatcher integration ───────────────────────────


class TestHandleGithubDispatcher:
    """Tests for the updated handle_github dispatcher routing to the
    new token/add/remove subcommands."""

    @pytest.mark.asyncio
    async def test_token_routes_to_handler(self):
        """/github token routes to _handle_github_token."""
        update = _make_update(text="/github token ghp_test")
        config = _make_config()
        ctx = _make_context(config=config, args=["token", "ghp_test"])

        with patch("kai.bot._handle_github_token", new_callable=AsyncMock) as mock_handler:
            await handle_github(update, ctx)

        mock_handler.assert_called_once_with(update, 12345, ["ghp_test"])

    @pytest.mark.asyncio
    async def test_add_routes_to_handler(self):
        """/github add routes to _handle_github_add."""
        update = _make_update(text="/github add alice/repo")
        config = _make_config()
        ctx = _make_context(config=config, args=["add", "alice/repo"])

        with patch("kai.bot._handle_github_add", new_callable=AsyncMock) as mock_handler:
            await handle_github(update, ctx)

        mock_handler.assert_called_once_with(update, 12345, ["alice/repo"], config)

    @pytest.mark.asyncio
    async def test_remove_routes_to_handler(self):
        """/github remove routes to _handle_github_remove."""
        update = _make_update(text="/github remove alice/repo")
        config = _make_config()
        ctx = _make_context(config=config, args=["remove", "alice/repo"])

        with patch("kai.bot._handle_github_remove", new_callable=AsyncMock) as mock_handler:
            await handle_github(update, ctx)

        mock_handler.assert_called_once_with(update, 12345, ["alice/repo"], config)

    @pytest.mark.asyncio
    async def test_unknown_subcommand_lists_all(self):
        """/github bogus lists all valid subcommands including new ones."""
        update = _make_update(text="/github bogus")
        config = _make_config()
        ctx = _make_context(config=config, args=["bogus"])
        mock_sessions = AsyncMock()

        with patch("kai.bot.sessions", mock_sessions):
            await handle_github(update, ctx)

        reply = update.message.reply_text.call_args[0][0]
        assert "unknown subcommand" in reply.lower()
        # All subcommands should be listed
        for cmd in ("notify", "reviews", "triage", "add", "remove", "token"):
            assert cmd in reply.lower()
