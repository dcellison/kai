"""Tests for webhook.py pure functions and GitHub event formatters."""

import asyncio
import hashlib
import hmac
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web

from kai.config import UserConfig
from kai.internal_api_auth import InternalAPIAuth
from kai.telegram_http import TelegramWebhookIngress
from kai.webhook import (
    ALLOWED_WORKSPACES_KEY,
    CONFIG_KEY,
    GITHUB_WEBHOOK_SECRET_KEY,
    INTERNAL_API_AUTH_KEY,
    WORKSPACE_BASE_KEY,
    _fmt_issue_comment,
    _fmt_issues,
    _fmt_pull_request,
    _fmt_pull_request_review,
    _fmt_push,
    _handle_github,
    _resolve_local_repo,
    _verify_github_signature,
)
from kai.workshop.domain import AgentId, ChannelId, PrincipalId
from kai.workshop.integration_notifications import WorkshopIntegrationNotificationService
from kai.workshop.internal_api_contexts import WorkshopInternalAPIExecutionContext
from kai.workshop.run_previews import WorkshopRunPreviewRegistry
from kai.workshop.storage_namespaces import (
    WorkshopPrincipalStorageNamespace,
    WorkshopPrincipalStorageRegistry,
)
from kai.workshop.store import WorkshopEventStore
from tests.workshop_profiles import profile_id

# Telegram-specific fixtures own these values; the shared HTTP host does not.
ALLOWED_USER_IDS_KEY: web.AppKey[set[int]] = web.AppKey("test_allowed_user_ids", set)
CHAT_ID_KEY: web.AppKey[int] = web.AppKey("test_chat_id", int)
TELEGRAM_APP_KEY: web.AppKey[object] = web.AppKey("test_telegram_app", object)
TELEGRAM_BOT_KEY: web.AppKey[object] = web.AppKey("test_telegram_bot", object)


def _internal_api_auth() -> InternalAPIAuth:
    context = WorkshopInternalAPIExecutionContext(
        principal_id=PrincipalId("prn_" + "1" * 32),
        channel_id=ChannelId("chn_" + "1" * 32),
        agent_id=AgentId("agt_" + "1" * 32),
        runtime_profile_id=profile_id(111),
    )
    return InternalAPIAuth({context: "secret"})


def _principal_storage_registry() -> WorkshopPrincipalStorageRegistry:
    return WorkshopPrincipalStorageRegistry(
        (
            WorkshopPrincipalStorageNamespace(
                PrincipalId("prn_" + "1" * 32),
                profile_id(111),
                111,
            ),
        )
    )


# ── _verify_github_signature ─────────────────────────────────────────


class TestVerifyGithubSignature:
    def test_valid_signature(self):
        secret = "mysecret"
        body = b"test body content"
        digest = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        assert _verify_github_signature(secret, body, f"sha256={digest}") is True

    def test_wrong_signature(self):
        assert _verify_github_signature("secret", b"body", "sha256=wrong") is False

    def test_missing_prefix(self):
        secret = "mysecret"
        body = b"body"
        digest = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        assert _verify_github_signature(secret, body, digest) is False


# ── _fmt_push ────────────────────────────────────────────────────────


def _push_payload(num_commits=2, compare="https://github.com/o/r/compare/a...b"):
    return {
        "pusher": {"name": "alice"},
        "ref": "refs/heads/main",
        "commits": [{"id": f"sha{i:010d}", "message": f"Commit {i}"} for i in range(num_commits)],
        "repository": {"full_name": "owner/repo"},
        "compare": compare,
    }


class TestFmtPush:
    def test_basic_format(self):
        result = _fmt_push(_push_payload(2))
        assert "owner/repo" in result
        assert "main" in result
        assert "alice" in result
        assert "Commit 0" in result
        assert "Commit 1" in result

    def test_more_than_five_commits(self):
        result = _fmt_push(_push_payload(7))
        assert "... and 2 more" in result
        # Only first 5 commit messages shown
        assert "Commit 4" in result
        assert "Commit 5" not in result

    def test_includes_compare_url(self):
        result = _fmt_push(_push_payload(1, "https://github.com/o/r/compare/x...y"))
        assert "https://github.com/o/r/compare/x...y" in result


# ── _fmt_pull_request ────────────────────────────────────────────────


def _pr_payload(action="opened", merged=False):
    return {
        "action": action,
        "pull_request": {
            "title": "Add feature",
            "number": 42,
            "user": {"login": "bob"},
            "html_url": "https://github.com/o/r/pull/42",
            "merged": merged,
        },
        "repository": {"full_name": "owner/repo"},
    }


class TestFmtPullRequest:
    def test_opened(self):
        result = _fmt_pull_request(_pr_payload("opened"))
        assert "opened" in result
        assert "#42" in result
        assert "bob" in result

    def test_closed_not_merged(self):
        result = _fmt_pull_request(_pr_payload("closed", merged=False))
        assert "closed" in result
        assert "merged" not in result

    def test_closed_and_merged(self):
        result = _fmt_pull_request(_pr_payload("closed", merged=True))
        assert "merged" in result

    def test_reopened(self):
        result = _fmt_pull_request(_pr_payload("reopened"))
        assert "reopened" in result

    def test_other_action_returns_none(self):
        assert _fmt_pull_request(_pr_payload("edited")) is None


# ── _fmt_issues ──────────────────────────────────────────────────────


def _issue_payload(action="opened"):
    return {
        "action": action,
        "issue": {
            "title": "Bug report",
            "number": 7,
            "user": {"login": "carol"},
            "html_url": "https://github.com/o/r/issues/7",
        },
        "repository": {"full_name": "owner/repo"},
    }


class TestFmtIssues:
    def test_opened(self):
        result = _fmt_issues(_issue_payload("opened"))
        assert "opened" in result
        assert "#7" in result

    def test_closed(self):
        result = _fmt_issues(_issue_payload("closed"))
        assert "closed" in result

    def test_reopened(self):
        result = _fmt_issues(_issue_payload("reopened"))
        assert "reopened" in result

    def test_other_action_returns_none(self):
        assert _fmt_issues(_issue_payload("labeled")) is None


# ── _fmt_issue_comment ───────────────────────────────────────────────


def _comment_payload(action="created", body="Nice work!"):
    return {
        "action": action,
        "comment": {
            "body": body,
            "user": {"login": "dave"},
            "html_url": "https://github.com/o/r/issues/7#comment-1",
        },
        "issue": {"number": 7},
        "repository": {"full_name": "owner/repo"},
    }


class TestFmtIssueComment:
    def test_created(self):
        result = _fmt_issue_comment(_comment_payload())
        assert "dave" in result
        assert "Nice work!" in result
        assert "#7" in result

    def test_long_body_truncated(self):
        long_body = "x" * 300
        result = _fmt_issue_comment(_comment_payload(body=long_body))
        assert "..." in result
        # Body truncated to 200 chars + "..."
        assert "x" * 200 in result

    def test_other_action_returns_none(self):
        assert _fmt_issue_comment(_comment_payload("deleted")) is None


# ── _fmt_pull_request_review ─────────────────────────────────────────


def _review_payload(action="submitted", state="approved"):
    return {
        "action": action,
        "review": {
            "state": state,
            "user": {"login": "eve"},
            "html_url": "https://github.com/o/r/pull/10#review-1",
        },
        "pull_request": {"number": 10},
        "repository": {"full_name": "owner/repo"},
    }


class TestFmtPullRequestReview:
    def test_approved(self):
        result = _fmt_pull_request_review(_review_payload("submitted", "approved"))
        assert "eve" in result
        assert "approved" in result
        assert "#10" in result

    def test_changes_requested(self):
        result = _fmt_pull_request_review(_review_payload("submitted", "changes_requested"))
        assert "requested changes on" in result

    def test_other_state_returns_none(self):
        assert _fmt_pull_request_review(_review_payload("submitted", "dismissed")) is None

    def test_non_submitted_action_returns_none(self):
        assert _fmt_pull_request_review(_review_payload("edited", "approved")) is None


# ── _should_skip_review / _record_review ────────────────────────────


# ── _prune_expired / cooldown dict cleanup ───────────────────────────


# ── PR review routing (integration tests) ──────────────────────────


# Shared secret used to sign GitHub webhook payloads in tests
_TEST_SECRET = "test-webhook-secret"


def _sign_payload(payload: dict) -> str:
    """Compute HMAC-SHA256 signature for a GitHub webhook payload."""
    body = json.dumps(payload).encode()
    digest = hmac.new(_TEST_SECRET.encode(), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def _make_pr_payload(action: str, pr_number: int = 42, merged: bool = False) -> dict:
    """Build a minimal pull_request webhook payload."""
    return {
        "action": action,
        "pull_request": {
            "title": "Test PR",
            "number": pr_number,
            "user": {"login": "testuser"},
            "html_url": f"https://github.com/owner/repo/pull/{pr_number}",
            "merged": merged,
        },
        "repository": {"full_name": "owner/repo"},
    }


@pytest.fixture(autouse=True)
def _default_github_token_lookup():
    """Webhook tests default to no stored per-user GitHub token."""
    with patch("kai.webhook.sessions.get_setting", new_callable=AsyncMock, return_value=None):
        yield


def _build_test_app(
    cooldown: int = 300,
    config: object | None = None,
) -> web.Application:
    """Build a minimal aiohttp app with _handle_github wired up.

    The config parameter controls per-user routing. When None, the legacy
    fallback-admin fixture is retained, but its synchronous user lookup
    returns an explicit ``owner/repo`` authorization so review/triage tests
    exercise the permitted path. To test other routing, pass a mock with
    custom user_configs.

    Feature flags (pr_review, issue_triage, notify_chat_id) are now resolved
    per-user via resolve_github_settings() instead of app dict globals.
    Tests should mock sessions.resolve_github_settings to control these.
    """
    app = web.Application()
    app[GITHUB_WEBHOOK_SECRET_KEY] = _TEST_SECRET
    app[INTERNAL_API_AUTH_KEY] = InternalAPIAuth()
    # Review subprocess resource limits (defaults match config.py).
    # Config needed by review background tasks
    # Workspace config for review agent repo resolution. Tests that need
    # _resolve_local_repo() to return a specific path populate workspace_base
    # or allowed_workspaces; the default is empty so _mock_resolve_repo is
    # used for routing tests that don't care about the resolution result.
    app[WORKSPACE_BASE_KEY] = None
    app[ALLOWED_WORKSPACES_KEY] = []
    # Mock bot that records sent messages
    mock_bot = AsyncMock()
    app[TELEGRAM_BOT_KEY] = mock_bot
    app[CHAT_ID_KEY] = 12345
    # Config for per-user routing. The default deliberately keeps an empty
    # routing map so these older tests exercise the fallback path without a DB,
    # while get_user_config returns the fallback admin's explicit operations
    # authorization.
    if config is None:
        # Config is a sync dataclass. Using AsyncMock here would make
        # every chained attribute call (e.g. config.default_models.get(...))
        # return a coroutine, which then leaks as the
        # AsyncMockMixin._execute_mock_call never-awaited warning.
        #
        # The dataclass-shaped attribute defaults below let
        # resolve_user_model() return a real string (the registry
        # default for the selected backend) rather than a chained
        # MagicMock. Without them, model_override would silently flow
        # through to review_pr / triage_issue as a MagicMock instance,
        # which the mocked downstream callers accept but real code
        # would not.
        mock_config = MagicMock()
        default_user = UserConfig(
            telegram_id=12345,
            name="test-admin",
            role="admin",
            github_repos=["owner/repo"],
        )
        mock_config.user_configs = {}
        mock_config.default_models = {}
        mock_config.default_backend = "claude"
        mock_config.default_provider = ""
        mock_config.protected_install = False
        # get_user_config is synchronous in real Config; it must not return a
        # coroutine or authorization could be tested against a mock object.
        mock_config.get_user_config = lambda uid: default_user if uid == 12345 else None
        app[CONFIG_KEY] = mock_config
    else:
        app[CONFIG_KEY] = config
    app.router.add_post("/webhook/github", _handle_github)
    return app


class TestResolveLocalRepo:
    """Tests for workspace-aware repo resolution."""

    @pytest.mark.asyncio
    async def test_workspace_base(self, tmp_path):
        """Resolves via WORKSPACE_BASE when a child dir matches repo name."""
        # Create ~/Projects/anvil/ structure
        anvil_dir = tmp_path / "anvil"
        anvil_dir.mkdir()

        app = web.Application()
        app[WORKSPACE_BASE_KEY] = str(tmp_path)
        app[ALLOWED_WORKSPACES_KEY] = []

        result = await _resolve_local_repo("dcellison/anvil", app)
        assert result == str(anvil_dir)

    @pytest.mark.asyncio
    async def test_allowed_workspaces(self, tmp_path):
        """Resolves via ALLOWED_WORKSPACES when dir name matches."""
        myrepo = tmp_path / "myrepo"
        myrepo.mkdir()

        app = web.Application()
        app[WORKSPACE_BASE_KEY] = None
        app[ALLOWED_WORKSPACES_KEY] = [str(myrepo)]

        result = await _resolve_local_repo("owner/myrepo", app)
        assert result == str(myrepo)

    @pytest.mark.asyncio
    async def test_workspace_history(self, tmp_path):
        """Resolves via workspace_history entries from the database."""
        history_repo = tmp_path / "historic"
        history_repo.mkdir()

        app = web.Application()
        app[WORKSPACE_BASE_KEY] = None
        app[ALLOWED_WORKSPACES_KEY] = []

        with patch(
            "kai.webhook.sessions.get_all_workspace_paths",
            new_callable=AsyncMock,
            return_value=[str(history_repo)],
        ):
            result = await _resolve_local_repo("owner/historic", app)
        assert result == str(history_repo)

    @pytest.mark.asyncio
    async def test_no_match(self, tmp_path):
        """Returns None when no workspace matches the repo."""
        app = web.Application()
        app[WORKSPACE_BASE_KEY] = str(tmp_path)
        app[ALLOWED_WORKSPACES_KEY] = []

        with patch(
            "kai.webhook.sessions.get_all_workspace_paths",
            new_callable=AsyncMock,
            return_value=[],
        ):
            result = await _resolve_local_repo("owner/nonexistent", app)
        assert result is None

    @pytest.mark.asyncio
    async def test_nonexistent_dir_skipped(self, tmp_path):
        """History entries pointing to deleted directories are skipped."""
        app = web.Application()
        app[WORKSPACE_BASE_KEY] = None
        app[ALLOWED_WORKSPACES_KEY] = []

        with patch(
            "kai.webhook.sessions.get_all_workspace_paths",
            new_callable=AsyncMock,
            return_value=["/gone/deleted-repo"],
        ):
            result = await _resolve_local_repo("owner/deleted-repo", app)
        assert result is None

    @pytest.mark.asyncio
    async def test_history_searches_all_users(self, tmp_path):
        """Workspace history resolution finds repos from any user, not just one."""
        other_user_repo = tmp_path / "other_user_project"
        other_user_repo.mkdir()

        app = web.Application()
        app[WORKSPACE_BASE_KEY] = None
        app[ALLOWED_WORKSPACES_KEY] = []

        with patch(
            "kai.webhook.sessions.get_all_workspace_paths",
            new_callable=AsyncMock,
            return_value=[str(other_user_repo)],
        ):
            result = await _resolve_local_repo("owner/other_user_project", app)
        assert result == str(other_user_repo)


# ── Issue triage routing ─────────────────────────────────────────────


def _make_issue_payload(action: str = "opened", issue_number: int = 10) -> dict:
    """Build a minimal issues webhook payload."""
    return {
        "action": action,
        "issue": {
            "number": issue_number,
            "title": "Test issue",
            "body": "Test body",
            "user": {"login": "testuser"},
            "html_url": f"https://github.com/owner/repo/issues/{issue_number}",
            "labels": [],
        },
        "repository": {"full_name": "owner/repo"},
    }


class TestWebhookHealthMonitor:
    """Tests for consecutive failure tracking and admin notification."""

    @pytest.mark.asyncio
    async def test_failure_counter_increments(self):
        """Consecutive failures increment on exception."""
        bot = AsyncMock()
        bot.get_webhook_info = AsyncMock(side_effect=RuntimeError("API down"))

        call_count = 0

        async def mock_sleep(_duration):
            nonlocal call_count
            call_count += 1
            if call_count > 2:
                raise asyncio.CancelledError

        ingress = object.__new__(TelegramWebhookIngress)
        ingress._bot = bot
        ingress._webhook_url = "https://example.com/webhook"
        ingress._webhook_secret = "secret"
        ingress._notification_chat_id = 12345
        with patch("kai.telegram_http.asyncio.sleep", side_effect=mock_sleep):
            try:
                await ingress._webhook_health_loop()
            except asyncio.CancelledError:
                pass

        # Two failed checks (first sleep skips initial check, then two iterations)
        # Bot should not have been asked to send a notification (threshold is 3)
        bot.send_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_notification_after_three_failures(self):
        """Admin is notified after 3 consecutive failures."""
        bot = AsyncMock()
        bot.get_webhook_info = AsyncMock(side_effect=RuntimeError("API down"))
        bot.send_message = AsyncMock()

        call_count = 0

        async def mock_sleep(_duration):
            nonlocal call_count
            call_count += 1
            # First sleep is the initial skip, then 4 more iterations
            if call_count > 4:
                raise asyncio.CancelledError

        ingress = object.__new__(TelegramWebhookIngress)
        ingress._bot = bot
        ingress._webhook_url = "https://example.com/webhook"
        ingress._webhook_secret = "secret"
        ingress._notification_chat_id = 12345
        with patch("kai.telegram_http.asyncio.sleep", side_effect=mock_sleep):
            try:
                await ingress._webhook_health_loop()
            except asyncio.CancelledError:
                pass

        # Notification sent exactly once after 3 failures
        bot.send_message.assert_called_once()
        args = bot.send_message.call_args
        assert args[0][0] == 12345
        assert "3 consecutive" in args[0][1]

    @pytest.mark.asyncio
    async def test_counter_resets_on_success(self):
        """Successful check resets failure counter and notification flag."""
        bot = AsyncMock()
        # First two calls fail, third succeeds
        mock_info = AsyncMock()
        mock_info.url = "https://example.com/webhook"
        mock_info.last_error_date = None
        mock_info.pending_update_count = 0
        bot.get_webhook_info = AsyncMock(side_effect=[RuntimeError("fail"), RuntimeError("fail"), mock_info])

        call_count = 0

        async def mock_sleep(_duration):
            nonlocal call_count
            call_count += 1
            if call_count > 3:
                raise asyncio.CancelledError

        ingress = object.__new__(TelegramWebhookIngress)
        ingress._bot = bot
        ingress._webhook_url = "https://example.com/webhook"
        ingress._webhook_secret = "secret"
        ingress._notification_chat_id = 12345
        with patch("kai.telegram_http.asyncio.sleep", side_effect=mock_sleep):
            try:
                await ingress._webhook_health_loop()
            except asyncio.CancelledError:
                pass

        # No notification (only 2 consecutive failures, then recovery)
        bot.send_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_failed_notification_does_not_crash(self):
        """If the notification itself fails, the loop continues."""
        bot = AsyncMock()
        bot.get_webhook_info = AsyncMock(side_effect=RuntimeError("API down"))
        bot.send_message = AsyncMock(side_effect=RuntimeError("Telegram unreachable"))

        call_count = 0

        async def mock_sleep(_duration):
            nonlocal call_count
            call_count += 1
            if call_count > 4:
                raise asyncio.CancelledError

        ingress = object.__new__(TelegramWebhookIngress)
        ingress._bot = bot
        ingress._webhook_url = "https://example.com/webhook"
        ingress._webhook_secret = "secret"
        ingress._notification_chat_id = 12345
        with patch("kai.telegram_http.asyncio.sleep", side_effect=mock_sleep):
            try:
                await ingress._webhook_health_loop()
            except asyncio.CancelledError:
                pass

        # send_message was attempted exactly once (not retried after failure)
        bot.send_message.assert_called_once()


# ── GitHub notification group routing ────────────────────────────────


class TestNotificationChatIdMutations:
    """Tests for the live notification-destination registry.

    These functions are called by bot.py when /github notify modifies
    a notification destination. The registry must remain detached from
    Config.allowed_user_ids so outbound changes cannot authorize senders.
    """

    @pytest.mark.asyncio
    async def test_start_keeps_notification_destinations_out_of_inbound_principals(self):
        """Configured and DB notification destinations do not become authorized users."""
        import kai.webhook as wh

        admin = UserConfig(
            telegram_id=111,
            name="alice",
            role="admin",
            github_notify_chat_id=-100111,
        )
        config = MagicMock()
        config.user_configs = {111: admin}
        config.allowed_user_ids = {111}
        config.get_admins.return_value = [admin]
        config.telegram_webhook_url = None
        config.telegram_webhook_secret = None
        config.github_webhook_secret = None
        config.generic_webhook_secret = None
        config.pr_review_cooldown = 0
        config.pr_review_timeout_s = 0
        config.webhook_port = 0
        config.workshop_lan_host = ""
        config.workspace_base = None
        config.allowed_workspaces = []
        config.spec_dir = "specs"

        private_execution = MagicMock()
        private_execution.recoverable_client_runs = AsyncMock(return_value=())
        core_services = SimpleNamespace(
            subprocess_pool=MagicMock(internal_api_auth=_internal_api_auth()),
            principal_storage=_principal_storage_registry(),
            client_store=MagicMock(),
            client_commands=MagicMock(),
            run_previews=WorkshopRunPreviewRegistry(),
            artifacts=MagicMock(),
            settings_workspaces=MagicMock(),
            memory_queries=MagicMock(),
            preference_documents=MagicMock(),
            github_automation=MagicMock(),
        )
        core_host = MagicMock()
        integration_notifications = MagicMock(spec=WorkshopIntegrationNotificationService)

        fake_runner = MagicMock()
        fake_runner.setup = AsyncMock()
        fake_runner.cleanup = AsyncMock()
        fake_site = MagicMock()
        fake_site.start = AsyncMock()

        old_app = wh._app
        old_runner = wh._runner
        try:
            with (
                patch("kai.webhook.sessions.get_setting", new_callable=AsyncMock, return_value="-100222"),
                patch("kai.webhook._register_workshop_client_api", new_callable=AsyncMock),
                patch("kai.webhook.web.AppRunner", return_value=fake_runner),
                patch("kai.webhook.web.TCPSite", return_value=fake_site),
            ):
                await wh.start(
                    config,
                    core_host=core_host,
                    core_services=core_services,
                    integration_notifications=integration_notifications,
                )

            assert ALLOWED_USER_IDS_KEY not in wh._app
        finally:
            await wh.stop()
            wh._app = old_app
            wh._runner = old_runner

    @pytest.mark.asyncio
    async def test_start_workshop_only_constructs_no_telegram_surface(
        self,
        tmp_path,
        monkeypatch,
    ):
        import kai.webhook as wh

        config = MagicMock()
        config.user_configs = {}
        config.allowed_user_ids = set()
        config.get_admins.return_value = []
        config.telegram_webhook_url = None
        config.telegram_webhook_secret = None
        config.github_webhook_secret = "configured-but-telegram-owned"
        config.generic_webhook_secret = "configured-but-telegram-owned"
        config.pr_review_cooldown = 0
        config.pr_review_timeout_s = 0
        config.webhook_port = 8080
        config.workshop_lan_host = ""
        config.workspace_base = None
        config.allowed_workspaces = []
        config.spec_dir = "specs"
        config.session_db_path = tmp_path / "kai.db"

        store = await WorkshopEventStore.open(config.session_db_path)
        core_services = SimpleNamespace(
            subprocess_pool=MagicMock(internal_api_auth=_internal_api_auth()),
            principal_storage=_principal_storage_registry(),
            client_store=store,
            client_commands=MagicMock(),
            run_previews=WorkshopRunPreviewRegistry(),
            artifacts=MagicMock(),
            settings_workspaces=MagicMock(),
            memory_queries=MagicMock(),
            preference_documents=MagicMock(),
            github_automation=MagicMock(),
        )
        fake_runner = MagicMock()
        fake_runner.setup = AsyncMock()
        fake_runner.cleanup = AsyncMock()
        fake_site = MagicMock()
        fake_site.start = AsyncMock()

        monkeypatch.setattr("kai.webhook.web.AppRunner", MagicMock(return_value=fake_runner))
        monkeypatch.setattr("kai.webhook.web.TCPSite", MagicMock(return_value=fake_site))

        await wh.start(
            config,
            core_host=MagicMock(),
            core_services=core_services,
            integration_notifications=MagicMock(spec=WorkshopIntegrationNotificationService),
            workshop_enabled=True,
        )
        try:
            assert TELEGRAM_APP_KEY not in wh._app
            assert TELEGRAM_BOT_KEY not in wh._app
            paths = {resource.canonical for resource in wh._app.router.resources()}
            assert "/workshop/" in paths
            assert "/webhook/telegram" not in paths
            assert "/webhook/github" in paths
            assert "/webhook" in paths
        finally:
            await wh.stop()
            await store.close()

    @pytest.mark.asyncio
    async def test_start_lan_listener_exposes_only_workshop_client_routes(
        self,
        tmp_path,
        monkeypatch,
    ):
        """Opt-in LAN access must not publish webhook or internal API routes."""
        import kai.webhook as wh

        admin = UserConfig(telegram_id=111, name="alice", role="admin")
        config = MagicMock()
        config.user_configs = {111: admin}
        config.allowed_user_ids = {111}
        config.get_admins.return_value = [admin]
        config.telegram_webhook_url = None
        config.telegram_webhook_secret = None
        config.github_webhook_secret = None
        config.generic_webhook_secret = None
        config.pr_review_cooldown = 0
        config.pr_review_timeout_s = 0
        config.webhook_port = 8080
        config.workshop_lan_host = "10.0.0.36"
        config.workspace_base = None
        config.allowed_workspaces = []
        config.spec_dir = "specs"
        config.session_db_path = tmp_path / "kai.db"

        private_execution = MagicMock()
        private_execution.recoverable_client_runs = AsyncMock(return_value=())
        core_store = await WorkshopEventStore.open(config.session_db_path)
        core_services = SimpleNamespace(
            subprocess_pool=MagicMock(internal_api_auth=_internal_api_auth()),
            principal_storage=_principal_storage_registry(),
            client_store=core_store,
            client_commands=MagicMock(),
            run_previews=WorkshopRunPreviewRegistry(),
            artifacts=MagicMock(),
            settings_workspaces=MagicMock(),
            memory_queries=MagicMock(),
            preference_documents=MagicMock(),
            github_automation=MagicMock(),
        )
        core_host = MagicMock()
        integration_notifications = MagicMock(spec=WorkshopIntegrationNotificationService)

        apps: list[web.Application] = []
        runner_shutdown_timeouts: list[float] = []
        runners: list[MagicMock] = []
        sites: list[tuple[MagicMock, str, int]] = []

        def fake_runner(app, *, access_log=None, shutdown_timeout):
            apps.append(app)
            runner_shutdown_timeouts.append(shutdown_timeout)
            runner = MagicMock()
            runner.setup = AsyncMock()
            runner.cleanup = AsyncMock()
            runners.append(runner)
            return runner

        def fake_site(runner, host, port):
            sites.append((runner, host, port))
            site = MagicMock()
            site.start = AsyncMock()
            return site

        monkeypatch.setattr("kai.webhook.web.AppRunner", fake_runner)
        monkeypatch.setattr("kai.webhook.web.TCPSite", fake_site)
        monkeypatch.setattr("kai.webhook.sessions.get_setting", AsyncMock(return_value=None))

        await wh.start(
            config,
            core_host=core_host,
            core_services=core_services,
            integration_notifications=integration_notifications,
        )
        try:
            assert [(host, port) for _, host, port in sites] == [
                ("127.0.0.1", 8080),
                ("10.0.0.36", 8080),
            ]
            assert len(apps) == 2
            assert runner_shutdown_timeouts == [
                wh._HTTP_RUNNER_SHUTDOWN_TIMEOUT,
                wh._HTTP_RUNNER_SHUTDOWN_TIMEOUT,
            ]
            lan_paths = {resource.canonical for resource in apps[1].router.resources()}
            assert lan_paths == {
                "/v1/client/enrollment/redeem",
                "/v1/client/navigation",
                "/v1/client/agents",
                "/v1/client/agents/events",
                "/v1/client/agents/{definition_id}",
                "/v1/client/agents/{definition_id}/revisions",
                "/v1/client/agents/{definition_id}/activate",
                "/v1/client/agents/{definition_id}/archive",
                "/v1/channels",
                "/v1/channels/{channel_id}/archive",
                "/v1/channels/{channel_id}/restore",
                "/v1/channels/{channel_id}/timeline",
                "/v1/channels/{channel_id}/threads/{root_message_id}",
                "/v1/channels/{channel_id}/events",
                "/v1/channels/{channel_id}/messages/{message_id}/reactions",
                "/v1/channels/{channel_id}/artifacts/{artifact_id}/content",
                "/v1/channels/{channel_id}/artifacts/{artifact_id}/download",
                "/v1/channels/{channel_id}/commands",
                "/v1/channels/{channel_id}/agents/{agent_id}/attach",
                "/v1/channels/{channel_id}/agents/{agent_id}/detach",
                "/v1/channels/{channel_id}/agents/{agent_id}/dismiss",
                "/v1/channels/{channel_id}/runs/{run_id}",
                "/v1/channels/{channel_id}/runs/{run_id}/cancel",
                "/v1/channels/{channel_id}/runs/{run_id}/trace",
                "/v1/channels/{channel_id}/settings",
                "/v1/channels/{channel_id}/models",
                "/v1/channels/{channel_id}/workspace",
                "/v1/channels/{channel_id}/workspace-config",
                "/v1/settings/model-catalogue/refresh-all",
                "/v1/preferences",
                "/v1/preferences/revisions",
                "/v1/preferences/revisions/{preference_revision}/restore",
                "/v1/memory/stats",
                "/v1/memory/records",
                "/v1/memory/search",
                "/v1/memory/records/{memory_id}",
                "/v1/memory/records/{memory_id}/source",
                "/v1/memory/records/{memory_id}/scope",
                "/v1/memory/actions/scope",
                "/v1/memory/actions/delete",
                "/workshop",
                "/workshop/",
                "/workshop/app.css",
                "/workshop/app.js",
            }
            assert not any(path.startswith("/api/") for path in lan_paths)
            assert not any(path.startswith("/webhook") for path in lan_paths)
        finally:
            await wh.stop()
            await core_store.close()
