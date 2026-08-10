"""Tests for webhook.py pure functions and GitHub event formatters."""

import asyncio
import dataclasses
import hashlib
import hmac
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from kai.config import UserConfig
from kai.internal_api_auth import InternalAPIAuth, InternalAPIScope
from kai.webhook import (
    ALLOWED_USER_IDS_KEY,
    ALLOWED_WORKSPACES_KEY,
    CHAT_ID_KEY,
    CONFIG_KEY,
    GITHUB_WEBHOOK_SECRET_KEY,
    INTERNAL_API_AUTH_KEY,
    NOTIFICATION_CHAT_IDS_KEY,
    PR_REVIEW_COOLDOWN_KEY,
    PR_REVIEW_TIMEOUT_S_KEY,
    SPEC_DIR_KEY,
    TELEGRAM_BOT_KEY,
    WEBHOOK_PORT_KEY,
    WORKSPACE_BASE_KEY,
    _fmt_issue_comment,
    _fmt_issues,
    _fmt_pull_request,
    _fmt_pull_request_review,
    _fmt_push,
    _get_subscribed_users,
    _handle_github,
    _prune_expired,
    _record_review,
    _resolve_local_repo,
    _review_cooldowns,
    _should_skip_review,
    _strip_markdown,
    _triage_cooldowns,
    _verify_github_signature,
    _webhook_health_loop,
    add_notification_chat_id,
    remove_notification_chat_id,
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


# ── _strip_markdown ──────────────────────────────────────────────────


class TestStripMarkdown:
    def test_converts_links(self):
        assert _strip_markdown("[click](https://example.com)") == "click (https://example.com)"

    def test_removes_bold(self):
        assert _strip_markdown("**bold text**") == "bold text"

    def test_removes_backticks(self):
        assert _strip_markdown("`inline code`") == "inline code"

    def test_removes_italic_preserves_snake_case(self):
        result = _strip_markdown("_italic_ and snake_case")
        assert result == "italic and snake_case"

    def test_combined(self):
        text = "**Push** to `main` by [alice](https://github.com/alice)"
        result = _strip_markdown(text)
        assert "**" not in result
        assert "`" not in result
        assert "alice (https://github.com/alice)" in result


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


class TestReviewCooldown:
    def setup_method(self):
        """Clear the cooldown dict before each test."""
        _review_cooldowns.clear()

    def test_first_review_not_skipped(self):
        """A PR that has never been reviewed should not be skipped."""
        assert _should_skip_review("owner/repo", 1, 300) is False

    def test_recent_review_skipped(self):
        """A PR reviewed within the cooldown window should be skipped."""
        _record_review("owner/repo", 1, 300)
        assert _should_skip_review("owner/repo", 1, 300) is True

    def test_different_pr_not_skipped(self):
        """Cooldown is per-PR, so a different PR number is not skipped."""
        _record_review("owner/repo", 1, 300)
        assert _should_skip_review("owner/repo", 2, 300) is False

    def test_different_repo_not_skipped(self):
        """Cooldown is per-repo+PR, so a different repo is not skipped."""
        _record_review("owner/repo", 1, 300)
        assert _should_skip_review("other/repo", 1, 300) is False

    def test_expired_cooldown_not_skipped(self):
        """After cooldown expires, the PR can be reviewed again."""
        from unittest.mock import patch

        _record_review("owner/repo", 1, 300)
        # Advance time past the cooldown
        import time

        future = time.time() + 301
        with patch("kai.webhook.time.time", return_value=future):
            assert _should_skip_review("owner/repo", 1, 300) is False


# ── _prune_expired / cooldown dict cleanup ───────────────────────────


class TestPruneExpired:
    def test_removes_old_entries(self):
        """Entries older than max_age are removed."""
        import time

        cooldowns: dict[tuple[str, int], float] = {
            ("repo", 1): time.time() - 500,
            ("repo", 2): time.time() - 100,
        }
        _prune_expired(cooldowns, 300)
        assert ("repo", 1) not in cooldowns
        assert ("repo", 2) in cooldowns

    def test_retains_recent_entries(self):
        """Entries newer than max_age are kept."""
        import time

        cooldowns: dict[tuple[str, int], float] = {
            ("repo", 1): time.time(),
            ("repo", 2): time.time() - 10,
        }
        _prune_expired(cooldowns, 300)
        assert len(cooldowns) == 2

    def test_empty_dict_is_noop(self):
        """Pruning an empty dict does nothing."""
        cooldowns: dict[tuple[str, int], float] = {}
        _prune_expired(cooldowns, 300)
        assert len(cooldowns) == 0

    def test_record_review_prunes_before_adding(self):
        """_record_review prunes expired entries before adding a new one."""
        import time

        _review_cooldowns.clear()
        # Add an old entry directly
        _review_cooldowns[("repo", 1)] = time.time() - 500

        # Record a new review - should prune the old entry first
        _record_review("repo", 2, 300)

        assert ("repo", 1) not in _review_cooldowns
        assert ("repo", 2) in _review_cooldowns

    def test_record_triage_prunes_before_adding(self):
        """_record_triage prunes expired entries before adding a new one."""
        import time

        from kai.webhook import _TRIAGE_COOLDOWN_SECONDS, _record_triage

        _triage_cooldowns.clear()
        # Add an entry past the cooldown threshold
        _triage_cooldowns[("repo", 10)] = time.time() - (_TRIAGE_COOLDOWN_SECONDS + 60)

        # Record a new triage - should prune the old entry first
        _record_triage("repo", 20)

        assert ("repo", 10) not in _triage_cooldowns
        assert ("repo", 20) in _triage_cooldowns


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
    app[PR_REVIEW_COOLDOWN_KEY] = cooldown
    # Review subprocess resource limits (defaults match config.py).
    app[PR_REVIEW_TIMEOUT_S_KEY] = 900
    # Config needed by review background tasks
    app[WEBHOOK_PORT_KEY] = 8080
    # Workspace config for review agent repo resolution. Tests that need
    # _resolve_local_repo() to return a specific path populate workspace_base
    # or allowed_workspaces; the default is empty so _mock_resolve_repo is
    # used for routing tests that don't care about the resolution result.
    app[WORKSPACE_BASE_KEY] = None
    app[ALLOWED_WORKSPACES_KEY] = []
    app[SPEC_DIR_KEY] = "specs"
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
        # get_user_config is synchronous in real Config; it must not return a
        # coroutine or authorization could be tested against a mock object.
        mock_config.get_user_config = lambda uid: default_user if uid == 12345 else None
        app[CONFIG_KEY] = mock_config
    else:
        app[CONFIG_KEY] = config
    app.router.add_post("/webhook/github", _handle_github)
    return app


@pytest.fixture
def _clear_cooldowns():
    """Clear review and triage cooldown dicts before each routing test."""
    _review_cooldowns.clear()
    _triage_cooldowns.clear()
    yield
    _review_cooldowns.clear()
    _triage_cooldowns.clear()


@pytest.fixture(autouse=False)
def _mock_resolve_repo():
    """Mock _resolve_local_repo so routing tests skip filesystem/DB checks."""
    with patch("kai.webhook._resolve_local_repo", new_callable=AsyncMock, return_value=None):
        yield


def _mock_settings(
    pr_review: bool = False,
    issue_triage: bool = False,
    notify_chat_id: int = 12345,
    repos: list | None = None,
):
    """Return a context manager that mocks resolve_github_settings.

    Provides a consistent way to control per-user feature flags in
    routing tests. The returned settings dict matches the GitHubSettings
    TypedDict shape from sessions.py.
    """
    settings = {
        "repos": repos or [],
        "notify_chat_id": notify_chat_id,
        "pr_review": pr_review,
        "issue_triage": issue_triage,
    }
    return patch(
        "kai.webhook.sessions.resolve_github_settings",
        new_callable=AsyncMock,
        return_value=settings,
    )


@pytest.fixture
def _mock_review_and_triage():
    """Mock review.review_pr and triage.triage_issue at module scope.

    The webhook routing tests trigger background-task creation
    (`asyncio.create_task(review.review_pr(...))`). Without the mock,
    the real review_pr spawns a `gh pr diff` subprocess via
    `fetch_pr_diff`. The test exits before that background task
    completes, the event loop closes, the subprocess transport never
    sees a clean close, and the OS-level pipes leak as
    ResourceWarning chatter (`unclosed transport`, `unclosed file`).
    Mocking both functions keeps the routing path fully simulated and
    avoids any real subprocess spawn.
    """
    with (
        patch("kai.webhook.review.review_pr", new_callable=AsyncMock) as mock_review,
        patch("kai.webhook.triage.triage_issue", new_callable=AsyncMock) as mock_triage,
    ):
        yield mock_review, mock_triage


class TestPRReviewRouting:
    """Integration tests for PR review routing in _handle_github.

    These tests use _mock_settings() to control per-user feature flags
    via resolve_github_settings(). The default _build_test_app() has no
    user_configs, so all events hit the fallback path (admin chat_id)
    and the mocked settings control whether review/triage triggers.
    """

    @pytest.fixture(autouse=True)
    def _stub_review_and_triage(self, _mock_review_and_triage):
        """Default routing tests do not need real review_pr / triage_issue."""
        yield

    @pytest.mark.asyncio
    async def test_routes_opened_when_enabled(self, _clear_cooldowns, _mock_resolve_repo):
        """Reviewable PR events are routed to review pipeline, not Telegram."""
        app = _build_test_app()
        payload = _make_pr_payload("opened")
        body = json.dumps(payload).encode()
        sig = _sign_payload(payload)

        with _mock_settings(pr_review=True):
            async with TestClient(TestServer(app)) as client:
                resp = await client.post(
                    "/webhook/github",
                    data=body,
                    headers={
                        "X-GitHub-Event": "pull_request",
                        "X-Hub-Signature-256": sig,
                    },
                )
                data = await resp.json()
                assert resp.status == 200
                assert data["status"] == "ok"
                # Should NOT have sent a Telegram notification (review task fires instead)
                app[TELEGRAM_BOT_KEY].send_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_falls_through_when_disabled(self, _clear_cooldowns):
        """With PR review disabled, opened events go to the notification formatter."""
        app = _build_test_app()
        payload = _make_pr_payload("opened")
        body = json.dumps(payload).encode()
        sig = _sign_payload(payload)

        with _mock_settings(pr_review=False):
            async with TestClient(TestServer(app)) as client:
                resp = await client.post(
                    "/webhook/github",
                    data=body,
                    headers={
                        "X-GitHub-Event": "pull_request",
                        "X-Hub-Signature-256": sig,
                    },
                )
                data = await resp.json()
                assert resp.status == 200
                # Falls through to _fmt_pull_request, which formats a notification
                assert data.get("status") == "ok"
                app[TELEGRAM_BOT_KEY].send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_cooldown_skips_recent(self, _clear_cooldowns, _mock_resolve_repo):
        """Second event for the same PR within cooldown is silently skipped."""
        app = _build_test_app(cooldown=300)
        payload = _make_pr_payload("synchronize", pr_number=10)
        body = json.dumps(payload).encode()
        sig = _sign_payload(payload)

        with _mock_settings(pr_review=True):
            async with TestClient(TestServer(app)) as client:
                # First request triggers a review
                resp1 = await client.post(
                    "/webhook/github",
                    data=body,
                    headers={
                        "X-GitHub-Event": "pull_request",
                        "X-Hub-Signature-256": sig,
                    },
                )
                data1 = await resp1.json()
                assert data1["status"] == "ok"

                # Second request hits cooldown - still returns ok since
                # _process_github_event_for_user returns None (no response)
                resp2 = await client.post(
                    "/webhook/github",
                    data=body,
                    headers={
                        "X-GitHub-Event": "pull_request",
                        "X-Hub-Signature-256": sig,
                    },
                )
                data2 = await resp2.json()
                assert data2["status"] == "ok"

    @pytest.mark.asyncio
    async def test_cooldown_allows_after_expiry(self, _clear_cooldowns, _mock_resolve_repo):
        """After cooldown expires, the same PR can be reviewed again."""
        app = _build_test_app(cooldown=60)
        payload = _make_pr_payload("synchronize", pr_number=10)
        body = json.dumps(payload).encode()
        sig = _sign_payload(payload)

        with _mock_settings(pr_review=True):
            async with TestClient(TestServer(app)) as client:
                # First request
                resp1 = await client.post(
                    "/webhook/github",
                    data=body,
                    headers={
                        "X-GitHub-Event": "pull_request",
                        "X-Hub-Signature-256": sig,
                    },
                )
                assert (await resp1.json())["status"] == "ok"

                # Advance time past the cooldown
                import time

                future = time.time() + 61
                with patch("kai.webhook.time.time", return_value=future):
                    resp2 = await client.post(
                        "/webhook/github",
                        data=body,
                        headers={
                            "X-GitHub-Event": "pull_request",
                            "X-Hub-Signature-256": sig,
                        },
                    )
                    assert (await resp2.json())["status"] == "ok"

    @pytest.mark.asyncio
    async def test_closed_still_notifies(self, _clear_cooldowns):
        """Closed PRs go through the standard notification path, not the review pipeline."""
        app = _build_test_app()
        payload = _make_pr_payload("closed", merged=False)
        body = json.dumps(payload).encode()
        sig = _sign_payload(payload)

        with _mock_settings(pr_review=True):
            async with TestClient(TestServer(app)) as client:
                resp = await client.post(
                    "/webhook/github",
                    data=body,
                    headers={
                        "X-GitHub-Event": "pull_request",
                        "X-Hub-Signature-256": sig,
                    },
                )
                data = await resp.json()
                assert resp.status == 200
                # Should fall through to _fmt_pull_request for the "closed" notification
                assert data.get("status") == "ok"
                app[TELEGRAM_BOT_KEY].send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_synchronize_routed(self, _clear_cooldowns, _mock_resolve_repo):
        """synchronize events (new push to existing PR) are routed to review."""
        app = _build_test_app()
        payload = _make_pr_payload("synchronize", pr_number=99)
        body = json.dumps(payload).encode()
        sig = _sign_payload(payload)

        with _mock_settings(pr_review=True):
            async with TestClient(TestServer(app)) as client:
                resp = await client.post(
                    "/webhook/github",
                    data=body,
                    headers={
                        "X-GitHub-Event": "pull_request",
                        "X-Hub-Signature-256": sig,
                    },
                )
                data = await resp.json()
                assert data["status"] == "ok"

    @pytest.mark.asyncio
    async def test_launches_background_task(self, _clear_cooldowns, _mock_resolve_repo):
        """Reviewable PR events launch review.review_pr as a background task."""
        app = _build_test_app()
        payload = _make_pr_payload("opened")
        body = json.dumps(payload).encode()
        sig = _sign_payload(payload)

        with (
            _mock_settings(pr_review=True),
            patch("kai.webhook.review.review_pr", new_callable=AsyncMock) as mock_review,
        ):
            async with TestClient(TestServer(app)) as client:
                resp = await client.post(
                    "/webhook/github",
                    data=body,
                    headers={
                        "X-GitHub-Event": "pull_request",
                        "X-Hub-Signature-256": sig,
                    },
                )
                assert (await resp.json())["status"] == "ok"

            # Allow the background task to complete
            await asyncio.sleep(0.01)

            mock_review.assert_called_once()
            call_kwargs = mock_review.call_args
            # Verify the payload and config were passed correctly
            assert call_kwargs[0][0] == payload
            assert call_kwargs[1]["webhook_port"] == 8080
            credential = call_kwargs[1]["webhook_secret"]
            assert credential != _TEST_SECRET
            principal = app[INTERNAL_API_AUTH_KEY].authenticate(credential)
            assert principal is not None
            assert principal.chat_id == 12345
            assert principal.allows(InternalAPIScope.MESSAGES_SEND)
            assert not principal.allows(InternalAPIScope.JOBS_WRITE)
            # model_override must be a real string, not a mock. The
            # dataclass-shaped defaults on the fixture config let
            # resolve_user_model return the registry default for the
            # claude backend (`sonnet`); a bare MagicMock config would
            # silently pass a MagicMock instance here, vacuously
            # satisfying mocked review_pr but failing under real code.
            assert isinstance(call_kwargs[1]["model_override"], str)
            # _resolve_local_repo is mocked to return None here;
            # dedicated tests for resolution logic are in TestResolveLocalRepo.
            assert call_kwargs[1]["local_repo_path"] is None


# ── _resolve_local_repo ─────────────────────────────────────────────


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

    @pytest.mark.asyncio
    async def test_handler_uses_resolve(self, _clear_cooldowns):
        """_handle_github calls _resolve_local_repo instead of old home_repo_name logic."""
        app = _build_test_app()
        payload = _make_pr_payload("opened")
        body = json.dumps(payload).encode()
        sig = _sign_payload(payload)

        with (
            _mock_settings(pr_review=True),
            patch(
                "kai.webhook._resolve_local_repo",
                new_callable=AsyncMock,
                return_value="/resolved/path",
            ) as mock_resolve,
            patch("kai.webhook.review.review_pr", new_callable=AsyncMock),
        ):
            async with TestClient(TestServer(app)) as client:
                resp = await client.post(
                    "/webhook/github",
                    data=body,
                    headers={
                        "X-GitHub-Event": "pull_request",
                        "X-Hub-Signature-256": sig,
                    },
                )
                assert (await resp.json())["status"] == "ok"

            await asyncio.sleep(0.01)

            # Verify _resolve_local_repo was called with the repo name
            mock_resolve.assert_called_once_with("owner/repo", app)


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


class TestIssueTriageRouting:
    """Integration tests for issue triage routing in _handle_github.

    Uses _mock_settings() to control per-user issue_triage flag.
    """

    @pytest.mark.asyncio
    async def test_routes_opened_when_enabled(self, _clear_cooldowns):
        """Opened issues are routed to triage pipeline when enabled."""
        app = _build_test_app()
        payload = _make_issue_payload("opened")
        body = json.dumps(payload).encode()
        sig = _sign_payload(payload)

        with (
            _mock_settings(issue_triage=True),
            patch("kai.webhook.triage.triage_issue", new_callable=AsyncMock),
        ):
            async with TestClient(TestServer(app)) as client:
                resp = await client.post(
                    "/webhook/github",
                    data=body,
                    headers={
                        "X-GitHub-Event": "issues",
                        "X-Hub-Signature-256": sig,
                    },
                )
                data = await resp.json()
                assert resp.status == 200
                assert data["status"] == "ok"
                # Should NOT have sent a standard Telegram notification
                app[TELEGRAM_BOT_KEY].send_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_falls_through_when_disabled(self, _clear_cooldowns):
        """With issue triage disabled, opened events go to the notification formatter."""
        app = _build_test_app()
        payload = _make_issue_payload("opened")
        body = json.dumps(payload).encode()
        sig = _sign_payload(payload)

        with _mock_settings(issue_triage=False):
            async with TestClient(TestServer(app)) as client:
                resp = await client.post(
                    "/webhook/github",
                    data=body,
                    headers={
                        "X-GitHub-Event": "issues",
                        "X-Hub-Signature-256": sig,
                    },
                )
                await resp.json()
                assert resp.status == 200
                # Falls through to standard formatter, which sends a Telegram message
                app[TELEGRAM_BOT_KEY].send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_cooldown(self, _clear_cooldowns):
        """Second opened event within cooldown is silently skipped."""
        app = _build_test_app()
        payload = _make_issue_payload("opened", issue_number=10)
        body = json.dumps(payload).encode()
        sig = _sign_payload(payload)

        with (
            _mock_settings(issue_triage=True),
            patch("kai.webhook.triage.triage_issue", new_callable=AsyncMock),
        ):
            async with TestClient(TestServer(app)) as client:
                # First event - triggers triage
                resp1 = await client.post(
                    "/webhook/github",
                    data=body,
                    headers={
                        "X-GitHub-Event": "issues",
                        "X-Hub-Signature-256": sig,
                    },
                )
                assert (await resp1.json())["status"] == "ok"

                # Second event - cooldown (silently skipped, still returns ok)
                resp2 = await client.post(
                    "/webhook/github",
                    data=body,
                    headers={
                        "X-GitHub-Event": "issues",
                        "X-Hub-Signature-256": sig,
                    },
                )
                assert (await resp2.json())["status"] == "ok"

    @pytest.mark.asyncio
    async def test_closed_still_notifies(self, _clear_cooldowns):
        """Closed issues still go through the standard notification path."""
        app = _build_test_app()
        payload = _make_issue_payload("closed")
        body = json.dumps(payload).encode()
        sig = _sign_payload(payload)

        with _mock_settings(issue_triage=True):
            async with TestClient(TestServer(app)) as client:
                resp = await client.post(
                    "/webhook/github",
                    data=body,
                    headers={
                        "X-GitHub-Event": "issues",
                        "X-Hub-Signature-256": sig,
                    },
                )
                await resp.json()
                assert resp.status == 200
                # Closed events fall through to standard formatter
                app[TELEGRAM_BOT_KEY].send_message.assert_called_once()


# ── _handle_github exception handling ──────────────────────────────


class TestGitHubExceptionHandler:
    """Verify the top-level exception handler in _handle_github."""

    @pytest.mark.asyncio
    async def test_exception_returns_500(self, _clear_cooldowns):
        """An unhandled exception in event processing returns 500."""
        app = _build_test_app()
        # Remove telegram_bot to trigger KeyError in _process_github_event
        del app[TELEGRAM_BOT_KEY]
        payload = {"action": "opened", "repository": {"full_name": "owner/repo"}}
        body = json.dumps(payload).encode()
        sig = _sign_payload(payload)

        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/webhook/github",
                data=body,
                headers={
                    "X-GitHub-Event": "push",
                    "X-Hub-Signature-256": sig,
                },
            )
            result = await resp.json()
            assert resp.status == 500
            assert result["msg"] == "internal_error"

    @pytest.mark.asyncio
    async def test_signature_validation_unaffected(self, _clear_cooldowns):
        """Signature validation still returns 401, not caught by the handler."""
        app = _build_test_app()
        body = json.dumps({"action": "opened"}).encode()

        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/webhook/github",
                data=body,
                headers={
                    "X-GitHub-Event": "push",
                    "X-Hub-Signature-256": "sha256=invalid",
                },
            )
            assert resp.status == 401

    @pytest.mark.asyncio
    async def test_happy_path_unaffected(self, _clear_cooldowns):
        """Normal event processing still returns 200."""
        app = _build_test_app()
        payload = {
            "ref": "refs/heads/main",
            "commits": [{"message": "test commit", "author": {"name": "dev"}}],
            "repository": {"full_name": "owner/repo"},
            "compare": "https://github.com/owner/repo/compare/abc...def",
        }
        body = json.dumps(payload).encode()
        sig = _sign_payload(payload)

        with _mock_settings():
            async with TestClient(TestServer(app)) as client:
                resp = await client.post(
                    "/webhook/github",
                    data=body,
                    headers={
                        "X-GitHub-Event": "push",
                        "X-Hub-Signature-256": sig,
                    },
                )
                assert resp.status == 200


# ── Webhook health monitor ─────────────────────────────────────────


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

        with patch("kai.webhook.asyncio.sleep", side_effect=mock_sleep):
            try:
                await _webhook_health_loop(bot, "https://example.com/webhook", "secret", 12345)
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

        with patch("kai.webhook.asyncio.sleep", side_effect=mock_sleep):
            try:
                await _webhook_health_loop(bot, "https://example.com/webhook", "secret", 12345)
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

        with patch("kai.webhook.asyncio.sleep", side_effect=mock_sleep):
            try:
                await _webhook_health_loop(bot, "https://example.com/webhook", "secret", 12345)
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

        with patch("kai.webhook.asyncio.sleep", side_effect=mock_sleep):
            try:
                await _webhook_health_loop(bot, "https://example.com/webhook", "secret", 12345)
            except asyncio.CancelledError:
                pass

        # send_message was attempted exactly once (not retried after failure)
        bot.send_message.assert_called_once()


# ── GitHub notification group routing ────────────────────────────────


class TestGitHubNotifyGroup:
    """Tests for routing GitHub notifications to a per-user notify_chat_id.

    The notify_chat_id is now resolved per-user via resolve_github_settings()
    instead of a global app dict key. These tests verify that notifications
    reach the correct destination based on the resolved settings.
    """

    @pytest.mark.asyncio
    async def test_notification_routes_to_group(self, _clear_cooldowns):
        """When notify_chat_id resolves to a group, notifications go there."""
        app = _build_test_app()
        payload = {
            "ref": "refs/heads/main",
            "commits": [{"message": "test", "author": {"name": "dev"}}],
            "repository": {"full_name": "owner/repo"},
            "compare": "https://github.com/owner/repo/compare/a...b",
        }
        body = json.dumps(payload).encode()
        sig = _sign_payload(payload)

        with _mock_settings(notify_chat_id=-100999):
            async with TestClient(TestServer(app)) as client:
                resp = await client.post(
                    "/webhook/github",
                    data=body,
                    headers={
                        "X-GitHub-Event": "push",
                        "X-Hub-Signature-256": sig,
                    },
                )
                assert resp.status == 200

        # Notification should go to the group chat_id, not the default DM
        call_args = app[TELEGRAM_BOT_KEY].send_message.call_args
        assert call_args[0][0] == -100999

    @pytest.mark.asyncio
    async def test_notification_routes_to_dm_when_unset(self, _clear_cooldowns):
        """When notify_chat_id resolves to user's DM, notifications go there."""
        app = _build_test_app()
        payload = {
            "ref": "refs/heads/main",
            "commits": [{"message": "test", "author": {"name": "dev"}}],
            "repository": {"full_name": "owner/repo"},
            "compare": "https://github.com/owner/repo/compare/a...b",
        }
        body = json.dumps(payload).encode()
        sig = _sign_payload(payload)

        # notify_chat_id matches the admin chat_id (DM, not group)
        with _mock_settings(notify_chat_id=12345):
            async with TestClient(TestServer(app)) as client:
                resp = await client.post(
                    "/webhook/github",
                    data=body,
                    headers={
                        "X-GitHub-Event": "push",
                        "X-Hub-Signature-256": sig,
                    },
                )
                assert resp.status == 200

        # Notification should go to the default admin DM (12345)
        call_args = app[TELEGRAM_BOT_KEY].send_message.call_args
        assert call_args[0][0] == 12345


# ── _get_subscribed_users ──────────────────────────────────────────


class TestGetSubscribedUsers:
    """Tests for the per-user repo subscription lookup.

    All tests mock get_effective_repos to return yaml repos as-is
    (no DB layer), isolating the subscription matching logic.
    """

    @pytest.fixture(autouse=True)
    def _mock_effective_repos(self):
        """Mock get_effective_repos to pass through yaml repos unchanged."""

        async def _passthrough(chat_id: int, yaml_repos: list[str]) -> list[str]:
            return sorted(r.lower() for r in yaml_repos)

        with patch("kai.webhook.sessions.get_effective_repos", side_effect=_passthrough):
            yield

    def _make_config(self, user_configs: dict | None = None) -> MagicMock:
        """Build a mock Config with the given user_configs dict.

        Post-#565 tranche A `Config.user_configs` is a non-optional
        dict; a None argument here is coerced to `{}` to preserve the
        empty-dict test ergonomics callers expect.

        Config is a sync dataclass; AsyncMock would make chained
        attribute calls return coroutines that never get awaited.

        The dataclass-shaped attribute defaults make
        `resolve_user_model()` return a real registry string rather
        than a chained MagicMock; without them, model_override would
        flow through routing tests as an opaque mock.
        """
        config = MagicMock()
        config.user_configs = user_configs if user_configs is not None else {}
        config.default_models = {}
        config.default_backend = "claude"
        config.default_provider = ""
        return config

    def _make_user(
        self,
        telegram_id: int,
        name: str = "testuser",
        repos: list[str] | None = None,
        role: str = "user",
    ) -> UserConfig:
        """Build a UserConfig with the given github_repos and role."""
        return UserConfig(
            telegram_id=telegram_id,
            name=name,
            github_repos=repos or [],
            role=role,
        )

    async def test_exact_match(self):
        """User with matching repo is returned."""
        user = self._make_user(111, repos=["dcellison/kai"])
        config = self._make_config({111: user})
        result = await _get_subscribed_users(config, "dcellison/kai")
        assert result == [user]

    async def test_case_insensitive(self):
        """Repo matching is case-insensitive (GitHub repos are)."""
        user = self._make_user(111, repos=["dcellison/kai"])
        config = self._make_config({111: user})
        result = await _get_subscribed_users(config, "Dcellison/Kai")
        assert result == [user]

    async def test_multiple_users(self):
        """Multiple users subscribed to the same repo are all returned."""
        user1 = self._make_user(111, name="alice", repos=["dcellison/kai"])
        user2 = self._make_user(222, name="bob", repos=["dcellison/kai"])
        config = self._make_config({111: user1, 222: user2})
        result = await _get_subscribed_users(config, "dcellison/kai")
        assert len(result) == 2
        assert user1 in result
        assert user2 in result

    async def test_no_match(self):
        """No users subscribed to the repo returns empty list."""
        user = self._make_user(111, repos=["dcellison/other"])
        config = self._make_config({111: user})
        result = await _get_subscribed_users(config, "dcellison/kai")
        assert result == []

    async def test_no_user_configs(self):
        """Config with empty user_configs returns empty list."""
        config = self._make_config({})
        result = await _get_subscribed_users(config, "dcellison/kai")
        assert result == []

    # ── Admin wildcard tests ─────────────────────────────────────

    async def test_admin_with_no_repos_is_wildcard(self):
        """Admin with empty github_repos receives all events."""
        admin = self._make_user(111, role="admin")
        config = self._make_config({111: admin})
        result = await _get_subscribed_users(config, "any/repo")
        assert result == [admin]

    async def test_admin_with_explicit_repos_not_wildcard(self):
        """Admin with explicit repos only receives those repos."""
        admin = self._make_user(111, role="admin", repos=["owner/other"])
        config = self._make_config({111: admin})
        result = await _get_subscribed_users(config, "owner/repo")
        assert result == []

    async def test_admin_with_matching_explicit_repo(self):
        """Admin with explicit matching repo appears once (no duplicate)."""
        admin = self._make_user(111, role="admin", repos=["owner/repo"])
        config = self._make_config({111: admin})
        result = await _get_subscribed_users(config, "owner/repo")
        assert result == [admin]

    async def test_regular_user_with_no_repos_not_wildcard(self):
        """Non-admin with empty github_repos receives nothing."""
        user = self._make_user(111, role="user")
        config = self._make_config({111: user})
        result = await _get_subscribed_users(config, "any/repo")
        assert result == []

    async def test_admin_wildcard_and_explicit_subscriber_both_returned(self):
        """Admin wildcard and explicit subscriber both appear."""
        admin = self._make_user(111, name="admin", role="admin")
        user = self._make_user(222, name="user", repos=["owner/repo"])
        config = self._make_config({111: admin, 222: user})
        result = await _get_subscribed_users(config, "owner/repo")
        assert len(result) == 2
        assert admin in result
        assert user in result

    async def test_admin_wildcard_with_case_insensitive_explicit_user(self):
        """Admin wildcard and case-insensitive explicit match coexist."""
        admin = self._make_user(111, name="admin", role="admin")
        user = self._make_user(222, name="user", repos=["Owner/Repo"])
        config = self._make_config({111: admin, 222: user})
        result = await _get_subscribed_users(config, "owner/repo")
        assert len(result) == 2
        assert admin in result
        assert user in result

    async def test_multiple_admin_wildcards(self):
        """Multiple admins with empty repos all receive events."""
        admin1 = self._make_user(111, name="alice", role="admin")
        admin2 = self._make_user(222, name="bob", role="admin")
        config = self._make_config({111: admin1, 222: admin2})
        result = await _get_subscribed_users(config, "any/repo")
        assert len(result) == 2
        assert admin1 in result
        assert admin2 in result

    @pytest.mark.asyncio
    async def test_no_subscribed_users_warning_includes_remediation_hint(self, _clear_cooldowns, caplog):
        """Warning for unmatched repo includes github_repos hint."""
        import logging

        # User has repos, but not the one in the payload
        user = self._make_user(111, repos=["owner/other-repo"])
        config = self._make_config({111: user})
        app = _build_test_app(config=config)
        payload = {
            "ref": "refs/heads/main",
            "commits": [{"message": "test", "author": {"name": "dev"}}],
            "repository": {"full_name": "owner/target-repo"},
            "compare": "https://github.com/owner/target-repo/compare/a...b",
        }
        body = json.dumps(payload).encode()
        sig = _sign_payload(payload)

        with caplog.at_level(logging.WARNING, logger="kai.webhook"):
            async with TestClient(TestServer(app)) as client:
                await client.post(
                    "/webhook/github",
                    data=body,
                    headers={
                        "X-GitHub-Event": "push",
                        "X-Hub-Signature-256": sig,
                    },
                )

        # Warning should mention github_repos and include the repo name
        # at least twice (once in the event description, once in the hint).
        warning_records = [
            r for r in caplog.records if r.levelno >= logging.WARNING and "no subscribed users" in r.message
        ]
        assert len(warning_records) == 1
        msg = warning_records[0].message
        assert "github_repos" in msg
        assert msg.count("owner/target-repo") >= 2


# ── Per-user webhook routing ────────────────────────────────────────


class TestPerUserRouting:
    """Tests for per-user repo-based event routing.

    These tests create apps with user_configs populated so that
    _get_subscribed_users returns specific users. Each user's
    feature flags are controlled via resolve_github_settings mock.
    get_effective_repos is mocked to pass through yaml repos (no DB).
    """

    @pytest.fixture(autouse=True)
    def _mock_effective_repos(self):
        """Mock get_effective_repos to pass through yaml repos unchanged."""

        async def _passthrough(chat_id: int, yaml_repos: list[str]) -> list[str]:
            return sorted(r.lower() for r in yaml_repos)

        with patch("kai.webhook.sessions.get_effective_repos", side_effect=_passthrough):
            yield

    def _make_user_config(
        self,
        telegram_id: int,
        name: str = "testuser",
        repos: list[str] | None = None,
        role: str = "user",
    ) -> UserConfig:
        """Build a UserConfig for routing tests."""
        return UserConfig(
            telegram_id=telegram_id,
            name=name,
            github_repos=repos or [],
            role=role,
        )

    def _make_config_with_users(self, users: list) -> MagicMock:
        """Build a mock Config with user_configs populated.

        Config is a sync dataclass; using AsyncMock would make chained
        attribute calls like `config.default_models.get(...)` return a
        coroutine instead of a value, which then surfaces as the
        AsyncMockMixin._execute_mock_call never-awaited warning.

        The dataclass-shaped attribute defaults make
        `resolve_user_model()` return a real registry string rather
        than a chained MagicMock; without them, the routing tests
        would pass model_override as an opaque mock into review_pr /
        triage_issue, hiding any future regression that depended on
        the override being a real string.
        """
        config = MagicMock()
        config.user_configs = {u.telegram_id: u for u in users}
        config.default_models = {}
        config.default_backend = "claude"
        config.default_provider = ""
        # get_user_config returns the UserConfig for a given ID
        config.get_user_config = lambda uid: config.user_configs.get(uid)
        return config

    @pytest.mark.asyncio
    async def test_event_routes_to_subscribed_user(self, _clear_cooldowns):
        """Event for a subscribed repo reaches the correct user."""
        user = self._make_user_config(111, repos=["owner/repo"])
        config = self._make_config_with_users([user])
        app = _build_test_app(config=config)
        payload = {
            "ref": "refs/heads/main",
            "commits": [{"message": "test", "author": {"name": "dev"}}],
            "repository": {"full_name": "owner/repo"},
            "compare": "https://github.com/owner/repo/compare/a...b",
        }
        body = json.dumps(payload).encode()
        sig = _sign_payload(payload)

        # Mock resolve_github_settings to route to user 111's DM
        with _mock_settings(notify_chat_id=111):
            async with TestClient(TestServer(app)) as client:
                resp = await client.post(
                    "/webhook/github",
                    data=body,
                    headers={
                        "X-GitHub-Event": "push",
                        "X-Hub-Signature-256": sig,
                    },
                )
                assert resp.status == 200

        # Notification should go to user 111, not the admin (12345)
        call_args = app[TELEGRAM_BOT_KEY].send_message.call_args
        assert call_args[0][0] == 111

    @pytest.mark.asyncio
    async def test_event_routes_to_multiple_users(self, _clear_cooldowns):
        """Event for a repo subscribed by two users reaches both."""
        user1 = self._make_user_config(111, name="alice", repos=["owner/repo"])
        user2 = self._make_user_config(222, name="bob", repos=["owner/repo"])
        config = self._make_config_with_users([user1, user2])
        app = _build_test_app(config=config)
        payload = {
            "ref": "refs/heads/main",
            "commits": [{"message": "test", "author": {"name": "dev"}}],
            "repository": {"full_name": "owner/repo"},
            "compare": "https://github.com/owner/repo/compare/a...b",
        }
        body = json.dumps(payload).encode()
        sig = _sign_payload(payload)

        # Return different notify_chat_id per user so we can verify both
        call_count = 0

        async def _per_user_settings(chat_id, config):
            nonlocal call_count
            call_count += 1
            return {
                "repos": [],
                "notify_chat_id": chat_id,
                "pr_review": False,
                "issue_triage": False,
            }

        with patch(
            "kai.webhook.sessions.resolve_github_settings",
            side_effect=_per_user_settings,
        ):
            async with TestClient(TestServer(app)) as client:
                resp = await client.post(
                    "/webhook/github",
                    data=body,
                    headers={
                        "X-GitHub-Event": "push",
                        "X-Hub-Signature-256": sig,
                    },
                )
                assert resp.status == 200

        # Resolver called once per subscribed user
        assert call_count == 2
        # Both users should receive notifications
        assert app[TELEGRAM_BOT_KEY].send_message.call_count == 2
        sent_to = {c[0][0] for c in app[TELEGRAM_BOT_KEY].send_message.call_args_list}
        assert sent_to == {111, 222}

    @pytest.mark.asyncio
    async def test_event_fallback_to_admin(self, _clear_cooldowns):
        """No subscribers for a repo routes to the admin chat_id."""
        # User subscribed to a different repo
        user = self._make_user_config(111, repos=["owner/other-repo"])
        config = self._make_config_with_users([user])
        app = _build_test_app(config=config)
        payload = {
            "ref": "refs/heads/main",
            "commits": [{"message": "test", "author": {"name": "dev"}}],
            "repository": {"full_name": "owner/repo"},
            "compare": "https://github.com/owner/repo/compare/a...b",
        }
        body = json.dumps(payload).encode()
        sig = _sign_payload(payload)

        # Fallback resolves settings for admin (12345)
        with _mock_settings(notify_chat_id=12345):
            async with TestClient(TestServer(app)) as client:
                resp = await client.post(
                    "/webhook/github",
                    data=body,
                    headers={
                        "X-GitHub-Event": "push",
                        "X-Hub-Signature-256": sig,
                    },
                )
                assert resp.status == 200

        # Should go to admin chat_id (12345), not user 111
        call_args = app[TELEGRAM_BOT_KEY].send_message.call_args
        assert call_args[0][0] == 12345

    @pytest.mark.asyncio
    async def test_pr_review_per_user_flag(self, _clear_cooldowns, _mock_resolve_repo):
        """Only users with pr_review=True trigger review."""
        user = self._make_user_config(111, repos=["owner/repo"])
        config = self._make_config_with_users([user])
        app = _build_test_app(config=config)
        payload = _make_pr_payload("opened")
        body = json.dumps(payload).encode()
        sig = _sign_payload(payload)

        with (
            _mock_settings(pr_review=True, notify_chat_id=111),
            patch("kai.webhook.review.review_pr", new_callable=AsyncMock) as mock_review,
        ):
            async with TestClient(TestServer(app)) as client:
                resp = await client.post(
                    "/webhook/github",
                    data=body,
                    headers={
                        "X-GitHub-Event": "pull_request",
                        "X-Hub-Signature-256": sig,
                    },
                )
                assert resp.status == 200

            # Allow background task to complete
            await asyncio.sleep(0.01)
            mock_review.assert_called_once()
            # Verify notify_chat_id is passed to the review agent
            assert mock_review.call_args[1]["notify_chat_id"] == 111

    @pytest.mark.asyncio
    async def test_issue_triage_per_user_flag(self, _clear_cooldowns):
        """Only users with issue_triage=True trigger triage."""
        user = self._make_user_config(111, repos=["owner/repo"])
        config = self._make_config_with_users([user])
        app = _build_test_app(config=config)
        payload = _make_issue_payload("opened")
        body = json.dumps(payload).encode()
        sig = _sign_payload(payload)

        with (
            _mock_settings(issue_triage=True, notify_chat_id=111),
            patch("kai.webhook.triage.triage_issue", new_callable=AsyncMock) as mock_triage,
        ):
            async with TestClient(TestServer(app)) as client:
                resp = await client.post(
                    "/webhook/github",
                    data=body,
                    headers={
                        "X-GitHub-Event": "issues",
                        "X-Hub-Signature-256": sig,
                    },
                )
                assert resp.status == 200

            # Allow background task to complete
            await asyncio.sleep(0.01)
            mock_triage.assert_called_once()
            # Verify notify_chat_id is passed to the triage agent
            assert mock_triage.call_args[1]["notify_chat_id"] == 111

    @pytest.mark.asyncio
    async def test_cooldown_shared_across_users(self, _clear_cooldowns, _mock_resolve_repo):
        """One review per PR per cooldown window regardless of subscriber count.

        The cooldown dict is server-level, so if user A triggers a review
        for PR #42, user B's subscription won't trigger a second review.
        """
        user1 = self._make_user_config(111, name="alice", repos=["owner/repo"])
        user2 = self._make_user_config(222, name="bob", repos=["owner/repo"])
        config = self._make_config_with_users([user1, user2])
        app = _build_test_app(config=config)
        payload = _make_pr_payload("opened", pr_number=42)
        body = json.dumps(payload).encode()
        sig = _sign_payload(payload)

        review_count = 0

        async def _per_user_settings(chat_id, config):
            return {
                "repos": [],
                "notify_chat_id": chat_id,
                "pr_review": True,
                "issue_triage": False,
            }

        async def _counting_review(*args, **kwargs):
            nonlocal review_count
            review_count += 1

        with (
            patch(
                "kai.webhook.sessions.resolve_github_settings",
                side_effect=_per_user_settings,
            ),
            patch("kai.webhook.review.review_pr", side_effect=_counting_review),
        ):
            async with TestClient(TestServer(app)) as client:
                resp = await client.post(
                    "/webhook/github",
                    data=body,
                    headers={
                        "X-GitHub-Event": "pull_request",
                        "X-Hub-Signature-256": sig,
                    },
                )
                assert resp.status == 200

            # Allow background tasks to complete
            await asyncio.sleep(0.01)

        # Only one review should have been launched despite two subscribers.
        # The second user hits the cooldown set by the first user's review.
        assert review_count == 1

    @pytest.mark.asyncio
    async def test_standard_notification_per_user(self, _clear_cooldowns):
        """Push events are formatted and sent to each subscribed user's notify_chat_id."""
        user = self._make_user_config(111, repos=["owner/repo"])
        config = self._make_config_with_users([user])
        app = _build_test_app(config=config)
        payload = {
            "ref": "refs/heads/main",
            "commits": [{"message": "test", "author": {"name": "dev"}}],
            "repository": {"full_name": "owner/repo"},
            "compare": "https://github.com/owner/repo/compare/a...b",
            "pusher": {"name": "testuser"},
        }
        body = json.dumps(payload).encode()
        sig = _sign_payload(payload)

        # Route notification to user's group chat
        with _mock_settings(notify_chat_id=-100999):
            async with TestClient(TestServer(app)) as client:
                resp = await client.post(
                    "/webhook/github",
                    data=body,
                    headers={
                        "X-GitHub-Event": "push",
                        "X-Hub-Signature-256": sig,
                    },
                )
                assert resp.status == 200

        # Notification should go to the user's notify_chat_id, not their DM
        call_args = app[TELEGRAM_BOT_KEY].send_message.call_args
        assert call_args[0][0] == -100999

    @pytest.mark.asyncio
    async def test_fan_out_exception_isolation(self, _clear_cooldowns):
        """A failure processing one user does not block subsequent users."""
        user1 = self._make_user_config(111, name="alice", repos=["owner/repo"])
        user2 = self._make_user_config(222, name="bob", repos=["owner/repo"])
        config = self._make_config_with_users([user1, user2])
        app = _build_test_app(config=config)
        payload = {
            "ref": "refs/heads/main",
            "commits": [{"message": "test", "author": {"name": "dev"}}],
            "repository": {"full_name": "owner/repo"},
            "compare": "https://github.com/owner/repo/compare/a...b",
        }
        body = json.dumps(payload).encode()
        sig = _sign_payload(payload)

        # First user raises, second user succeeds
        call_count = 0

        async def _failing_then_ok(chat_id, config):
            nonlocal call_count
            call_count += 1
            if chat_id == 111:
                raise RuntimeError("transient DB failure")
            return {
                "repos": [],
                "notify_chat_id": chat_id,
                "pr_review": False,
                "issue_triage": False,
            }

        with patch(
            "kai.webhook.sessions.resolve_github_settings",
            side_effect=_failing_then_ok,
        ):
            async with TestClient(TestServer(app)) as client:
                resp = await client.post(
                    "/webhook/github",
                    data=body,
                    headers={
                        "X-GitHub-Event": "push",
                        "X-Hub-Signature-256": sig,
                    },
                )
                assert resp.status == 200

        # Both users were attempted (resolve called twice)
        assert call_count == 2
        # User 2 still got their notification despite user 1's failure
        call_args = app[TELEGRAM_BOT_KEY].send_message.call_args
        assert call_args[0][0] == 222

    # ── Per-user os_user resolution ─────────────────────────────

    @pytest.mark.asyncio
    async def test_review_passes_per_user_claude_user(self, _clear_cooldowns, _mock_resolve_repo):
        """PR review uses the subscribing user's os_user, not the global."""
        user = UserConfig(
            telegram_id=111,
            name="alice",
            github_repos=["owner/repo"],
            os_user="alice",
        )
        config = self._make_config_with_users([user])
        config.claude_user = "global-user"
        app = _build_test_app(config=config)
        payload = _make_pr_payload("opened")
        body = json.dumps(payload).encode()
        sig = _sign_payload(payload)

        with (
            _mock_settings(pr_review=True, notify_chat_id=111),
            patch("kai.webhook.review.review_pr", new_callable=AsyncMock) as mock_review,
        ):
            async with TestClient(TestServer(app)) as client:
                resp = await client.post(
                    "/webhook/github",
                    data=body,
                    headers={
                        "X-GitHub-Event": "pull_request",
                        "X-Hub-Signature-256": sig,
                    },
                )
                assert resp.status == 200

            await asyncio.sleep(0.01)
            mock_review.assert_called_once()
            assert mock_review.call_args[1]["claude_user"] == "alice"

    @pytest.mark.asyncio
    async def test_triage_passes_per_user_claude_user(self, _clear_cooldowns):
        """Issue triage uses the subscribing user's os_user, not the global."""
        user = UserConfig(
            telegram_id=111,
            name="bob",
            github_repos=["owner/repo"],
            os_user="bob",
        )
        config = self._make_config_with_users([user])
        config.claude_user = "global-user"
        app = _build_test_app(config=config)
        payload = _make_issue_payload("opened")
        body = json.dumps(payload).encode()
        sig = _sign_payload(payload)

        with (
            _mock_settings(issue_triage=True, notify_chat_id=111),
            patch("kai.webhook.triage.triage_issue", new_callable=AsyncMock) as mock_triage,
        ):
            async with TestClient(TestServer(app)) as client:
                resp = await client.post(
                    "/webhook/github",
                    data=body,
                    headers={
                        "X-GitHub-Event": "issues",
                        "X-Hub-Signature-256": sig,
                    },
                )
                assert resp.status == 200

            await asyncio.sleep(0.01)
            mock_triage.assert_called_once()
            assert mock_triage.call_args[1]["claude_user"] == "bob"

    @pytest.mark.asyncio
    async def test_review_user_without_os_user_runs_as_bot(self, _clear_cooldowns, _mock_resolve_repo):
        """UserConfig with os_user=None means run as bot user (claude_user=None)."""
        user = self._make_user_config(111, repos=["owner/repo"])
        config = self._make_config_with_users([user])
        app = _build_test_app(config=config)
        payload = _make_pr_payload("opened")
        body = json.dumps(payload).encode()
        sig = _sign_payload(payload)

        with (
            _mock_settings(pr_review=True, notify_chat_id=111),
            patch("kai.webhook.review.review_pr", new_callable=AsyncMock) as mock_review,
        ):
            async with TestClient(TestServer(app)) as client:
                resp = await client.post(
                    "/webhook/github",
                    data=body,
                    headers={
                        "X-GitHub-Event": "pull_request",
                        "X-Hub-Signature-256": sig,
                    },
                )
                assert resp.status == 200

            await asyncio.sleep(0.01)
            mock_review.assert_called_once()
            assert mock_review.call_args[1]["claude_user"] is None

    # ── Admin wildcard integration tests ─────────────────────────

    @pytest.mark.asyncio
    async def test_admin_wildcard_receives_event(self, _clear_cooldowns):
        """Admin with empty github_repos receives push events via wildcard."""
        admin = self._make_user_config(111, role="admin")
        config = self._make_config_with_users([admin])
        app = _build_test_app(config=config)
        payload = {
            "ref": "refs/heads/main",
            "commits": [{"message": "test", "author": {"name": "dev"}}],
            "repository": {"full_name": "any/repo"},
            "compare": "https://github.com/any/repo/compare/a...b",
        }
        body = json.dumps(payload).encode()
        sig = _sign_payload(payload)

        # Return different notify_chat_id per caller so we can distinguish
        # wildcard routing (chat_id=111) from fallback routing (chat_id=12345).
        # The fallback calls resolve_github_settings with app[CHAT_ID_KEY]=12345,
        # the wildcard path calls with the admin's telegram_id=111.
        async def _per_caller_settings(chat_id, config):
            return {
                "repos": [],
                "notify_chat_id": chat_id,
                "pr_review": False,
                "issue_triage": False,
            }

        with patch("kai.webhook.sessions.resolve_github_settings", side_effect=_per_caller_settings):
            async with TestClient(TestServer(app)) as client:
                resp = await client.post(
                    "/webhook/github",
                    data=body,
                    headers={
                        "X-GitHub-Event": "push",
                        "X-Hub-Signature-256": sig,
                    },
                )
                assert resp.status == 200

        # Admin received the event via wildcard (111), not fallback (12345)
        call_args = app[TELEGRAM_BOT_KEY].send_message.call_args
        assert call_args[0][0] == 111

    @pytest.mark.asyncio
    async def test_admin_wildcard_cannot_trigger_pr_review(self, _clear_cooldowns, _mock_resolve_repo, caplog):
        """Subscription wildcard never grants shared-identity review authority."""
        admin = self._make_user_config(111, role="admin")
        config = self._make_config_with_users([admin])
        app = _build_test_app(config=config)
        payload = _make_pr_payload("opened")
        body = json.dumps(payload).encode()
        sig = _sign_payload(payload)

        with (
            _mock_settings(pr_review=True, notify_chat_id=111),
            patch("kai.webhook.review.review_pr", new_callable=AsyncMock) as mock_review,
        ):
            async with TestClient(TestServer(app)) as client:
                resp = await client.post(
                    "/webhook/github",
                    data=body,
                    headers={
                        "X-GitHub-Event": "pull_request",
                        "X-Hub-Signature-256": sig,
                    },
                )
                assert resp.status == 200

            # Allow background task to complete
            await asyncio.sleep(0.01)
            mock_review.assert_not_called()
            app[TELEGRAM_BOT_KEY].send_message.assert_called_once()
            assert "not admin-authorized" in caplog.text

    @pytest.mark.asyncio
    async def test_db_added_subscription_cannot_trigger_pr_review(
        self,
        _clear_cooldowns,
        _mock_resolve_repo,
        caplog,
    ):
        """An effective notification subscription does not grant review."""
        user = self._make_user_config(111, repos=[])
        config = self._make_config_with_users([user])
        app = _build_test_app(config=config)
        payload = _make_pr_payload("opened")
        body = json.dumps(payload).encode()
        sig = _sign_payload(payload)

        with (
            patch(
                "kai.webhook.sessions.get_effective_repos",
                new=AsyncMock(return_value=["owner/repo"]),
            ),
            _mock_settings(pr_review=True, notify_chat_id=111),
            patch("kai.webhook.review.review_pr", new_callable=AsyncMock) as mock_review,
        ):
            async with TestClient(TestServer(app)) as client:
                resp = await client.post(
                    "/webhook/github",
                    data=body,
                    headers={
                        "X-GitHub-Event": "pull_request",
                        "X-Hub-Signature-256": sig,
                    },
                )
                assert resp.status == 200

            await asyncio.sleep(0.01)
            mock_review.assert_not_called()
            app[TELEGRAM_BOT_KEY].send_message.assert_called_once()
            assert "not admin-authorized" in caplog.text

    @pytest.mark.asyncio
    async def test_db_added_subscription_cannot_trigger_issue_triage(self, _clear_cooldowns, caplog):
        """Mutable notification state cannot grant GitHub mutations."""
        user = self._make_user_config(111, repos=[])
        config = self._make_config_with_users([user])
        app = _build_test_app(config=config)
        payload = _make_issue_payload("opened")
        body = json.dumps(payload).encode()
        sig = _sign_payload(payload)

        with (
            patch(
                "kai.webhook.sessions.get_effective_repos",
                new=AsyncMock(return_value=["owner/repo"]),
            ),
            _mock_settings(issue_triage=True, notify_chat_id=111),
            patch("kai.webhook.triage.triage_issue", new_callable=AsyncMock) as mock_triage,
        ):
            async with TestClient(TestServer(app)) as client:
                resp = await client.post(
                    "/webhook/github",
                    data=body,
                    headers={
                        "X-GitHub-Event": "issues",
                        "X-Hub-Signature-256": sig,
                    },
                )
                assert resp.status == 200

            await asyncio.sleep(0.01)
            mock_triage.assert_not_called()
            app[TELEGRAM_BOT_KEY].send_message.assert_called_once()
            assert "not admin-authorized" in caplog.text

    @pytest.mark.asyncio
    async def test_regular_user_no_repos_receives_nothing(self, _clear_cooldowns):
        """Non-admin with empty github_repos does not receive events."""
        user = self._make_user_config(111, role="user")
        config = self._make_config_with_users([user])
        app = _build_test_app(config=config)
        payload = {
            "ref": "refs/heads/main",
            "commits": [{"message": "test", "author": {"name": "dev"}}],
            "repository": {"full_name": "any/repo"},
            "compare": "https://github.com/any/repo/compare/a...b",
        }
        body = json.dumps(payload).encode()
        sig = _sign_payload(payload)

        # Fallback resolves settings for the default admin (12345)
        with _mock_settings(notify_chat_id=12345):
            async with TestClient(TestServer(app)) as client:
                resp = await client.post(
                    "/webhook/github",
                    data=body,
                    headers={
                        "X-GitHub-Event": "push",
                        "X-Hub-Signature-256": sig,
                    },
                )
                assert resp.status == 200

        # Event goes to fallback admin (12345), not user 111
        call_args = app[TELEGRAM_BOT_KEY].send_message.call_args
        assert call_args[0][0] == 12345

    @pytest.mark.asyncio
    async def test_review_receives_user_backend(self, _clear_cooldowns, _mock_resolve_repo):
        """Per-user backend/provider are passed to review_pr."""
        base = self._make_user_config(111, repos=["owner/repo"])
        # Frozen dataclass - use replace to set per-user backend override
        user = dataclasses.replace(base, backend="goose", provider="openai")
        config = self._make_config_with_users([user])
        # Global defaults (should be overridden by per-user values)
        config.default_backend = "claude"
        config.default_provider = ""
        app = _build_test_app(config=config)
        payload = _make_pr_payload("opened")
        body = json.dumps(payload).encode()
        sig = _sign_payload(payload)

        with (
            _mock_settings(pr_review=True, notify_chat_id=111),
            patch("kai.webhook.review.review_pr", new_callable=AsyncMock) as mock_review,
        ):
            async with TestClient(TestServer(app)) as client:
                resp = await client.post(
                    "/webhook/github",
                    data=body,
                    headers={
                        "X-GitHub-Event": "pull_request",
                        "X-Hub-Signature-256": sig,
                    },
                )
                assert resp.status == 200

            await asyncio.sleep(0.01)
            mock_review.assert_called_once()
            assert mock_review.call_args[1]["agent_backend"] == "goose"
            assert mock_review.call_args[1]["provider"] == "openai"

    @pytest.mark.asyncio
    async def test_triage_receives_user_backend(self, _clear_cooldowns):
        """Per-user backend/provider are passed to triage_issue."""
        base = self._make_user_config(111, repos=["owner/repo"])
        user = dataclasses.replace(
            base,
            backend="goose",
            provider="anthropic",
            allowed_triage_projects=["Sprint 1"],
        )
        config = self._make_config_with_users([user])
        config.default_backend = "claude"
        config.default_provider = ""
        app = _build_test_app(config=config)
        payload = _make_issue_payload("opened")
        body = json.dumps(payload).encode()
        sig = _sign_payload(payload)

        with (
            _mock_settings(issue_triage=True, notify_chat_id=111),
            patch("kai.webhook.triage.triage_issue", new_callable=AsyncMock) as mock_triage,
        ):
            async with TestClient(TestServer(app)) as client:
                resp = await client.post(
                    "/webhook/github",
                    data=body,
                    headers={
                        "X-GitHub-Event": "issues",
                        "X-Hub-Signature-256": sig,
                    },
                )
                assert resp.status == 200

            await asyncio.sleep(0.01)
            mock_triage.assert_called_once()
            assert mock_triage.call_args[1]["agent_backend"] == "goose"
            assert mock_triage.call_args[1]["provider"] == "anthropic"
            assert mock_triage.call_args[1]["allowed_triage_projects"] == ["Sprint 1"]

    @pytest.mark.asyncio
    async def test_review_uses_global_backend_when_no_user_override(self, _clear_cooldowns, _mock_resolve_repo):
        """Without per-user override, review_pr gets the global backend."""
        user = self._make_user_config(111, repos=["owner/repo"])
        # No backend/provider set on user - defaults are None
        config = self._make_config_with_users([user])
        config.default_backend = "goose"
        config.default_provider = "google"
        app = _build_test_app(config=config)
        payload = _make_pr_payload("opened")
        body = json.dumps(payload).encode()
        sig = _sign_payload(payload)

        with (
            _mock_settings(pr_review=True, notify_chat_id=111),
            patch("kai.webhook.review.review_pr", new_callable=AsyncMock) as mock_review,
        ):
            async with TestClient(TestServer(app)) as client:
                resp = await client.post(
                    "/webhook/github",
                    data=body,
                    headers={
                        "X-GitHub-Event": "pull_request",
                        "X-Hub-Signature-256": sig,
                    },
                )
                assert resp.status == 200

            await asyncio.sleep(0.01)
            mock_review.assert_called_once()
            # Falls back to global config values
            assert mock_review.call_args[1]["agent_backend"] == "goose"
            assert mock_review.call_args[1]["provider"] == "google"


# ── add_notification_chat_id / remove_notification_chat_id ───────────


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
        config.webhook_secret = None
        config.telegram_webhook_url = None
        config.telegram_webhook_secret = None
        config.github_webhook_secret = None
        config.generic_webhook_secret = None
        config.pr_review_cooldown = 0
        config.pr_review_timeout_s = 0
        config.webhook_port = 0
        config.workspace_base = None
        config.allowed_workspaces = []
        config.spec_dir = "specs"

        telegram_app = MagicMock()
        telegram_app.bot = AsyncMock()
        telegram_app.bot_data = {"pool": MagicMock(internal_api_auth=InternalAPIAuth({111: "secret"}))}

        fake_runner = MagicMock()
        fake_runner.setup = AsyncMock()
        fake_site = MagicMock()
        fake_site.start = AsyncMock()

        old_app = wh._app
        old_runner = wh._runner
        try:
            with (
                patch("kai.webhook.sessions.get_setting", new_callable=AsyncMock, return_value="-100222"),
                patch("kai.webhook.web.AppRunner", return_value=fake_runner),
                patch("kai.webhook.web.TCPSite", return_value=fake_site),
            ):
                await wh.start(telegram_app, config)

            assert wh._app[ALLOWED_USER_IDS_KEY] == {111}
            assert wh._app[NOTIFICATION_CHAT_IDS_KEY] == {-100111, -100222}
        finally:
            wh._app = old_app
            wh._runner = old_runner

    def test_add_notification_chat_id(self):
        """add_notification_chat_id adds a chat_id to the outbound registry."""
        import kai.webhook as wh

        app = web.Application()
        app[ALLOWED_USER_IDS_KEY] = {100}
        app[NOTIFICATION_CHAT_IDS_KEY] = set()
        old_app = wh._app
        wh._app = app
        try:
            add_notification_chat_id(999)
            assert app[NOTIFICATION_CHAT_IDS_KEY] == {999}
            assert app[ALLOWED_USER_IDS_KEY] == {100}
        finally:
            wh._app = old_app

    def test_add_notification_chat_id_idempotent(self):
        """Adding the same chat_id twice does not duplicate it."""
        import kai.webhook as wh

        app = web.Application()
        app[ALLOWED_USER_IDS_KEY] = {100}
        app[NOTIFICATION_CHAT_IDS_KEY] = set()
        old_app = wh._app
        wh._app = app
        try:
            add_notification_chat_id(999)
            add_notification_chat_id(999)
            # Sets don't have duplicates; just verify it's present
            assert app[NOTIFICATION_CHAT_IDS_KEY] == {999}
            assert app[ALLOWED_USER_IDS_KEY] == {100}
        finally:
            wh._app = old_app

    def test_add_does_not_mutate_inbound_principals(self):
        """Adding an outbound destination cannot authorize a Telegram user."""
        import kai.webhook as wh

        inbound_principals = {100}
        app = web.Application()
        app[ALLOWED_USER_IDS_KEY] = set(inbound_principals)
        app[NOTIFICATION_CHAT_IDS_KEY] = set()
        old_app = wh._app
        wh._app = app
        try:
            add_notification_chat_id(999)
            assert app[ALLOWED_USER_IDS_KEY] == {100}
            assert app[NOTIFICATION_CHAT_IDS_KEY] == {999}
            assert inbound_principals == {100}
        finally:
            wh._app = old_app

    def test_add_notification_chat_id_no_app(self):
        """add_notification_chat_id is a no-op when _app is None (polling mode)."""
        import kai.webhook as wh

        old_app = wh._app
        wh._app = None
        try:
            # Should not raise
            add_notification_chat_id(999)
        finally:
            wh._app = old_app

    def test_remove_notification_chat_id_group(self):
        """remove_notification_chat_id removes a group chat_id from the outbound registry."""
        import kai.webhook as wh

        app = web.Application()
        app[ALLOWED_USER_IDS_KEY] = {100}
        app[NOTIFICATION_CHAT_IDS_KEY] = {-100123}
        # No config needed for this case - group IDs are never in user_configs
        mock_config = MagicMock()
        mock_config.user_configs = {}
        app[CONFIG_KEY] = mock_config
        old_app = wh._app
        wh._app = app
        try:
            remove_notification_chat_id(-100123)
            assert -100123 not in app[NOTIFICATION_CHAT_IDS_KEY]
            assert app[ALLOWED_USER_IDS_KEY] == {100}
        finally:
            wh._app = old_app

    def test_remove_notification_chat_id_preserves_users(self):
        """remove_notification_chat_id does NOT remove a real user's telegram_id."""
        import kai.webhook as wh

        user = UserConfig(telegram_id=12345, name="alice")
        mock_config = MagicMock()
        mock_config.user_configs = {12345: user}

        app = web.Application()
        app[ALLOWED_USER_IDS_KEY] = {12345}
        app[NOTIFICATION_CHAT_IDS_KEY] = {12345, -100999}
        app[CONFIG_KEY] = mock_config
        old_app = wh._app
        wh._app = app
        try:
            remove_notification_chat_id(12345)
            # 12345 is a real user - must NOT be removed
            assert 12345 in app[ALLOWED_USER_IDS_KEY]
            assert 12345 in app[NOTIFICATION_CHAT_IDS_KEY]
        finally:
            wh._app = old_app

    def test_remove_notification_chat_id_no_app(self):
        """remove_notification_chat_id is a no-op when _app is None (polling mode)."""
        import kai.webhook as wh

        old_app = wh._app
        wh._app = None
        try:
            # Should not raise
            remove_notification_chat_id(-100123)
        finally:
            wh._app = old_app
