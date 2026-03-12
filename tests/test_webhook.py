"""Tests for webhook.py pure functions and GitHub event formatters."""

import hashlib
import hmac
import json
from unittest.mock import AsyncMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from kai.webhook import (
    _fmt_issue_comment,
    _fmt_issues,
    _fmt_pull_request,
    _fmt_pull_request_review,
    _fmt_push,
    _handle_github,
    _record_review,
    _review_cooldowns,
    _should_skip_review,
    _strip_markdown,
    _verify_github_signature,
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
        _record_review("owner/repo", 1)
        assert _should_skip_review("owner/repo", 1, 300) is True

    def test_different_pr_not_skipped(self):
        """Cooldown is per-PR, so a different PR number is not skipped."""
        _record_review("owner/repo", 1)
        assert _should_skip_review("owner/repo", 2, 300) is False

    def test_different_repo_not_skipped(self):
        """Cooldown is per-repo+PR, so a different repo is not skipped."""
        _record_review("owner/repo", 1)
        assert _should_skip_review("other/repo", 1, 300) is False

    def test_expired_cooldown_not_skipped(self):
        """After cooldown expires, the PR can be reviewed again."""
        from unittest.mock import patch

        _record_review("owner/repo", 1)
        # Advance time past the cooldown
        import time

        future = time.time() + 301
        with patch("kai.webhook.time.time", return_value=future):
            assert _should_skip_review("owner/repo", 1, 300) is False


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


def _build_test_app(pr_review_enabled: bool = True, cooldown: int = 300) -> web.Application:
    """Build a minimal aiohttp app with _handle_github wired up."""
    app = web.Application()
    app["webhook_secret"] = _TEST_SECRET
    app["pr_review_enabled"] = pr_review_enabled
    app["pr_review_cooldown"] = cooldown
    # Mock bot that records sent messages
    mock_bot = AsyncMock()
    app["telegram_bot"] = mock_bot
    app["chat_id"] = 12345
    app.router.add_post("/webhook/github", _handle_github)
    return app


@pytest.fixture
def _clear_cooldowns():
    """Clear the review cooldown dict before each routing test."""
    _review_cooldowns.clear()
    yield
    _review_cooldowns.clear()


class TestPRReviewRouting:
    """Integration tests for PR review routing in _handle_github."""

    @pytest.mark.asyncio
    async def test_routes_opened_when_enabled(self, _clear_cooldowns):
        """Reviewable PR events are routed to review pipeline, not Telegram."""
        app = _build_test_app(pr_review_enabled=True)
        payload = _make_pr_payload("opened")
        body = json.dumps(payload).encode()
        sig = _sign_payload(payload)

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
            assert data["status"] == "review_triggered"
            # Should NOT have sent a Telegram notification
            app["telegram_bot"].send_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_falls_through_when_disabled(self, _clear_cooldowns):
        """With PR review disabled, opened events go to the notification formatter."""
        app = _build_test_app(pr_review_enabled=False)
        payload = _make_pr_payload("opened")
        body = json.dumps(payload).encode()
        sig = _sign_payload(payload)

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
            app["telegram_bot"].send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_cooldown_skips_recent(self, _clear_cooldowns):
        """Second event for the same PR within cooldown returns review_cooldown."""
        app = _build_test_app(pr_review_enabled=True, cooldown=300)
        payload = _make_pr_payload("synchronize", pr_number=10)
        body = json.dumps(payload).encode()
        sig = _sign_payload(payload)

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
            assert data1["status"] == "review_triggered"

            # Second request hits cooldown
            resp2 = await client.post(
                "/webhook/github",
                data=body,
                headers={
                    "X-GitHub-Event": "pull_request",
                    "X-Hub-Signature-256": sig,
                },
            )
            data2 = await resp2.json()
            assert data2["msg"] == "review_cooldown"

    @pytest.mark.asyncio
    async def test_cooldown_allows_after_expiry(self, _clear_cooldowns):
        """After cooldown expires, the same PR can be reviewed again."""
        from unittest.mock import patch

        app = _build_test_app(pr_review_enabled=True, cooldown=60)
        payload = _make_pr_payload("synchronize", pr_number=10)
        body = json.dumps(payload).encode()
        sig = _sign_payload(payload)

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
            assert (await resp1.json())["status"] == "review_triggered"

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
                assert (await resp2.json())["status"] == "review_triggered"

    @pytest.mark.asyncio
    async def test_closed_still_notifies(self, _clear_cooldowns):
        """Closed PRs go through the standard notification path, not the review pipeline."""
        app = _build_test_app(pr_review_enabled=True)
        payload = _make_pr_payload("closed", merged=False)
        body = json.dumps(payload).encode()
        sig = _sign_payload(payload)

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
            app["telegram_bot"].send_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_synchronize_routed(self, _clear_cooldowns):
        """synchronize events (new push to existing PR) are routed to review."""
        app = _build_test_app(pr_review_enabled=True)
        payload = _make_pr_payload("synchronize", pr_number=99)
        body = json.dumps(payload).encode()
        sig = _sign_payload(payload)

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
            assert data["status"] == "review_triggered"
