"""Tests for webhook.py pure functions and GitHub event formatters."""

import hashlib
import hmac

from kai.webhook import (
    _fmt_issue_comment,
    _fmt_issues,
    _fmt_pull_request,
    _fmt_pull_request_review,
    _fmt_push,
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
        "commits": [
            {"id": f"sha{i:010d}", "message": f"Commit {i}"}
            for i in range(num_commits)
        ],
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
