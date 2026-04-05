"""
GitHub API client for webhook management.

Provides low-level functions to check, register, and deregister GitHub
webhooks on repositories. Used by the bot's /github add and /github remove
command handlers to automate webhook lifecycle when the user has stored a
GitHub PAT.

All functions create a fresh aiohttp.ClientSession per call, matching the
project-wide pattern (review.py, services.py, triage.py). This is fine for
low-volume GitHub API calls - a persistent session would save connection
overhead but adds lifecycle complexity for no practical gain here.

HTTP errors are raised as GitHubAPIError with the status code attached,
so callers can branch on specific codes (401, 403, 404, 422).
"""

from __future__ import annotations

import logging

import aiohttp

log = logging.getLogger(__name__)

# GitHub API base URL. Kept as a module constant so tests can verify
# the correct URL is called without hitting the real API.
GITHUB_API_BASE = "https://api.github.com"

# Events to subscribe to when registering a webhook. These match the
# event types that webhook.py knows how to process.
WEBHOOK_EVENTS = [
    "push",
    "pull_request",
    "issues",
    "issue_comment",
    "pull_request_review",
]


class GitHubAPIError(Exception):
    """
    An error returned by the GitHub API.

    Attributes:
        status: HTTP status code from the response (0 for non-HTTP errors
            like network timeouts or missing configuration).
        message: Human-readable error description.
    """

    def __init__(self, status: int, message: str) -> None:
        self.status = status
        super().__init__(message)


def _auth_headers(token: str) -> dict[str, str]:
    """Build authorization headers for the GitHub API."""
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


async def check_webhook_exists(
    owner: str,
    repo: str,
    token: str,
    target_url: str,
) -> tuple[bool, int | None]:
    """
    Check if a webhook pointing to target_url exists on owner/repo.

    Paginates through all hooks (requesting 100 per page) and
    compares each hook's config.url against target_url. The comparison
    is case-insensitive since URLs are case-insensitive in the scheme
    and authority components.

    Returns:
        (exists, hook_id) - hook_id is None if not found.

    Raises:
        GitHubAPIError: On HTTP errors (401 unauthorized, 404 repo not
            found, etc.) or network failures.
    """
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/hooks"
    target_lower = target_url.lower()

    try:
        async with aiohttp.ClientSession() as session:
            page = 1
            while True:
                async with session.get(
                    url,
                    headers=_auth_headers(token),
                    params={"per_page": "100", "page": str(page)},
                ) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        raise GitHubAPIError(resp.status, f"Failed to list hooks: {text}")
                    hooks = await resp.json()

                # Empty page means we've exhausted all hooks
                if not hooks:
                    return (False, None)

                for hook in hooks:
                    hook_url = hook.get("config", {}).get("url", "")
                    if hook_url.lower() == target_lower:
                        return (True, hook["id"])

                # GitHub caps at 100 per page; fewer means last page
                if len(hooks) < 100:
                    return (False, None)
                page += 1

    except aiohttp.ClientError as e:
        raise GitHubAPIError(0, f"Network error checking hooks: {e}") from e


async def register_webhook(
    owner: str,
    repo: str,
    token: str,
    webhook_url: str,
    webhook_secret: str,
) -> int:
    """
    Register a Kai webhook on owner/repo.

    Creates a webhook that sends JSON payloads for push, PR, issue, and
    review events. The secret is used by webhook.py to verify incoming
    payloads via HMAC-SHA256.

    Returns:
        The hook_id on success.

    Raises:
        GitHubAPIError: On failure. Notable status codes:
            - 401: Token is invalid or expired.
            - 403: Token lacks admin:repo_hook scope.
            - 404: Repository not found.
            - 422: Validation failed (often means hook already exists).
    """
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/hooks"
    payload = {
        "name": "web",
        "active": True,
        "events": WEBHOOK_EVENTS,
        "config": {
            "url": webhook_url,
            "content_type": "json",
            "secret": webhook_secret,
        },
    }

    try:
        async with (
            aiohttp.ClientSession() as session,
            session.post(url, headers=_auth_headers(token), json=payload) as resp,
        ):
            if resp.status == 201:
                data = await resp.json()
                return data["id"]

            # 422 often means the webhook already exists (same URL).
            # Treat as success - caller should look up the hook ID.
            if resp.status == 422:
                log.info(
                    "Webhook already exists on %s/%s (422 from GitHub)",
                    owner,
                    repo,
                )
                # Try to find the existing hook so we can return its ID
                exists, hook_id = await check_webhook_exists(
                    owner,
                    repo,
                    token,
                    payload["config"]["url"],
                )
                if exists and hook_id is not None:
                    return hook_id
                # Couldn't find it despite 422 - unusual, but raise
                text = await resp.text()
                raise GitHubAPIError(422, f"Webhook exists but could not find ID: {text}")

            text = await resp.text()
            raise GitHubAPIError(resp.status, f"Failed to create hook: {text}")

    except aiohttp.ClientError as e:
        raise GitHubAPIError(0, f"Network error registering hook: {e}") from e


async def deregister_webhook(
    owner: str,
    repo: str,
    hook_id: int,
    token: str,
) -> None:
    """
    Delete a webhook from owner/repo by hook_id.

    No-op if the hook is already deleted (404 treated as success, since
    the goal is "this hook should not exist" and it doesn't).

    Raises:
        GitHubAPIError: On errors other than 404.
    """
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/hooks/{hook_id}"

    try:
        async with (
            aiohttp.ClientSession() as session,
            session.delete(url, headers=_auth_headers(token)) as resp,
        ):
            # 204 = deleted, 404 = already gone. Both are success.
            if resp.status in (204, 404):
                return
            text = await resp.text()
            raise GitHubAPIError(resp.status, f"Failed to delete hook: {text}")

    except aiohttp.ClientError as e:
        raise GitHubAPIError(0, f"Network error deleting hook: {e}") from e
