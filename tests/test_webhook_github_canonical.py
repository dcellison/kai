"""HTTP contracts for canonical GitHub webhook routing."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from aiohttp.test_utils import TestClient, TestServer

from kai.webhook import (
    TELEGRAM_BOT_KEY,
    WORKSHOP_GITHUB_AUTOMATION_KEY,
    WORKSHOP_INTEGRATION_NOTIFICATIONS_KEY,
)
from kai.workshop.domain import ChannelId, PrincipalId
from kai.workshop.github_automation import GitHubSubscriptionRoute
from tests.test_webhook import (
    _build_test_app,
    _make_issue_payload,
    _make_pr_payload,
    _sign_payload,
)
from tests.workshop_profiles import profile_id


def _route(*, review: bool = False, triage: bool = False, authorized: bool = True):
    return GitHubSubscriptionRoute(
        principal_id=PrincipalId("prn_" + "1" * 32),
        execution_channel_id=ChannelId("chn_" + "2" * 32),
        notification_channel_id=ChannelId("chn_" + "3" * 32),
        runtime_profile_id=profile_id(101),
        pr_review_enabled=review,
        issue_triage_enabled=triage,
        operations_authorized=authorized,
    )


def _canonical_app(route):
    app = _build_test_app()
    automation = MagicMock()
    automation.routes_for_repository = AsyncMock(return_value=(route,))
    automation.enqueue = AsyncMock(return_value=SimpleNamespace(inserted=True))
    notifications = MagicMock()
    notifications.record_for_channel = AsyncMock(return_value=SimpleNamespace())
    app[WORKSHOP_GITHUB_AUTOMATION_KEY] = automation
    app[WORKSHOP_INTEGRATION_NOTIFICATIONS_KEY] = notifications
    return app, automation, notifications


async def _post(client, payload, event_type, *, delivery: str | None = "delivery-1"):
    headers = {
        "X-GitHub-Event": event_type,
        "X-Hub-Signature-256": _sign_payload(payload),
    }
    if delivery is not None:
        headers["X-GitHub-Delivery"] = delivery
    return await client.post("/webhook/github", data=json.dumps(payload).encode(), headers=headers)


async def test_reviewable_delivery_enqueues_durable_canonical_work():
    app, automation, notifications = _canonical_app(_route(review=True))
    with patch("kai.webhook._resolve_local_repo", new=AsyncMock(return_value="/resolved/kai")):
        async with TestClient(TestServer(app)) as client:
            response = await _post(client, _make_pr_payload("opened"), "pull_request")

    assert response.status == 200
    automation.enqueue.assert_awaited_once()
    assert automation.enqueue.call_args.kwargs["kind"] == "pr_review"
    assert automation.enqueue.call_args.kwargs["delivery_id"] == "delivery-1"
    assert automation.enqueue.call_args.kwargs["local_repo_path"] == "/resolved/kai"
    notifications.record_for_channel.assert_not_awaited()
    app[TELEGRAM_BOT_KEY].send_message.assert_not_awaited()


async def test_issue_triage_delivery_enqueues_durable_canonical_work():
    app, automation, notifications = _canonical_app(_route(triage=True))
    async with TestClient(TestServer(app)) as client:
        response = await _post(client, _make_issue_payload("opened"), "issues")

    assert response.status == 200
    assert automation.enqueue.call_args.kwargs["kind"] == "issue_triage"
    notifications.record_for_channel.assert_not_awaited()


async def test_unprivileged_subscription_cannot_trigger_mutation_and_gets_notification():
    route = _route(review=True, authorized=False)
    app, automation, notifications = _canonical_app(route)
    async with TestClient(TestServer(app)) as client:
        response = await _post(client, _make_pr_payload("opened"), "pull_request")

    assert response.status == 200
    automation.enqueue.assert_not_awaited()
    notifications.record_for_channel.assert_awaited_once()
    assert notifications.record_for_channel.call_args.args[1] == route.notification_channel_id
    app[TELEGRAM_BOT_KEY].send_message.assert_not_awaited()


async def test_standard_delivery_records_canonical_notification_without_telegram():
    app, _automation, notifications = _canonical_app(_route())
    del app[TELEGRAM_BOT_KEY]
    async with TestClient(TestServer(app)) as client:
        response = await _post(client, _make_pr_payload("closed"), "pull_request")

    assert response.status == 200
    notifications.record_for_channel.assert_awaited_once()


async def test_missing_delivery_identity_fails_before_routing():
    app, automation, notifications = _canonical_app(_route(review=True))
    async with TestClient(TestServer(app)) as client:
        response = await _post(
            client,
            _make_pr_payload("opened"),
            "pull_request",
            delivery=None,
        )

    assert response.status == 400
    automation.enqueue.assert_not_awaited()
    notifications.record_for_channel.assert_not_awaited()
