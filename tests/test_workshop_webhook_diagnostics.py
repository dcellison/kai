"""Canonical webhook diagnostics contracts shared by Workshop and Telegram."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kai.config import Config
from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.domain import PrincipalId
from kai.workshop.store import WorkshopEventStore
from kai.workshop.webhook_diagnostics import (
    WebhookDiagnosticsAccessDenied,
    WebhookDiagnosticState,
    WorkshopWebhookDiagnosticsService,
    render_telegram_webhook_diagnostics,
)
from tests.workshop_profiles import profile_id


async def _seed(path: Path) -> tuple[WorkshopEventStore, PrincipalId, PrincipalId]:
    store = await WorkshopEventStore.open(path)
    await bootstrap_default_workshop(
        store,
        (
            BootstrapHuman("Daniel", "admin", "telegram", "101", "101", profile_id(101)),
            BootstrapHuman("Scott", "member", "telegram", "202", "202", profile_id(202)),
        ),
    )
    async with store.connection.execute(
        "SELECT external_subject, principal_id FROM external_identities "
        "WHERE provider = 'telegram' ORDER BY external_subject"
    ) as cursor:
        principals = {str(row[0]): PrincipalId(str(row[1])) for row in await cursor.fetchall()}
    return store, principals["101"], principals["202"]


def _config(**overrides: object) -> Config:
    values: dict[str, object] = {
        "telegram_bot_token": "telegram-secret-token",
        "allowed_user_ids": {101, 202},
        "telegram_webhook_url": "https://private.example/telegram-secret-path",
        "github_webhook_secret": "github-secret",
        "generic_webhook_secret": "generic-secret",
        "webhook_port": 8123,
    }
    values.update(overrides)
    return Config(**values)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_admin_receives_redacted_canonical_snapshot(tmp_path: Path) -> None:
    store, admin_id, _member_id = await _seed(tmp_path / "kai.db")
    service = WorkshopWebhookDiagnosticsService(_config(), store)
    service.bind_listener_probe(lambda: True)
    try:
        snapshot = await service.inspect(admin_id)
        serialized = json.dumps(snapshot.as_dict())
        telegram = render_telegram_webhook_diagnostics(snapshot)

        assert snapshot.state == WebhookDiagnosticState.HEALTHY
        assert snapshot.listener_state == WebhookDiagnosticState.HEALTHY
        assert snapshot.listener_port == 8123
        assert {endpoint.endpoint_class for endpoint in snapshot.endpoints} == {
            "health",
            "internal_api",
            "workshop_client",
            "telegram_webhook",
            "github_webhook",
            "generic_webhook",
        }
        assert snapshot.deliveries.state == WebhookDiagnosticState.HEALTHY
        assert "telegram-secret-token" not in serialized
        assert "telegram-secret-path" not in serialized
        assert "github-secret" not in serialized
        assert "generic-secret" not in serialized
        assert "/Users/" not in serialized
        assert "Shared listener: healthy (port 8123)" in telegram
        assert "GitHub webhook ingress: healthy" in telegram
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_member_is_denied_host_diagnostics(tmp_path: Path) -> None:
    store, _admin_id, member_id = await _seed(tmp_path / "kai.db")
    service = WorkshopWebhookDiagnosticsService(_config(), store)
    try:
        with pytest.raises(WebhookDiagnosticsAccessDenied, match="Administrator access required"):
            await service.inspect(member_id)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_listener_and_endpoint_states_are_stable(tmp_path: Path) -> None:
    store, admin_id, _member_id = await _seed(tmp_path / "kai.db")
    try:
        unavailable = WorkshopWebhookDiagnosticsService(
            _config(
                telegram_webhook_url=None,
                github_webhook_secret=None,
                generic_webhook_secret=None,
            ),
            store,
        )
        unavailable_snapshot = await unavailable.inspect(admin_id)
        assert unavailable_snapshot.state == WebhookDiagnosticState.UNAVAILABLE
        endpoint_states = {endpoint.endpoint_class: endpoint.state for endpoint in unavailable_snapshot.endpoints}
        assert endpoint_states["telegram_webhook"] == WebhookDiagnosticState.DISABLED
        assert endpoint_states["github_webhook"] == WebhookDiagnosticState.DISABLED
        assert endpoint_states["generic_webhook"] == WebhookDiagnosticState.DISABLED

        degraded = WorkshopWebhookDiagnosticsService(_config(), store)
        degraded.bind_listener_probe(lambda: False)
        degraded_snapshot = await degraded.inspect(admin_id)
        assert degraded_snapshot.state == WebhookDiagnosticState.DEGRADED
        assert degraded_snapshot.listener_state == WebhookDiagnosticState.DEGRADED
        assert all(
            endpoint.state in {WebhookDiagnosticState.DEGRADED, WebhookDiagnosticState.DISABLED}
            for endpoint in degraded_snapshot.endpoints
        )
    finally:
        await store.close()
