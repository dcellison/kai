"""Canonical, administrator-scoped webhook and integration diagnostics."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from kai.config import Config
from kai.workshop.domain import PrincipalId
from kai.workshop.store import WorkshopEventStore


class WebhookDiagnosticState(StrEnum):
    """Stable, presentation-neutral diagnostic states."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DISABLED = "disabled"
    UNAVAILABLE = "unavailable"


class WebhookDiagnosticsAccessDenied(RuntimeError):
    """Raised when a non-administrator requests host diagnostics."""


@dataclass(frozen=True, slots=True)
class WebhookEndpointDiagnostic:
    """Redacted readiness for one endpoint class, never one concrete URL."""

    endpoint_class: str
    display_name: str
    state: WebhookDiagnosticState
    description: str

    def as_dict(self) -> dict[str, object]:
        return {
            "endpoint_class": self.endpoint_class,
            "display_name": self.display_name,
            "state": self.state.value,
            "description": self.description,
        }


@dataclass(frozen=True, slots=True)
class WebhookDeliveryHealth:
    """Safe aggregate delivery health for a bounded recent window."""

    state: WebhookDiagnosticState
    window_hours: int
    pending: int
    executing: int
    retrying: int
    succeeded: int
    failed: int

    def as_dict(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "window_hours": self.window_hours,
            "pending": self.pending,
            "executing": self.executing,
            "retrying": self.retrying,
            "succeeded": self.succeeded,
            "failed": self.failed,
        }


@dataclass(frozen=True, slots=True)
class WebhookDiagnosticsSnapshot:
    """One redacted canonical diagnostic snapshot."""

    state: WebhookDiagnosticState
    listener_state: WebhookDiagnosticState
    listener_port: int
    endpoints: tuple[WebhookEndpointDiagnostic, ...]
    deliveries: WebhookDeliveryHealth
    guidance: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "state": self.state.value,
            "listener": {
                "state": self.listener_state.value,
                "port": self.listener_port,
            },
            "endpoints": [endpoint.as_dict() for endpoint in self.endpoints],
            "deliveries": self.deliveries.as_dict(),
            "guidance": list(self.guidance),
        }


class WorkshopWebhookDiagnosticsService:
    """Read-only authority shared by Workshop and Telegram presentations."""

    _DELIVERY_WINDOW_HOURS = 24

    def __init__(self, config: Config, store: WorkshopEventStore) -> None:
        self._store = store
        self._port = config.webhook_port
        self._github_configured = bool(config.github_webhook_secret)
        self._generic_configured = bool(config.generic_webhook_secret)
        self._telegram_webhook_configured = bool(config.telegram_enabled and config.telegram_webhook_url)
        self._workshop_enabled = config.workshop_enabled
        self._listener_probe: Callable[[], bool] | None = None

    def bind_listener_probe(self, probe: Callable[[], bool]) -> None:
        """Bind the host-owned listener probe without coupling core to HTTP globals."""
        self._listener_probe = probe

    async def inspect(self, principal_id: PrincipalId) -> WebhookDiagnosticsSnapshot:
        """Return diagnostics only when the principal is a Workshop administrator."""
        async with self._store.connection.execute(
            "SELECT 1 FROM workshop_memberships WHERE principal_id = ? AND role = 'admin' LIMIT 1",
            (principal_id,),
        ) as cursor:
            if await cursor.fetchone() is None:
                raise WebhookDiagnosticsAccessDenied("Administrator access required")

        listener_running = self._listener_probe() if self._listener_probe is not None else None
        listener_state = (
            WebhookDiagnosticState.UNAVAILABLE
            if listener_running is None
            else WebhookDiagnosticState.HEALTHY
            if listener_running
            else WebhookDiagnosticState.DEGRADED
        )
        deliveries = await self._delivery_health()
        endpoints = self._endpoint_diagnostics(listener_running)

        if listener_state == WebhookDiagnosticState.UNAVAILABLE:
            state = WebhookDiagnosticState.UNAVAILABLE
        elif listener_state == WebhookDiagnosticState.DEGRADED or deliveries.state in {
            WebhookDiagnosticState.DEGRADED,
            WebhookDiagnosticState.UNAVAILABLE,
        }:
            state = WebhookDiagnosticState.DEGRADED
        else:
            state = WebhookDiagnosticState.HEALTHY

        guidance: list[str] = []
        if listener_running is False:
            guidance.append("Restart Kai and verify that the service becomes ready.")
        if listener_running is None:
            guidance.append("Listener health is not available from the running host.")
        if not self._github_configured:
            guidance.append("Configure the named GitHub webhook secret to accept GitHub events.")
        if not self._generic_configured:
            guidance.append("Configure the named generic webhook secret to accept generic events.")
        if self._github_configured or self._generic_configured or self._telegram_webhook_configured:
            guidance.append("Publish external webhook ingress only through an operator-managed HTTPS endpoint.")
        if deliveries.failed:
            guidance.append("Review recent integration or adapter delivery failures in service logs.")

        return WebhookDiagnosticsSnapshot(
            state=state,
            listener_state=listener_state,
            listener_port=self._port,
            endpoints=endpoints,
            deliveries=deliveries,
            guidance=tuple(guidance),
        )

    def _endpoint_diagnostics(
        self,
        listener_running: bool | None,
    ) -> tuple[WebhookEndpointDiagnostic, ...]:
        def configured(
            endpoint_class: str,
            display_name: str,
            enabled: bool,
            enabled_description: str,
            disabled_description: str,
        ) -> WebhookEndpointDiagnostic:
            if not enabled:
                state = WebhookDiagnosticState.DISABLED
                description = disabled_description
            elif listener_running:
                state = WebhookDiagnosticState.HEALTHY
                description = enabled_description
            elif listener_running is None:
                state = WebhookDiagnosticState.UNAVAILABLE
                description = "Configured; listener health is unavailable."
            else:
                state = WebhookDiagnosticState.DEGRADED
                description = "Configured; the shared listener is not running."
            return WebhookEndpointDiagnostic(
                endpoint_class=endpoint_class,
                display_name=display_name,
                state=state,
                description=description,
            )

        return (
            configured(
                "health",
                "Service health",
                True,
                "Available on the shared listener.",
                "Disabled.",
            ),
            configured(
                "internal_api",
                "Principal-bound internal APIs",
                True,
                "Available with per-principal process credentials.",
                "Disabled.",
            ),
            configured(
                "workshop_client",
                "Workshop client",
                self._workshop_enabled,
                "Workshop client routes are enabled.",
                "Workshop is disabled by host policy.",
            ),
            configured(
                "telegram_webhook",
                "Telegram webhook ingress",
                self._telegram_webhook_configured,
                "Telegram webhook delivery is configured.",
                "Telegram uses polling or is disabled.",
            ),
            configured(
                "github_webhook",
                "GitHub webhook ingress",
                self._github_configured,
                "GitHub event ingress is configured.",
                "No GitHub webhook secret is configured.",
            ),
            configured(
                "generic_webhook",
                "Generic webhook ingress",
                self._generic_configured,
                "Generic event ingress is configured.",
                "No generic webhook secret is configured.",
            ),
        )

    async def _delivery_health(self) -> WebhookDeliveryHealth:
        try:
            async with self._store.connection.execute(
                "SELECT "
                "SUM(pending), SUM(executing), SUM(retrying), SUM(succeeded), SUM(failed) "
                "FROM ("
                "SELECT "
                "CASE WHEN status = 'pending' THEN 1 ELSE 0 END AS pending, "
                "CASE WHEN status = 'executing' THEN 1 ELSE 0 END AS executing, "
                "0 AS retrying, "
                "CASE WHEN status = 'succeeded' THEN 1 ELSE 0 END AS succeeded, "
                "CASE WHEN status IN ('failed', 'uncertain') THEN 1 ELSE 0 END AS failed "
                "FROM workshop_github_automation_work "
                "WHERE julianday(updated_at) >= julianday('now', '-24 hours') "
                "UNION ALL "
                "SELECT "
                "CASE WHEN status = 'pending' THEN 1 ELSE 0 END, "
                "CASE WHEN status = 'leased' THEN 1 ELSE 0 END, "
                "CASE WHEN status = 'retry_wait' THEN 1 ELSE 0 END, "
                "CASE WHEN status = 'succeeded' THEN 1 ELSE 0 END, "
                "CASE WHEN status IN ('failed', 'uncertain') THEN 1 ELSE 0 END "
                "FROM delivery_outbox "
                "WHERE julianday(updated_at) >= julianday('now', '-24 hours')"
                ")"
            ) as cursor:
                row = await cursor.fetchone()
        except sqlite3.Error:
            return WebhookDeliveryHealth(
                state=WebhookDiagnosticState.UNAVAILABLE,
                window_hours=self._DELIVERY_WINDOW_HOURS,
                pending=0,
                executing=0,
                retrying=0,
                succeeded=0,
                failed=0,
            )
        counts = tuple(int(value or 0) for value in (row or (0, 0, 0, 0, 0)))
        return WebhookDeliveryHealth(
            state=(WebhookDiagnosticState.DEGRADED if counts[4] else WebhookDiagnosticState.HEALTHY),
            window_hours=self._DELIVERY_WINDOW_HOURS,
            pending=counts[0],
            executing=counts[1],
            retrying=counts[2],
            succeeded=counts[3],
            failed=counts[4],
        )


def render_telegram_webhook_diagnostics(snapshot: WebhookDiagnosticsSnapshot) -> str:
    """Render the canonical snapshot for Telegram without adding transport facts."""
    lines = [
        f"Webhook and integration diagnostics: {snapshot.state.value}",
        f"Shared listener: {snapshot.listener_state.value} (port {snapshot.listener_port})",
        "",
        "Endpoint classes:",
    ]
    lines.extend(
        f"  {endpoint.display_name}: {endpoint.state.value} — {endpoint.description}" for endpoint in snapshot.endpoints
    )
    deliveries = snapshot.deliveries
    lines.extend(
        (
            "",
            f"Recent delivery health ({deliveries.window_hours}h): {deliveries.state.value}",
            f"  pending={deliveries.pending}, executing={deliveries.executing}, "
            f"retrying={deliveries.retrying}, succeeded={deliveries.succeeded}, failed={deliveries.failed}",
        )
    )
    if snapshot.guidance:
        lines.extend(("", "Guidance:"))
        lines.extend(f"  {item}" for item in snapshot.guidance)
    return "\n".join(lines)
