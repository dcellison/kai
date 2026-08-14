"""Explicit lifecycle boundary for Kai's mixed HTTP and Workshop listeners."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import StrEnum

from kai import webhook
from kai.application_host import KaiApplicationHost, KaiCoreServices
from kai.config import Config
from kai.telegram_adapter import TelegramAdapter

log = logging.getLogger(__name__)


class HttpAdapterState(StrEnum):
    """Observable lifecycle state for the HTTP adapter."""

    NEW = "new"
    STARTING = "starting"
    READY = "ready"
    DRAINING = "draining"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class HttpAdapterReadiness:
    """Non-sensitive readiness for HTTP-owned listener components."""

    state: HttpAdapterState
    loopback_listener: bool
    workshop_lan_listener: bool | None

    @property
    def ready(self) -> bool:
        return (
            self.state == HttpAdapterState.READY and self.loopback_listener and self.workshop_lan_listener is not False
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.state.value,
            "ready": self.ready,
            "components": {
                "loopback_listener": self.loopback_listener,
                "workshop_lan_listener": self.workshop_lan_listener,
            },
        }


class HttpAdapter:
    """Own HTTP listener startup, readiness, supervision, and shutdown.

    Telegram-owned routes and dependencies are installed only when a started
    Telegram adapter is supplied.  The loopback host and Workshop client can
    therefore run without constructing any Telegram application.
    """

    def __init__(
        self,
        config: Config,
        core_host: KaiApplicationHost,
        core_services: KaiCoreServices,
        telegram_adapter: TelegramAdapter | None,
    ) -> None:
        self._config = config
        self._core_host = core_host
        self._core_services = core_services
        self._telegram_adapter = telegram_adapter
        self._state = HttpAdapterState.NEW
        self._stop_event = asyncio.Event()

    @property
    def readiness(self) -> HttpAdapterReadiness:
        lan_configured = bool(self._config.workshop_lan_host)
        return HttpAdapterReadiness(
            state=self._state,
            loopback_listener=webhook.is_running(),
            workshop_lan_listener=(webhook.is_workshop_lan_running() if lan_configured else None),
        )

    async def start(self) -> None:
        if self._state != HttpAdapterState.NEW:
            raise RuntimeError(f"HTTP adapter cannot start from {self._state.value}")
        self._state = HttpAdapterState.STARTING
        try:
            await webhook.start(
                self._telegram_adapter.application if self._telegram_adapter else None,
                self._config,
                core_host=self._core_host,
                core_services=self._core_services,
                github_notifications=(self._telegram_adapter.notification_delivery if self._telegram_adapter else None),
                workshop_enabled=self._config.workshop_enabled,
            )
        except BaseException:
            self._state = HttpAdapterState.FAILED
            try:
                await webhook.stop()
            except Exception:
                log.exception("HTTP adapter cleanup failed after startup error")
            raise
        self._state = HttpAdapterState.READY

    async def wait(self) -> None:
        if self._state != HttpAdapterState.READY:
            raise RuntimeError(f"HTTP adapter cannot be supervised while {self._state.value}")
        await self._stop_event.wait()

    async def stop(self) -> None:
        if self._state in {HttpAdapterState.NEW, HttpAdapterState.STOPPED}:
            self._state = HttpAdapterState.STOPPED
            self._stop_event.set()
            return
        self._state = HttpAdapterState.DRAINING
        try:
            await webhook.stop()
        except BaseException:
            self._state = HttpAdapterState.FAILED
            raise
        finally:
            self._stop_event.set()
        self._state = HttpAdapterState.STOPPED
