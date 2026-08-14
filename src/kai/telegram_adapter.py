"""Explicit lifecycle boundary for Kai's Telegram adapter."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import StrEnum

from telegram import Bot, BotCommand
from telegram.error import NetworkError

from kai import cron
from kai.application_host import KaiCoreServices
from kai.bot import create_bot
from kai.config import Config
from kai.telegram_context import KaiTelegramApplication
from kai.workshop.telegram_delivery_runtime import (
    WorkshopTelegramConversationDeliveryService,
    WorkshopTelegramNotificationService,
)

log = logging.getLogger(__name__)


class TelegramAdapterState(StrEnum):
    """Observable lifecycle state for the Telegram adapter."""

    NEW = "new"
    STARTING = "starting"
    READY = "ready"
    DRAINING = "draining"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class TelegramAdapterReadiness:
    """Non-sensitive readiness for Telegram-owned components."""

    state: TelegramAdapterState
    application: bool
    ingress: bool
    conversation_delivery: bool
    notification_delivery: bool

    @property
    def ready(self) -> bool:
        return self.state == TelegramAdapterState.READY and all(
            (
                self.application,
                self.ingress,
                self.conversation_delivery,
                self.notification_delivery,
            )
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.state.value,
            "ready": self.ready,
            "components": {
                "application": self.application,
                "ingress": self.ingress,
                "conversation_delivery": self.conversation_delivery,
                "notification_delivery": self.notification_delivery,
            },
        }


_TELEGRAM_COMMANDS = (
    BotCommand("stop", "Interrupt current response"),
    BotCommand("new", "Start a fresh session"),
    BotCommand("models", "Choose a model"),
    BotCommand("model", "Switch model directly"),
    BotCommand("settings", "Show or change your settings"),
    BotCommand("memory", "Browse, search, and manage remembered facts"),
    BotCommand("workspace", "Switch working directory"),
    BotCommand("workspaces", "List recent workspaces"),
    BotCommand("github", "Show GitHub settings"),
    BotCommand("voice", "Toggle voice or set voice name"),
    BotCommand("voices", "Choose a voice"),
    BotCommand("stats", "Show session info and cost"),
    BotCommand("job", "Manage scheduled jobs"),
    BotCommand("webhooks", "Show webhook server status"),
    BotCommand("help", "Show available commands"),
)


class TelegramAdapter:
    """Own Telegram application, polling, scheduler bridge, and delivery workers.

    HTTP webhook registration remains in the HTTP adapter during this
    milestone. The core host starts, supervises, reports, and stops this
    boundary without importing a Telegram package itself.
    """

    def __init__(
        self,
        config: Config,
        core_services: KaiCoreServices,
        *,
        use_webhook: bool,
    ) -> None:
        self._config = config
        self._core_services = core_services
        self._use_webhook = use_webhook
        self._state = TelegramAdapterState.NEW
        self._application: KaiTelegramApplication | None = None
        self._application_initialized = False
        self._application_started = False
        self._ingress_started = False
        self._conversation_delivery: WorkshopTelegramConversationDeliveryService | None = None
        self._notification_delivery: WorkshopTelegramNotificationService | None = None

    @property
    def application(self) -> KaiTelegramApplication:
        application = self._application
        if application is None:
            raise RuntimeError("Telegram application is not started")
        return application

    @property
    def bot(self) -> Bot:
        return self.application.bot

    @property
    def notification_delivery(self) -> WorkshopTelegramNotificationService:
        service = self._notification_delivery
        if service is None:
            raise RuntimeError("Telegram notification delivery is not started")
        return service

    @property
    def readiness(self) -> TelegramAdapterReadiness:
        return TelegramAdapterReadiness(
            state=self._state,
            application=self._application_started,
            ingress=self._ingress_started,
            conversation_delivery=(self._conversation_delivery is not None and self._conversation_delivery.ready),
            notification_delivery=(self._notification_delivery is not None and self._notification_delivery.ready),
        )

    async def start(self) -> None:
        if self._state != TelegramAdapterState.NEW:
            raise RuntimeError(f"Telegram adapter cannot start from {self._state.value}")
        self._state = TelegramAdapterState.STARTING
        try:
            application = create_bot(
                self._config,
                use_webhook=self._use_webhook,
                core_services=self._core_services,
            )
            if not isinstance(application, KaiTelegramApplication):
                raise RuntimeError("Telegram factory returned an untyped application")
            self._application = application

            for attempt in range(1, 13):
                try:
                    await application.initialize()
                    self._application_initialized = True
                    break
                except NetworkError:
                    if attempt == 12:
                        raise
                    wait = min(30, 2**attempt)
                    log.warning(
                        "Telegram network not ready (attempt %d/12), retrying in %ds…",
                        attempt,
                        wait,
                    )
                    await asyncio.sleep(wait)

            await application.start()
            self._application_started = True
            await application.bot.set_my_commands(_TELEGRAM_COMMANDS)
            await cron.init_jobs(application)

            self._conversation_delivery = await WorkshopTelegramConversationDeliveryService.open_and_start(
                self._config.session_db_path,
                application.bot,
                authority_epoch_id=self._core_services.delivery_authority_epoch.epoch_id,
            )
            log.info("Workshop Telegram conversation delivery is ready")
            self._notification_delivery = await WorkshopTelegramNotificationService.open_and_start(
                self._config.session_db_path,
                application.bot,
            )
            log.info("Workshop Telegram notification delivery is ready")
        except BaseException:
            self._state = TelegramAdapterState.FAILED
            await self._stop_components()
            raise

    async def activate_ingress(self) -> None:
        """Mark webhook ingress ready or start the polling updater.

        In webhook mode the HTTP adapter has already registered Telegram's URL
        before calling this method. The adapter therefore remains STARTING for
        the brief handoff between confirmed registration and this readiness
        transition, even though Telegram may begin delivering during it.
        """
        if self._state != TelegramAdapterState.STARTING:
            raise RuntimeError(f"Telegram ingress cannot start while {self._state.value}")
        try:
            if not self._use_webhook:
                updater = self.application.updater
                if updater is None:
                    raise RuntimeError("Telegram polling mode has no updater")
                await updater.start_polling(allowed_updates=["message", "callback_query"])
                log.info("Telegram polling started")
            self._ingress_started = True
            self._state = TelegramAdapterState.READY
        except BaseException:
            self._state = TelegramAdapterState.FAILED
            raise

    async def wait(self) -> None:
        if self._state != TelegramAdapterState.READY:
            raise RuntimeError(f"Telegram adapter cannot be supervised while {self._state.value}")
        conversation_delivery = self._conversation_delivery
        notification_delivery = self._notification_delivery
        if conversation_delivery is None or notification_delivery is None:
            raise RuntimeError("Telegram delivery services are unavailable")
        await asyncio.gather(
            conversation_delivery.wait(),
            notification_delivery.wait(),
        )

    async def stop(self) -> None:
        if self._state in {TelegramAdapterState.NEW, TelegramAdapterState.STOPPED}:
            self._state = TelegramAdapterState.STOPPED
            return
        self._state = TelegramAdapterState.DRAINING
        errors = await self._stop_components()
        self._state = TelegramAdapterState.STOPPED
        if errors:
            raise ExceptionGroup("Telegram adapter shutdown failed", errors)

    async def _stop_components(self) -> list[Exception]:
        errors: list[Exception] = []
        application = self._application
        if self._ingress_started and not self._use_webhook and application is not None:
            updater = application.updater
            if updater is not None:
                try:
                    await updater.stop()
                except Exception as exc:
                    errors.append(exc)
        self._ingress_started = False

        for service in (self._notification_delivery, self._conversation_delivery):
            if service is not None:
                try:
                    await service.stop()
                except Exception as exc:
                    errors.append(exc)
        self._notification_delivery = None
        self._conversation_delivery = None

        if application is not None and self._application_started:
            try:
                await application.stop()
            except Exception as exc:
                errors.append(exc)
        self._application_started = False
        if application is not None and self._application_initialized:
            try:
                await application.shutdown()
            except Exception as exc:
                errors.append(exc)
        self._application_initialized = False
        self._application = None
        return errors
