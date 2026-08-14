"""Typed dependency access for Telegram adapter callbacks."""

from __future__ import annotations

from typing import Any, cast

from telegram.ext import Application, ContextTypes

from kai.application_host import KaiCoreServices


class KaiTelegramApplication(Application):
    """Telegram application carrying an explicit typed core boundary."""

    __slots__ = ("core_services",)

    def __init__(self, *, core_services: KaiCoreServices, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.core_services = core_services


def get_core_services(context: ContextTypes.DEFAULT_TYPE) -> KaiCoreServices:
    """Resolve core services without Telegram's untyped bot_data map."""
    application = context.application
    if not isinstance(application, KaiTelegramApplication):
        raise RuntimeError("Telegram adapter core services are unavailable")
    return cast(KaiCoreServices, application.core_services)
