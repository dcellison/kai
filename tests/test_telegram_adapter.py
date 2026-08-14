"""Lifecycle contracts for the explicit Telegram adapter boundary."""

from types import SimpleNamespace

import pytest

import kai.telegram_adapter as adapter_module
from kai.telegram_adapter import TelegramAdapter, TelegramAdapterState


class _FakeBot:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def set_my_commands(self, _commands) -> None:
        self.events.append("commands:set")


class _FakeUpdater:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def start_polling(self, *, allowed_updates) -> None:
        assert allowed_updates == ["message", "callback_query"]
        self.events.append("polling:start")

    async def stop(self) -> None:
        self.events.append("polling:stop")


class _FakeApplication:
    def __init__(self, events: list[str], *, polling: bool) -> None:
        self.events = events
        self.bot = _FakeBot(events)
        self.updater = _FakeUpdater(events) if polling else None

    async def initialize(self) -> None:
        self.events.append("application:initialize")

    async def start(self) -> None:
        self.events.append("application:start")

    async def stop(self) -> None:
        self.events.append("application:stop")

    async def shutdown(self) -> None:
        self.events.append("application:shutdown")


class _FakeDelivery:
    def __init__(self, events: list[str], name: str) -> None:
        self.events = events
        self.name = name
        self.ready = True

    async def wait(self) -> None:
        self.events.append(f"{self.name}:wait")

    async def stop(self) -> None:
        self.events.append(f"{self.name}:stop")
        self.ready = False


@pytest.fixture
def adapter_dependencies(monkeypatch):
    events: list[str] = []
    application: _FakeApplication | None = None

    def fake_create_bot(_config, *, use_webhook, core_services):
        nonlocal application
        assert core_services is not None
        events.append("application:create")
        application = _FakeApplication(events, polling=not use_webhook)
        return application

    class FakeConversationDelivery:
        @classmethod
        async def open_and_start(cls, _path, _bot, *, authority_epoch_id):
            assert authority_epoch_id == "dae_test"
            events.append("conversation:start")
            return _FakeDelivery(events, "conversation")

    class FakeNotificationDelivery:
        @classmethod
        async def open_and_start(cls, _path, _bot):
            events.append("notification:start")
            return _FakeDelivery(events, "notification")

    async def fake_init_jobs(app) -> None:
        assert app is application
        events.append("jobs:start")

    monkeypatch.setattr(adapter_module, "KaiTelegramApplication", _FakeApplication)
    monkeypatch.setattr(adapter_module, "create_bot", fake_create_bot)
    monkeypatch.setattr(
        adapter_module,
        "WorkshopTelegramConversationDeliveryService",
        FakeConversationDelivery,
    )
    monkeypatch.setattr(
        adapter_module,
        "WorkshopTelegramNotificationService",
        FakeNotificationDelivery,
    )
    monkeypatch.setattr(adapter_module.cron, "init_jobs", fake_init_jobs)
    return events


def _adapter(*, use_webhook: bool) -> TelegramAdapter:
    config = SimpleNamespace(session_db_path="/tmp/kai.db")
    core_services = SimpleNamespace(delivery_authority_epoch=SimpleNamespace(epoch_id="dae_test"))
    return TelegramAdapter(  # type: ignore[arg-type]
        config,
        core_services,
        use_webhook=use_webhook,
    )


async def test_webhook_adapter_owns_application_and_delivery_lifecycle(
    adapter_dependencies,
) -> None:
    adapter = _adapter(use_webhook=True)

    await adapter.start()
    assert adapter.readiness.state == TelegramAdapterState.STARTING
    assert adapter.readiness.ready is False
    assert adapter.readiness.application is True
    assert adapter.readiness.ingress is False

    await adapter.activate_ingress()
    assert adapter.readiness.as_dict() == {
        "status": "ready",
        "ready": True,
        "components": {
            "application": True,
            "ingress": True,
            "conversation_delivery": True,
            "notification_delivery": True,
        },
    }

    await adapter.stop()
    assert adapter.readiness.state == TelegramAdapterState.STOPPED
    assert adapter_dependencies == [
        "application:create",
        "application:initialize",
        "application:start",
        "commands:set",
        "jobs:start",
        "conversation:start",
        "notification:start",
        "notification:stop",
        "conversation:stop",
        "application:stop",
        "application:shutdown",
    ]


async def test_polling_ingress_is_started_and_stopped_inside_adapter(
    adapter_dependencies,
) -> None:
    adapter = _adapter(use_webhook=False)

    await adapter.start()
    await adapter.activate_ingress()
    await adapter.stop()

    assert "polling:start" in adapter_dependencies
    assert adapter_dependencies.index("polling:stop") < adapter_dependencies.index("notification:stop")


async def test_adapter_rejects_supervision_before_ingress(adapter_dependencies) -> None:
    adapter = _adapter(use_webhook=True)
    await adapter.start()

    with pytest.raises(RuntimeError, match="cannot be supervised while starting"):
        await adapter.wait()

    await adapter.stop()
