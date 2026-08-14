"""Lifecycle contracts for the explicit HTTP adapter boundary."""

import asyncio
from types import SimpleNamespace

import pytest

import kai.http_adapter as adapter_module
from kai.http_adapter import HttpAdapter, HttpAdapterState


@pytest.fixture
def http_dependencies(monkeypatch):
    events: list[str] = []
    state = {"loopback": False, "lan": False}

    async def fake_start(
        application,
        config,
        *,
        core_host,
        core_services,
        github_notifications,
    ) -> None:
        assert application == "telegram-application"
        assert config.workshop_lan_host in {None, "10.0.0.36"}
        assert core_host == "core-host"
        assert core_services == "core-services"
        assert github_notifications == "notification-delivery"
        events.append("http:start")
        state["loopback"] = True
        state["lan"] = True

    async def fake_stop() -> None:
        events.append("http:stop")
        state["loopback"] = False
        state["lan"] = False

    monkeypatch.setattr(adapter_module.webhook, "start", fake_start)
    monkeypatch.setattr(adapter_module.webhook, "stop", fake_stop)
    monkeypatch.setattr(
        adapter_module.webhook,
        "is_running",
        lambda: state["loopback"],
    )
    monkeypatch.setattr(
        adapter_module.webhook,
        "is_workshop_lan_running",
        lambda: state["lan"],
    )
    return events


def _adapter(*, workshop_lan_host: str | None = "10.0.0.36") -> HttpAdapter:
    config = SimpleNamespace(workshop_lan_host=workshop_lan_host)
    telegram = SimpleNamespace(
        application="telegram-application",
        notification_delivery="notification-delivery",
    )
    return HttpAdapter(  # type: ignore[arg-type]
        config,
        "core-host",  # type: ignore[arg-type]
        "core-services",  # type: ignore[arg-type]
        telegram,  # type: ignore[arg-type]
    )


async def test_http_adapter_owns_listener_lifecycle_and_readiness(
    http_dependencies,
) -> None:
    adapter = _adapter()

    await adapter.start()

    assert adapter.readiness.as_dict() == {
        "status": "ready",
        "ready": True,
        "components": {
            "loopback_listener": True,
            "workshop_lan_listener": True,
        },
    }
    wait_task = asyncio.create_task(adapter.wait())
    await asyncio.sleep(0)
    assert wait_task.done() is False

    await adapter.stop()
    await wait_task

    assert adapter.readiness.state == HttpAdapterState.STOPPED
    assert http_dependencies == ["http:start", "http:stop"]


async def test_http_adapter_reports_unconfigured_lan_listener_as_not_applicable(
    http_dependencies,
) -> None:
    adapter = _adapter(workshop_lan_host=None)

    await adapter.start()

    assert adapter.readiness.ready is True
    assert adapter.readiness.workshop_lan_listener is None
    await adapter.stop()


async def test_http_adapter_rejects_supervision_before_start(http_dependencies) -> None:
    adapter = _adapter()

    with pytest.raises(RuntimeError, match="cannot be supervised while new"):
        await adapter.wait()

    await adapter.stop()


async def test_http_adapter_cleans_partial_start_and_reports_failure(
    http_dependencies,
    monkeypatch,
) -> None:
    async def failing_start(*_args, **_kwargs) -> None:
        http_dependencies.append("http:start:failed")
        raise RuntimeError("listener startup failed")

    monkeypatch.setattr(adapter_module.webhook, "start", failing_start)
    adapter = _adapter()

    with pytest.raises(RuntimeError, match="listener startup failed"):
        await adapter.start()

    assert adapter.readiness.state == HttpAdapterState.FAILED
    assert adapter.readiness.ready is False
    assert http_dependencies == ["http:start:failed", "http:stop"]

    # Host attachment cleanup may call stop after start already cleaned the
    # partial listener state. The second webhook stop remains safe.
    await adapter.stop()
    assert adapter.readiness.state == HttpAdapterState.STOPPED
    assert http_dependencies == ["http:start:failed", "http:stop", "http:stop"]


async def test_http_adapter_propagates_stop_failure_and_unblocks_supervision(
    http_dependencies,
    monkeypatch,
) -> None:
    adapter = _adapter()
    await adapter.start()
    wait_task = asyncio.create_task(adapter.wait())
    await asyncio.sleep(0)

    async def failing_stop() -> None:
        http_dependencies.append("http:stop:failed")
        raise RuntimeError("listener shutdown failed")

    monkeypatch.setattr(adapter_module.webhook, "stop", failing_stop)

    with pytest.raises(RuntimeError, match="listener shutdown failed"):
        await adapter.stop()

    await wait_task
    assert adapter.readiness.state == HttpAdapterState.FAILED
    assert adapter.readiness.ready is False
    assert http_dependencies == ["http:start", "http:stop:failed"]
