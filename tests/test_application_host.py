"""Transport-neutral construction and lifecycle tests for Kai's core host."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

import kai.application_host as host_module
from kai.application_host import KaiApplicationHost, KaiApplicationState
from kai.config import Config
from kai.workshop.bootstrap import (
    BootstrapHuman,
    bootstrap_default_workshop,
    bootstrap_human_principal_id,
)
from kai.workshop.conversation_commands import WorkshopConversationCommandService
from kai.workshop.delivery_authority import WorkshopConversationDeliveryAuthority
from kai.workshop.domain import PrincipalId, RunExecutionOwnerId
from kai.workshop.inbound import InboundMessage
from kai.workshop.run_execution_authority import (
    RunAttemptStatus,
    RunExecutionSelection,
    WorkshopRunExecutionAuthority,
)
from kai.workshop.run_lifecycle import RunStatus, WorkshopRunLifecycle
from kai.workshop.storage_namespaces import (
    WorkshopPrincipalStorageNamespace,
    WorkshopPrincipalStorageRegistry,
)
from kai.workshop.store import WorkshopEventStore
from tests.workshop_profiles import profile_id, profile_registry


class _FakePool:
    def __init__(self, *, events: list[str], **_kwargs) -> None:
        self.events = events

    def start(self) -> None:
        self.events.append("pool:start")

    async def shutdown(self) -> None:
        self.events.append("pool:stop")


class _FakeExecution:
    ready = True

    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def wait(self) -> None:
        self.events.append("execution:wait")

    async def stop(self) -> None:
        self.events.append("execution:stop")


class _FakeExecutionFactory:
    events: list[str]

    @classmethod
    async def open_and_start(cls, *_args, **_kwargs):
        cls.events.append("execution:start")
        return _FakeExecution(cls.events)


class _FakeDeliveryAuthority:
    def __init__(self, _store) -> None:
        self.events = _FakeExecutionFactory.events

    async def activate(self):
        self.events.append("authority:activate")
        return SimpleNamespace(epoch=SimpleNamespace(epoch_id="dae_" + "1" * 32))


class _FakeStore:
    events: list[str]

    @classmethod
    async def open(cls, _path: Path):
        cls.events.append("store:open")
        return cls(cls.events)

    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def close(self) -> None:
        self.events.append("store:close")


class _FakeClientCommands:
    ready = False

    def __init__(self, _execution, _compatibility) -> None:
        self.events = _FakeExecutionFactory.events

    async def start(self) -> None:
        self.events.append("client:start")
        self.ready = True

    async def stop(self) -> None:
        self.events.append("client:stop")
        self.ready = False


@pytest.fixture
def host_dependencies(monkeypatch):
    events: list[str] = []
    _FakeExecutionFactory.events = events
    _FakeStore.events = events
    monkeypatch.setattr(
        host_module,
        "SubprocessPool",
        lambda **kwargs: _FakePool(events=events, **kwargs),
    )
    monkeypatch.setattr(host_module, "WorkshopRuntimePool", lambda pool, profiles: (pool, profiles))
    monkeypatch.setattr(host_module, "WorkshopConversationRunService", lambda pool, resolver: (pool, resolver))
    monkeypatch.setattr(host_module, "WorkshopPrivateTextExecutionService", _FakeExecutionFactory)
    monkeypatch.setattr(host_module, "WorkshopEventStore", _FakeStore)
    monkeypatch.setattr(host_module, "WorkshopConversationDeliveryAuthority", _FakeDeliveryAuthority)
    monkeypatch.setattr(host_module, "WorkshopClientCommandExecutor", _FakeClientCommands)
    monkeypatch.setattr(host_module, "WorkshopCompatibilityStateWriter", lambda config, pool: (config, pool))
    return events


def _host() -> KaiApplicationHost:
    return KaiApplicationHost(
        config=SimpleNamespace(session_db_path=Path("/tmp/kai-test.db")),  # type: ignore[arg-type]
        runtime_profiles=SimpleNamespace(),  # type: ignore[arg-type]
        principal_storage=SimpleNamespace(),  # type: ignore[arg-type]
        services_info=[],
        registered_backend_ids=frozenset({"codex"}),
    )


async def test_core_starts_and_stops_without_a_telegram_application(host_dependencies) -> None:
    host = _host()

    services = await host.start()

    assert host.readiness.ready is True
    assert host.readiness.as_dict() == {
        "status": "ready",
        "ready": True,
        "components": {
            "runtime": True,
            "executor": True,
            "client_api": True,
            "store": True,
        },
    }
    assert services.subprocess_pool is not None
    assert host_dependencies == [
        "pool:start",
        "store:open",
        "authority:activate",
        "execution:start",
        "client:start",
    ]

    await host.wait()
    await host.stop()

    assert host.readiness.state == KaiApplicationState.STOPPED
    assert host.readiness.ready is False
    assert host_dependencies[-5:] == [
        "execution:wait",
        "client:stop",
        "store:close",
        "execution:stop",
        "pool:stop",
    ]


def test_core_host_module_has_no_telegram_dependency() -> None:
    source = Path(host_module.__file__).read_text(encoding="utf-8")

    assert "from telegram" not in source
    assert "import telegram" not in source


async def test_core_rejects_double_start(host_dependencies) -> None:
    host = _host()
    await host.start()

    with pytest.raises(RuntimeError, match="cannot start from ready"):
        await host.start()

    await host.stop()


async def test_real_core_lifecycle_uses_workshop_identity_without_telegram_application(tmp_path) -> None:
    database = tmp_path / "kai.db"
    profiles = profile_registry(101)
    store = await WorkshopEventStore.open(database)
    try:
        bootstrap = await bootstrap_default_workshop(
            store,
            (
                BootstrapHuman(
                    display_name="Browser Human",
                    role="admin",
                    transport="workshop",
                    external_subject="browser-human",
                    external_channel_id="browser-direct",
                    runtime_profile_id=profile_id(101),
                ),
            ),
        )
        principal_id = bootstrap_human_principal_id(
            bootstrap.workshop_id,
            "workshop",
            "browser-human",
        )
    finally:
        await store.close()

    config = Config(
        telegram_bot_token="unused-by-core-host",
        allowed_user_ids=set(),
        session_db_path=database,
        agent_idle_timeout=0,
        default_backend="codex",
        default_model="gpt-5.6-sol",
    )
    principal_storage = WorkshopPrincipalStorageRegistry(
        (
            WorkshopPrincipalStorageNamespace(
                PrincipalId(str(principal_id)),
                profile_id(101),
                101,
            ),
        )
    )
    host = KaiApplicationHost(
        config=config,
        runtime_profiles=profiles,
        principal_storage=principal_storage,
        services_info=[],
        registered_backend_ids=frozenset({"codex"}),
    )

    services = await host.start()
    try:
        assert host.readiness.ready is True
        assert services.runtime_pool.runtime_profile(profile_id(101)).backend == "codex"
        assert services.client_commands.ready is True
        assert services.private_text_execution.ready is True
    finally:
        await host.stop()

    assert host.readiness.state == KaiApplicationState.STOPPED


async def test_core_activates_delivery_authority_before_recovering_expired_started_run(tmp_path) -> None:
    database = tmp_path / "kai.db"
    profiles = profile_registry(101)
    store = await WorkshopEventStore.open(database)
    now = datetime.now(UTC)
    try:
        bootstrap = await bootstrap_default_workshop(
            store,
            (
                BootstrapHuman(
                    display_name="Browser Human",
                    role="admin",
                    transport="telegram",
                    external_subject="101",
                    external_channel_id="101",
                    runtime_profile_id=profile_id(101),
                ),
            ),
        )
        principal_id = bootstrap_human_principal_id(
            bootstrap.workshop_id,
            "telegram",
            "101",
        )
        accepted = await WorkshopConversationCommandService(store).accept(
            InboundMessage(
                transport="telegram",
                update_id="expired-run-command",
                message_id="expired-run-message",
                sender_subject="101",
                channel_subject="101",
                body="Recover me after restart",
                occurred_at=now - timedelta(minutes=10),
            )
        )
        delivery_authority = WorkshopConversationDeliveryAuthority(store)
        old_epoch = (await delivery_authority.activate()).epoch
        await delivery_authority.deactivate()

        selection = RunExecutionSelection("codex", "gpt-5.6-sol", "openai")
        run_authority = WorkshopRunExecutionAuthority(
            store,
            selection_resolver=lambda _run: selection,
            registered_backend_ids=frozenset({"codex"}),
        )
        granted = await run_authority.grant(
            accepted.run.run_id,
            owner_id=RunExecutionOwnerId.new(),
            occurred_at=now - timedelta(minutes=9),
            lease_expires_at=now - timedelta(minutes=8),
        )
        await run_authority.start(
            granted.claim,
            occurred_at=now - timedelta(minutes=8, seconds=30),
        )
    finally:
        await store.close()

    host = KaiApplicationHost(
        config=Config(
            telegram_bot_token="unused-by-core-host",
            allowed_user_ids=set(),
            session_db_path=database,
            agent_idle_timeout=0,
            default_backend="codex",
            default_model="gpt-5.6-sol",
        ),
        runtime_profiles=profiles,
        principal_storage=WorkshopPrincipalStorageRegistry(
            (
                WorkshopPrincipalStorageNamespace(
                    PrincipalId(str(principal_id)),
                    profile_id(101),
                    101,
                ),
            )
        ),
        services_info=[],
        registered_backend_ids=frozenset({"codex"}),
    )

    services = await host.start()
    try:
        assert services.delivery_authority_epoch.epoch_id != old_epoch.epoch_id
        inspection = services.client_store
        recovered_run = await WorkshopRunLifecycle(inspection).state(accepted.run.run_id)
        recovered_attempt = await WorkshopRunExecutionAuthority(
            inspection,
            selection_resolver=lambda _run: selection,
            registered_backend_ids=frozenset({"codex"}),
        ).attempt(granted.claim.attempt_id)
        assert recovered_run.status == RunStatus.FAILED
        assert recovered_attempt is not None
        assert recovered_attempt.status == RunAttemptStatus.INTERRUPTED
        async with inspection.connection.execute(
            "SELECT d.authority_epoch_id FROM delivery_outbox d "
            "JOIN messages m ON m.id = d.message_id "
            "WHERE m.reply_to_message_id = ?",
            (recovered_run.inbound_message_id,),
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None
        assert str(row[0]) == str(services.delivery_authority_epoch.epoch_id)
    finally:
        await host.stop()
