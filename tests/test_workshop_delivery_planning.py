"""Capability-aware, transport-neutral Workshop delivery planning contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from kai.workshop.artifacts import WorkshopArtifactService
from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.delivery_authority import WorkshopConversationDeliveryAuthority
from kai.workshop.delivery_fragments import WorkshopDeliveryFragments
from kai.workshop.delivery_outbox import (
    CONVERSATION_REPLY_PURPOSE,
    NOTIFICATION_PURPOSE,
    SEND_FRAGMENTS_CONTRACT,
    STREAMING_FINALIZATION_CONTRACT,
    DeliveryClaim,
    DeliveryId,
    DeliveryPurpose,
    WorkshopDeliveryOutbox,
)
from kai.workshop.delivery_policy import (
    DeliveryAdapterCapabilities,
    WorkshopDeliveryBindingPolicy,
)
from kai.workshop.domain import (
    ChannelBindingId,
    EventEnvelope,
    ExternalIdentityId,
    MessageId,
    PrincipalId,
    WorkshopEventType,
    WorkshopId,
)
from kai.workshop.inbound import InboundMessage, record_inbound_message
from kai.workshop.internal_api_contexts import WorkshopInternalAPIContextRegistry
from kai.workshop.outbound import (
    OutboundMessage,
    record_outbound_message_with_streaming_finalization,
)
from kai.workshop.proactive_publication import (
    ProactivePublicationAuthority,
    WorkshopProactivePublicationService,
)
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.scheduled_notifications import (
    ScheduledReminder,
    WorkshopScheduledReminderRecorder,
)
from kai.workshop.storage_namespaces import WorkshopPrincipalStorageRegistry
from kai.workshop.store import WorkshopEventStore
from tests.workshop_delivery import TELEGRAM_DELIVERY_POLICY
from tests.workshop_profiles import profile_id, profile_registry

_NOW = datetime(2026, 8, 25, 12, 0, tzinfo=UTC)
_DESKTOP_CAPABILITIES = DeliveryAdapterCapabilities(transport="desktop")


class _DesktopDeliveryAdapter:
    """Small test adapter proving core delivery work needs no feature branch."""

    def __init__(self, store: WorkshopEventStore) -> None:
        self._outbox = WorkshopDeliveryOutbox(store)
        self._fragments = WorkshopDeliveryFragments(store)
        self.delivered: list[str] = []

    async def deliver(
        self,
        delivery_id: DeliveryId,
        *,
        purpose: DeliveryPurpose,
    ) -> DeliveryClaim:
        claim = await self._outbox.claim_next(
            "desktop-test-adapter",
            purposes=(purpose,),
            execution_contracts=(SEND_FRAGMENTS_CONTRACT,),
            transport="desktop",
            delivery_id=delivery_id,
        )
        assert claim is not None
        await self._fragments.prepare(claim, (claim.body,))
        fragment = await self._fragments.begin_next(claim)
        assert fragment is not None
        self.delivered.append(fragment.body)
        await self._fragments.mark_sent(
            claim,
            fragment,
            external_message_id=f"desktop-{len(self.delivered)}",
        )
        state = await self._outbox.mark_succeeded(claim)
        assert state.status == "succeeded"
        return claim


async def _desktop_store(path: Path) -> tuple[WorkshopEventStore, PrincipalId, MessageId]:
    store = await WorkshopEventStore.open(path)
    await bootstrap_default_workshop(
        store,
        (
            BootstrapHuman(
                "Desktop Human",
                "admin",
                "desktop",
                "desktop-human",
                "desktop-human",
                profile_id(101),
            ),
        ),
    )
    async with store.connection.execute(
        "SELECT principal_id FROM external_identities WHERE provider = 'desktop' AND external_subject = 'desktop-human'"
    ) as cursor:
        row = await cursor.fetchone()
    assert row is not None
    principal_id = PrincipalId(str(row[0]))
    inbound = await record_inbound_message(
        store,
        InboundMessage(
            transport="desktop",
            update_id="desktop-update-1",
            message_id="desktop-message-1",
            sender_subject="desktop-human",
            channel_subject="desktop-human",
            body="Hello from desktop",
            occurred_at=_NOW,
        ),
    )
    return store, principal_id, MessageId(str(inbound.event.envelope.aggregate_id))


async def _add_telegram_destination(
    store: WorkshopEventStore,
    principal_id: PrincipalId,
    inbound_id: MessageId,
) -> None:
    async with store.connection.execute(
        "SELECT c.workshop_id, m.channel_id FROM messages m JOIN channels c ON c.id = m.channel_id WHERE m.id = ?",
        (inbound_id,),
    ) as cursor:
        row = await cursor.fetchone()
    assert row is not None
    workshop_id = WorkshopId(str(row[0]))
    channel_id = str(row[1])
    await store.append(
        EventEnvelope.create(
            event_type=WorkshopEventType.EXTERNAL_IDENTITY_BOUND,
            event_version=1,
            workshop_id=workshop_id,
            aggregate_type="external_identity",
            aggregate_id=ExternalIdentityId.new(),
            occurred_at=_NOW,
            idempotency_key="test:desktop-human:telegram-identity",
            payload={
                "principal_id": principal_id,
                "provider": "telegram",
                "external_subject": "101",
            },
        )
    )
    await store.append(
        EventEnvelope.create(
            event_type=WorkshopEventType.TRANSPORT_CHANNEL_BOUND,
            event_version=1,
            workshop_id=workshop_id,
            aggregate_type="channel_binding",
            aggregate_id=ChannelBindingId.new(),
            occurred_at=_NOW,
            idempotency_key="test:desktop-channel:telegram-binding",
            payload={
                "channel_id": channel_id,
                "transport": "telegram",
                "external_channel_id": "101",
            },
        )
    )
    await store.project_pending(CanonicalConversationProjection())


def _desktop_policy() -> WorkshopDeliveryBindingPolicy:
    return WorkshopDeliveryBindingPolicy(
        frozenset({"desktop"}),
        (_DESKTOP_CAPABILITIES,),
    )


async def test_non_telegram_adapter_delivers_ordinary_reply_without_core_changes(tmp_path: Path):
    store, _, inbound_id = await _desktop_store(tmp_path / "kai.db")
    try:
        result = await record_outbound_message_with_streaming_finalization(
            store,
            OutboundMessage(inbound_id, "Desktop reply", _NOW),
            delivery_policy=_desktop_policy(),
        )

        assert len(result.deliveries) == 1
        delivery = result.deliveries[0].delivery
        assert (delivery.transport, delivery.execution_contract) == (
            "desktop",
            SEND_FRAGMENTS_CONTRACT,
        )
        adapter = _DesktopDeliveryAdapter(store)
        await adapter.deliver(delivery.delivery_id, purpose=CONVERSATION_REPLY_PURPOSE)
        assert adapter.delivered == ["Desktop reply"]
    finally:
        await store.close()


async def test_multiple_adapters_have_independent_durable_outcomes(tmp_path: Path):
    store, principal_id, inbound_id = await _desktop_store(tmp_path / "kai.db")
    try:
        await _add_telegram_destination(store, principal_id, inbound_id)
        await WorkshopConversationDeliveryAuthority(store).activate()
        policy = WorkshopDeliveryBindingPolicy(
            frozenset({"desktop", "telegram"}),
            (
                _DESKTOP_CAPABILITIES,
                TELEGRAM_DELIVERY_POLICY.adapter_capabilities[0],
            ),
        )
        result = await record_outbound_message_with_streaming_finalization(
            store,
            OutboundMessage(inbound_id, "Fanout reply", _NOW),
            delivery_policy=policy,
        )
        by_transport = {item.delivery.transport: item.delivery for item in result.deliveries}
        assert set(by_transport) == {"desktop", "telegram"}
        assert by_transport["desktop"].execution_contract == SEND_FRAGMENTS_CONTRACT
        assert by_transport["telegram"].execution_contract == STREAMING_FINALIZATION_CONTRACT

        adapter = _DesktopDeliveryAdapter(store)
        await adapter.deliver(
            by_transport["desktop"].delivery_id,
            purpose=CONVERSATION_REPLY_PURPOSE,
        )
        epoch = await WorkshopConversationDeliveryAuthority(store).active_epoch()
        telegram_claim = await WorkshopDeliveryOutbox(store).claim_next(
            "telegram-test-failure",
            purposes=(CONVERSATION_REPLY_PURPOSE,),
            execution_contracts=(STREAMING_FINALIZATION_CONTRACT,),
            transport="telegram",
            delivery_id=by_transport["telegram"].delivery_id,
            authority_epoch_id=epoch.epoch_id,
        )
        assert telegram_claim is not None
        await WorkshopDeliveryOutbox(store).mark_failed(
            telegram_claim,
            retryable=False,
            error_code="telegram_test_failure",
        )
        assert (await WorkshopDeliveryOutbox(store).state(by_transport["desktop"].delivery_id)).status == ("succeeded")
        assert (await WorkshopDeliveryOutbox(store).state(by_transport["telegram"].delivery_id)).status == "failed"
    finally:
        await store.close()


async def test_non_telegram_adapter_delivers_scheduled_and_proactive_messages(tmp_path: Path):
    store, principal_id, _ = await _desktop_store(tmp_path / "kai.db")
    profiles = profile_registry(101)
    try:
        contexts = await WorkshopInternalAPIContextRegistry.from_store(store, profiles)
        context = contexts.for_runtime_profile(profile_id(101))
        async with store.connection.execute(
            "SELECT ca.agent_id FROM channel_agents ca WHERE ca.channel_id = ?",
            (context.channel_id,),
        ) as cursor:
            agent_row = await cursor.fetchone()
        assert agent_row is not None
        await store.connection.execute(
            "INSERT INTO workshop_scheduled_jobs "
            "(id, principal_id, channel_id, agent_id, runtime_profile_id, name, "
            "job_type, prompt, schedule_type, schedule_data) "
            "VALUES (901, ?, ?, ?, ?, 'Desktop reminder', 'reminder', 'Remember', "
            "'once', '{\"run_at\":\"2036-01-01T00:00:00Z\"}')",
            (principal_id, context.channel_id, str(agent_row[0]), profile_id(101)),
        )
        await store.connection.commit()
        scheduled = await WorkshopScheduledReminderRecorder(store, _desktop_policy()).record(
            ScheduledReminder(901, "occurrence-1", "Scheduled desktop", _NOW)
        )

        storage = await WorkshopPrincipalStorageRegistry.from_store(store, profiles)
        artifacts = WorkshopArtifactService(
            store,
            data_dir=tmp_path,
            principal_storage=storage,
            runtime_profiles=profiles,
        )
        proactive = await WorkshopProactivePublicationService(
            store,
            artifacts,
            artifact_storage_root=tmp_path / "files",
            delivery_policy=_desktop_policy(),
        ).publish_text(
            ProactivePublicationAuthority(
                context.principal_id,
                context.channel_id,
                context.agent_id,
                context.runtime_profile_id,
            ),
            request_id="desktop-proactive",
            body="Proactive desktop",
            occurred_at=_NOW,
        )

        adapter = _DesktopDeliveryAdapter(store)
        await adapter.deliver(scheduled.deliveries[0].delivery.delivery_id, purpose=NOTIFICATION_PURPOSE)
        await adapter.deliver(proactive.deliveries[0].delivery.delivery_id, purpose=NOTIFICATION_PURPOSE)
        assert adapter.delivered == ["Scheduled desktop", "Proactive desktop"]
    finally:
        await store.close()
