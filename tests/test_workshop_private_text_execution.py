"""Integrated contracts for the production private-text execution owner."""

from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from kai.backend import AgentResponse, StreamEvent
from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop, bootstrap_human_principal_id
from kai.workshop.delivery_authority import WorkshopConversationDeliveryAuthority
from kai.workshop.domain import ChannelId, PrincipalId
from kai.workshop.execution_coordinator import (
    CanonicalCancellationDisposition,
    CanonicalExecutionDisposition,
)
from kai.workshop.inbound import ClientInboundMessage, InboundMessage
from kai.workshop.internal_api_contexts import WorkshopInternalAPIExecutionContext
from kai.workshop.private_text_execution import (
    RecoverableClientRun,
    WorkshopPrivateTextExecutionService,
)
from kai.workshop.routing_eligibility import (
    CapabilityAssessment,
    CapabilitySupport,
    EligibilityReason,
    RoutingEligibilityAuthority,
    RoutingTaskClass,
    RuntimeCapability,
    RuntimeEligibilityCandidate,
    RuntimeEligibilityReport,
)
from kai.workshop.run_lifecycle import RunStatus
from kai.workshop.runtime_pool import WorkshopRuntimePool
from kai.workshop.store import WorkshopEventStore
from tests.workshop_delivery import TELEGRAM_DELIVERY_POLICY
from tests.workshop_profiles import profile_id, profile_registry

_NOW = datetime(2026, 1, 1, 23, 0, tzinfo=UTC)


class _Runtime:
    def __init__(self, workspace: Path, *, wait: asyncio.Event | None = None) -> None:
        self.selection = SimpleNamespace(backend="codex", provider="openai", model="gpt-5.6-sol")
        self.workspace = workspace
        self.wait = wait
        self.validated = False
        self.cancelled = False
        self.canonical_histories: list[str] = []
        self.agent_definition_contexts: list[str] = []

    def stage_canonical_history(self, history: str) -> None:
        self.canonical_histories.append(history)

    def stage_canonical_agent_context(self, context: str) -> None:
        self.agent_definition_contexts.append(context)

    def validate_current(self) -> None:
        self.validated = True

    async def cancel(self) -> None:
        self.cancelled = True
        if self.wait is not None:
            self.wait.set()

    async def stream(self, prompt: str):
        yield StreamEvent(text_so_far="Stable preview.", done=False)
        if self.wait is not None:
            await self.wait.wait()
            if self.cancelled:
                raise RuntimeError("runtime stopped")
        yield StreamEvent(
            text_so_far="Durable answer",
            done=True,
            response=AgentResponse(success=True, text="Durable answer", session_id="session-1"),
        )


class _Eligibility:
    def __init__(self, authority: RoutingEligibilityAuthority) -> None:
        self.authority = authority

    def authority_for_principal_channel(self, principal_id, channel_id):
        assert principal_id == self.authority.principal_id
        assert channel_id == self.authority.channel_id
        return self.authority

    def authority_for_principal_runtime(self, principal_id, runtime_profile_id):
        assert principal_id == self.authority.principal_id
        assert runtime_profile_id == self.authority.runtime_profile_id
        return self.authority

    async def inspect(self, authority, task_class, *, additional_required=()):
        assert not additional_required or additional_required[0].value == "text_generation"
        return RuntimeEligibilityReport(
            version=1,
            task_class=RoutingTaskClass(task_class),
            required_capabilities=(RuntimeCapability.TEXT_GENERATION,),
            principal_id=authority.principal_id,
            channel_id=authority.channel_id,
            agent_id=authority.agent_id,
            runtime_profile_id=authority.runtime_profile_id,
            workspace="/workspace",
            candidates=(
                RuntimeEligibilityCandidate(
                    option_id="codex:openai",
                    backend="codex",
                    provider="openai",
                    allowed_services=(),
                    model_id="gpt-5.6-sol",
                    model_source="current_selection",
                    selected=True,
                    eligible=True,
                    capabilities=(
                        CapabilityAssessment(
                            RuntimeCapability.TEXT_GENERATION,
                            CapabilitySupport.SUPPORTED,
                            "test",
                        ),
                    ),
                    reasons=(EligibilityReason("eligible", "test"),),
                ),
            ),
        )


async def _foundation(database: Path) -> _Eligibility:
    store = await WorkshopEventStore.open(database)
    try:
        result = await bootstrap_default_workshop(
            store,
            (
                BootstrapHuman(
                    display_name="Workshop Human",
                    role="admin",
                    transport="telegram",
                    external_subject="101",
                    external_channel_id="101",
                    runtime_profile_id=profile_id(101),
                ),
            ),
        )
        await WorkshopConversationDeliveryAuthority(store).activate()
        principal_id = bootstrap_human_principal_id(result.workshop_id, "telegram", "101")
        channel_token = hashlib.sha256(b"telegram\x00101").hexdigest()
        channel_id = ChannelId.derived(result.workshop_id, f"direct-channel:{channel_token}")
        return _Eligibility(
            RoutingEligibilityAuthority(
                principal_id,
                channel_id,
                result.agent_id,
                profile_id(101),
            )
        )
    finally:
        await store.close()


def _message(*, suffix: str = "1") -> InboundMessage:
    return InboundMessage(
        transport="telegram",
        update_id=f"update-{suffix}",
        message_id=f"message-{suffix}",
        sender_subject="101",
        channel_subject="101",
        body=f"Canonical prompt {suffix}",
        occurred_at=_NOW,
    )


async def test_owner_accepts_executes_and_atomically_enqueues_terminal_reply(tmp_path: Path):
    database = tmp_path / "kai.db"
    eligibility = await _foundation(database)
    runtime = _Runtime(tmp_path)
    pool = SimpleNamespace(prepare_routed_execution=AsyncMock(return_value=runtime))
    service = await WorkshopPrivateTextExecutionService.open_and_start(
        database,
        WorkshopRuntimePool(pool, profile_registry(101)),  # type: ignore[arg-type]
        registered_backend_ids=frozenset({"codex"}),
        delivery_policy=TELEGRAM_DELIVERY_POLICY,
        routing_eligibility=eligibility,  # type: ignore[arg-type]
    )
    observed: list[str] = []
    try:
        accepted = await service.accept(_message())

        async def observe(event: StreamEvent) -> None:
            observed.append(event.text_so_far)

        result = await service.execute(accepted.run.run_id, stream_observer=observe)

        assert result.disposition == CanonicalExecutionDisposition.COMPLETED
        assert result.run.status == RunStatus.COMPLETED
        assert result.session_id == "session-1"
        assert result.workspace == str(tmp_path)
        assert observed == ["Stable preview."]
        assert runtime.validated is True
        assert len(runtime.canonical_histories) == 1
        assert "canonical-transcript.ndjson" in runtime.canonical_histories[0]
        assert "untrusted conversation data" in runtime.canonical_histories[0]
        pool.prepare_routed_execution.assert_awaited_once_with(
            WorkshopInternalAPIExecutionContext(
                accepted.run.requested_by_principal_id,
                accepted.run.channel_id,
                accepted.run.agent_id,
                profile_id(101),
            ),
            "codex:openai",
            "gpt-5.6-sol",
        )
    finally:
        await service.stop()

    inspection = await WorkshopEventStore.open(database)
    try:
        async with inspection.connection.execute(
            "SELECT m.body, d.status FROM messages m "
            "JOIN delivery_outbox d ON d.message_id = m.id "
            "WHERE m.reply_to_message_id IS NOT NULL"
        ) as cursor:
            assert tuple(await cursor.fetchone()) == ("Durable answer", "pending")
        async with inspection.connection.execute(
            "SELECT status, runtime_profile_id, workspace, provider_session_id "
            "FROM workshop_post_run_effects WHERE run_id = ?",
            (result.run.run_id,),
        ) as cursor:
            assert tuple(await cursor.fetchone()) == (
                "pending",
                str(profile_id(101)),
                str(tmp_path),
                "session-1",
            )
    finally:
        await inspection.close()


async def test_owner_routes_stop_to_exact_active_runtime_and_terminal_cancellation(tmp_path: Path):
    database = tmp_path / "kai.db"
    eligibility = await _foundation(database)
    release = asyncio.Event()
    runtime = _Runtime(tmp_path, wait=release)
    pool = SimpleNamespace(prepare_routed_execution=AsyncMock(return_value=runtime))
    service = await WorkshopPrivateTextExecutionService.open_and_start(
        database,
        WorkshopRuntimePool(pool, profile_registry(101)),  # type: ignore[arg-type]
        registered_backend_ids=frozenset({"codex"}),
        delivery_policy=TELEGRAM_DELIVERY_POLICY,
        routing_eligibility=eligibility,  # type: ignore[arg-type]
    )
    try:
        accepted = await service.accept(_message())
        execution = asyncio.create_task(service.execute(accepted.run.run_id))
        while not runtime.validated:
            await asyncio.sleep(0)

        cancellation = await service.request_transport_cancellation(
            transport="telegram",
            sender_subject="101",
            channel_subject="101",
        )
        result = await execution

        assert cancellation == CanonicalCancellationDisposition.REQUESTED
        assert runtime.cancelled is True
        assert result.disposition == CanonicalExecutionDisposition.CANCELLED
        assert result.run.status == RunStatus.CANCELLED
    finally:
        await service.stop()


async def test_owner_discovers_only_durably_accepted_workshop_client_runs(tmp_path: Path):
    database = tmp_path / "kai.db"
    eligibility = await _foundation(database)
    inspection = await WorkshopEventStore.open(database)
    try:
        async with inspection.connection.execute(
            "SELECT p.id, c.id FROM principals p "
            "JOIN external_identities ei ON ei.principal_id = p.id "
            "JOIN channel_memberships cm ON cm.principal_id = p.id "
            "JOIN channels c ON c.id = cm.channel_id AND c.kind = 'direct' "
            "WHERE ei.provider = 'telegram' AND ei.external_subject = '101'"
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None
        principal_id = PrincipalId(str(row[0]))
        channel_id = ChannelId(str(row[1]))
    finally:
        await inspection.close()

    pool = SimpleNamespace(prepare_routed_execution=AsyncMock())
    service = await WorkshopPrivateTextExecutionService.open_and_start(
        database,
        WorkshopRuntimePool(pool, profile_registry(101)),  # type: ignore[arg-type]
        registered_backend_ids=frozenset({"codex"}),
        delivery_policy=TELEGRAM_DELIVERY_POLICY,
        routing_eligibility=eligibility,  # type: ignore[arg-type]
    )
    try:
        accepted = await service.accept_client(
            ClientInboundMessage(
                principal_id=principal_id,
                channel_id=channel_id,
                client_message_id="recoverable-browser-command",
                body="Resume this browser run after restart",
                occurred_at=_NOW,
            )
        )

        assert await service.recoverable_client_runs() == (
            RecoverableClientRun(
                accepted.run.run_id,
                profile_id(101),
                accepted.command.message.event.envelope.aggregate_id,
                "Resume this browser run after restart",
            ),
        )
    finally:
        await service.stop()
