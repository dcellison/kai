"""End-to-end contracts for the Workshop execution coordinator."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from kai.agent_failure import AgentFailureKind
from kai.backend import AgentResponse, ContextAssemblyObservation, StreamEvent, TraceEntry
from kai.principal_documents import (
    PrincipalDocument,
    PrincipalDocumentKind,
    PrincipalDocumentReport,
    PrincipalDocumentState,
)
from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.channel_lifecycle import WorkshopChannelLifecycleService
from kai.workshop.context_manifests import CONTEXT_SOURCE_ORDER, WorkshopContextManifestService
from kai.workshop.conversation_commands import WorkshopConversationCommandService
from kai.workshop.delivery_authority import WorkshopConversationDeliveryAuthority
from kai.workshop.diagnostics import (
    workshop_context_manifest_status,
    workshop_conversation_observation_status,
    workshop_runtime_session_status,
)
from kai.workshop.domain import ChannelId, PrincipalId, RunExecutionOwnerId, RuntimeProfileId
from kai.workshop.execution_coordinator import (
    CanonicalCancellationDisposition,
    CanonicalExecutionDisposition,
    WorkshopCanonicalExecutionCoordinator,
)
from kai.workshop.inbound import ClientInboundMessage, InboundMessage
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.protected_execution import ProtectedExecutionRoutingRejected
from kai.workshop.routing_eligibility import RoutingTaskClass
from kai.workshop.routing_policy import (
    RoutingDecisionDisposition,
    RunRoutingDecision,
)
from kai.workshop.run_execution_authority import (
    RunAttemptStatus,
    RunExecutionSelection,
    WorkshopRunExecutionAuthority,
)
from kai.workshop.run_lifecycle import RunStatus, WorkshopRunLifecycle
from kai.workshop.runtime_sessions import load_runtime_session
from kai.workshop.store import WorkshopEventStore
from tests.workshop_delivery import TELEGRAM_DELIVERY_POLICY

_NOW = datetime(2026, 8, 12, 22, 0, tzinfo=UTC)
_RUNTIME_PROFILE_ID = RuntimeProfileId.new()


class _Prepared:
    def __init__(
        self,
        run,
        *,
        response: AgentResponse | None = None,
        wait: asyncio.Event | None = None,
        on_stream: Callable[[], Awaitable[None]] | None = None,
        selection: RunExecutionSelection | None = None,
        ambient_context_discovery_enabled: bool | None = None,
    ) -> None:
        self.run = run
        self.runtime_profile_id = _RUNTIME_PROFILE_ID
        self.selection = selection or RunExecutionSelection("codex", "gpt-5.6-sol")
        self.workspace = Path("/private/tmp/kai-workshop-test-workspace")
        self.home_workspace = self.workspace
        self.response = response or AgentResponse(success=True, text="Canonical answer")
        self.wait = wait
        self.on_stream = on_stream
        self.prompts: list[str] = []
        self.validated = False
        self.cancelled = False
        self.reject_validation = False
        self.canonical_histories: list[str] = []
        self.canonical_history_options: list[dict[str, object]] = []
        self.agent_definition_contexts: list[str] = []
        self.collaboration_invocations = []
        self.discarded_collaboration_invocations = []
        self.context_observer = None
        self.ambient_context_discovery_enabled = ambient_context_discovery_enabled
        self.retained_context_revision = "c" * 64

    def stage_canonical_history(self, history: str, **_kwargs: object) -> None:
        self.canonical_histories.append(history)
        self.canonical_history_options.append(dict(_kwargs))

    def stage_agent_definition_context(self, context: str) -> None:
        self.agent_definition_contexts.append(context)

    def stage_context_assembly_observer(self, observer) -> None:
        self.context_observer = observer

    def stage_collaboration_invocation(self, invocation) -> None:
        self.collaboration_invocations.append(invocation)

    def discard_collaboration_invocation(self, invocation) -> None:
        assert self.collaboration_invocations[-1] == invocation
        self.discarded_collaboration_invocations.append(invocation)

    def validate_current(self) -> None:
        self.validated = True
        if self.reject_validation:
            raise RuntimeError("runtime drift")

    async def cancel(self) -> None:
        self.cancelled = True
        if self.wait is not None:
            self.wait.set()

    async def stream(self, prompt: str) -> AsyncIterator[StreamEvent]:
        self.prompts.append(prompt)
        if self.context_observer is not None:
            observer, self.context_observer = self.context_observer, None
            await observer(
                ContextAssemblyObservation(
                    provider_dispatch_reached=True,
                    session_context_delivered=True,
                    session_context_revision="0" * 64,
                    semantic_recall_attempted=True,
                    semantic_recall_delivered=False,
                    semantic_recall_reason="no_matches",
                    semantic_recall_revision=None,
                    workspace_reminder_delivered=False,
                    principal_documents=PrincipalDocumentReport(
                        policy=PrincipalDocument(
                            PrincipalDocumentKind.POLICY,
                            PrincipalDocumentState.PRESENT,
                            "redacted",
                            "1" * 64,
                            "verified_owner_read",
                        ),
                        preferences=PrincipalDocument(
                            PrincipalDocumentKind.PREFERENCES,
                            PrincipalDocumentState.MISSING,
                            None,
                            None,
                            "missing",
                        ),
                    ),
                    ambient_context_discovery_enabled=self.ambient_context_discovery_enabled,
                    canonical_conversation_delivered=True,
                    canonical_conversation_mode="snapshot",
                    canonical_conversation_revision=str(self.canonical_history_options[-1].get("snapshot_revision")),
                )
            )
        if self.on_stream is not None:
            await self.on_stream()
        if self.wait is not None:
            await self.wait.wait()
            if self.cancelled:
                raise RuntimeError("runtime stopped")
        yield StreamEvent(text_so_far=self.response.text, done=True, response=self.response)


class _Preparation:
    def __init__(self, prepared: _Prepared) -> None:
        self.prepared = prepared
        self.calls = 0

    async def prepare(self, run_id):
        assert run_id == self.prepared.run.run_id
        self.calls += 1
        return self.prepared


class _PreparationByRun:
    def __init__(self, prepared: tuple[_Prepared, ...]) -> None:
        self.prepared = {item.run.run_id: item for item in prepared}

    async def prepare(self, run_id):
        prepared = self.prepared[run_id]
        assert prepared.run.runtime_profile_id is not None
        prepared.runtime_profile_id = prepared.run.runtime_profile_id
        return prepared


class _RejectedPreparation:
    def __init__(self, run) -> None:
        self.run = run

    async def prepare(self, run_id):
        assert run_id == self.run.run_id
        raise ProtectedExecutionRoutingRejected(
            self.run,
            RunRoutingDecision(
                run_id=self.run.run_id,
                runtime_profile_id=_RUNTIME_PROFILE_ID,
                requested_task_class=RoutingTaskClass.CODING,
                requested_backend_option_id="codex:openai",
                selected_backend_option_id=None,
                disposition=RoutingDecisionDisposition.REJECTED,
                reason_code="capability_unknown",
                policy_revision=1,
                selection=RunExecutionSelection("codex", "gpt-5.6-sol"),
                evidence_version=1,
                decided_at=_NOW,
            ),
        )


async def _accepted(path: Path, *, suffix: str = "1"):
    store = await WorkshopEventStore.open(path)
    await bootstrap_default_workshop(
        store,
        (
            BootstrapHuman(
                display_name="Workshop Human",
                role="admin",
                transport="telegram",
                external_subject="101",
                external_channel_id="101",
                runtime_profile_id=_RUNTIME_PROFILE_ID,
            ),
        ),
    )
    result = await WorkshopConversationCommandService(store).accept(
        InboundMessage(
            transport="telegram",
            update_id=f"command-{suffix}",
            message_id=f"message-{suffix}",
            sender_subject="101",
            channel_subject="101",
            body=f"Canonical prompt {suffix}",
            occurred_at=_NOW,
        )
    )
    await WorkshopConversationDeliveryAuthority(store).activate()
    return store, result.run


async def _accepted_group_pair(path: Path):
    store = await WorkshopEventStore.open(path)
    bootstrap = await bootstrap_default_workshop(
        store,
        (
            BootstrapHuman("Daniel", "admin", "desktop", "daniel", "daniel", _RUNTIME_PROFILE_ID),
            BootstrapHuman("Scott", "member", "desktop", "scott", "scott", RuntimeProfileId.new()),
        ),
    )
    async with store.connection.execute(
        "SELECT e.external_subject, e.principal_id, cb.channel_id "
        "FROM external_identities e JOIN channel_bindings cb "
        "ON cb.transport = e.provider AND cb.external_channel_id = e.external_subject "
        "ORDER BY e.external_subject"
    ) as cursor:
        rows = list(await cursor.fetchall())
    identities = {str(row[0]): (PrincipalId(str(row[1])), ChannelId(str(row[2]))) for row in rows}
    daniel_id, daniel_direct = identities["daniel"]
    scott_id, _scott_direct = identities["scott"]
    lifecycle = WorkshopChannelLifecycleService(store)
    group = await lifecycle.create_group(
        daniel_id,
        name="Shared execution lane",
        agent_ids=[bootstrap.agent_id],
        origin_channel_id=daniel_direct,
    )
    membership = await lifecycle.human_members(daniel_id, group.channel_id)
    await lifecycle.add_human_member(
        daniel_id,
        group.channel_id,
        scott_id,
        expected_state_version=membership.state_version,
        client_operation_id="add-scott-to-shared-execution-lane",
    )
    commands = WorkshopConversationCommandService(store)
    first = await commands.accept_client(
        ClientInboundMessage(
            daniel_id,
            group.channel_id,
            "shared-lane-daniel",
            "@Kai first",
            _NOW,
        )
    )
    second = await commands.accept_client(
        ClientInboundMessage(
            scott_id,
            group.channel_id,
            "shared-lane-scott",
            "@Kai second",
            _NOW + timedelta(seconds=1),
        )
    )
    await WorkshopConversationDeliveryAuthority(store).activate()
    return store, first.command.runs[0], second.command.runs[0]


def _coordinator(
    store,
    preparation,
    *,
    lease_seconds: int = 60,
    registered_backend_ids: frozenset[str] = frozenset({"codex"}),
):
    return WorkshopCanonicalExecutionCoordinator(
        store,
        preparation,
        registered_backend_ids=registered_backend_ids,
        clock=lambda: _NOW + timedelta(seconds=10),
        lease_duration=timedelta(seconds=lease_seconds),
        delivery_policy=TELEGRAM_DELIVERY_POLICY,
    )


async def _terminal_bodies(store: WorkshopEventStore) -> list[str]:
    async with store.connection.execute("SELECT body FROM messages ORDER BY created_event_position") as cursor:
        return [str(row[0]) for row in await cursor.fetchall()]


class TestCanonicalExecutionCoordinator:
    async def test_records_redacted_context_manifest_before_dispatch_and_replays_it(
        self,
        tmp_path: Path,
    ) -> None:
        store, run = await _accepted(tmp_path / "kai.db")
        prepared = _Prepared(run)
        coordinator = _coordinator(store, _Preparation(prepared))
        try:
            result = await coordinator.execute(run.run_id)
            assert result.disposition == CanonicalExecutionDisposition.COMPLETED

            manifests = await WorkshopContextManifestService(store).load_run(run.run_id)
            assert len(manifests) == 1
            manifest = manifests[0]
            assert tuple(source.kind for source in manifest.draft.sources) == CONTEXT_SOURCE_ORDER
            assert manifest.draft.workspace_kind == "home"
            assert manifest.draft.selection == prepared.selection
            assert manifest.draft.sources[2].reason == "run_bound_revision"
            assert manifest.draft.sources[0].refresh_class.value == "provider_session"
            assert manifest.draft.sources[0].state.value == "newly_delivered"
            assert manifest.draft.sources[1].delivery_shape == "inline_verified_document"
            assert manifest.draft.sources[1].revision == "1" * 64
            assert manifest.draft.sources[4].reason == "missing"
            assert manifest.draft.sources[5].reason == "semantic_memory_enabled_or_shared"
            assert manifest.draft.sources[7].history_boundary == 0
            assert manifest.draft.sources[8].state.value == "newly_delivered"
            assert manifest.draft.sources[10].reason == "accepted_input"
            assert manifest.draft.sources[11].reason == "not_observable"
            assert manifest.draft.sources[11].delivery_shape == "provider_managed_unknown"
            authority_source = manifest.draft.sources[9]
            assert authority_source.state.value == "server_attached"
            assert authority_source.reason == "server_binding_active"
            assert authority_source.delivery_shape == "server_attached_no_bearer"
            assert authority_source.authorization_operations == ("agent_delegation",)
            assert "X-Kai-Collaboration-Proof" not in repr(manifest)
            assert workshop_context_manifest_status(tmp_path / "kai.db").startswith(
                "Workshop context manifests: active; post-cutover attempts=1, manifests=1, missing=0, malformed=0, "
                "budget violations=0"
            )

            digest = manifest.manifest_sha256
            await store.rebuild_projection(CanonicalConversationProjection())
            replayed = await WorkshopContextManifestService(store).load_run(run.run_id)
            assert len(replayed) == 1
            assert replayed[0].manifest_sha256 == digest
        finally:
            await store.close()

    async def test_pi_manifest_records_disabled_ambient_context_discovery(
        self,
        tmp_path: Path,
    ) -> None:
        store, run = await _accepted(tmp_path / "kai.db")
        prepared = _Prepared(
            run,
            selection=RunExecutionSelection(
                "pi",
                "anthropic/claude-sonnet-4-6",
                provider="anthropic",
            ),
            ambient_context_discovery_enabled=False,
        )
        coordinator = _coordinator(
            store,
            _Preparation(prepared),
            registered_backend_ids=frozenset({"codex", "pi"}),
        )
        try:
            result = await coordinator.execute(run.run_id)
            assert result.disposition == CanonicalExecutionDisposition.COMPLETED

            manifests = await WorkshopContextManifestService(store).load_run(run.run_id)
            assert len(manifests) == 1
            provider_native = manifests[0].draft.sources[11]
            assert provider_native.reason == "ambient_discovery_disabled"
            assert provider_native.delivery_shape == "provider_managed_ambient_disabled"
        finally:
            await store.close()

    async def test_two_humans_alternate_in_one_group_agent_lane_in_order(
        self,
        tmp_path: Path,
    ) -> None:
        store, first_run, second_run = await _accepted_group_pair(tmp_path / "kai.db")
        first_started = asyncio.Event()
        release_first = asyncio.Event()
        dispatch_order: list[PrincipalId] = []

        async def start_first() -> None:
            dispatch_order.append(first_run.requested_by_principal_id)
            first_started.set()

        async def start_second() -> None:
            dispatch_order.append(second_run.requested_by_principal_id)

        first = _Prepared(first_run, wait=release_first, on_stream=start_first)
        second = _Prepared(second_run, on_stream=start_second)
        coordinator = _coordinator(store, _PreparationByRun((first, second)))
        try:
            first_execution = asyncio.create_task(coordinator.execute(first_run.run_id))
            await first_started.wait()
            second_execution = asyncio.create_task(coordinator.execute(second_run.run_id))
            await asyncio.sleep(0)

            assert second.prompts == []
            release_first.set()
            first_result, second_result = await asyncio.gather(first_execution, second_execution)

            assert first_result.disposition == CanonicalExecutionDisposition.COMPLETED
            assert second_result.disposition == CanonicalExecutionDisposition.COMPLETED
            assert dispatch_order == [
                first_run.requested_by_principal_id,
                second_run.requested_by_principal_id,
            ]
            first_manifests = await WorkshopContextManifestService(store).load_run(first_run.run_id)
            second_manifests = await WorkshopContextManifestService(store).load_run(second_run.run_id)
            assert len(first_manifests) == len(second_manifests) == 1
            assert first_manifests[0].requested_by_principal_id == first_run.requested_by_principal_id
            assert second_manifests[0].requested_by_principal_id == second_run.requested_by_principal_id
            assert first_manifests[0].requested_by_principal_id != second_manifests[0].requested_by_principal_id
        finally:
            await store.close()

    async def test_live_delta_contains_only_intervening_canonical_messages(self, tmp_path: Path) -> None:
        store, first_run, second_run = await _accepted_group_pair(tmp_path / "kai.db")
        first = _Prepared(first_run)
        second = _Prepared(second_run)
        coordinator = _coordinator(store, _PreparationByRun((first, second)))
        try:
            assert (await coordinator.execute(first_run.run_id)).disposition == CanonicalExecutionDisposition.COMPLETED
            assert (await coordinator.execute(second_run.run_id)).disposition == CanonicalExecutionDisposition.COMPLETED
            async with store.connection.execute(
                "SELECT requested_by_principal_id FROM runs WHERE id = ?",
                (second_run.run_id,),
            ) as cursor:
                scott_row = await cursor.fetchone()
            assert scott_row is not None
            scott_id = PrincipalId(str(scott_row[0]))
            commands = WorkshopConversationCommandService(store)
            intervening = await commands.accept_client(
                ClientInboundMessage(
                    scott_id,
                    second_run.channel_id,
                    "shared-lane-intervening",
                    "INTERVENING_CANONICAL_MESSAGE",
                    _NOW + timedelta(seconds=2),
                )
            )
            # Accepted but not dispatched: an accepted run must not advance
            # the observation boundary or hide its source message.
            assert len(intervening.command.runs) == 1
            third_acceptance = await commands.accept_client(
                ClientInboundMessage(
                    scott_id,
                    second_run.channel_id,
                    "shared-lane-third",
                    "@Kai third",
                    _NOW + timedelta(seconds=3),
                )
            )
            third_run = third_acceptance.command.runs[0]
            third = _Prepared(third_run)
            coordinator = _coordinator(store, _Preparation(third))

            assert (await coordinator.execute(third_run.run_id)).disposition == CanonicalExecutionDisposition.COMPLETED
            delta = str(third.canonical_history_options[0]["live_delta"])
            assert '"mode":"delta"' in delta
            assert '"body":"INTERVENING_CANONICAL_MESSAGE"' in delta
            assert '"body":"@Kai third"' not in delta
            assert '"body":"Canonical answer"' not in delta
        finally:
            await store.close()

    async def test_thread_context_and_cursor_are_exactly_thread_scoped(self, tmp_path: Path) -> None:
        store, first_run, second_run = await _accepted_group_pair(tmp_path / "kai.db")
        first = _Prepared(first_run)
        second = _Prepared(second_run)
        coordinator = _coordinator(store, _PreparationByRun((first, second)))
        try:
            assert (await coordinator.execute(first_run.run_id)).disposition == CanonicalExecutionDisposition.COMPLETED
            assert (await coordinator.execute(second_run.run_id)).disposition == CanonicalExecutionDisposition.COMPLETED
            thread_acceptance = await WorkshopConversationCommandService(store).accept_client(
                ClientInboundMessage(
                    second_run.requested_by_principal_id,
                    second_run.channel_id,
                    "shared-lane-thread",
                    "@Kai thread request",
                    _NOW + timedelta(seconds=2),
                    thread_root_id=first_run.inbound_message_id,
                )
            )
            thread_run = thread_acceptance.command.runs[0]
            prepared = _Prepared(thread_run)

            assert (
                await _coordinator(store, _Preparation(prepared)).execute(thread_run.run_id)
            ).disposition == CanonicalExecutionDisposition.COMPLETED
            snapshot = prepared.canonical_histories[0]
            assert '"scope_kind":"thread"' in snapshot
            assert f'"scope_id":"{first_run.inbound_message_id}"' in snapshot
            assert '"body":"@Kai first"' in snapshot
            assert '"body":"@Kai second"' not in snapshot
            assert '"body":"@Kai thread request"' not in snapshot
            async with store.connection.execute(
                "SELECT scope_kind, scope_id FROM channel_agent_conversation_observations WHERE last_run_id = ?",
                (thread_run.run_id,),
            ) as cursor:
                assert tuple(await cursor.fetchone()) == ("thread", first_run.inbound_message_id)
        finally:
            await store.close()

    async def test_foreign_workspace_manifest_records_only_redacted_workspace_identity(
        self,
        tmp_path: Path,
    ) -> None:
        store, run = await _accepted(tmp_path / "kai.db")
        prepared = _Prepared(run)
        prepared.home_workspace = Path("/private/home/principal")
        prepared.workspace = Path("/private/workspaces/secret-project-name")
        try:
            result = await _coordinator(store, _Preparation(prepared)).execute(run.run_id)

            assert result.disposition == CanonicalExecutionDisposition.COMPLETED
            manifests = await WorkshopContextManifestService(store).load_run(run.run_id)
            assert len(manifests) == 1
            manifest = manifests[0]
            assert manifest.draft.workspace_kind == "foreign"
            assert manifest.draft.workspace_digest is not None
            assert "secret-project-name" not in repr(manifest)
            assert all("/private/" not in repr(source) for source in manifest.draft.sources)
        finally:
            await store.close()

    async def test_ineligible_explicit_route_fails_without_backend_dispatch(self, tmp_path: Path):
        store, run = await _accepted(tmp_path / "kai.db")
        try:
            result = await _coordinator(store, _RejectedPreparation(run)).execute(run.run_id)

            assert result.disposition == CanonicalExecutionDisposition.FAILED
            assert result.run.status == RunStatus.FAILED
            assert result.run.terminal_code == "routing_ineligible"
            assert (await _terminal_bodies(store))[-1] == (
                "The requested task route is not eligible under your routing policy. Kai did not dispatch this request."
            )
            async with store.connection.execute(
                "SELECT status FROM run_attempts WHERE run_id = ?",
                (run.run_id,),
            ) as cursor:
                assert [str(row[0]) for row in await cursor.fetchall()] == ["failed"]
            manifests = await WorkshopContextManifestService(store).load_run(run.run_id)
            assert len(manifests) == 1
            assert manifests[0].draft.workspace_kind == "unknown"
            assert manifests[0].draft.workspace_digest is None
            assert {source.reason for source in manifests[0].draft.sources} >= {"dispatch_not_reached"}
            assert "not_observable" not in {source.reason for source in manifests[0].draft.sources}
        finally:
            await store.close()

    async def test_success_uses_stored_prompt_and_starts_before_exact_dispatch(self, tmp_path: Path):
        store, run = await _accepted(tmp_path / "kai.db")

        async def assert_started() -> None:
            assert (await WorkshopRunLifecycle(store).state(run.run_id)).status == RunStatus.STARTED

        prepared = _Prepared(run, on_stream=assert_started)
        coordinator = _coordinator(store, _Preparation(prepared))
        try:
            result = await coordinator.execute(run.run_id)

            assert result.disposition == CanonicalExecutionDisposition.COMPLETED
            assert result.run.status == RunStatus.COMPLETED
            assert prepared.validated is True
            assert prepared.prompts == ["Canonical prompt 1"]
            assert len(prepared.collaboration_invocations) == 1
            assert prepared.discarded_collaboration_invocations == prepared.collaboration_invocations
            assert await _terminal_bodies(store) == ["Canonical prompt 1", "Canonical answer"]
            async with store.connection.execute(
                "SELECT requested_operations_json, effective_operations_json, "
                "revocation_code, length(proof_fingerprint) "
                "FROM collaboration_grants WHERE run_id = ?",
                (run.run_id,),
            ) as cursor:
                assert tuple(await cursor.fetchone()) == (
                    '["agent_delegation"]',
                    '["agent_delegation"]',
                    "attempt_terminal",
                    64,
                )
            session = await load_runtime_session(store, run.channel_id, run.agent_id)
            assert session is not None
            assert session.last_run_id == run.run_id
            assert session.runtime_profile_id == _RUNTIME_PROFILE_ID
            assert session.retained_context_revision == "c" * 64
            async with store.connection.execute(
                "SELECT observed_through_event_position, last_run_id, last_inbound_message_id, scope_kind "
                "FROM channel_agent_conversation_observations WHERE channel_id = ? AND agent_id = ?",
                (run.channel_id, run.agent_id),
            ) as cursor:
                observation = await cursor.fetchone()
            assert observation is not None
            async with store.connection.execute(
                "SELECT created_event_position FROM messages WHERE id = ?",
                (run.inbound_message_id,),
            ) as cursor:
                source_position = int((await cursor.fetchone())[0])
            assert tuple(observation) == (
                source_position,
                run.run_id,
                run.inbound_message_id,
                "channel",
            )
            assert workshop_conversation_observation_status(tmp_path / "kai.db").startswith(
                "Workshop conversation observation: active; cursors=1 (channel=1, thread=0), integrity gaps=0"
            )
            await store.rebuild_projection(CanonicalConversationProjection())
            assert await load_runtime_session(store, run.channel_id, run.agent_id) == session
            assert workshop_runtime_session_status(tmp_path / "kai.db").startswith(
                "Workshop conversation continuity: active; successful lanes=1, sessions=1"
            )
        finally:
            await store.close()

    async def test_owner_runtime_continuity_ignores_a_distinct_access_profile(
        self,
        tmp_path: Path,
    ) -> None:
        store, run = await _accepted(tmp_path / "kai.db")
        access_profile = RuntimeProfileId.new()
        try:
            await store.connection.execute(
                "UPDATE channel_agent_runtime_assignments SET runtime_profile_id = ? "
                "WHERE channel_id = ? AND agent_id = ?",
                (access_profile, run.channel_id, run.agent_id),
            )
            await store.connection.execute(
                "UPDATE principal_agent_enablements SET runtime_profile_id = ? "
                "WHERE direct_channel_id = ? AND agent_id = ?",
                (access_profile, run.channel_id, run.agent_id),
            )
            await store.connection.commit()

            result = await _coordinator(store, _Preparation(_Prepared(run))).execute(run.run_id)

            assert result.disposition == CanonicalExecutionDisposition.COMPLETED
            session = await load_runtime_session(store, run.channel_id, run.agent_id)
            assert session is not None
            assert session.runtime_profile_id == _RUNTIME_PROFILE_ID
            assert workshop_runtime_session_status(tmp_path / "kai.db").startswith(
                "Workshop conversation continuity: active; successful lanes=1, sessions=1"
            )
        finally:
            await store.close()

    async def test_version_fifty_nine_retires_a_pre_owner_runtime_session(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from kai.workshop import schema

        path = tmp_path / "kai.db"
        replacement_profile = RuntimeProfileId.new()
        with monkeypatch.context() as migration_context:
            migration_context.setattr(schema, "WORKSHOP_SCHEMA_VERSION", 58)
            migration_context.setattr(schema, "_MIGRATIONS", schema._MIGRATIONS[:58])
            legacy, run = await _accepted(path)
            try:
                result = await _coordinator(legacy, _Preparation(_Prepared(run))).execute(run.run_id)
                assert result.disposition == CanonicalExecutionDisposition.COMPLETED
                assert await load_runtime_session(legacy, run.channel_id, run.agent_id) is not None

                await legacy.connection.execute(
                    "UPDATE agent_definitions SET owner_runtime_profile_id = ? WHERE agent_id = ?",
                    (replacement_profile, run.agent_id),
                )
                await legacy.connection.commit()
            finally:
                await legacy.close()

        before = workshop_runtime_session_status(path)
        assert before.startswith("Workshop conversation continuity: INCOMPLETE; successful lanes=0, sessions=1")
        assert "missing=0, stale=1" in before

        upgraded = await WorkshopEventStore.open(path)
        try:
            assert await upgraded.schema_version() == 82
            assert await load_runtime_session(upgraded, run.channel_id, run.agent_id) is None
            after = workshop_runtime_session_status(path)
            assert after.startswith("Workshop conversation continuity: active; successful lanes=0, sessions=0")
            assert "missing=0, stale=0" in after
        finally:
            await upgraded.close()

    async def test_version_fifty_nine_retains_a_current_owner_runtime_session(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from kai.workshop import schema

        path = tmp_path / "kai.db"
        with monkeypatch.context() as migration_context:
            migration_context.setattr(schema, "WORKSHOP_SCHEMA_VERSION", 58)
            migration_context.setattr(schema, "_MIGRATIONS", schema._MIGRATIONS[:58])
            legacy, run = await _accepted(path)
            try:
                result = await _coordinator(legacy, _Preparation(_Prepared(run))).execute(run.run_id)
                assert result.disposition == CanonicalExecutionDisposition.COMPLETED
            finally:
                await legacy.close()

        upgraded = await WorkshopEventStore.open(path)
        try:
            assert await upgraded.schema_version() == 82
            session = await load_runtime_session(upgraded, run.channel_id, run.agent_id)
            assert session is not None
            assert session.runtime_profile_id == _RUNTIME_PROFILE_ID
            status = workshop_runtime_session_status(path)
            assert status.startswith("Workshop conversation continuity: active; successful lanes=1, sessions=1")
            assert "context refresh pending=1" in status
            assert "missing=0, stale=0" in status
            async with upgraded.connection.execute(
                "SELECT observed_through_event_position, last_run_id, last_inbound_message_id "
                "FROM channel_agent_conversation_observations WHERE channel_id = ? AND agent_id = ?",
                (run.channel_id, run.agent_id),
            ) as cursor:
                observation = await cursor.fetchone()
            assert observation is not None
            async with upgraded.connection.execute(
                "SELECT created_event_position FROM messages WHERE id = ?",
                (run.inbound_message_id,),
            ) as cursor:
                source = await cursor.fetchone()
            assert source is not None
            assert tuple(observation) == (
                int(source[0]),
                run.run_id,
                run.inbound_message_id,
            )
            assert workshop_conversation_observation_status(path).startswith(
                "Workshop conversation observation: active; cursors=1 (channel=1, thread=0), integrity gaps=0"
            )
        finally:
            await upgraded.close()

    async def test_retired_successful_lane_does_not_require_a_live_provider_session(
        self,
        tmp_path: Path,
    ) -> None:
        store, run = await _accepted(tmp_path / "kai.db")
        try:
            result = await _coordinator(store, _Preparation(_Prepared(run))).execute(run.run_id)
            assert result.disposition == CanonicalExecutionDisposition.COMPLETED

            await store.connection.execute(
                "UPDATE agent_definitions SET lifecycle_state = 'archived' WHERE agent_id = ?",
                (run.agent_id,),
            )
            await store.connection.execute(
                "UPDATE principal_agent_enablements SET lifecycle_state = 'disabled' "
                "WHERE direct_channel_id = ? AND agent_id = ?",
                (run.channel_id, run.agent_id),
            )
            await store.connection.commit()

            retained = workshop_runtime_session_status(tmp_path / "kai.db")
            assert retained.startswith("Workshop conversation continuity: INCOMPLETE; successful lanes=0, sessions=1")
            assert "missing=0, stale=1" in retained

            await store.connection.execute(
                "DELETE FROM channel_agent_runtime_sessions WHERE channel_id = ? AND agent_id = ?",
                (run.channel_id, run.agent_id),
            )
            await store.connection.commit()

            retired = workshop_runtime_session_status(tmp_path / "kai.db")
            assert retired.startswith("Workshop conversation continuity: active; successful lanes=0, sessions=0")
            assert "missing=0, stale=0" in retired
        finally:
            await store.close()

    async def test_cold_restart_bootstraps_only_prior_canonical_timeline(self, tmp_path: Path):
        store, first_run = await _accepted(tmp_path / "kai.db")
        first = _Prepared(first_run)
        try:
            first_result = await _coordinator(store, _Preparation(first)).execute(first_run.run_id)
            assert first_result.disposition == CanonicalExecutionDisposition.COMPLETED
            assert len(first.canonical_histories) == 1
            assert '"record_type":"canonical_conversation_window"' in first.canonical_histories[0]
            assert '"selected_message_count":0' in first.canonical_histories[0]
            assert len(first.agent_definition_contexts) == 1
            assert "Handle: @kai" in first.agent_definition_contexts[0]
            assert "Definition revision: 2" in first.agent_definition_contexts[0]

            second_acceptance = await WorkshopConversationCommandService(store).accept(
                InboundMessage(
                    transport="telegram",
                    update_id="command-2",
                    message_id="message-2",
                    sender_subject="101",
                    channel_subject="101",
                    body="Canonical prompt 2",
                    occurred_at=_NOW,
                )
            )
            second_run = second_acceptance.run
            second = _Prepared(
                second_run,
                response=AgentResponse(
                    success=True,
                    text="Second canonical answer",
                    session_id="provider-session-2",
                ),
            )
            second_result = await _coordinator(store, _Preparation(second)).execute(second_run.run_id)

            assert second_result.disposition == CanonicalExecutionDisposition.COMPLETED
            assert len(second.canonical_histories) == 1
            history = second.canonical_histories[0]
            assert '"author_display_name":"Workshop Human"' in history
            assert '"body":"Canonical prompt 1"' in history
            assert '"author_display_name":"Kai"' in history
            assert '"body":"Canonical answer"' in history
            assert "Canonical prompt 2" not in history
            session = await load_runtime_session(store, second_run.channel_id, second_run.agent_id)
            assert session is not None
            assert session.last_run_id == second_run.run_id
            assert session.provider_session_id == "provider-session-2"
            assert session.last_result_message_id == second_result.run.result_message_id
        finally:
            await store.close()

    async def test_live_runtime_receives_restart_context_without_prompt_duplication(self, tmp_path: Path):
        store, run = await _accepted(tmp_path / "kai.db")
        prepared = _Prepared(run)
        try:
            result = await _coordinator(store, _Preparation(prepared)).execute(run.run_id)

            assert result.disposition == CanonicalExecutionDisposition.COMPLETED
            assert len(prepared.canonical_histories) == 1
            assert '"selected_message_count":0' in prepared.canonical_histories[0]
            assert prepared.canonical_history_options[0]["live_delta"] == ""
            assert prepared.prompts == ["Canonical prompt 1"]
        finally:
            await store.close()

    async def test_stream_observer_sees_only_nonterminal_events(self, tmp_path: Path):
        store, run = await _accepted(tmp_path / "kai.db")
        prepared = _Prepared(run)
        observed: list[StreamEvent] = []

        async def observe(event: StreamEvent) -> None:
            observed.append(event)

        original_stream = prepared.stream

        async def stream_with_preview(prompt: str) -> AsyncIterator[StreamEvent]:
            yield StreamEvent(text_so_far="Stable preview.", done=False)
            async for event in original_stream(prompt):
                yield event

        prepared.stream = stream_with_preview  # type: ignore[method-assign]
        try:
            result = await _coordinator(store, _Preparation(prepared)).execute(
                run.run_id,
                stream_observer=observe,
            )

            assert result.disposition == CanonicalExecutionDisposition.COMPLETED
            assert [event.text_so_far for event in observed] == ["Stable preview."]
        finally:
            await store.close()

    async def test_trace_bearing_events_persist_to_run_traces(self, tmp_path: Path):
        store, run = await _accepted(tmp_path / "kai.db")
        prepared = _Prepared(run)
        original_stream = prepared.stream

        async def stream_with_trace(prompt: str) -> AsyncIterator[StreamEvent]:
            yield StreamEvent(
                text_so_far="",
                trace=TraceEntry(
                    kind="tool_call",
                    tool_use_id="toolu_1",
                    summary="Bash: ls",
                    detail="{}",
                    tool_name="Bash",
                ),
            )
            async for event in original_stream(prompt):
                yield event

        prepared.stream = stream_with_trace  # type: ignore[method-assign]
        try:
            result = await _coordinator(store, _Preparation(prepared)).execute(run.run_id)

            assert result.disposition == CanonicalExecutionDisposition.COMPLETED
            async with store.connection.execute(
                "SELECT seq, kind, summary FROM run_traces WHERE run_id = ?",
                (run.run_id,),
            ) as cursor:
                rows = list(await cursor.fetchall())
            assert [tuple(row) for row in rows] == [(1, "tool_call", "Bash: ls")]
        finally:
            await store.close()

    async def test_attempt_authority_contains_no_model_visible_bearer_material(self, tmp_path: Path):
        store, run = await _accepted(tmp_path / "kai.db")
        prepared = _Prepared(run)
        observed: list[StreamEvent] = []

        async def stream_without_bearer(prompt: str) -> AsyncIterator[StreamEvent]:
            invocation = prepared.collaboration_invocations[-1]
            assert not hasattr(invocation, "token")
            assert "X-Kai-Collaboration-Proof" not in prompt
            yield StreamEvent(text_so_far="preview safe")
            yield StreamEvent(
                text_so_far="",
                trace=TraceEntry(
                    kind="tool_call",
                    tool_use_id="authority-tool",
                    summary="curl collaboration endpoint",
                    detail="ordinary typed arguments",
                    tool_name="curl",
                ),
            )
            yield StreamEvent(
                text_so_far="answer safe",
                done=True,
                response=AgentResponse(success=True, text="answer safe"),
            )

        async def observe(event: StreamEvent) -> None:
            observed.append(event)

        prepared.stream = stream_without_bearer  # type: ignore[method-assign]
        try:
            result = await _coordinator(store, _Preparation(prepared)).execute(
                run.run_id,
                stream_observer=observe,
            )
            assert result.disposition == CanonicalExecutionDisposition.COMPLETED
            assert "X-Kai-Collaboration-Proof" not in "".join(event.text_so_far for event in observed)
            async with store.connection.execute(
                "SELECT summary, detail FROM run_traces WHERE run_id = ?",
                (run.run_id,),
            ) as cursor:
                trace_row = tuple(await cursor.fetchone())
            assert trace_row == (
                "curl collaboration endpoint",
                "ordinary typed arguments",
            )
            assert (await _terminal_bodies(store))[-1] == "answer safe"
        finally:
            await store.close()

    async def test_concurrent_duplicate_dispatches_backend_once(self, tmp_path: Path):
        store, run = await _accepted(tmp_path / "kai.db")
        release = asyncio.Event()
        prepared = _Prepared(run, wait=release)
        preparation = _Preparation(prepared)
        coordinator = _coordinator(store, preparation)
        try:
            first = asyncio.create_task(coordinator.execute(run.run_id))
            while not prepared.prompts:
                await asyncio.sleep(0)
            second = asyncio.create_task(coordinator.execute(run.run_id))
            release.set()
            first_result, second_result = await asyncio.gather(first, second)

            assert first_result.disposition == CanonicalExecutionDisposition.COMPLETED
            assert second_result.disposition == CanonicalExecutionDisposition.TERMINAL_REPLAY
            assert preparation.calls == 1
            assert prepared.prompts == ["Canonical prompt 1"]
        finally:
            await store.close()

    async def test_native_failure_is_replaced_by_bounded_canonical_text(self, tmp_path: Path):
        store, run = await _accepted(tmp_path / "kai.db")
        prepared = _Prepared(
            run,
            response=AgentResponse(
                success=False,
                text="",
                error="native secret-bearing payload",
                failure_kind=AgentFailureKind.AUTHENTICATION_REQUIRED,
            ),
        )
        try:
            result = await _coordinator(store, _Preparation(prepared)).execute(run.run_id)

            assert result.disposition == CanonicalExecutionDisposition.FAILED
            bodies = await _terminal_bodies(store)
            assert (
                bodies[-1] == "Authentication for the configured agent is required. Kai did not complete this request."
            )
            assert "native" not in bodies[-1]
            async with store.connection.execute(
                "SELECT COUNT(*) FROM channel_agent_conversation_observations"
            ) as cursor:
                assert int((await cursor.fetchone())[0]) == 0
        finally:
            await store.close()

    async def test_runtime_drift_leaves_retryable_grant_until_expiry(self, tmp_path: Path, caplog):
        store, run = await _accepted(tmp_path / "kai.db")
        prepared = _Prepared(run)
        prepared.reject_validation = True
        preparation = _Preparation(prepared)
        coordinator = _coordinator(store, preparation, lease_seconds=5)
        try:
            deferred = await coordinator.execute(run.run_id)
            assert deferred.disposition == CanonicalExecutionDisposition.PREPARATION_DEFERRED
            assert (await WorkshopRunLifecycle(store).state(run.run_id)).status == RunStatus.ACCEPTED
            assert f"Workshop run {run.run_id} preparation deferred" in caplog.text
            assert "runtime drift" in caplog.text

            recovered = await coordinator.recover_expired(occurred_at=_NOW + timedelta(seconds=16))
            assert recovered.expired_before_dispatch == 1
            prepared.reject_validation = False
            completed = await coordinator.execute(run.run_id)
            assert completed.disposition == CanonicalExecutionDisposition.COMPLETED
            assert preparation.calls == 2
        finally:
            await store.close()

    async def test_cancellation_is_confirmed_only_after_exact_runtime_stops(self, tmp_path: Path):
        store, run = await _accepted(tmp_path / "kai.db")
        release = asyncio.Event()
        prepared = _Prepared(run, wait=release)
        coordinator = _coordinator(store, _Preparation(prepared))
        try:
            execution = asyncio.create_task(coordinator.execute(run.run_id))
            while not prepared.prompts:
                await asyncio.sleep(0)
            cancellation = await coordinator.request_cancellation(run.run_id)
            result = await execution

            assert cancellation == CanonicalCancellationDisposition.REQUESTED
            assert prepared.cancelled is True
            assert result.disposition == CanonicalExecutionDisposition.CANCELLED
            assert result.run.status == RunStatus.CANCELLED
            assert (await _terminal_bodies(store))[-1] == "This request was cancelled."
        finally:
            await store.close()

    async def test_accepted_run_cancels_durably_before_backend_dispatch(self, tmp_path: Path):
        store, run = await _accepted(tmp_path / "kai.db")
        prepared = _Prepared(run)
        preparation = _Preparation(prepared)
        coordinator = _coordinator(store, preparation)
        try:
            cancellation = await coordinator.request_cancellation(run.run_id)
            current = await WorkshopRunLifecycle(store).state(run.run_id)

            assert cancellation == CanonicalCancellationDisposition.REQUESTED
            assert current.status == RunStatus.CANCELLED
            assert current.terminal_code == "requested_by_human"
            assert preparation.calls == 0
            assert await _terminal_bodies(store) == ["Canonical prompt 1"]
        finally:
            await store.close()

    async def test_expired_started_attempt_gets_visible_interruption_without_redispatch(self, tmp_path: Path):
        store, run = await _accepted(tmp_path / "kai.db")
        selection = RunExecutionSelection("codex", "gpt-5.6-sol")
        authority = WorkshopRunExecutionAuthority(
            store,
            selection_resolver=lambda _run: selection,
            registered_backend_ids=frozenset({"codex"}),
        )
        granted = await authority.grant(
            run.run_id,
            owner_id=RunExecutionOwnerId.new(),
            occurred_at=_NOW + timedelta(seconds=1),
            lease_expires_at=_NOW + timedelta(seconds=2),
        )
        await authority.start(granted.claim, occurred_at=_NOW + timedelta(seconds=1, milliseconds=500))
        prepared = _Prepared(run)
        preparation = _Preparation(prepared)
        coordinator = _coordinator(store, preparation)
        try:
            recovered = await coordinator.recover_expired(occurred_at=_NOW + timedelta(seconds=3))

            assert recovered.interrupted_after_dispatch == 1
            assert (await authority.attempt(granted.claim.attempt_id)).status == RunAttemptStatus.INTERRUPTED
            assert (await WorkshopRunLifecycle(store).state(run.run_id)).status == RunStatus.FAILED
            assert (await _terminal_bodies(store))[-1] == (
                "Kai was interrupted while the configured agent was working. This request was not retried."
            )
            manifests = await WorkshopContextManifestService(store).load_run(run.run_id)
            assert len(manifests) == 1
            assert manifests[0].draft.workspace_kind == "unknown"
            assert {source.reason for source in manifests[0].draft.sources} >= {
                "dispatch_state_unknown",
                "not_observable",
            }
            replay = await coordinator.execute(run.run_id)
            assert replay.disposition == CanonicalExecutionDisposition.TERMINAL_REPLAY
            assert preparation.calls == 0
        finally:
            await store.close()

    async def test_coordinator_remains_absent_from_production_construction(self):
        source_root = Path(__file__).parents[1] / "src" / "kai"
        for relative_path in ("main.py", "bot.py", "sessions.py"):
            source = (source_root / relative_path).read_text(encoding="utf-8")
            assert "WorkshopCanonicalExecutionCoordinator" not in source
