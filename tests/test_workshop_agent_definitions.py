"""Contracts for canonical, versioned Workshop agent definitions."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from kai.workshop.agent_definitions import (
    MAX_AGENT_INSTRUCTIONS,
    active_agent_definition_revision,
    load_agent_definition_revision,
    render_agent_definition_context,
)
from kai.workshop.agent_lifecycle import (
    WorkshopAgentLifecycleConflict,
    WorkshopAgentLifecycleService,
)
from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.diagnostics import workshop_agent_authority_status
from kai.workshop.domain import (
    AgentDefinitionId,
    AgentDefinitionRevisionId,
    AgentId,
    EventEnvelope,
    MessageId,
    PrincipalId,
    WorkshopEventType,
)
from kai.workshop.inbound import InboundMessage, record_inbound_message, resolve_message_mentions
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.run_lifecycle import WorkshopRunLifecycle
from kai.workshop.store import WorkshopEventStore

_NOW = datetime(2026, 8, 29, 12, 0, tzinfo=UTC)


async def _open(path: Path) -> tuple[WorkshopEventStore, AgentId]:
    store = await WorkshopEventStore.open(path)
    result = await bootstrap_default_workshop(
        store,
        (
            BootstrapHuman(
                display_name="Workshop Human",
                role="admin",
                transport="desktop",
                external_subject="human-1",
                external_channel_id="human-1",
            ),
        ),
    )
    return store, result.agent_id


async def _append_and_project(store: WorkshopEventStore, envelope: EventEnvelope) -> None:
    await store.append(envelope)
    await store.project_pending(CanonicalConversationProjection())


async def _record(store: WorkshopEventStore, number: int) -> MessageId:
    result = await record_inbound_message(
        store,
        InboundMessage(
            transport="desktop",
            update_id=f"command-{number}",
            message_id=f"message-{number}",
            sender_subject="human-1",
            channel_subject="human-1",
            body=f"Canonical prompt {number}",
            occurred_at=_NOW + timedelta(minutes=number),
        ),
    )
    message_id = result.event.envelope.aggregate_id
    assert isinstance(message_id, MessageId)
    return message_id


class TestAgentDefinitionBootstrap:
    async def test_agent_handle_cannot_collide_with_a_human_handle(self, tmp_path: Path):
        store, _ = await _open(tmp_path / "kai.db")
        try:
            async with store.connection.execute(
                "SELECT principal_id FROM external_identities "
                "WHERE provider = 'desktop' AND external_subject = 'human-1'"
            ) as cursor:
                row = await cursor.fetchone()
            assert row is not None
            with pytest.raises(WorkshopAgentLifecycleConflict, match="used by a human"):
                await WorkshopAgentLifecycleService(store).create_draft(
                    PrincipalId(str(row[0])),
                    idempotency_key="human-handle-collision",
                    handle="workshop_human",
                    display_name="Collision",
                    description="Must not be created.",
                    presentation={"avatar": "C"},
                    purpose="Test shared namespace.",
                    instructions="Do nothing.",
                    capabilities=["text_generation"],
                )
        finally:
            await store.close()

    async def test_bootstrap_creates_one_active_kai_revision_and_replays(self, tmp_path: Path):
        store, agent_id = await _open(tmp_path / "kai.db")
        try:
            revision = await active_agent_definition_revision(store, agent_id)
            assert revision is not None
            assert revision.handle == "kai"
            assert revision.display_name == "Kai"
            assert revision.revision_number == 2
            assert revision.capabilities == ("agent_delegation", "text_generation")

            await store.rebuild_projection(CanonicalConversationProjection())
            assert await active_agent_definition_revision(store, agent_id) == revision
            assert workshop_agent_authority_status(tmp_path / "kai.db") == (
                "Workshop agent authority: active; definitions=1 (active=1, draft=0, archived=0), "
                "revisions=2, enablements=0 (enabled=0), direct channels=0, attachments=0 "
                "(detached=0), runtime sponsorships=0, delegation trees=0, delegations=0 "
                "(nonterminal=0); integrity gaps=0 (definitions=0, missing revisions=0, "
                "stale revisions=0, ambiguous revisions=0, principals=0, handles=0, "
                "enablements=0, runtime bindings=0, namespaces=0, attachments=0, "
                "delegations=0); authority=canonical"
            )
        finally:
            await store.close()

    async def test_diagnostics_fail_closed_for_invalid_handle_and_active_revision(
        self,
        tmp_path: Path,
    ) -> None:
        store, agent_id = await _open(tmp_path / "kai.db")
        try:
            await store.connection.execute(
                "UPDATE agent_definitions SET handle = 'Kai', active_revision_id = NULL WHERE agent_id = ?",
                (agent_id,),
            )
            await store.connection.commit()

            status = workshop_agent_authority_status(tmp_path / "kai.db")

            assert status.startswith("Workshop agent authority: INCOMPLETE;")
            assert "stale revisions=1" in status
            assert "handles=1" in status
        finally:
            await store.close()

    async def test_handle_mentions_are_case_insensitive(self, tmp_path: Path):
        store, agent_id = await _open(tmp_path / "kai.db")
        try:
            async with store.connection.execute(
                "SELECT channel_id FROM channel_agents WHERE agent_id = ?", (agent_id,)
            ) as cursor:
                channel_id = (await cursor.fetchone())[0]
            mentions = await resolve_message_mentions(store, channel_id, "@kAi please respond")
            assert len(mentions) == 1
            assert mentions[0].kind == "agent"
            assert (mentions[0].start, mentions[0].length) == (0, 4)
        finally:
            await store.close()


class TestAgentDefinitionRevisions:
    async def test_runs_keep_exact_revision_after_later_activation(self, tmp_path: Path):
        store, _agent_id = await _open(tmp_path / "kai.db")
        try:
            first = await WorkshopRunLifecycle(store).accept(
                await _record(store, 1), occurred_at=_NOW + timedelta(minutes=1, seconds=1)
            )
            revision_one_id = first.run.agent_definition_revision_id
            assert revision_one_id is not None
            revision_one = await load_agent_definition_revision(store, revision_one_id)
            assert revision_one is not None

            definition_id = revision_one.definition_id
            revision_two_id = AgentDefinitionRevisionId.derived(definition_id, "revision:3")
            await _append_and_project(
                store,
                EventEnvelope.create(
                    event_type=WorkshopEventType.AGENT_DEFINITION_REVISION_ADDED,
                    event_version=1,
                    workshop_id=first.run.workshop_id,
                    aggregate_type="agent_definition_revision",
                    aggregate_id=revision_two_id,
                    occurred_at=_NOW + timedelta(minutes=2),
                    payload={
                        "definition_id": definition_id,
                        "revision_number": 3,
                        "purpose": "Exercise the second immutable definition.",
                        "instructions": "Use the second definition without changing older runs.",
                        "capabilities": ["text_generation", "tool_activity"],
                    },
                ),
            )
            await _append_and_project(
                store,
                EventEnvelope.create(
                    event_type=WorkshopEventType.AGENT_DEFINITION_REVISION_ACTIVATED,
                    event_version=1,
                    workshop_id=first.run.workshop_id,
                    aggregate_type="agent_definition",
                    aggregate_id=definition_id,
                    occurred_at=_NOW + timedelta(minutes=2, seconds=1),
                    payload={"revision_id": revision_two_id},
                ),
            )

            second = await WorkshopRunLifecycle(store).accept(
                await _record(store, 3), occurred_at=_NOW + timedelta(minutes=3, seconds=1)
            )
            assert first.run.agent_definition_revision_id == revision_one_id
            assert second.run.agent_definition_revision_id == revision_two_id
            assert (await load_agent_definition_revision(store, revision_one_id)) == revision_one
            async with store.connection.execute(
                "SELECT COUNT(*) FROM agent_definition_revisions WHERE agent_definition_id = ?",
                (definition_id,),
            ) as cursor:
                assert (await cursor.fetchone())[0] == 3
        finally:
            await store.close()

    async def test_rendered_definition_explicitly_carries_no_authority(self, tmp_path: Path):
        store, agent_id = await _open(tmp_path / "kai.db")
        try:
            revision = await active_agent_definition_revision(store, agent_id)
            assert revision is not None
            rendered = render_agent_definition_context(revision)
            assert f"Definition revision: 2 ({revision.revision_id})" in rendered
            assert "does not grant tools, credentials, data access, identity, or permission" in rendered
        finally:
            await store.close()

    async def test_archival_disables_new_resolution_but_preserves_run_provenance(
        self,
        tmp_path: Path,
    ):
        store, agent_id = await _open(tmp_path / "kai.db")
        try:
            active = await active_agent_definition_revision(store, agent_id)
            assert active is not None
            accepted = await WorkshopRunLifecycle(store).accept(
                await _record(store, 1), occurred_at=_NOW + timedelta(minutes=1, seconds=1)
            )
            assert accepted.run.agent_definition_revision_id == active.revision_id
            async with store.connection.execute(
                "SELECT principal_id FROM external_identities "
                "WHERE provider = 'desktop' AND external_subject = 'human-1'"
            ) as cursor:
                principal_row = await cursor.fetchone()
            assert principal_row is not None
            principal_id = PrincipalId(str(principal_row[0]))
            service = WorkshopAgentLifecycleService(store)
            current = await service.get_visible(principal_id, active.definition_id)
            archived = await service.archive(
                principal_id,
                active.definition_id,
                idempotency_key="archive-kai-provenance-test",
                expected_version=current.state_version,
            )
            assert archived.lifecycle_state == "archived"
            assert archived.active_revision_id == active.revision_id
            assert await active_agent_definition_revision(store, agent_id) is None
            preserved = await load_agent_definition_revision(store, active.revision_id)
            assert preserved is not None
            assert preserved.revision_id == active.revision_id
            assert preserved.instructions == active.instructions
            async with store.connection.execute(
                "SELECT agent_definition_revision_id FROM runs WHERE id = ?",
                (accepted.run.run_id,),
            ) as cursor:
                run_row = await cursor.fetchone()
            assert run_row is not None
            assert AgentDefinitionRevisionId(str(run_row[0])) == active.revision_id

            await store.rebuild_projection(CanonicalConversationProjection())
            replayed_revision = await load_agent_definition_revision(store, active.revision_id)
            assert replayed_revision is not None
            assert replayed_revision.revision_id == active.revision_id
            assert replayed_revision.instructions == active.instructions
            async with store.connection.execute(
                "SELECT agent_definition_revision_id FROM runs WHERE id = ?",
                (accepted.run.run_id,),
            ) as cursor:
                replayed_run = await cursor.fetchone()
            assert replayed_run is not None
            assert AgentDefinitionRevisionId(str(replayed_run[0])) == active.revision_id
        finally:
            await store.close()


class TestAgentDefinitionValidation:
    @pytest.mark.parametrize(
        "payload, message",
        (
            (
                {
                    "agent_id": None,
                    "handle": "kai",
                    "display_name": "Kai",
                    "description": "description",
                    "presentation": {"avatar": "K"},
                    "lifecycle_state": "active",
                    "credentials": {"token": "secret"},
                },
                "exactly",
            ),
            (
                {
                    "agent_id": None,
                    "handle": "not a handle",
                    "display_name": "Kai",
                    "description": "description",
                    "presentation": {"avatar": "K"},
                    "lifecycle_state": "active",
                },
                "handle",
            ),
        ),
    )
    async def test_definition_rejects_authority_fields_and_bad_handles(
        self, tmp_path: Path, payload: dict[str, object], message: str
    ):
        store, agent_id = await _open(tmp_path / f"{message}.db")
        try:
            payload["agent_id"] = agent_id
            definition_id = AgentDefinitionId.new()
            await store.append(
                EventEnvelope.create(
                    event_type=WorkshopEventType.AGENT_DEFINITION_CREATED,
                    event_version=1,
                    workshop_id=(await active_agent_definition_revision(store, agent_id)).workshop_id,
                    aggregate_type="agent_definition",
                    aggregate_id=definition_id,
                    occurred_at=_NOW,
                    payload=payload,
                )
            )
            with pytest.raises((TypeError, ValueError), match=message):
                await store.project_pending(CanonicalConversationProjection())
        finally:
            await store.close()

    @pytest.mark.parametrize(
        "instructions, capabilities, message",
        (
            ("x" * (MAX_AGENT_INSTRUCTIONS + 1), ["text_generation"], "instructions"),
            ("Otherwise valid instructions", ["root_access"], "unsupported value"),
        ),
    )
    async def test_revision_rejects_oversized_instructions_and_unknown_capability(
        self,
        tmp_path: Path,
        instructions: str,
        capabilities: list[str],
        message: str,
    ):
        store, agent_id = await _open(tmp_path / f"invalid-revision-{message}.db")
        try:
            active = await active_agent_definition_revision(store, agent_id)
            assert active is not None
            revision_id = AgentDefinitionRevisionId.new()
            await store.append(
                EventEnvelope.create(
                    event_type=WorkshopEventType.AGENT_DEFINITION_REVISION_ADDED,
                    event_version=1,
                    workshop_id=active.workshop_id,
                    aggregate_type="agent_definition_revision",
                    aggregate_id=revision_id,
                    occurred_at=_NOW,
                    payload={
                        "definition_id": active.definition_id,
                        "revision_number": 2,
                        "purpose": "Invalid revision",
                        "instructions": instructions,
                        "capabilities": capabilities,
                    },
                )
            )
            with pytest.raises(ValueError, match=message):
                await store.project_pending(CanonicalConversationProjection())
        finally:
            await store.close()
