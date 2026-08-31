"""Canonical Workshop human-handle authority contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.diagnostics import workshop_human_handle_status
from kai.workshop.domain import (
    EventEnvelope,
    PrincipalId,
    WorkshopEventType,
    WorkshopId,
    WorkshopMembershipId,
)
from kai.workshop.human_handles import (
    WorkshopHumanHandleError,
    derive_human_handle,
    normalize_human_handle,
    reconcile_human_handles,
)
from kai.workshop.inbound import InboundMessage, record_inbound_message
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.store import WorkshopEventStore

_NOW = datetime(2026, 8, 31, 10, 0, tzinfo=UTC)


class TestHumanHandleValues:
    @pytest.mark.parametrize(
        ("display_name", "expected"),
        [
            ("Daniel", "daniel"),
            ("Milestone 1 qualification", "milestone_1_qualification"),
            ("Élodie Example", "elodie_example"),
            ("42 Tester", "human_42_tester"),
        ],
    )
    def test_derives_transport_independent_initial_handles(
        self,
        display_name: str,
        expected: str,
    ) -> None:
        assert derive_human_handle(display_name) == expected

    def test_normalizes_case_but_rejects_noncanonical_shapes(self) -> None:
        assert normalize_human_handle("  DaNiEl  ") == "daniel"
        for value in ("two words", "2daniel", "daniel-name", "x" * 33):
            with pytest.raises(WorkshopHumanHandleError):
                normalize_human_handle(value)


class TestHumanHandleAuthority:
    async def test_case_punctuation_unicode_spans_and_nonmember_isolation(
        self,
        tmp_path: Path,
    ) -> None:
        store = await WorkshopEventStore.open(tmp_path / "kai.db")
        try:
            await bootstrap_default_workshop(
                store,
                (BootstrapHuman("Dániel", "admin", "desktop", "one", "one", handle="daniel"),),
            )
            async with store.connection.execute(
                "SELECT c.workshop_id, c.id, ei.principal_id FROM channel_bindings cb "
                "JOIN channels c ON c.id = cb.channel_id "
                "JOIN external_identities ei ON ei.provider = cb.transport "
                "AND ei.external_subject = cb.external_channel_id "
                "WHERE cb.transport = 'desktop' AND cb.external_channel_id = 'one'"
            ) as cursor:
                context = await cursor.fetchone()
            assert context is not None
            workshop_id = WorkshopId(str(context[0]))
            daniel_id = PrincipalId(str(context[2]))

            outsider_id = PrincipalId.new()
            await store.append(
                EventEnvelope.create(
                    event_type=WorkshopEventType.PRINCIPAL_CREATED,
                    event_version=2,
                    workshop_id=workshop_id,
                    aggregate_type="principal",
                    aggregate_id=outsider_id,
                    occurred_at=_NOW,
                    payload={
                        "kind": "human",
                        "display_name": "Outsider",
                        "handle": "outsider",
                    },
                )
            )
            await store.append(
                EventEnvelope.create(
                    event_type=WorkshopEventType.WORKSHOP_MEMBER_ADDED,
                    event_version=1,
                    workshop_id=workshop_id,
                    aggregate_type="workshop_membership",
                    aggregate_id=WorkshopMembershipId.new(),
                    occurred_at=_NOW,
                    payload={"principal_id": outsider_id, "role": "member"},
                )
            )
            other_workshop_id = WorkshopId.new()
            other_daniel_id = PrincipalId.new()
            await store.append(
                EventEnvelope.create(
                    event_type=WorkshopEventType.WORKSHOP_CREATED,
                    event_version=1,
                    workshop_id=other_workshop_id,
                    aggregate_type="workshop",
                    aggregate_id=other_workshop_id,
                    occurred_at=_NOW,
                    payload={"name": "Other Workshop"},
                )
            )
            await store.append(
                EventEnvelope.create(
                    event_type=WorkshopEventType.PRINCIPAL_CREATED,
                    event_version=2,
                    workshop_id=other_workshop_id,
                    aggregate_type="principal",
                    aggregate_id=other_daniel_id,
                    occurred_at=_NOW,
                    payload={
                        "kind": "human",
                        "display_name": "Other Daniel",
                        "handle": "daniel",
                    },
                )
            )
            await store.append(
                EventEnvelope.create(
                    event_type=WorkshopEventType.WORKSHOP_MEMBER_ADDED,
                    event_version=1,
                    workshop_id=other_workshop_id,
                    aggregate_type="workshop_membership",
                    aggregate_id=WorkshopMembershipId.new(),
                    occurred_at=_NOW,
                    payload={"principal_id": other_daniel_id, "role": "member"},
                )
            )
            await store.project_pending(CanonicalConversationProjection())

            body = "😀 (@DaNiEl), ignore @outsider and email@example.com."
            result = await record_inbound_message(
                store,
                InboundMessage(
                    transport="desktop",
                    update_id="unicode-mention",
                    message_id="unicode-mention",
                    sender_subject="one",
                    channel_subject="one",
                    body=body,
                    occurred_at=_NOW,
                ),
            )
            assert result.event.envelope.payload["mentions"] == [
                {
                    "principal_id": daniel_id,
                    "kind": "human",
                    "start": body.index("@DaNiEl"),
                    "length": len("@DaNiEl"),
                }
            ]
            message_id = result.event.envelope.aggregate_id
            async with store.connection.execute(
                "SELECT mentions_json FROM messages WHERE id = ?",
                (message_id,),
            ) as cursor:
                before = str((await cursor.fetchone())[0])
            await store.connection.execute(
                "UPDATE principals SET display_name = 'Renamed Daniel' WHERE id = ?",
                (daniel_id,),
            )
            await store.connection.commit()
            async with store.connection.execute(
                "SELECT mentions_json FROM messages WHERE id = ?",
                (message_id,),
            ) as cursor:
                assert str((await cursor.fetchone())[0]) == before
            await store.rebuild_projection(CanonicalConversationProjection())
            async with store.connection.execute(
                "SELECT mentions_json FROM messages WHERE id = ?",
                (message_id,),
            ) as cursor:
                assert str((await cursor.fetchone())[0]) == before
        finally:
            await store.close()

    async def test_fresh_bootstrap_projects_versioned_handles_and_replays(
        self,
        tmp_path: Path,
    ) -> None:
        store = await WorkshopEventStore.open(tmp_path / "kai.db")
        try:
            await bootstrap_default_workshop(
                store,
                (
                    BootstrapHuman("Daniel", "admin", "desktop", "one", "one"),
                    BootstrapHuman("Scott", "member", "desktop", "two", "two"),
                ),
            )
            async with store.connection.execute(
                "SELECT p.display_name, hh.handle FROM human_handles hh "
                "JOIN principals p ON p.id = hh.principal_id ORDER BY hh.handle"
            ) as cursor:
                assert [tuple(row) for row in await cursor.fetchall()] == [
                    ("Daniel", "daniel"),
                    ("Scott", "scott"),
                ]
            async with store.connection.execute(
                "SELECT DISTINCT event_version FROM event_log "
                "WHERE event_type = 'principal.created' "
                "AND json_extract(payload_json, '$.kind') = 'human'"
            ) as cursor:
                assert [int(row[0]) for row in await cursor.fetchall()] == [2]

            await store.rebuild_projection(CanonicalConversationProjection())
            async with store.connection.execute("SELECT handle FROM human_handles ORDER BY handle") as cursor:
                assert [str(row[0]) for row in await cursor.fetchall()] == [
                    "daniel",
                    "scott",
                ]
        finally:
            await store.close()

    async def test_legacy_human_receives_one_replayable_migration_fact(
        self,
        tmp_path: Path,
    ) -> None:
        store = await WorkshopEventStore.open(tmp_path / "kai.db")
        workshop_id = WorkshopId.new()
        principal_id = PrincipalId.new()
        try:
            await store.append(
                EventEnvelope.create(
                    event_type=WorkshopEventType.WORKSHOP_CREATED,
                    event_version=1,
                    workshop_id=workshop_id,
                    aggregate_type="workshop",
                    aggregate_id=workshop_id,
                    occurred_at=_NOW,
                    payload={"name": "Migration"},
                )
            )
            await store.append(
                EventEnvelope.create(
                    event_type=WorkshopEventType.PRINCIPAL_CREATED,
                    event_version=1,
                    workshop_id=workshop_id,
                    aggregate_type="principal",
                    aggregate_id=principal_id,
                    occurred_at=_NOW,
                    payload={"kind": "human", "display_name": "Legacy Human"},
                )
            )
            await store.append(
                EventEnvelope.create(
                    event_type=WorkshopEventType.WORKSHOP_MEMBER_ADDED,
                    event_version=1,
                    workshop_id=workshop_id,
                    aggregate_type="workshop_membership",
                    aggregate_id=WorkshopMembershipId.new(),
                    occurred_at=_NOW,
                    payload={"principal_id": principal_id, "role": "member"},
                )
            )
            await store.rebuild_projection(CanonicalConversationProjection())

            first = await reconcile_human_handles(store)
            second = await reconcile_human_handles(store)
            assert (first.eligible, first.assigned, first.migrated, first.missing) == (
                1,
                1,
                1,
                0,
            )
            assert second.migrated == 0
            async with store.connection.execute(
                "SELECT handle FROM human_handles WHERE principal_id = ?",
                (principal_id,),
            ) as cursor:
                assert str((await cursor.fetchone())[0]) == "legacy_human"

            await store.rebuild_projection(CanonicalConversationProjection())
            async with store.connection.execute(
                "SELECT handle FROM human_handles WHERE principal_id = ?",
                (principal_id,),
            ) as cursor:
                assert str((await cursor.fetchone())[0]) == "legacy_human"
            assert "active" in workshop_human_handle_status(tmp_path / "kai.db")
        finally:
            await store.close()

    async def test_ambiguous_legacy_names_fail_closed_without_partial_assignment(
        self,
        tmp_path: Path,
    ) -> None:
        store = await WorkshopEventStore.open(tmp_path / "kai.db")
        workshop_id = WorkshopId.new()
        try:
            await store.append(
                EventEnvelope.create(
                    event_type=WorkshopEventType.WORKSHOP_CREATED,
                    event_version=1,
                    workshop_id=workshop_id,
                    aggregate_type="workshop",
                    aggregate_id=workshop_id,
                    occurred_at=_NOW,
                    payload={"name": "Collision"},
                )
            )
            for _index in range(2):
                principal_id = PrincipalId.new()
                await store.append(
                    EventEnvelope.create(
                        event_type=WorkshopEventType.PRINCIPAL_CREATED,
                        event_version=1,
                        workshop_id=workshop_id,
                        aggregate_type="principal",
                        aggregate_id=principal_id,
                        occurred_at=_NOW,
                        payload={"kind": "human", "display_name": "Same Name"},
                    )
                )
                await store.append(
                    EventEnvelope.create(
                        event_type=WorkshopEventType.WORKSHOP_MEMBER_ADDED,
                        event_version=1,
                        workshop_id=workshop_id,
                        aggregate_type="workshop_membership",
                        aggregate_id=WorkshopMembershipId.new(),
                        occurred_at=_NOW,
                        payload={"principal_id": principal_id, "role": "member"},
                    )
                )
            await store.rebuild_projection(CanonicalConversationProjection())

            result = await reconcile_human_handles(store)
            assert (result.eligible, result.assigned, result.conflicting) == (2, 0, 2)
            status = workshop_human_handle_status(tmp_path / "kai.db")
            assert "INCOMPLETE" in status
            assert "conflicting=2" in status
        finally:
            await store.close()
