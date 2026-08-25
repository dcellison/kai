"""Canonical authority and idempotency for protected proactive publication."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from kai.workshop.artifacts import WorkshopArtifactService, artifact_for_delivery
from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.delivery_policy import WorkshopDeliveryBindingPolicy
from kai.workshop.domain import AgentId
from kai.workshop.internal_api_contexts import WorkshopInternalAPIContextRegistry
from kai.workshop.proactive_publication import (
    PROACTIVE_ARTIFACT_MODE,
    ProactivePublicationAuthority,
    ProactivePublicationError,
    WorkshopProactivePublicationService,
)
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.storage_namespaces import WorkshopPrincipalStorageRegistry
from kai.workshop.store import IdempotencyConflictError, WorkshopEventStore
from tests.workshop_profiles import profile_id, profile_registry

_NOW = datetime(2026, 8, 24, 12, 0, tzinfo=UTC)


async def _publication_service(
    tmp_path: Path,
    *,
    delivery_transports: frozenset[str] = frozenset({"telegram"}),
):
    store = await WorkshopEventStore.open(tmp_path / "kai.db")
    profiles = profile_registry(101)
    await bootstrap_default_workshop(
        store,
        (
            BootstrapHuman(
                "Daniel",
                "admin",
                "telegram",
                "101",
                "101",
                profile_id(101),
            ),
        ),
    )
    contexts = await WorkshopInternalAPIContextRegistry.from_store(store, profiles)
    context = contexts.for_runtime_profile(profile_id(101))
    storage = await WorkshopPrincipalStorageRegistry.from_store(store, profiles)
    artifacts = WorkshopArtifactService(
        store,
        data_dir=tmp_path,
        principal_storage=storage,
        runtime_profiles=profiles,
    )
    service = WorkshopProactivePublicationService(
        store,
        artifacts,
        artifact_storage_root=tmp_path / "files",
        delivery_policy=WorkshopDeliveryBindingPolicy(delivery_transports),
    )
    authority = ProactivePublicationAuthority(
        context.principal_id,
        context.channel_id,
        context.agent_id,
        context.runtime_profile_id,
    )
    return store, service, authority


async def test_workshop_only_text_is_canonical_without_adapter_delivery(tmp_path: Path):
    store, service, authority = await _publication_service(
        tmp_path,
        delivery_transports=frozenset(),
    )
    try:
        result = await service.publish_text(
            authority,
            request_id="workshop-only-text",
            body="Canonical without Telegram",
            occurred_at=_NOW,
        )

        assert result.inserted is True
        assert result.deliveries == ()
        assert result.delivery_status == "not_configured"
        async with store.connection.execute(
            "SELECT m.body, p.kind, e.metadata_json FROM messages m "
            "JOIN principals p ON p.id = m.author_principal_id "
            "JOIN event_log e ON e.position = m.created_event_position WHERE m.id = ?",
            (result.message_id,),
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None
        assert row[0:2] == ("Canonical without Telegram", "agent")
        assert json.loads(str(row[2])) == {
            "publication_kind": "text",
            "request_id": "workshop-only-text",
            "runtime_profile_id": authority.runtime_profile_id,
            "source": "internal_api",
        }
    finally:
        await store.close()


async def test_hybrid_text_records_once_and_requests_bound_delivery(tmp_path: Path):
    store, service, authority = await _publication_service(tmp_path)
    try:
        first = await service.publish_text(
            authority,
            request_id="hybrid-text",
            body="Canonical then adapter",
            occurred_at=_NOW,
        )
        replay = await service.publish_text(
            authority,
            request_id="hybrid-text",
            body="Canonical then adapter",
            occurred_at=_NOW,
        )

        assert first.inserted is True
        assert replay.inserted is False
        assert replay.message_id == first.message_id
        assert len(first.deliveries) == len(replay.deliveries) == 1
        assert first.deliveries[0].inserted is True
        assert replay.deliveries[0].inserted is False
        assert first.delivery_status == "queued"
        async with store.connection.execute(
            "SELECT COUNT(*) FROM messages WHERE id = ?",
            (first.message_id,),
        ) as cursor:
            assert (await cursor.fetchone())[0] == 1
        async with store.connection.execute("SELECT transport, mode, purpose, status FROM delivery_outbox") as cursor:
            assert tuple(await cursor.fetchone()) == ("telegram", "text", "notification", "pending")

        with pytest.raises(IdempotencyConflictError):
            await service.publish_text(
                authority,
                request_id="hybrid-text",
                body="Conflicting retry",
                occurred_at=_NOW,
            )
    finally:
        await store.close()


async def test_file_is_one_agent_artifact_and_one_durable_adapter_request(tmp_path: Path):
    store, service, authority = await _publication_service(tmp_path)
    source = tmp_path / "report.txt"
    source.write_text("publication artifact", encoding="utf-8")
    try:
        first = await service.publish_file(
            authority,
            request_id="hybrid-file",
            path=source.resolve(),
            caption="Qualification report",
            occurred_at=_NOW,
        )
        replay = await service.publish_file(
            authority,
            request_id="hybrid-file",
            path=source.resolve(),
            caption="Qualification report",
            occurred_at=_NOW,
        )

        assert first.inserted is True
        assert replay.inserted is False
        assert first.message_id == replay.message_id
        assert len(first.deliveries) == 1
        assert first.deliveries[0].delivery.mode == PROACTIVE_ARTIFACT_MODE
        await store.rebuild_projection(CanonicalConversationProjection())
        async with store.connection.execute(
            "SELECT m.body, p.kind, COUNT(a.id) FROM messages m "
            "JOIN principals p ON p.id = m.author_principal_id "
            "LEFT JOIN artifacts a ON a.message_id = m.id WHERE m.id = ? GROUP BY m.id",
            (first.message_id,),
        ) as cursor:
            assert tuple(await cursor.fetchone()) == ("Qualification report", "agent", 1)
        artifact, caption = await artifact_for_delivery(
            store,
            first.message_id,
            storage_root=tmp_path / "files",
        )
        assert artifact.storage_path.read_text(encoding="utf-8") == "publication artifact"
        assert artifact.summary.original_filename == "report.txt"
        assert caption == "Qualification report"
    finally:
        await store.close()


async def test_invalid_canonical_lane_fails_before_any_publication(tmp_path: Path):
    store, service, authority = await _publication_service(tmp_path)
    invalid = replace(authority, agent_id=AgentId("agt_" + "f" * 32))
    try:
        with pytest.raises(ProactivePublicationError, match="missing or ambiguous"):
            await service.publish_text(
                invalid,
                request_id="forbidden",
                body="Do not record",
                occurred_at=_NOW,
            )
        async with store.connection.execute("SELECT COUNT(*) FROM messages") as cursor:
            assert (await cursor.fetchone())[0] == 0
        async with store.connection.execute("SELECT COUNT(*) FROM delivery_outbox") as cursor:
            assert (await cursor.fetchone())[0] == 0
    finally:
        await store.close()
