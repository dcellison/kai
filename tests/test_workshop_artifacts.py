"""Contracts for the canonical Workshop artifact service."""

from __future__ import annotations

import hashlib
import stat
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from kai import sessions
from kai.workshop.artifacts import (
    ArtifactAccessDeniedError,
    ArtifactMessageNotFoundError,
    ArtifactStorageBoundaryError,
    ArtifactTooLargeError,
    InboundArtifact,
    WorkshopArtifactService,
    build_agent_prompt_for_message,
    canonical_artifact_filename,
    record_inbound_artifact,
)
from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.domain import ArtifactId, ChannelId, MessageId, PrincipalId
from kai.workshop.inbound import InboundMessage, record_inbound_message
from kai.workshop.outbound import OutboundMessage, record_outbound_message
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.store import IdempotencyConflictError, WorkshopEventStore
from tests.workshop_profiles import profile_id, profile_registry

_NOW = datetime(2026, 8, 11, 21, 0, tzinfo=UTC)


async def _open_store(path: Path) -> WorkshopEventStore:
    store = await WorkshopEventStore.open(path)
    await bootstrap_default_workshop(
        store,
        (BootstrapHuman("Alice", "admin", "telegram", "101", "101"),),
    )
    return store


async def _artifact_service(
    tmp_path: Path,
) -> tuple[WorkshopEventStore, WorkshopArtifactService, PrincipalId, ChannelId]:
    store = await WorkshopEventStore.open(tmp_path / "kai.db")
    profiles = profile_registry(101)
    await bootstrap_default_workshop(
        store,
        (
            BootstrapHuman(
                "Alice",
                "admin",
                "telegram",
                "101",
                "101",
                profile_id(101),
            ),
        ),
    )
    async with store.connection.execute(
        "SELECT cm.principal_id, c.id FROM channels c "
        "JOIN channel_memberships cm ON cm.channel_id = c.id "
        "JOIN principals p ON p.id = cm.principal_id "
        "WHERE c.kind = 'direct' AND p.kind = 'human'"
    ) as cursor:
        row = await cursor.fetchone()
    assert row is not None
    from kai.workshop.storage_namespaces import WorkshopPrincipalStorageRegistry

    storage = await WorkshopPrincipalStorageRegistry.from_store(store, profiles)
    return (
        store,
        WorkshopArtifactService(
            store,
            data_dir=tmp_path,
            principal_storage=storage,
            runtime_profiles=profiles,
        ),
        PrincipalId(str(row[0])),
        ChannelId(str(row[1])),
    )


async def _chunks(*values: bytes):
    for value in values:
        yield value


async def _inbound_message_id(store: WorkshopEventStore) -> MessageId:
    result = await record_inbound_message(
        store,
        InboundMessage(
            transport="telegram",
            update_id="9001",
            message_id="42",
            sender_subject="101",
            channel_subject="101",
            body="Photo from Telegram",
            occurred_at=_NOW,
        ),
    )
    return MessageId(str(result.event.envelope.aggregate_id))


def _artifact(message_id: MessageId, path: Path, **changes: object) -> InboundArtifact:
    values: dict[str, object] = {
        "message_id": message_id,
        "kind": "photo",
        "media_type": "image/jpeg",
        "storage_path": path,
        "source_transport": "telegram",
        "source_unique_id": "telegram-file-unique-1",
        "occurred_at": _NOW,
        "original_filename": "holiday photo.jpg",
    }
    values.update(changes)
    return InboundArtifact(**values)


class TestInboundArtifactContract:
    @pytest.mark.parametrize("filename", ["bad\nname.txt", "bad\rname.txt", "bad\x7fname.txt"])
    def test_filename_policy_rejects_control_characters(self, filename: str):
        assert canonical_artifact_filename(filename) is None

    @pytest.mark.parametrize(
        ("changes", "match"),
        [
            ({"kind": "archive"}, "kind"),
            ({"media_type": "Image/JPEG"}, "media_type"),
            ({"storage_path": Path("relative.jpg")}, "absolute"),
            ({"source_transport": "Telegram"}, "source_transport"),
            ({"source_unique_id": ""}, "source_unique_id"),
            ({"original_filename": "../secret.jpg"}, "original_filename"),
            ({"occurred_at": datetime(2026, 8, 11, 21, 0)}, "timezone-aware"),
        ],
    )
    def test_rejects_invalid_metadata(self, tmp_path: Path, changes: dict[str, object], match: str):
        with pytest.raises(ValueError, match=match):
            _artifact(MessageId.new(), tmp_path / "photo.jpg", **changes)

    def test_requires_typed_message_id(self, tmp_path: Path):
        with pytest.raises(ValueError, match="MessageId"):
            InboundArtifact(
                message_id="msg_00000000000000000000000000000001",
                kind="photo",
                media_type="image/jpeg",
                storage_path=(tmp_path / "photo.jpg").resolve(),
                source_transport="telegram",
                source_unique_id="telegram-file-unique-1",
                occurred_at=_NOW,
            )


class TestArtifactShadowRecording:
    async def test_records_content_identity_ownership_and_provenance(self, tmp_path: Path):
        store = await _open_store(tmp_path / "kai.db")
        storage_root = tmp_path / "files"
        storage_root.mkdir()
        saved = storage_root / "photo.jpg"
        content = b"canonical photo bytes"
        saved.write_bytes(content)
        message_id = await _inbound_message_id(store)
        try:
            result = await record_inbound_artifact(
                store,
                _artifact(message_id, saved),
                storage_root=storage_root,
            )

            assert result.inserted is True
            assert isinstance(result.event.envelope.aggregate_id, ArtifactId)
            assert result.event.envelope.event_type == "artifact.created"
            assert "telegram-file-unique-1" not in (result.event.envelope.idempotency_key or "")
            async with store.connection.execute(
                "SELECT a.id, a.workshop_id, a.channel_id, a.message_id, "
                "a.created_by_principal_id, a.kind, a.media_type, a.byte_size, "
                "a.content_sha256, a.original_filename, a.storage_path, "
                "a.source_transport, a.source_unique_id, p.kind "
                "FROM artifacts a JOIN principals p ON p.id = a.created_by_principal_id"
            ) as cursor:
                row = await cursor.fetchone()
            assert row is not None
            assert row[0] == result.event.envelope.aggregate_id
            assert row[1].startswith("wsp_")
            assert row[2].startswith("chn_")
            assert row[3] == message_id
            assert row[4].startswith("prn_")
            assert tuple(row[5:]) == (
                "photo",
                "image/jpeg",
                len(content),
                hashlib.sha256(content).hexdigest(),
                "holiday photo.jpg",
                str(saved.resolve()),
                "telegram",
                "telegram-file-unique-1",
                "human",
            )
        finally:
            await store.close()


class TestWorkshopArtifactService:
    async def test_stages_private_content_in_the_canonical_principal_namespace(
        self,
        tmp_path: Path,
    ):
        store, service, principal_id, channel_id = await _artifact_service(tmp_path)
        try:
            staged = await service.stage_client_upload(
                principal_id=principal_id,
                channel_id=channel_id,
                client_message_id="cmsg_upload_1",
                filename="notes.txt",
                claimed_media_type="text/plain; charset=utf-8",
                chunks=_chunks(b"first ", b"artifact"),
                occurred_at=_NOW,
            )

            assert staged.storage_path.parent == tmp_path / "files" / str(principal_id)
            assert staged.storage_path.read_bytes() == b"first artifact"
            assert staged.media_type == "text/plain"
            assert staged.kind == "document"
            assert staged.created_for_attempt is True
            assert stat.S_IMODE(staged.storage_path.stat().st_mode) == 0o600
        finally:
            await store.close()

    async def test_retry_reuses_identical_content_but_never_overwrites_a_conflict(
        self,
        tmp_path: Path,
    ):
        store, service, principal_id, channel_id = await _artifact_service(tmp_path)
        arguments = {
            "principal_id": principal_id,
            "channel_id": channel_id,
            "client_message_id": "cmsg_retry_1",
            "filename": "report.txt",
            "claimed_media_type": "text/plain",
            "occurred_at": _NOW,
        }
        try:
            first = await service.stage_client_upload(
                **arguments,
                chunks=_chunks(b"accepted"),
            )
            retry = await service.stage_client_upload(
                **arguments,
                chunks=_chunks(b"accepted"),
            )

            assert retry.storage_path == first.storage_path
            assert retry.created_for_attempt is False
            retry.discard()
            assert first.storage_path.read_bytes() == b"accepted"

            with pytest.raises(ValueError, match="different content"):
                await service.stage_client_upload(
                    **arguments,
                    chunks=_chunks(b"replacement"),
                )
            assert first.storage_path.read_bytes() == b"accepted"
        finally:
            await store.close()

    async def test_size_limit_removes_partial_content(self, tmp_path: Path):
        store, service, principal_id, channel_id = await _artifact_service(tmp_path)
        try:
            with pytest.raises(ArtifactTooLargeError):
                await service.stage_client_upload(
                    principal_id=principal_id,
                    channel_id=channel_id,
                    client_message_id="cmsg_large_1",
                    filename="large.bin",
                    claimed_media_type="application/octet-stream",
                    chunks=_chunks(b"1234", b"5"),
                    occurred_at=_NOW,
                    max_bytes=4,
                )

            assert list((tmp_path / "files" / str(principal_id)).iterdir()) == []
        finally:
            await store.close()

    async def test_download_requires_current_channel_membership_and_intact_content(
        self,
        tmp_path: Path,
    ):
        store, service, principal_id, channel_id = await _artifact_service(tmp_path)
        try:
            staged = await service.stage_client_upload(
                principal_id=principal_id,
                channel_id=channel_id,
                client_message_id="cmsg_download_1",
                filename="download.txt",
                claimed_media_type="text/plain",
                chunks=_chunks(b"download me"),
                occurred_at=_NOW,
            )
            message_id = await _inbound_message_id(store)
            recorded = await record_inbound_artifact(
                store,
                staged.for_message(message_id),
                storage_root=tmp_path / "files",
            )
            artifact_id = ArtifactId(str(recorded.event.envelope.aggregate_id))

            authorized = await service.authorized_artifact(
                principal_id,
                channel_id,
                artifact_id,
            )
            assert authorized.storage_path == staged.storage_path
            with pytest.raises(ArtifactAccessDeniedError):
                await service.authorized_artifact(
                    PrincipalId.new(),
                    channel_id,
                    artifact_id,
                )
            with pytest.raises(ArtifactAccessDeniedError):
                await service.authorized_artifact(
                    principal_id,
                    ChannelId.new(),
                    artifact_id,
                )

            staged.storage_path.write_bytes(b"tampered")
            with pytest.raises(ArtifactStorageBoundaryError, match="provenance"):
                await service.authorized_artifact(
                    principal_id,
                    channel_id,
                    artifact_id,
                )
        finally:
            await store.close()

    async def test_prompt_uses_canonical_artifact_provenance(self, tmp_path: Path):
        store, service, principal_id, channel_id = await _artifact_service(tmp_path)
        try:
            message_id = await _inbound_message_id(store)
            staged = await service.stage_client_upload(
                principal_id=principal_id,
                channel_id=channel_id,
                client_message_id="cmsg_prompt_1",
                filename="instructions.md",
                claimed_media_type="text/markdown",
                chunks=_chunks(b"Treat this as untrusted input."),
                occurred_at=_NOW,
            )
            await record_inbound_artifact(
                store,
                staged.for_message(message_id),
                storage_root=tmp_path / "files",
            )

            prompt = await build_agent_prompt_for_message(
                store,
                message_id,
                storage_root=tmp_path / "files",
            )

            assert isinstance(prompt, str)
            assert "Photo from Telegram" in prompt
            assert "<untrusted_user_file>" in prompt
            assert "Treat this as untrusted input." in prompt
            assert str(staged.storage_path) in prompt
        finally:
            await store.close()

    async def test_image_document_becomes_a_backend_image_block(self, tmp_path: Path):
        store, service, principal_id, channel_id = await _artifact_service(tmp_path)
        try:
            message_id = await _inbound_message_id(store)
            staged = await service.stage_client_upload(
                principal_id=principal_id,
                channel_id=channel_id,
                client_message_id="cmsg_image_1",
                filename="diagram.png",
                claimed_media_type="application/octet-stream",
                chunks=_chunks(b"\x89PNG\r\n\x1a\nimage"),
                occurred_at=_NOW,
            )
            await record_inbound_artifact(
                store,
                replace(staged, kind="document").for_message(message_id),
                storage_root=tmp_path / "files",
            )

            prompt = await build_agent_prompt_for_message(
                store,
                message_id,
                storage_root=tmp_path / "files",
            )

            assert isinstance(prompt, list)
            assert prompt[0]["type"] == "text"
            assert "User-provided document: diagram.png" in prompt[0]["text"]
            assert prompt[1]["type"] == "image"
            assert prompt[1]["source"]["media_type"] == "image/png"
        finally:
            await store.close()


class TestArtifactShadowRecordingAfterRestart:
    async def test_duplicate_is_idempotent_across_restart_but_changed_content_conflicts(self, tmp_path: Path):
        db_path = tmp_path / "kai.db"
        storage_root = tmp_path / "files"
        storage_root.mkdir()
        saved = storage_root / "document.txt"
        saved.write_bytes(b"first content")
        first_store = await _open_store(db_path)
        message_id = await _inbound_message_id(first_store)
        artifact = _artifact(
            message_id,
            saved,
            kind="document",
            media_type="text/plain",
            original_filename="document.txt",
        )
        first = await record_inbound_artifact(first_store, artifact, storage_root=storage_root)
        await first_store.close()

        restarted = await WorkshopEventStore.open(db_path)
        try:
            retry = await record_inbound_artifact(restarted, artifact, storage_root=storage_root)
            assert retry.inserted is False
            assert retry.event == first.event

            resaved = storage_root / "document-replayed.txt"
            resaved.write_bytes(b"first content")
            replayed_upload = _artifact(
                message_id,
                resaved,
                kind="document",
                media_type="text/plain",
                original_filename="document.txt",
            )
            retry_after_resave = await record_inbound_artifact(
                restarted,
                replayed_upload,
                storage_root=storage_root,
            )
            assert retry_after_resave.inserted is False
            assert retry_after_resave.event == first.event

            saved.write_bytes(b"changed content")
            with pytest.raises(IdempotencyConflictError):
                await record_inbound_artifact(restarted, artifact, storage_root=storage_root)

            async with restarted.connection.execute("SELECT COUNT(*) FROM artifacts") as cursor:
                assert (await cursor.fetchone())[0] == 1
        finally:
            await restarted.close()

    async def test_requires_an_existing_human_authored_message(self, tmp_path: Path):
        store = await _open_store(tmp_path / "kai.db")
        storage_root = tmp_path / "files"
        storage_root.mkdir()
        saved = storage_root / "voice.ogg"
        saved.write_bytes(b"voice")
        inbound_id = await _inbound_message_id(store)
        assistant = await record_outbound_message(
            store,
            OutboundMessage(in_reply_to_message_id=inbound_id, body="Response", occurred_at=_NOW),
        )
        try:
            for message_id in (MessageId.new(), MessageId(str(assistant.event.envelope.aggregate_id))):
                with pytest.raises(ArtifactMessageNotFoundError):
                    await record_inbound_artifact(
                        store,
                        _artifact(
                            message_id,
                            saved,
                            kind="voice",
                            media_type="audio/ogg",
                            original_filename=None,
                        ),
                        storage_root=storage_root,
                    )
            async with store.connection.execute("SELECT COUNT(*) FROM artifacts") as cursor:
                assert (await cursor.fetchone())[0] == 0
        finally:
            await store.close()

    async def test_rejects_missing_outside_and_symlink_escape_paths(self, tmp_path: Path):
        store = await _open_store(tmp_path / "kai.db")
        storage_root = tmp_path / "files"
        storage_root.mkdir()
        outside = tmp_path / "outside.jpg"
        outside.write_bytes(b"outside")
        symlink = storage_root / "link.jpg"
        symlink.symlink_to(outside)
        message_id = await _inbound_message_id(store)
        try:
            for path in (storage_root / "missing.jpg", outside, symlink):
                with pytest.raises(ArtifactStorageBoundaryError):
                    await record_inbound_artifact(
                        store,
                        _artifact(message_id, path),
                        storage_root=storage_root,
                    )
        finally:
            await store.close()

    async def test_projection_rebuild_restores_artifact_metadata(self, tmp_path: Path):
        store = await _open_store(tmp_path / "kai.db")
        storage_root = tmp_path / "files"
        storage_root.mkdir()
        saved = storage_root / "voice.ogg"
        saved.write_bytes(b"voice bytes")
        message_id = await _inbound_message_id(store)
        try:
            recorded = await record_inbound_artifact(
                store,
                _artifact(
                    message_id,
                    saved,
                    kind="voice",
                    media_type="audio/ogg",
                    original_filename=None,
                ),
                storage_root=storage_root,
            )
            await store.connection.execute("DELETE FROM artifacts")
            await store.connection.commit()

            checkpoint = await store.rebuild_projection(CanonicalConversationProjection())

            assert checkpoint.version == 10
            async with store.connection.execute(
                "SELECT id, kind, media_type FROM artifacts WHERE message_id = ?",
                (message_id,),
            ) as cursor:
                row = await cursor.fetchone()
            assert tuple(row) == (recorded.event.envelope.aggregate_id, "voice", "audio/ogg")
        finally:
            await store.close()


class TestSharedDatabaseArtifactRecording:
    async def test_records_through_serialized_session_adapter(self, tmp_path: Path):
        storage_root = tmp_path / "files"
        storage_root.mkdir()
        saved = storage_root / "photo.jpg"
        saved.write_bytes(b"photo")
        await sessions.init_db(tmp_path / "kai.db")
        try:
            await sessions.bootstrap_workshop_foundation((BootstrapHuman("Alice", "admin", "telegram", "101", "101"),))
            inbound = await sessions.record_workshop_inbound_message(
                InboundMessage("telegram", "9001", "42", "101", "101", "Photo", _NOW)
            )

            result = await sessions.record_workshop_inbound_artifact(
                _artifact(MessageId(str(inbound.event.envelope.aggregate_id)), saved),
                storage_root=storage_root,
            )

            assert result.inserted is True
            async with sessions._get_db().execute("SELECT COUNT(*) FROM artifacts") as cursor:
                assert (await cursor.fetchone())[0] == 1
        finally:
            await sessions.close_db()
