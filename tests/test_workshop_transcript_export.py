"""Canonical transcript export and prior-context contracts."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from kai import history
from kai.conversation_compatibility import CanonicalMemoryProvenance, schedule_memory_ingestion
from kai.memory import TranscriptProvenance
from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.conversation_commands import WorkshopConversationCommandService
from kai.workshop.conversation_context import assemble_canonical_prior_pairs
from kai.workshop.delivery_authority import WorkshopConversationDeliveryAuthority
from kai.workshop.diagnostics import workshop_transcript_authority_status
from kai.workshop.domain import MessageId, RunExecutionOwnerId, RunId, RuntimeProfileId
from kai.workshop.inbound import InboundMessage
from kai.workshop.outbound import OutboundMessage, record_outbound_message
from kai.workshop.run_execution_authority import (
    RunExecutionSelection,
    WorkshopRunExecutionAuthority,
)
from kai.workshop.store import WorkshopEventStore
from kai.workshop.terminal_transactions import WorkshopRunTerminalTransactionCoordinator
from kai.workshop.transcript_export import (
    build_canonical_transcript_export,
    write_canonical_transcript_snapshot,
)
from kai.workshop_cli import _parser

_NOW = datetime(2026, 8, 15, 9, 0, tzinfo=UTC)


async def _accepted(store: WorkshopEventStore, *, suffix: str = "1"):
    return await WorkshopConversationCommandService(store).accept(
        InboundMessage(
            transport="telegram",
            update_id=f"update-{suffix}",
            message_id=f"message-{suffix}",
            sender_subject="919191",
            channel_subject="919191",
            body=f"Canonical prompt {suffix}",
            occurred_at=_NOW,
        )
    )


async def _complete(store: WorkshopEventStore, acceptance, *, answer: str = "Canonical answer 1"):
    authority = WorkshopRunExecutionAuthority(
        store,
        selection_resolver=lambda _run: RunExecutionSelection("codex", "gpt-5.6-sol"),
        registered_backend_ids=frozenset({"codex"}),
    )
    granted = await authority.grant(
        acceptance.run.run_id,
        owner_id=RunExecutionOwnerId.new(),
        occurred_at=_NOW + timedelta(seconds=1),
        lease_expires_at=_NOW + timedelta(minutes=2),
    )
    started = await authority.start(granted.claim, occurred_at=_NOW + timedelta(seconds=2))
    await WorkshopConversationDeliveryAuthority(store).activate()
    return await WorkshopRunTerminalTransactionCoordinator(authority).complete(
        started.claim,
        body=answer,
        occurred_at=_NOW + timedelta(seconds=3),
    )


async def test_export_contains_only_canonical_identity_and_message_fields(tmp_path: Path):
    store = await WorkshopEventStore.open(tmp_path / "kai.db")
    try:
        await bootstrap_default_workshop(
            store,
            (
                BootstrapHuman(
                    display_name="Daniel",
                    role="admin",
                    transport="telegram",
                    external_subject="919191",
                    external_channel_id="919191",
                    runtime_profile_id=RuntimeProfileId.new(),
                ),
            ),
        )
        acceptance = await _accepted(store)
        export = await build_canonical_transcript_export(store, acceptance.run.channel_id)

        rows = [json.loads(line) for line in export.jsonl().splitlines()]
        assert len(rows) == 1
        assert rows[0]["format"] == "kai-workshop-transcript"
        assert rows[0]["format_version"] == 1
        assert rows[0]["channel_id"] == acceptance.run.channel_id
        assert rows[0]["body"] == "Canonical prompt 1"
        assert rows[0]["author"]["kind"] == "human"
        assert "919191" not in export.jsonl()
    finally:
        await store.close()


async def test_snapshot_is_atomic_and_ignored_by_compatibility_jsonl_readers(
    tmp_path: Path,
    monkeypatch,
):
    store = await WorkshopEventStore.open(tmp_path / "kai.db")
    try:
        await bootstrap_default_workshop(
            store,
            (
                BootstrapHuman(
                    display_name="Daniel",
                    role="admin",
                    transport="telegram",
                    external_subject="919191",
                    external_channel_id="919191",
                    runtime_profile_id=RuntimeProfileId.new(),
                ),
            ),
        )
        acceptance = await _accepted(store)
        export = await build_canonical_transcript_export(store, acceptance.run.channel_id)
    finally:
        await store.close()

    access: list[tuple[Path, str | None, bool]] = []
    monkeypatch.setattr(
        "kai.workshop.transcript_export.replace_named_read_access",
        lambda path, user, *, directory: access.append((path, user, directory)),
    )
    root = tmp_path / "history"
    target = write_canonical_transcript_snapshot(export, root, reader_user="daniel")

    assert target == root / str(export.channel_id) / "canonical-transcript.ndjson"
    assert target.read_text() == export.jsonl()
    assert not list(target.parent.glob(".transcript-*.tmp"))
    assert not list(target.parent.glob("*.jsonl"))
    assert access == [
        (target.parent, "daniel", True),
        (target, "daniel", False),
    ]


async def test_prior_pairs_are_canonical_and_exclude_current_inbound(tmp_path: Path):
    store = await WorkshopEventStore.open(tmp_path / "kai.db")
    try:
        await bootstrap_default_workshop(
            store,
            (
                BootstrapHuman(
                    display_name="Daniel",
                    role="admin",
                    transport="telegram",
                    external_subject="919191",
                    external_channel_id="919191",
                    runtime_profile_id=RuntimeProfileId.new(),
                ),
            ),
        )
        first = await _accepted(store)
        await record_outbound_message(
            store,
            OutboundMessage(
                in_reply_to_message_id=first.run.inbound_message_id,
                body="Canonical answer 1",
                occurred_at=_NOW,
            ),
        )
        second = await _accepted(store, suffix="2")

        pairs = await assemble_canonical_prior_pairs(store, second.run, limit=3)
        assert pairs == (("Canonical prompt 1", "Canonical answer 1"),)
        assert all("Canonical prompt 2" not in text for pair in pairs for text in pair)
    finally:
        await store.close()


async def test_canonical_memory_source_reads_sqlite_without_jsonl(tmp_path: Path, monkeypatch):
    store = await WorkshopEventStore.open(tmp_path / "kai.db")
    try:
        await bootstrap_default_workshop(
            store,
            (
                BootstrapHuman(
                    display_name="Daniel",
                    role="admin",
                    transport="telegram",
                    external_subject="919191",
                    external_channel_id="919191",
                    runtime_profile_id=RuntimeProfileId.new(),
                ),
            ),
        )
        acceptance = await _accepted(store)
        terminal = await _complete(store, acceptance)
        run = terminal.execution.run
        assert isinstance(run.result_message_id, MessageId)
    finally:
        await store.close()

    provenance = TranscriptProvenance(
        present=True,
        chat_id=None,
        date=None,
        user_ts=None,
        user_text_sha256=None,
        assistant_ts=None,
        date_end=None,
        canonical_present=True,
        principal_id=str(run.requested_by_principal_id),
        channel_id=str(run.channel_id),
        run_id=str(run.run_id),
        source_message_id=str(run.inbound_message_id),
        result_message_id=str(run.result_message_id),
    )
    monkeypatch.setattr(history, "DATA_DIR", tmp_path)
    monkeypatch.setattr(
        history,
        "_read_history_day",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("JSONL read")),
    )

    lookup = history.fetch_transcript_context(
        provenance,
        expected_principal_id=str(run.requested_by_principal_id),
    )

    assert lookup.reason == "ok"
    assert lookup.context is not None
    assert lookup.context.chat_id is None
    assert lookup.context.principal_id == run.requested_by_principal_id
    assert lookup.context.channel_id == run.channel_id
    assert lookup.context.target_user.text == "Canonical prompt 1"
    assert lookup.context.target_assistant is not None
    assert lookup.context.target_assistant.text == "Canonical answer 1"


async def test_canonical_memory_ingestion_never_reads_recent_jsonl(monkeypatch):
    extraction = AsyncMock()
    config = MagicMock()
    config.memory_extraction_enabled = True
    config.episode_classifier_context_turns = 2
    config.default_backend = "codex"
    config.get_user_config.return_value = None
    monkeypatch.setattr("kai.memory.is_enabled", lambda: True)
    monkeypatch.setattr("kai.memory_extraction.extract_and_store", extraction)
    monkeypatch.setattr(
        "kai.history.get_recent_pairs",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("JSONL read")),
    )
    provenance = CanonicalMemoryProvenance(
        run_id=RunId.new(),
        source_message_id=MessageId.new(),
        result_message_id=MessageId.new(),
    )

    schedule_memory_ingestion(
        prompt="Current canonical prompt",
        assistant_text="Current canonical answer",
        chat_id=919191,
        session_id="session",
        config=config,
        workspace="/workspace",
        user_log=None,
        assistant_log=None,
        canonical_provenance=provenance,
        canonical_prior_pairs=(("Prior prompt", "Prior answer"),),
        reasoner_backends=frozenset({"codex"}),
        effective_backend="codex",
    )
    from kai import conversation_compatibility

    await asyncio.gather(*tuple(conversation_compatibility._pending_memory_tasks))

    extraction.assert_awaited_once()
    assert extraction.await_args.kwargs["prior_pairs"] == [("Prior prompt", "Prior answer")]
    assert extraction.await_args.kwargs["user_log"] is None
    assert extraction.await_args.kwargs["assistant_log"] is None


async def test_transcript_diagnostic_and_cli_parser_expose_canonical_authority(tmp_path: Path):
    database = tmp_path / "kai.db"
    store = await WorkshopEventStore.open(database)
    try:
        await bootstrap_default_workshop(
            store,
            (
                BootstrapHuman(
                    display_name="Daniel",
                    role="admin",
                    transport="telegram",
                    external_subject="919191",
                    external_channel_id="919191",
                    runtime_profile_id=RuntimeProfileId.new(),
                ),
            ),
        )
        acceptance = await _accepted(store)
    finally:
        await store.close()

    status = workshop_transcript_authority_status(database)
    assert status.startswith("Workshop transcript authority: active;")
    assert "protected JSONL reads=disabled, writes=disabled" in status
    parsed = _parser().parse_args(["transcript", "export", "--channel-id", str(acceptance.run.channel_id)])
    assert parsed.command == "transcript"
    assert parsed.action == "export"
    assert parsed.channel_id == acceptance.run.channel_id
