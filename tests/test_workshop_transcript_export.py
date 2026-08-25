"""Canonical Workshop transcript-authority contracts."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from kai import history
from kai.conversation_compatibility import schedule_memory_ingestion
from kai.memory import TranscriptProvenance, read_transcript_provenance
from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.conversation_commands import WorkshopConversationCommandService
from kai.workshop.conversation_context import assemble_canonical_prior_pairs
from kai.workshop.delivery_authority import WorkshopConversationDeliveryAuthority
from kai.workshop.diagnostics import (
    workshop_canonical_message_integrity_status,
    workshop_legacy_jsonl_archive_status,
    workshop_transcript_authority_status,
)
from kai.workshop.domain import CanonicalMemoryProvenance, MessageId, RunExecutionOwnerId, RunId
from kai.workshop.inbound import InboundMessage
from kai.workshop.run_execution_authority import RunExecutionSelection, WorkshopRunExecutionAuthority
from kai.workshop.store import WorkshopEventStore
from kai.workshop.terminal_transactions import WorkshopRunTerminalTransactionCoordinator
from kai.workshop.transcript_export import (
    CanonicalTranscriptProjection,
    build_canonical_transcript_export,
    inspect_canonical_transcript_snapshot,
)
from kai.workshop_cli import _parser
from tests.workshop_delivery import TELEGRAM_DELIVERY_POLICY
from tests.workshop_profiles import profile_id

_NOW = datetime(2026, 8, 15, 9, 0, tzinfo=UTC)


async def _foundation(database: Path) -> WorkshopEventStore:
    store = await WorkshopEventStore.open(database)
    await bootstrap_default_workshop(
        store,
        (
            BootstrapHuman(
                display_name="Daniel",
                role="admin",
                transport="telegram",
                external_subject="919191",
                external_channel_id="919191",
                runtime_profile_id=profile_id(101),
            ),
        ),
    )
    await WorkshopConversationDeliveryAuthority(store).activate()
    return store


async def _accepted(store: WorkshopEventStore, suffix: str):
    return await WorkshopConversationCommandService(store).accept(
        InboundMessage(
            transport="telegram",
            update_id=f"update-{suffix}",
            message_id=f"message-{suffix}",
            sender_subject="919191",
            channel_subject="919191",
            body=f"Canonical prompt {suffix}",
            occurred_at=_NOW + timedelta(minutes=int(suffix)),
        )
    )


async def _complete(store: WorkshopEventStore, acceptance, suffix: str):
    authority = WorkshopRunExecutionAuthority(
        store,
        selection_resolver=lambda _run: RunExecutionSelection(
            backend="codex",
            model="gpt-5.6-sol",
            provider="openai",
        ),
        registered_backend_ids=frozenset({"codex"}),
    )
    granted = await authority.grant(
        acceptance.run.run_id,
        owner_id=RunExecutionOwnerId.new(),
        occurred_at=_NOW + timedelta(minutes=int(suffix), seconds=1),
        lease_expires_at=_NOW + timedelta(minutes=int(suffix) + 2),
    )
    started = await authority.start(granted.claim, occurred_at=_NOW + timedelta(minutes=int(suffix), seconds=2))
    return await WorkshopRunTerminalTransactionCoordinator(
        authority,
        delivery_policy=TELEGRAM_DELIVERY_POLICY,
    ).complete(
        started.claim,
        body=f"Canonical answer {suffix}",
        occurred_at=_NOW + timedelta(minutes=int(suffix), seconds=3),
    )


async def test_export_is_deterministic_and_transport_independent(tmp_path: Path):
    store = await _foundation(tmp_path / "kai.db")
    try:
        acceptance = await _accepted(store, "1")
        export = await build_canonical_transcript_export(store, acceptance.run.channel_id)
    finally:
        await store.close()

    rows = [json.loads(line) for line in export.ndjson().splitlines()]
    assert rows[0]["body"] == "Canonical prompt 1"
    assert rows[0]["author"]["kind"] == "human"
    assert rows[0]["format"] == "kai-workshop-transcript"
    assert "919191" not in export.ndjson()


async def test_projection_appends_and_recovers_from_truncation(tmp_path: Path, monkeypatch):
    from kai.workshop import transcript_export as transcript_module

    monkeypatch.setattr("kai.workshop.transcript_export.replace_named_read_access", lambda *_a, **_kw: None)
    store = await _foundation(tmp_path / "kai.db")
    projection = CanonicalTranscriptProjection(tmp_path / "history")
    lock = asyncio.Lock()
    writes_while_database_locked: list[bool] = []
    original_writer = transcript_module.write_canonical_transcript_snapshot

    def observed_writer(*args, **kwargs):
        writes_while_database_locked.append(lock.locked())
        return original_writer(*args, **kwargs)

    monkeypatch.setattr(transcript_module, "write_canonical_transcript_snapshot", observed_writer)
    try:
        first = await _accepted(store, "1")
        target = await projection.refresh(
            store,
            first.run.channel_id,
            reader_user=None,
            database_lock=lock,
        )
        first_text = target.read_text()
        await _complete(store, first, "1")
        await projection.refresh(store, first.run.channel_id, reader_user=None, database_lock=lock)
        assert target.read_text().startswith(first_text)
        assert "Canonical answer 1" in target.read_text()
        assert target.stat().st_mode & 0o777 == 0o600
        assert target.parent.stat().st_mode & 0o777 == 0o700

        target.write_text(target.read_text() + '{"partial":')
        await projection.refresh(store, first.run.channel_id, reader_user=None, database_lock=lock)
        assert inspect_canonical_transcript_snapshot(tmp_path / "history", first.run.channel_id).valid
        assert '"partial"' not in target.read_text()
        assert writes_while_database_locked and not any(writes_while_database_locked)
    finally:
        await store.close()


async def test_prior_pairs_follow_completed_owner_lane(tmp_path: Path):
    store = await _foundation(tmp_path / "kai.db")
    try:
        first = await _accepted(store, "1")
        await _complete(store, first, "1")
        await _accepted(store, "2")
        target = await _accepted(store, "3")
        assert await assemble_canonical_prior_pairs(store, target.run, limit=3) == (
            ("Canonical prompt 1", "Canonical answer 1"),
        )
    finally:
        await store.close()


async def test_canonical_memory_source_reads_sqlite_without_jsonl(tmp_path: Path, monkeypatch):
    store = await _foundation(tmp_path / "kai.db")
    try:
        acceptance = await _accepted(store, "1")
        terminal = await _complete(store, acceptance, "1")
        run = terminal.execution.run
        assert isinstance(run.result_message_id, MessageId)
        await store.connection.execute(
            "INSERT INTO workshop_memory_authority_migrations ("
            "runtime_profile_id, runtime_config_id, principal_id, channel_id, agent_id, "
            "moved_count, stamped_count, total_count) VALUES (?, ?, ?, ?, ?, 0, 0, 0)",
            (
                profile_id(101),
                101,
                run.requested_by_principal_id,
                run.channel_id,
                run.agent_id,
            ),
        )
        await store.connection.commit()
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
        agent_id=str(run.agent_id),
        runtime_profile_id=str(profile_id(101)),
        run_id=str(run.run_id),
        source_message_id=str(run.inbound_message_id),
        result_message_id=str(run.result_message_id),
    )
    monkeypatch.setattr(history, "DATA_DIR", tmp_path)
    monkeypatch.setattr(
        history,
        "_read_history_day",
        lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("JSONL read")),
    )
    lookup = history.fetch_transcript_context(
        provenance,
        expected_principal_id=str(run.requested_by_principal_id),
    )
    assert lookup.reason == "ok"
    assert lookup.context is not None
    assert lookup.context.target_user.text == "Canonical prompt 1"
    assert lookup.context.target_assistant is not None
    assert lookup.context.target_assistant.text == "Canonical answer 1"
    assert (
        history.fetch_transcript_context(
            replace(provenance, runtime_profile_id=str(profile_id(202))),
            expected_principal_id=str(run.requested_by_principal_id),
        ).reason
        == "canonical_missing"
    )


def test_partial_canonical_provenance_never_falls_back_to_jsonl(monkeypatch):
    provenance = read_transcript_provenance(
        {
            "workshop_principal_id": "prn_incomplete",
            "workshop_channel_id": "chn_incomplete",
            "workshop_agent_id": "agt_incomplete",
            "workshop_runtime_profile_id": "rtp_incomplete",
            "workshop_run_id": "run_incomplete",
            "source_chat_id": 919191,
            "source_date": "2026-08-15",
            "source_user_ts": "2026-08-15T09:00:00Z",
            "source_user_text_sha256": "a" * 64,
        }
    )
    monkeypatch.setattr(
        history,
        "_read_history_day",
        lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("JSONL read")),
    )
    assert history.fetch_transcript_context(provenance).reason == "provenance_invalid"


async def test_canonical_memory_ingestion_uses_supplied_prior_pairs(monkeypatch):
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
        lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("JSONL read")),
    )
    provenance = CanonicalMemoryProvenance(RunId.new(), MessageId.new(), MessageId.new())
    schedule_memory_ingestion(
        prompt="Current prompt",
        assistant_text="Current answer",
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
    assert extraction.await_args.kwargs["prior_pairs"] == [("Prior prompt", "Prior answer")]
    assert extraction.await_args.kwargs["user_log"] is None
    assert extraction.await_args.kwargs["assistant_log"] is None


async def test_durable_authority_diagnostic_and_cli(tmp_path: Path):
    database = tmp_path / "kai.db"
    store = await _foundation(database)
    try:
        acceptance = await _accepted(store, "1")
    finally:
        await store.close()
    status = workshop_transcript_authority_status(database)
    assert status.startswith("Workshop transcript authority: active;")
    assert "protected inputs=text/media, JSONL reads=disabled, writes=disabled" in status
    assert workshop_canonical_message_integrity_status(database).startswith(
        "Workshop canonical message integrity: clean;"
    )
    assert workshop_legacy_jsonl_archive_status(database, tmp_path / "history").startswith(
        "Workshop legacy JSONL archive: classified;"
    )
    parsed = _parser().parse_args(["transcript", "export", "--channel-id", str(acceptance.run.channel_id)])
    assert parsed.command == "transcript"
    assert parsed.action == "export"
