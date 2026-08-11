"""Read-only parity diagnostics for the Workshop conversation shadow."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.diagnostics import workshop_message_parity_status
from kai.workshop.domain import MessageId
from kai.workshop.inbound import InboundMessage, record_inbound_message
from kai.workshop.outbound import OutboundMessage, record_outbound_message
from kai.workshop.store import WorkshopEventStore

_NOW = datetime(2026, 8, 11, 12, 0, tzinfo=UTC)


def _write_history(history_root: Path, records: list[dict], *, malformed: str | None = None) -> None:
    directory = history_root / "101"
    directory.mkdir(parents=True)
    lines = [json.dumps(record) for record in records]
    if malformed is not None:
        lines.append(malformed)
    (directory / "2026-08-11.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _record(direction: str, text: str, *, seconds: int, media: dict | None = None) -> dict:
    return {
        "ts": (_NOW + timedelta(seconds=seconds)).isoformat(),
        "dir": direction,
        "chat_id": 101,
        "text": text,
        "media": media,
    }


async def _build_conversation(db_path: Path) -> None:
    store = await WorkshopEventStore.open(db_path)
    try:
        await bootstrap_default_workshop(
            store,
            (
                BootstrapHuman(
                    display_name="Private Operator Name",
                    role="admin",
                    transport="telegram",
                    external_subject="101",
                    external_channel_id="101",
                ),
            ),
        )
        inbound = await record_inbound_message(
            store,
            InboundMessage("telegram", "9001", "42", "101", "101", "Secret question", _NOW),
        )
        await record_outbound_message(
            store,
            OutboundMessage(
                MessageId(str(inbound.event.envelope.aggregate_id)),
                "Secret answer",
                _NOW + timedelta(seconds=2),
            ),
        )
    finally:
        await store.close()


class TestWorkshopMessageParityStatus:
    def test_missing_database_reports_pending(self, tmp_path: Path):
        status = workshop_message_parity_status(tmp_path / "kai.db", tmp_path / "history")

        assert status == "Workshop message parity: pending; canonical message schema unavailable"

    async def test_clean_status_reports_counts_without_content_or_identity(self, tmp_path: Path):
        db_path = tmp_path / "kai.db"
        history_root = tmp_path / "history"
        await _build_conversation(db_path)
        _write_history(
            history_root,
            [_record("user", "Secret question", seconds=0), _record("assistant", "Secret answer", seconds=2)],
        )

        status = workshop_message_parity_status(db_path, history_root)

        assert status == (
            "Workshop message parity: clean; canonical=2, projected=2, replay mismatches=0, "
            "JSONL matched=2, JSONL missing=0, JSONL unmatched=0, Telegram channels=1"
        )
        assert "Secret" not in status
        assert "101" not in status
        assert "Private Operator Name" not in status

    async def test_legacy_prefix_and_unrelated_records_do_not_create_false_divergence(self, tmp_path: Path):
        db_path = tmp_path / "kai.db"
        history_root = tmp_path / "history"
        await _build_conversation(db_path)
        _write_history(
            history_root,
            [
                _record("user", "Older question", seconds=-20),
                _record("assistant", "Older answer", seconds=-19),
                _record("assistant", "Scheduled notification", seconds=-10),
                _record("assistant", "[no response]", seconds=-9),
                _record("user", "Secret question", seconds=0),
                _record("assistant", "Secret answer", seconds=2),
                _record("assistant", "[Job: check] Scheduled result", seconds=3),
                {
                    **_record("user", "Non-shadowed photo caption", seconds=4),
                    "media": {"type": "photo"},
                },
                _record("assistant", "Non-shadowed photo response", seconds=5),
            ],
        )

        status = workshop_message_parity_status(db_path, history_root)

        assert "parity: clean" in status
        assert "JSONL matched=2, JSONL missing=0" in status

    async def test_includes_only_explicitly_shadowed_photo_history(self, tmp_path: Path):
        db_path = tmp_path / "kai.db"
        history_root = tmp_path / "history"
        await _build_conversation(db_path)
        _write_history(
            history_root,
            [
                _record(
                    "user",
                    "Old unshadowed photo",
                    seconds=-5,
                    media={"type": "photo"},
                ),
                _record("assistant", "Old photo response", seconds=-4),
                _record(
                    "user",
                    "Secret question",
                    seconds=0,
                    media={"type": "photo", "workshop_message_shadowed": True},
                ),
                _record("assistant", "Secret answer", seconds=2),
            ],
        )

        status = workshop_message_parity_status(db_path, history_root)

        assert "parity: clean" in status
        assert "JSONL matched=2, JSONL missing=0, JSONL unmatched=0" in status

    async def test_projection_drift_is_reported_without_leaking_changed_text(self, tmp_path: Path):
        db_path = tmp_path / "kai.db"
        history_root = tmp_path / "history"
        await _build_conversation(db_path)
        _write_history(
            history_root,
            [_record("user", "Secret question", seconds=0), _record("assistant", "Secret answer", seconds=2)],
        )
        store = await WorkshopEventStore.open(db_path)
        try:
            await store.connection.execute(
                "UPDATE messages SET body = 'projection-only secret mutation' WHERE body = 'Secret answer'"
            )
            await store.connection.commit()
        finally:
            await store.close()

        status = workshop_message_parity_status(db_path, history_root)

        assert "parity: diverged" in status
        assert "replay mismatches=1" in status
        assert "projection-only" not in status
        assert "Secret" not in status

    async def test_binding_projection_drift_cannot_hide_the_bound_history(self, tmp_path: Path):
        db_path = tmp_path / "kai.db"
        history_root = tmp_path / "history"
        await _build_conversation(db_path)
        _write_history(
            history_root,
            [_record("user", "Secret question", seconds=0), _record("assistant", "Secret answer", seconds=2)],
        )
        store = await WorkshopEventStore.open(db_path)
        try:
            await store.connection.execute(
                "UPDATE channel_bindings SET external_channel_id = '202' WHERE transport = 'telegram'"
            )
            await store.connection.commit()
        finally:
            await store.close()

        status = workshop_message_parity_status(db_path, history_root)

        assert "parity: diverged" in status
        assert "replay mismatches=1" in status
        assert "JSONL matched=2, JSONL missing=0" in status
        assert "101" not in status
        assert "202" not in status

    async def test_channel_authorization_projection_drift_is_reported(self, tmp_path: Path):
        db_path = tmp_path / "kai.db"
        history_root = tmp_path / "history"
        await _build_conversation(db_path)
        _write_history(
            history_root,
            [_record("user", "Secret question", seconds=0), _record("assistant", "Secret answer", seconds=2)],
        )
        store = await WorkshopEventStore.open(db_path)
        try:
            await store.connection.execute("DELETE FROM channel_memberships WHERE role = 'owner'")
            await store.connection.commit()
        finally:
            await store.close()

        status = workshop_message_parity_status(db_path, history_root)

        assert "parity: diverged" in status
        assert "replay mismatches=1" in status
        assert "Secret" not in status

    async def test_workshop_membership_projection_drift_is_reported(self, tmp_path: Path):
        db_path = tmp_path / "kai.db"
        history_root = tmp_path / "history"
        await _build_conversation(db_path)
        _write_history(
            history_root,
            [_record("user", "Secret question", seconds=0), _record("assistant", "Secret answer", seconds=2)],
        )
        store = await WorkshopEventStore.open(db_path)
        try:
            await store.connection.execute("DELETE FROM workshop_memberships WHERE role = 'admin'")
            await store.connection.commit()
        finally:
            await store.close()

        status = workshop_message_parity_status(db_path, history_root)

        assert "parity: diverged" in status
        assert "replay mismatches=1" in status
        assert "Secret" not in status

    async def test_missing_jsonl_record_is_reported(self, tmp_path: Path):
        db_path = tmp_path / "kai.db"
        history_root = tmp_path / "history"
        await _build_conversation(db_path)
        _write_history(history_root, [_record("user", "Secret question", seconds=0)])

        status = workshop_message_parity_status(db_path, history_root)

        assert "parity: diverged" in status
        assert "JSONL matched=1, JSONL missing=1" in status

    async def test_newer_unshadowed_jsonl_turn_is_reported(self, tmp_path: Path):
        db_path = tmp_path / "kai.db"
        history_root = tmp_path / "history"
        await _build_conversation(db_path)
        _write_history(
            history_root,
            [
                _record("user", "Secret question", seconds=0),
                _record("assistant", "Secret answer", seconds=2),
                _record("user", "Unshadowed question", seconds=3),
                _record("assistant", "Unshadowed answer", seconds=4),
            ],
        )

        status = workshop_message_parity_status(db_path, history_root)

        assert "parity: diverged" in status
        assert "JSONL matched=2, JSONL missing=0, JSONL unmatched=2" in status
        assert "Unshadowed" not in status

    async def test_malformed_history_is_not_silently_declared_clean(self, tmp_path: Path):
        db_path = tmp_path / "kai.db"
        history_root = tmp_path / "history"
        await _build_conversation(db_path)
        _write_history(
            history_root,
            [_record("user", "Secret question", seconds=0), _record("assistant", "Secret answer", seconds=2)],
            malformed='{ "text": "must not leak"',
        )

        status = workshop_message_parity_status(db_path, history_root)

        assert "parity: NOT VERIFIED" in status
        assert "malformed JSONL records=1" in status
        assert "must not leak" not in status

    async def test_event_integrity_failure_is_not_silently_declared_clean(self, tmp_path: Path):
        db_path = tmp_path / "kai.db"
        history_root = tmp_path / "history"
        await _build_conversation(db_path)
        _write_history(
            history_root,
            [_record("user", "Secret question", seconds=0), _record("assistant", "Secret answer", seconds=2)],
        )
        store = await WorkshopEventStore.open(db_path)
        try:
            await store.connection.execute(
                "UPDATE event_log SET payload_json = '{}' WHERE event_type = 'message.created'"
            )
            await store.connection.commit()
        finally:
            await store.close()

        status = workshop_message_parity_status(db_path, history_root)

        assert status == "Workshop message parity: NOT VERIFIED (ValueError)"
        assert "Secret" not in status

    async def test_symlinked_history_source_is_not_followed(self, tmp_path: Path):
        db_path = tmp_path / "kai.db"
        history_root = tmp_path / "history"
        await _build_conversation(db_path)
        directory = history_root / "101"
        directory.mkdir(parents=True)
        target = tmp_path / "outside.jsonl"
        target.write_text(
            json.dumps(_record("user", "Secret question", seconds=0)) + "\n",
            encoding="utf-8",
        )
        (directory / "2026-08-11.jsonl").symlink_to(target)

        status = workshop_message_parity_status(db_path, history_root)

        assert status == "Workshop message parity: NOT VERIFIED; unreadable history sources=1"
        assert "outside" not in status

    async def test_diagnostic_does_not_modify_database(self, tmp_path: Path):
        db_path = tmp_path / "kai.db"
        history_root = tmp_path / "history"
        await _build_conversation(db_path)
        _write_history(
            history_root,
            [_record("user", "Secret question", seconds=0), _record("assistant", "Secret answer", seconds=2)],
        )
        before = db_path.read_bytes()

        workshop_message_parity_status(db_path, history_root)

        assert db_path.read_bytes() == before
