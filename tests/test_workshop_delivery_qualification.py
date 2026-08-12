"""Installed, explicitly invoked Workshop delivery qualification contract."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from kai import workshop_cli
from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.delivery_qualification import DeliveryQualificationError, WorkshopDeliveryQualification
from kai.workshop.domain import DeliveryId, MessageId
from kai.workshop.inbound import InboundMessage, record_inbound_message
from kai.workshop.outbound import OutboundMessage, record_outbound_message
from kai.workshop.store import WorkshopEventStore

_NOW = datetime(2026, 8, 12, 12, 0, tzinfo=UTC)


async def _store(path: Path) -> WorkshopEventStore:
    store = await WorkshopEventStore.open(path)
    await bootstrap_default_workshop(
        store,
        (
            BootstrapHuman(
                display_name="Operator",
                role="admin",
                transport="telegram",
                external_subject="101",
                external_channel_id="101",
            ),
        ),
    )
    return store


async def _exchange(store: WorkshopEventStore, sequence: int) -> MessageId:
    inbound = await record_inbound_message(
        store,
        InboundMessage(
            transport="telegram",
            update_id=str(9000 + sequence),
            message_id=str(40 + sequence),
            sender_subject="101",
            channel_subject="101",
            body=f"Prompt {sequence}",
            occurred_at=_NOW + timedelta(seconds=sequence),
        ),
    )
    outbound = await record_outbound_message(
        store,
        OutboundMessage(
            in_reply_to_message_id=MessageId(str(inbound.event.envelope.aggregate_id)),
            body=f"Reply {sequence}",
            occurred_at=_NOW + timedelta(seconds=sequence),
        ),
    )
    return MessageId(str(outbound.event.envelope.aggregate_id))


class TestDeliveryQualification:
    async def test_prepare_selects_latest_canonical_reply_without_claiming_or_sending(self, tmp_path: Path):
        store = await _store(tmp_path / "kai.db")
        await _exchange(store, 1)
        latest = await _exchange(store, 2)
        qualification = WorkshopDeliveryQualification(store)
        try:
            result = await qualification.prepare(101)

            assert result.inserted is True
            assert result.delivery.message_id == latest
            assert result.delivery.transport == "telegram"
            assert result.delivery.mode == "text"
            assert result.delivery.status == "pending"
            assert result.delivery.attempt_count == 0
            assert result.delivery.max_attempts == 3
        finally:
            await store.close()

    async def test_prepare_is_idempotent_for_same_latest_reply(self, tmp_path: Path):
        store = await _store(tmp_path / "kai.db")
        await _exchange(store, 1)
        qualification = WorkshopDeliveryQualification(store)
        try:
            first = await qualification.prepare(101)
            second = await qualification.prepare(101)

            assert first.inserted is True
            assert second.inserted is False
            assert second.delivery.delivery_id == first.delivery.delivery_id
        finally:
            await store.close()


class TestDeliveryQualificationCLI:
    def test_requires_an_explicit_action(self):
        with pytest.raises(SystemExit):
            workshop_cli._parser().parse_args(["delivery-qualification"])

    def test_prepare_requires_a_telegram_user(self):
        with pytest.raises(SystemExit):
            workshop_cli._parser().parse_args(["delivery-qualification", "prepare"])

    @pytest.mark.parametrize("action", ["status", "run", "simulate-interruption"])
    def test_delivery_actions_require_an_exact_delivery_id(self, action: str):
        with pytest.raises(SystemExit):
            workshop_cli._parser().parse_args(["delivery-qualification", action])

    def test_interruption_lease_is_bounded(self):
        with pytest.raises(SystemExit):
            workshop_cli._parser().parse_args(
                [
                    "delivery-qualification",
                    "simulate-interruption",
                    "--delivery-id",
                    "dlv_00000000000000000000000000000000",
                    "--lease-seconds",
                    "301",
                ]
            )

    def test_invalid_delivery_id_is_bounded(self):
        with pytest.raises(DeliveryQualificationError, match="Invalid delivery ID"):
            workshop_cli._delivery_id("101")

    def test_missing_deployed_database_is_rejected_without_creation(self, tmp_path: Path):
        with pytest.raises(DeliveryQualificationError, match="was not found"):
            workshop_cli._qualification_database(tmp_path)
        assert not (tmp_path / "kai.db").exists()

    def test_database_must_be_owned_by_the_invoking_account(self, tmp_path: Path, monkeypatch):
        database = tmp_path / "kai.db"
        database.touch()
        monkeypatch.setattr(os, "geteuid", lambda: database.stat().st_uid + 1)

        with pytest.raises(DeliveryQualificationError, match="account that owns"):
            workshop_cli._qualification_database(tmp_path)

    def test_regular_database_owned_by_the_invoking_account_is_accepted(self, tmp_path: Path):
        database = tmp_path / "kai.db"
        database.touch()

        assert workshop_cli._qualification_database(tmp_path) == database


class TestDeliveryQualificationFailures:
    async def test_prepare_requires_configured_direct_user_with_canonical_reply(self, tmp_path: Path):
        store = await _store(tmp_path / "kai.db")
        qualification = WorkshopDeliveryQualification(store)
        try:
            with pytest.raises(DeliveryQualificationError, match="No canonical Kai reply"):
                await qualification.prepare(101)
            with pytest.raises(DeliveryQualificationError, match="No canonical Kai reply"):
                await qualification.prepare(999)
        finally:
            await store.close()

    @pytest.mark.parametrize("user_id", [0, -1, True, 2**63])
    async def test_prepare_rejects_invalid_telegram_user_id(self, tmp_path: Path, user_id):
        store = await _store(tmp_path / "kai.db")
        try:
            with pytest.raises(ValueError, match="positive signed 64-bit"):
                await WorkshopDeliveryQualification(store).prepare(user_id)
        finally:
            await store.close()

    async def test_simulated_interruption_claims_exact_delivery_without_sending(self, tmp_path: Path):
        store = await _store(tmp_path / "kai.db")
        await _exchange(store, 1)
        qualification = WorkshopDeliveryQualification(store)
        try:
            prepared = await qualification.prepare(101)
            claim = await qualification.simulate_interruption(
                prepared.delivery.delivery_id,
                worker_id="qualification:test",
                lease_duration=timedelta(seconds=5),
            )

            assert isinstance(claim.delivery_id, DeliveryId)
            assert claim.delivery_id == prepared.delivery.delivery_id
            state = await qualification.status(claim.delivery_id)
            assert state.status == "leased"
            assert state.attempt_count == 1
        finally:
            await store.close()
