"""Core delivery eligibility is adapter enablement plus persisted bindings."""

from pathlib import Path

import pytest

from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.delivery_policy import WorkshopDeliveryBindingPolicy
from kai.workshop.domain import ChannelId
from kai.workshop.store import WorkshopEventStore


async def _store_with_bindings(path: Path) -> tuple[WorkshopEventStore, ChannelId]:
    store = await WorkshopEventStore.open(path)
    await bootstrap_default_workshop(
        store,
        (
            BootstrapHuman(
                display_name="Daniel",
                role="admin",
                transport="telegram",
                external_subject="101",
                external_channel_id="101",
            ),
        ),
    )
    async with store.connection.execute(
        "SELECT channel_id FROM channel_bindings WHERE transport = 'telegram'"
    ) as cursor:
        row = await cursor.fetchone()
    assert row is not None
    channel_id = ChannelId(str(row[0]))
    await store.connection.execute(
        "INSERT INTO channel_bindings (id, channel_id, transport, external_channel_id, created_at) "
        "VALUES (?, ?, 'desktop', 'desktop-101', '2026-08-25T00:00:00Z')",
        ("cbd_00000000000000000000000000000001", channel_id),
    )
    await store.connection.commit()
    return store, channel_id


def test_policy_rejects_invalid_transport_identifiers():
    with pytest.raises(ValueError, match="lowercase"):
        WorkshopDeliveryBindingPolicy(frozenset({"Telegram"}))
    with pytest.raises(ValueError, match="lowercase"):
        WorkshopDeliveryBindingPolicy(frozenset({1}))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="frozenset"):
        WorkshopDeliveryBindingPolicy({"telegram"})  # type: ignore[arg-type]


async def test_disabled_policy_retains_bindings_without_making_them_eligible(tmp_path: Path):
    store, channel_id = await _store_with_bindings(tmp_path / "kai.db")
    try:
        policy = WorkshopDeliveryBindingPolicy.disabled()

        assert await policy.binding_ids(store, channel_id) == ()
        assert await policy.binding_ids(store, channel_id, transport="telegram") == ()
        async with store.connection.execute(
            "SELECT transport FROM channel_bindings WHERE channel_id = ? ORDER BY transport",
            (channel_id,),
        ) as cursor:
            assert [str(row[0]) for row in await cursor.fetchall()] == ["desktop", "telegram"]
    finally:
        await store.close()


async def test_policy_selects_only_bindings_for_enabled_transports(tmp_path: Path):
    store, channel_id = await _store_with_bindings(tmp_path / "kai.db")
    try:
        telegram_only = WorkshopDeliveryBindingPolicy(frozenset({"telegram"}))
        mixed = WorkshopDeliveryBindingPolicy(frozenset({"desktop", "telegram"}))

        telegram_bindings = await telegram_only.binding_ids(store, channel_id)
        assert len(telegram_bindings) == 1
        assert await telegram_only.binding_ids(store, channel_id, transport="desktop") == ()
        assert len(await mixed.binding_ids(store, channel_id)) == 2
    finally:
        await store.close()
