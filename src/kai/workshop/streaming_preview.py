"""Durable Telegram streaming-preview bindings."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from sqlite3 import Row

from kai.workshop.domain import ChannelBindingId, ChannelId, MessageId, WorkshopId
from kai.workshop.store import WorkshopEventStore

_MAX_SQLITE_INTEGER = (1 << 63) - 1
_CONFIRMED_NON_FINAL = "confirmed_non_final"


class StreamingPreviewTargetError(LookupError):
    """The canonical inbound message is not eligible for a direct-chat preview."""


class StreamingPreviewBindingError(LookupError):
    """The direct channel does not have exactly one canonical Telegram binding."""


class StreamingPreviewConflictError(RuntimeError):
    """A durable preview identity was reused with different canonical content."""


@dataclass(frozen=True, slots=True)
class ConfirmedTelegramStreamingPreview:
    """One Telegram message already confirmed as a non-final streaming preview.

    The caller supplies no channel, binding, transport, or chat destination.
    Those identities are resolved from the canonical inbound message.
    """

    inbound_message_id: MessageId
    external_message_id: int
    confirmed_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.inbound_message_id, MessageId):
            raise ValueError("inbound_message_id must be a MessageId")
        if (
            not isinstance(self.external_message_id, int)
            or isinstance(self.external_message_id, bool)
            or not 1 <= self.external_message_id <= _MAX_SQLITE_INTEGER
        ):
            raise ValueError("external_message_id must be a positive signed 64-bit integer")
        if self.confirmed_at.tzinfo is None or self.confirmed_at.utcoffset() is None:
            raise ValueError("confirmed_at must be timezone-aware")


@dataclass(frozen=True, slots=True)
class TelegramStreamingPreviewBinding:
    inbound_message_id: MessageId
    workshop_id: WorkshopId
    channel_id: ChannelId
    channel_binding_id: ChannelBindingId
    external_message_id: int
    state: str
    confirmed_at: datetime
    inserted: bool


@dataclass(frozen=True, slots=True)
class ResolvedTelegramStreamingTarget:
    workshop_id: WorkshopId
    channel_id: ChannelId
    channel_binding_id: ChannelBindingId


@dataclass(frozen=True, slots=True)
class ResolvedTelegramFinalizationTarget:
    """A direct Telegram delivery target plus preview eligibility."""

    workshop_id: WorkshopId
    channel_id: ChannelId
    channel_binding_id: ChannelBindingId
    preview_eligible: bool


def _format_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


async def _resolve_telegram_target(
    store: WorkshopEventStore,
    inbound_message_id: MessageId,
    *,
    allow_workshop_client: bool,
    require_binding: bool,
) -> ResolvedTelegramFinalizationTarget | None:
    async with store.connection.execute(
        "SELECT c.workshop_id, m.channel_id, c.kind, p.kind, m.reply_to_message_id, "
        "json_extract(e.metadata_json, '$.source'), "
        "json_extract(e.metadata_json, '$.transport_message_id'), "
        "json_extract(e.metadata_json, '$.client_message_id'), p.id "
        "FROM messages m "
        "JOIN channels c ON c.id = m.channel_id "
        "JOIN principals p ON p.id = m.author_principal_id "
        "JOIN event_log e ON e.position = m.created_event_position "
        "WHERE m.id = ?",
        (inbound_message_id,),
    ) as cursor:
        row = await cursor.fetchone()
    valid_telegram = row is not None and row[5] == "telegram" and isinstance(row[6], str) and bool(row[6])
    valid_workshop_client = (
        allow_workshop_client
        and row is not None
        and row[5] == "workshop_client"
        and row[6] is None
        and isinstance(row[7], str)
        and bool(row[7])
    )
    if (
        row is None
        or row[2] != "direct"
        or row[3] != "human"
        or row[4] is not None
        or not (valid_telegram or valid_workshop_client)
    ):
        expected = "Telegram or Workshop client" if allow_workshop_client else "Telegram inbound"
        raise StreamingPreviewTargetError(
            f"Target must be an existing {expected} message from a human in a direct channel"
        )

    channel_id = ChannelId(str(row[1]))
    async with store.connection.execute(
        "SELECT b.id, EXISTS (SELECT 1 FROM external_identities e "
        "WHERE e.provider = 'telegram' AND e.principal_id = ? "
        "AND e.external_subject = b.external_channel_id) "
        "FROM channel_bindings b WHERE b.channel_id = ? "
        "AND b.transport = 'telegram' ORDER BY b.id",
        (str(row[8]), channel_id),
    ) as cursor:
        bindings = list(await cursor.fetchall())
    if not bindings and not require_binding:
        return None
    if len(bindings) != 1 or int(bindings[0][1]) != 1:
        raise StreamingPreviewBindingError("Canonical direct channel must have exactly one Telegram binding")
    return ResolvedTelegramFinalizationTarget(
        workshop_id=WorkshopId(str(row[0])),
        channel_id=channel_id,
        channel_binding_id=ChannelBindingId(str(bindings[0][0])),
        preview_eligible=valid_telegram,
    )


async def resolve_telegram_streaming_target(
    store: WorkshopEventStore,
    inbound_message_id: MessageId,
) -> ResolvedTelegramStreamingTarget:
    target = await _resolve_telegram_target(
        store,
        inbound_message_id,
        allow_workshop_client=False,
        require_binding=True,
    )
    assert target is not None
    return ResolvedTelegramStreamingTarget(
        workshop_id=target.workshop_id,
        channel_id=target.channel_id,
        channel_binding_id=target.channel_binding_id,
    )


async def resolve_telegram_finalization_target(
    store: WorkshopEventStore,
    inbound_message_id: MessageId,
) -> ResolvedTelegramFinalizationTarget | None:
    """Resolve optional Telegram final delivery for supported ingress."""
    return await _resolve_telegram_target(
        store,
        inbound_message_id,
        allow_workshop_client=True,
        require_binding=False,
    )


def _from_row(row: Row, *, inserted: bool) -> TelegramStreamingPreviewBinding:
    values = tuple(row)
    return TelegramStreamingPreviewBinding(
        inbound_message_id=MessageId(str(values[0])),
        workshop_id=WorkshopId(str(values[1])),
        channel_id=ChannelId(str(values[2])),
        channel_binding_id=ChannelBindingId(str(values[3])),
        external_message_id=int(values[4]),
        state=str(values[5]),
        confirmed_at=_parse_timestamp(str(values[6])),
        inserted=inserted,
    )


async def bind_confirmed_telegram_streaming_preview(
    store: WorkshopEventStore,
    preview: ConfirmedTelegramStreamingPreview,
) -> TelegramStreamingPreviewBinding:
    """Persist one confirmed preview without sending or editing Telegram.

    The canonical inbound message is the only routing input. The direct channel
    and unique Telegram binding are resolved under the same write transaction,
    so this boundary cannot be used to select an arbitrary destination.
    """

    connection = store.connection
    try:
        await connection.execute("BEGIN IMMEDIATE")
        target = await resolve_telegram_streaming_target(store, preview.inbound_message_id)
        async with connection.execute(
            "SELECT inbound_message_id, workshop_id, channel_id, channel_binding_id, "
            "external_message_id, state, confirmed_at "
            "FROM telegram_streaming_previews WHERE inbound_message_id = ?",
            (preview.inbound_message_id,),
        ) as cursor:
            existing = await cursor.fetchone()
        if existing is not None:
            bound = _from_row(existing, inserted=False)
            if (
                bound.workshop_id != target.workshop_id
                or bound.channel_id != target.channel_id
                or bound.channel_binding_id != target.channel_binding_id
                or bound.external_message_id != preview.external_message_id
                or bound.state != _CONFIRMED_NON_FINAL
            ):
                raise StreamingPreviewConflictError("Inbound message already has a different streaming-preview binding")
            await connection.commit()
            return bound

        async with connection.execute(
            "SELECT inbound_message_id FROM telegram_streaming_previews "
            "WHERE channel_binding_id = ? AND external_message_id = ?",
            (target.channel_binding_id, preview.external_message_id),
        ) as cursor:
            reused = await cursor.fetchone()
        if reused is not None:
            raise StreamingPreviewConflictError(
                "Telegram preview message is already bound to a different canonical inbound message"
            )

        confirmed_at = _format_timestamp(preview.confirmed_at)
        await connection.execute(
            "INSERT INTO telegram_streaming_previews "
            "(inbound_message_id, workshop_id, channel_id, channel_binding_id, "
            "external_message_id, state, confirmed_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                preview.inbound_message_id,
                target.workshop_id,
                target.channel_id,
                target.channel_binding_id,
                preview.external_message_id,
                _CONFIRMED_NON_FINAL,
                confirmed_at,
            ),
        )
        await connection.commit()
        return TelegramStreamingPreviewBinding(
            inbound_message_id=preview.inbound_message_id,
            workshop_id=target.workshop_id,
            channel_id=target.channel_id,
            channel_binding_id=target.channel_binding_id,
            external_message_id=preview.external_message_id,
            state=_CONFIRMED_NON_FINAL,
            confirmed_at=_parse_timestamp(confirmed_at),
            inserted=True,
        )
    except Exception:
        await connection.rollback()
        raise
