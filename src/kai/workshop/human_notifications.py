"""Canonical, principal-private human mention notifications."""

from __future__ import annotations

import base64
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from kai.workshop.delivery_outbox import NOTIFICATION_PURPOSE
from kai.workshop.delivery_planning import CanonicalDeliveryIntent, WorkshopDeliveryPlanner
from kai.workshop.delivery_policy import WorkshopDeliveryBindingPolicy
from kai.workshop.domain import (
    ChannelId,
    EventEnvelope,
    HumanNotificationId,
    MessageId,
    PrincipalId,
    WorkshopEventType,
    WorkshopId,
)
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.store import AppendResult, StoredEvent, WorkshopEventStore

MAX_NOTIFICATION_PAGE_SIZE = 100
MAX_NOTIFICATION_MUTATION_BATCH = 100
HUMAN_NOTIFICATION_DELIVERY_MODE = "human_notification"


class WorkshopHumanNotificationError(RuntimeError):
    """A human notification request could not be completed."""


class WorkshopHumanNotificationAccessDenied(WorkshopHumanNotificationError):
    """The notification is not in the authenticated principal's accessible inbox."""


class WorkshopHumanNotificationConflict(WorkshopHumanNotificationError):
    """The notification state changed before the requested mutation."""


class WorkshopHumanNotificationValidationError(WorkshopHumanNotificationError):
    """The notification request is malformed."""


@dataclass(frozen=True, slots=True)
class HumanNotification:
    notification_id: HumanNotificationId
    kind: str
    source_message_id: MessageId
    source_channel_id: ChannelId
    source_thread_root_id: MessageId | None
    source_author_principal_id: PrincipalId
    source_author_display_name: str
    channel_name: str | None
    created_at: datetime
    created_event_position: int
    read: bool
    read_at: datetime | None
    state_version: int
    last_event_position: int


@dataclass(frozen=True, slots=True)
class HumanNotificationCounts:
    total: int
    unread: int
    read: int
    unread_by_channel: tuple[tuple[ChannelId, int], ...] = ()


@dataclass(frozen=True, slots=True)
class HumanNotificationPage:
    notifications: tuple[HumanNotification, ...]
    counts: HumanNotificationCounts
    next_cursor: str | None
    through_position: int


@dataclass(frozen=True, slots=True)
class HumanNotificationMutation:
    notification: HumanNotification
    changed: bool
    replayed: bool


@dataclass(frozen=True, slots=True)
class HumanNotificationStateRequest:
    notification_id: HumanNotificationId
    expected_state_version: int


@dataclass(frozen=True, slots=True)
class HumanNotificationEvent:
    notification: HumanNotification
    transition: WorkshopEventType
    event_position: int


@dataclass(frozen=True, slots=True)
class HumanNotificationEventBatch:
    events: tuple[HumanNotificationEvent, ...]
    next_position: int


def _parse_timestamp(value: object) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)


def _bounded_operation_id(value: object) -> str:
    if not isinstance(value, str):
        raise WorkshopHumanNotificationValidationError("client_operation_id must be text")
    normalized = value.strip()
    if not normalized or len(normalized) > 200:
        raise WorkshopHumanNotificationValidationError("client_operation_id must contain 1 through 200 characters")
    return normalized


def _state_version(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise WorkshopHumanNotificationValidationError("expected_state_version must be a non-negative integer")
    return value


def _notification_from_row(row: object) -> HumanNotification:
    values = tuple(row)  # type: ignore[arg-type]
    return HumanNotification(
        notification_id=HumanNotificationId(str(values[0])),
        kind=str(values[1]),
        source_message_id=MessageId(str(values[2])),
        source_channel_id=ChannelId(str(values[3])),
        source_thread_root_id=(MessageId(str(values[4])) if values[4] is not None else None),
        source_author_principal_id=PrincipalId(str(values[5])),
        source_author_display_name=str(values[6]),
        channel_name=(str(values[7]) if values[7] is not None else None),
        created_at=_parse_timestamp(values[8]),
        created_event_position=int(values[9]),
        read=values[10] is not None,
        read_at=(_parse_timestamp(values[10]) if values[10] is not None else None),
        state_version=int(values[11]),
        last_event_position=int(values[12]),
    )


_INBOX_COLUMNS = (
    "n.id, n.kind, n.source_message_id, n.source_channel_id, "
    "n.source_thread_root_id, m.author_principal_id, p.display_name, c.name, "
    "n.created_at, n.created_event_position, n.read_at, n.state_version, "
    "n.last_event_position"
)


def _encode_cursor(position: int, notification_id: HumanNotificationId) -> str:
    raw = json.dumps([position, str(notification_id)], separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _decode_cursor(value: object) -> tuple[int, HumanNotificationId]:
    if not isinstance(value, str) or not value or len(value) > 256:
        raise WorkshopHumanNotificationValidationError("Invalid notification cursor")
    try:
        padded = value + "=" * (-len(value) % 4)
        decoded = json.loads(base64.urlsafe_b64decode(padded.encode()).decode())
        if (
            not isinstance(decoded, list)
            or len(decoded) != 2
            or not isinstance(decoded[0], int)
            or isinstance(decoded[0], bool)
            or decoded[0] < 1
            or not isinstance(decoded[1], str)
        ):
            raise ValueError
        return decoded[0], HumanNotificationId(decoded[1])
    except (ValueError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkshopHumanNotificationValidationError("Invalid notification cursor") from exc


def _safe_label(value: object, *, fallback: str) -> str:
    normalized = " ".join(str(value).split()).strip()
    return normalized[:200] if normalized else fallback


async def _external_delivery_policy_result(
    store: WorkshopEventStore,
    principal_id: PrincipalId,
    occurred_at: datetime,
) -> str:
    """Apply personal DND only to adapter publication, never the inbox."""
    async with store.connection.execute(
        "SELECT dnd_enabled, dnd_timezone, dnd_start_minute, dnd_end_minute "
        "FROM principal_human_notification_policies WHERE principal_id = ?",
        (principal_id,),
    ) as cursor:
        row = await cursor.fetchone()
    if row is None or not bool(row[0]):
        return "eligible"
    try:
        local = occurred_at.astimezone(ZoneInfo(str(row[1])))
    except ZoneInfoNotFoundError:
        return "suppressed_dnd"
    minute = local.hour * 60 + local.minute
    start = int(row[2])
    end = int(row[3])
    inside = start <= minute < end if start < end else minute >= start or minute < end
    return "suppressed_dnd" if inside else "eligible"


async def _adapter_delivery_decisions(
    store: WorkshopEventStore,
    delivery_policy: WorkshopDeliveryBindingPolicy,
    workshop_id: WorkshopId,
    principal_id: PrincipalId,
) -> tuple[dict[str, object], ...]:
    """Resolve adapter eligibility from canonical recipient-owned preferences."""
    bindings = await delivery_policy.principal_bindings(store, workshop_id, principal_id)
    transports = sorted({binding.transport for binding in bindings})
    decisions: list[dict[str, object]] = []
    for transport in transports:
        async with store.connection.execute(
            "SELECT enabled FROM principal_human_notification_adapter_preferences "
            "WHERE principal_id = ? AND transport = ?",
            (principal_id, transport),
        ) as cursor:
            preference = await cursor.fetchone()
        enabled = preference is None or bool(preference[0])
        decisions.append(
            {
                "transport": transport,
                "policy_result": "eligible" if enabled else "suppressed_preference",
            }
        )
    return tuple(decisions)


async def _publication_body(
    store: WorkshopEventStore,
    *,
    author_principal_id: PrincipalId,
    source_channel_id: ChannelId,
    source_thread_root_id: MessageId | None,
    kind: str,
    deep_link: str | None,
) -> str:
    """Render bounded adapter-safe text without copying message content."""
    async with store.connection.execute(
        "SELECT p.display_name, c.name FROM principals p CROSS JOIN channels c WHERE p.id = ? AND c.id = ?",
        (author_principal_id, source_channel_id),
    ) as cursor:
        row = await cursor.fetchone()
    if row is None:
        raise RuntimeError("Canonical human notification source context is unavailable")
    author = _safe_label(row[0], fallback="Someone")
    channel = _safe_label(row[1], fallback="a Workshop channel")
    location = f"#{channel}" if row[1] is not None else channel
    if source_thread_root_id is not None:
        location = f"a thread in {location}"
    verb = {
        "mention": "mentioned you in",
        "reply": "replied to you in",
        "message": "posted a message in",
    }.get(kind)
    if verb is None:
        raise RuntimeError("Canonical human notification kind is unsupported")
    body = f"{author} {verb} {location}."
    if deep_link is not None:
        body = f"{body}\nOpen Workshop: {deep_link}"
    return body


async def append_human_notifications_in_transaction(
    store: WorkshopEventStore,
    message_event: StoredEvent,
    *,
    delivery_policy: WorkshopDeliveryBindingPolicy | None = None,
) -> tuple[AppendResult, ...]:
    """Append policy-qualified notification facts for one canonical message."""
    if not store.connection.in_transaction:
        raise RuntimeError("Notification creation requires an active transaction")
    envelope = message_event.envelope
    if envelope.event_type != WorkshopEventType.MESSAGE_CREATED or not isinstance(
        envelope.aggregate_id,
        MessageId,
    ):
        raise ValueError("Notification creation requires a canonical message event")
    raw_mentions = envelope.payload.get("mentions", [])
    if not isinstance(raw_mentions, list):
        raise RuntimeError("Canonical message mentions are malformed")
    source_channel_id = ChannelId(str(envelope.payload["channel_id"]))
    async with store.connection.execute(
        "SELECT kind FROM channels WHERE id = ?",
        (source_channel_id,),
    ) as cursor:
        source_channel = await cursor.fetchone()
    if source_channel is None or str(source_channel[0]) != "group":
        return ()
    source_thread_root = envelope.payload.get("thread_root_id")
    if source_thread_root is not None:
        source_thread_root = MessageId(str(source_thread_root))
    author_principal_id = PrincipalId(str(envelope.payload["author_principal_id"]))
    mentioned_recipients: set[PrincipalId] = set()
    for raw in raw_mentions:
        if not isinstance(raw, dict) or raw.get("kind") != "human":
            continue
        try:
            mentioned_recipients.add(PrincipalId(str(raw["principal_id"])))
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("Canonical human mention is malformed") from exc

    async with store.connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' "
        "AND name IN ('human_notifications', 'human_notification_publications', "
        "'human_notification_adapter_delivery_decisions', "
        "'principal_human_notification_adapter_preferences', "
        "'principal_channel_notification_policies', 'principal_human_notification_policies')"
    ) as cursor:
        notification_tables = {str(row[0]) for row in await cursor.fetchall()}
    if "human_notifications" not in notification_tables:
        return ()
    policy_available = {
        "principal_channel_notification_policies",
        "principal_human_notification_policies",
    } <= notification_tables
    publication_available = "human_notification_publications" in notification_tables
    adapter_decisions_available = {
        "human_notification_adapter_delivery_decisions",
        "principal_human_notification_adapter_preferences",
    } <= notification_tables

    reply_recipient: PrincipalId | None = None
    reply_target = envelope.payload.get("reply_to_message_id") or source_thread_root
    if reply_target is not None:
        async with store.connection.execute(
            "SELECT author_principal_id FROM messages WHERE id = ? AND channel_id = ?",
            (reply_target, source_channel_id),
        ) as cursor:
            root = await cursor.fetchone()
        if root is not None:
            reply_recipient = PrincipalId(str(root[0]))

    if policy_available:
        async with store.connection.execute(
            "SELECT p.id, COALESCE(cp.level, 'mentions_replies'), "
            "COALESCE(pp.muted_mentions_notify, 1) FROM channel_memberships cm "
            "JOIN principals p ON p.id = cm.principal_id AND p.kind = 'human' "
            "LEFT JOIN principal_channel_notification_policies cp "
            "ON cp.principal_id = p.id AND cp.channel_id = cm.channel_id "
            "LEFT JOIN principal_human_notification_policies pp ON pp.principal_id = p.id "
            "WHERE cm.channel_id = ? AND p.id != ? ORDER BY p.id",
            (source_channel_id, author_principal_id),
        ) as cursor:
            policy_rows = tuple(await cursor.fetchall())
    else:
        policy_rows = tuple((recipient_id, "mentions_replies", 1) for recipient_id in mentioned_recipients)

    recipients: list[tuple[PrincipalId, str]] = []
    for row in policy_rows:
        recipient_id = PrincipalId(str(row[0]))
        level = str(row[1])
        is_mention = recipient_id in mentioned_recipients
        is_reply = recipient_id == reply_recipient
        notify = (
            level == "all"
            or (level == "mentions_replies" and (is_mention or is_reply))
            or (level == "muted" and is_mention and bool(row[2]))
        )
        if notify:
            recipients.append((recipient_id, "mention" if is_mention else "reply" if is_reply else "message"))
    results: list[AppendResult] = []
    for recipient_id, kind in recipients:
        notification_id = HumanNotificationId.derived(
            envelope.workshop_id,
            f"human-mention:{envelope.aggregate_id}:{recipient_id}",
        )
        publication: dict[str, object] | None = None
        adapter_decisions: tuple[dict[str, object], ...] = ()
        if publication_available:
            policy_result = await _external_delivery_policy_result(
                store,
                recipient_id,
                envelope.occurred_at,
            )
            publication = {
                "policy_result": policy_result,
                "alert_body": await _publication_body(
                    store,
                    author_principal_id=author_principal_id,
                    source_channel_id=source_channel_id,
                    source_thread_root_id=source_thread_root,
                    kind=kind,
                    deep_link=(delivery_policy.notification_deep_link if delivery_policy is not None else None),
                ),
                "deep_link": delivery_policy.notification_deep_link if delivery_policy is not None else None,
            }
            if adapter_decisions_available and delivery_policy is not None:
                adapter_decisions = await _adapter_delivery_decisions(
                    store,
                    delivery_policy,
                    envelope.workshop_id,
                    recipient_id,
                )
            if adapter_decisions_available:
                publication["adapter_decisions"] = list(adapter_decisions)
        notification = EventEnvelope.create(
            event_type=WorkshopEventType.HUMAN_NOTIFICATION_CREATED,
            event_version=(
                4
                if publication is not None and adapter_decisions_available
                else 3
                if publication is not None
                else 2
                if policy_available
                else 1
            ),
            workshop_id=envelope.workshop_id,
            aggregate_type="human_notification",
            aggregate_id=notification_id,
            actor_principal_id=author_principal_id,
            occurred_at=envelope.occurred_at,
            idempotency_key=f"human-notification:v1:{notification_id}",
            payload={
                "recipient_principal_id": recipient_id,
                "source_message_id": envelope.aggregate_id,
                "source_channel_id": source_channel_id,
                "source_thread_root_id": source_thread_root,
                "kind": kind,
                **({"publication": publication} if publication is not None else {}),
            },
            metadata={"source": "canonical_message_notification_policy"},
        )
        appended = await store.append_in_transaction(notification)
        results.append(appended)
        if appended.inserted and publication is not None:
            await store.project_pending_in_transaction(CanonicalConversationProjection())
            if publication["policy_result"] == "eligible":
                effective_policy = delivery_policy or WorkshopDeliveryBindingPolicy.disabled()
                await WorkshopDeliveryPlanner(store, effective_policy).plan_in_transaction(
                    CanonicalDeliveryIntent(
                        message_id=MessageId(str(envelope.aggregate_id)),
                        channel_id=source_channel_id,
                        mode=HUMAN_NOTIFICATION_DELIVERY_MODE,
                        purpose=NOTIFICATION_PURPOSE,
                        occurred_at=envelope.occurred_at,
                        recipient_principal_id=recipient_id,
                        human_notification_id=notification_id,
                        workshop_id=envelope.workshop_id,
                        eligible_transports=frozenset(
                            str(decision["transport"])
                            for decision in adapter_decisions
                            if decision["policy_result"] == "eligible"
                        )
                        if adapter_decisions_available
                        else None,
                    )
                )
    return tuple(results)


async def append_human_mention_notifications_in_transaction(
    store: WorkshopEventStore,
    message_event: StoredEvent,
    *,
    delivery_policy: WorkshopDeliveryBindingPolicy | None = None,
) -> tuple[AppendResult, ...]:
    """Compatibility name for canonical policy-qualified notification creation."""
    return await append_human_notifications_in_transaction(
        store,
        message_event,
        delivery_policy=delivery_policy,
    )


class WorkshopHumanNotificationService:
    """Query and mutate one authenticated human principal's canonical inbox."""

    def __init__(self, store: WorkshopEventStore) -> None:
        self._store = store

    async def counts(self, principal_id: PrincipalId) -> HumanNotificationCounts:
        self._validate_principal(principal_id)
        async with self._store.connection.execute(
            "SELECT COUNT(*), SUM(CASE WHEN n.read_at IS NULL THEN 1 ELSE 0 END) "
            "FROM human_notifications n JOIN channel_memberships cm "
            "ON cm.channel_id = n.source_channel_id AND cm.principal_id = ? "
            "WHERE n.recipient_principal_id = ?",
            (principal_id, principal_id),
        ) as cursor:
            row = await cursor.fetchone()
        total = int(row[0] or 0) if row is not None else 0
        unread = int(row[1] or 0) if row is not None else 0
        async with self._store.connection.execute(
            "SELECT n.source_channel_id, COUNT(*) FROM human_notifications n "
            "JOIN channel_memberships cm ON cm.channel_id = n.source_channel_id "
            "AND cm.principal_id = ? WHERE n.recipient_principal_id = ? "
            "AND n.read_at IS NULL GROUP BY n.source_channel_id "
            "ORDER BY n.source_channel_id",
            (principal_id, principal_id),
        ) as cursor:
            channel_rows = list(await cursor.fetchall())
        return HumanNotificationCounts(
            total,
            unread,
            total - unread,
            tuple((ChannelId(str(channel_id)), int(count)) for channel_id, count in channel_rows),
        )

    async def list(
        self,
        principal_id: PrincipalId,
        *,
        limit: int = 50,
        cursor: str | None = None,
        unread_only: bool = False,
    ) -> HumanNotificationPage:
        self._validate_principal(principal_id)
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= MAX_NOTIFICATION_PAGE_SIZE:
            raise WorkshopHumanNotificationValidationError(
                f"limit must be an integer from 1 through {MAX_NOTIFICATION_PAGE_SIZE}"
            )
        if not isinstance(unread_only, bool):
            raise WorkshopHumanNotificationValidationError("unread_only must be a boolean")
        async with self._store.connection.execute("SELECT COALESCE(MAX(position), 0) FROM event_log") as tip_cursor:
            tip_row = await tip_cursor.fetchone()
        through_position = int(tip_row[0]) if tip_row is not None else 0
        cursor_clause = ""
        parameters: list[object] = [principal_id, principal_id, through_position]
        if cursor is not None:
            position, notification_id = _decode_cursor(cursor)
            cursor_clause = "AND (n.created_event_position < ? OR (n.created_event_position = ? AND n.id < ?)) "
            parameters.extend((position, position, notification_id))
        unread_clause = "AND n.read_at IS NULL " if unread_only else ""
        parameters.append(limit + 1)
        async with self._store.connection.execute(
            f"SELECT {_INBOX_COLUMNS} FROM human_notifications n "
            "JOIN channel_memberships cm ON cm.channel_id = n.source_channel_id "
            "AND cm.principal_id = ? JOIN messages m ON m.id = n.source_message_id "
            "JOIN principals p ON p.id = m.author_principal_id "
            "JOIN channels c ON c.id = n.source_channel_id "
            "WHERE n.recipient_principal_id = ? AND n.created_event_position <= ? "
            + unread_clause
            + cursor_clause
            + "ORDER BY n.created_event_position DESC, n.id DESC LIMIT ?",
            tuple(parameters),
        ) as query_cursor:
            rows = list(await query_cursor.fetchall())
        has_more = len(rows) > limit
        page_rows = rows[:limit]
        notifications = tuple(_notification_from_row(row) for row in page_rows)
        next_cursor = None
        if has_more and notifications:
            last = notifications[-1]
            next_cursor = _encode_cursor(last.created_event_position, last.notification_id)
        return HumanNotificationPage(
            notifications,
            await self.counts(principal_id),
            next_cursor,
            through_position,
        )

    async def set_read_state(
        self,
        principal_id: PrincipalId,
        notification_id: HumanNotificationId,
        *,
        read: bool,
        expected_state_version: object,
        client_operation_id: object,
        occurred_at: datetime | None = None,
    ) -> HumanNotificationMutation:
        results = await self.set_many_read(
            principal_id,
            (HumanNotificationStateRequest(notification_id, _state_version(expected_state_version)),),
            read=read,
            client_operation_id=client_operation_id,
            occurred_at=occurred_at,
        )
        return results[0]

    async def set_many_read(
        self,
        principal_id: PrincipalId,
        requests: Sequence[HumanNotificationStateRequest],
        *,
        read: bool = True,
        client_operation_id: object,
        occurred_at: datetime | None = None,
    ) -> tuple[HumanNotificationMutation, ...]:
        self._validate_principal(principal_id)
        if not isinstance(read, bool):
            raise WorkshopHumanNotificationValidationError("read must be a boolean")
        if not requests or len(requests) > MAX_NOTIFICATION_MUTATION_BATCH:
            raise WorkshopHumanNotificationValidationError(
                f"notifications must contain 1 through {MAX_NOTIFICATION_MUTATION_BATCH} items"
            )
        operation_id = _bounded_operation_id(client_operation_id)
        normalized = tuple(requests)
        if any(not isinstance(item, HumanNotificationStateRequest) for item in normalized):
            raise WorkshopHumanNotificationValidationError("Invalid notification state request")
        if len({item.notification_id for item in normalized}) != len(normalized):
            raise WorkshopHumanNotificationValidationError("Notification state requests must be unique")
        now = occurred_at or datetime.now(UTC)
        if now.tzinfo is None or now.utcoffset() is None:
            raise WorkshopHumanNotificationValidationError("Notification state time must be timezone-aware")
        now = now.astimezone(UTC)
        connection = self._store.connection
        changed_ids: set[HumanNotificationId] = set()
        replayed_ids: set[HumanNotificationId] = set()
        try:
            await connection.execute("BEGIN IMMEDIATE")
            for item in normalized:
                idempotency_key = f"human-notification-state:v1:{principal_id}:{operation_id}:{item.notification_id}"
                existing = await self._store.event_by_idempotency_key(idempotency_key)
                if existing is not None:
                    expected_type = (
                        WorkshopEventType.HUMAN_NOTIFICATION_READ
                        if read
                        else WorkshopEventType.HUMAN_NOTIFICATION_UNREAD
                    )
                    if (
                        existing.envelope.event_type != expected_type
                        or existing.envelope.aggregate_id != item.notification_id
                        or existing.envelope.actor_principal_id != principal_id
                        or existing.envelope.payload.get("expected_state_version") != item.expected_state_version
                    ):
                        raise WorkshopHumanNotificationConflict(
                            "client_operation_id is already bound to a different notification mutation"
                        )
                    replayed_ids.add(item.notification_id)
                    continue
                current = await self._load_accessible(principal_id, item.notification_id)
                if current.state_version != item.expected_state_version:
                    raise WorkshopHumanNotificationConflict("Notification state revision is stale")
                if current.read == read:
                    continue
                event = EventEnvelope.create(
                    event_type=(
                        WorkshopEventType.HUMAN_NOTIFICATION_READ
                        if read
                        else WorkshopEventType.HUMAN_NOTIFICATION_UNREAD
                    ),
                    event_version=1,
                    workshop_id=await self._workshop_id(item.notification_id),
                    aggregate_type="human_notification",
                    aggregate_id=item.notification_id,
                    actor_principal_id=principal_id,
                    occurred_at=now,
                    idempotency_key=idempotency_key,
                    payload={
                        "recipient_principal_id": principal_id,
                        "expected_state_version": item.expected_state_version,
                    },
                    metadata={"source": "workshop_client"},
                )
                result = await self._store.append_in_transaction(event)
                if not result.inserted:
                    raise WorkshopHumanNotificationConflict("Notification mutation replay was inconsistent")
                changed_ids.add(item.notification_id)
            await self._store.project_pending_in_transaction(CanonicalConversationProjection())
            loaded_items: list[HumanNotificationMutation] = []
            for item in normalized:
                loaded_items.append(
                    HumanNotificationMutation(
                        await self._load_accessible(principal_id, item.notification_id),
                        item.notification_id in changed_ids,
                        item.notification_id in replayed_ids,
                    )
                )
            loaded = tuple(loaded_items)
            await connection.commit()
            return loaded
        except Exception:
            await connection.rollback()
            raise

    async def events(
        self,
        principal_id: PrincipalId,
        *,
        after_position: int | None,
        limit: int = 100,
    ) -> HumanNotificationEventBatch:
        self._validate_principal(principal_id)
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 100:
            raise WorkshopHumanNotificationValidationError("event limit must be from 1 through 100")
        async with self._store.connection.execute("SELECT COALESCE(MAX(position), 0) FROM event_log") as cursor:
            tip_row = await cursor.fetchone()
        tip = int(tip_row[0]) if tip_row is not None else 0
        if after_position is None:
            return HumanNotificationEventBatch((), tip)
        if (
            not isinstance(after_position, int)
            or isinstance(after_position, bool)
            or after_position < 0
            or after_position > tip
        ):
            raise WorkshopHumanNotificationValidationError("Invalid notification event resume position")
        event_types = (
            WorkshopEventType.HUMAN_NOTIFICATION_CREATED,
            WorkshopEventType.HUMAN_NOTIFICATION_READ,
            WorkshopEventType.HUMAN_NOTIFICATION_UNREAD,
        )
        placeholders = ", ".join("?" for _ in event_types)
        async with self._store.connection.execute(
            f"SELECT e.position, e.event_type, {_INBOX_COLUMNS} FROM event_log e "
            "JOIN human_notifications n ON n.id = e.aggregate_id "
            "JOIN channel_memberships cm ON cm.channel_id = n.source_channel_id "
            "AND cm.principal_id = ? JOIN messages m ON m.id = n.source_message_id "
            "JOIN principals p ON p.id = m.author_principal_id "
            "JOIN channels c ON c.id = n.source_channel_id "
            f"WHERE n.recipient_principal_id = ? AND e.event_type IN ({placeholders}) "
            "AND e.position > ? ORDER BY e.position LIMIT ?",
            (
                principal_id,
                principal_id,
                *(event_type.value for event_type in event_types),
                after_position,
                limit,
            ),
        ) as cursor:
            rows = list(await cursor.fetchall())
        events = tuple(
            HumanNotificationEvent(
                notification=_notification_from_row(tuple(row)[2:]),
                transition=WorkshopEventType(str(row[1])),
                event_position=int(row[0]),
            )
            for row in rows
        )
        return HumanNotificationEventBatch(
            events,
            events[-1].event_position if events else after_position,
        )

    async def _load_accessible(
        self,
        principal_id: PrincipalId,
        notification_id: HumanNotificationId,
    ) -> HumanNotification:
        async with self._store.connection.execute(
            f"SELECT {_INBOX_COLUMNS} FROM human_notifications n "
            "JOIN channel_memberships cm ON cm.channel_id = n.source_channel_id "
            "AND cm.principal_id = ? JOIN messages m ON m.id = n.source_message_id "
            "JOIN principals p ON p.id = m.author_principal_id "
            "JOIN channels c ON c.id = n.source_channel_id "
            "WHERE n.id = ? AND n.recipient_principal_id = ?",
            (principal_id, notification_id, principal_id),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            raise WorkshopHumanNotificationAccessDenied("Notification access denied")
        return _notification_from_row(row)

    async def _workshop_id(self, notification_id: HumanNotificationId) -> WorkshopId:
        async with self._store.connection.execute(
            "SELECT workshop_id FROM human_notifications WHERE id = ?",
            (notification_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            raise WorkshopHumanNotificationAccessDenied("Notification access denied")
        return WorkshopId(str(row[0]))

    @staticmethod
    def _validate_principal(principal_id: PrincipalId) -> None:
        if not isinstance(principal_id, PrincipalId):
            raise WorkshopHumanNotificationValidationError("Invalid notification principal")
