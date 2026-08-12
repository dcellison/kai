"""Non-secret operator diagnostics for Workshop transition state."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kai.workshop.domain import (
    EventEnvelope,
    EventId,
    PrincipalId,
    WorkshopEventType,
    WorkshopId,
    parse_opaque_id,
)

_REQUIRED_TABLES = {
    "workshops",
    "principals",
    "workshop_memberships",
    "agents",
    "channel_bindings",
    "channel_memberships",
    "projection_checkpoints",
}

_PARITY_TABLES = {
    "channel_memberships",
    "event_log",
    "messages",
    "principals",
    "channel_bindings",
    "workshop_memberships",
}
_DELIVERY_AUTHORITY_TABLES = {
    "delivery_authority_epochs",
    "delivery_outbox",
    "delivery_fragments",
}
_TELEGRAM_SUBJECT_PATTERN = re.compile(r"^-?[0-9]+$")
_SYNTHETIC_ASSISTANT_PATTERN = re.compile(
    r"\[(stopped by user|no response|error: .+)\]",
    re.DOTALL,
)
_SCHEDULED_ASSISTANT_PREFIXES = ("[Reminder:", "[Job:")


@dataclass(frozen=True, slots=True)
class _ReplayMessage:
    message_id: str
    channel_id: str
    author_principal_id: str
    reply_to_message_id: str | None
    body: str
    event_position: int
    created_at: str
    direction: str

    @property
    def legacy_key(self) -> tuple[str, str]:
        return self.direction, hashlib.sha256(self.body.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class _ReplayState:
    principal_kinds: dict[str, str]
    channel_kinds: dict[str, str]
    workshop_memberships: dict[str, tuple[str, str, str]]
    channel_bindings: dict[str, tuple[str, str, str]]
    channel_memberships: dict[str, tuple[str, str, str]]
    messages: tuple[_ReplayMessage, ...]


def _pending_status(expected_humans: int | None) -> str:
    if expected_humans is None:
        return "Workshop bootstrap: pending; configured-human count unavailable"
    return (
        "Workshop bootstrap: pending; service startup will seed "
        f"1 workshop, {expected_humans} human principal(s), "
        f"{expected_humans} Telegram direct channel binding(s), and 1 Kai agent"
    )


def _scalar(connection: sqlite3.Connection, query: str, parameters: tuple[object, ...] = ()) -> int:
    row = connection.execute(query, parameters).fetchone()
    if row is None:
        return 0
    return int(row[0])


def workshop_bootstrap_status(db_path: Path, *, expected_humans: int | None) -> str:
    """Describe canonical bootstrap state without exposing identity data."""
    if not db_path.is_file():
        return _pending_status(expected_humans)

    try:
        connection = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
        try:
            rows = connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
            tables = {str(row[0]) for row in rows}
            if not tables >= _REQUIRED_TABLES:
                return _pending_status(expected_humans)

            workshop_count = _scalar(connection, "SELECT COUNT(*) FROM workshops")
            human_count = _scalar(connection, "SELECT COUNT(*) FROM principals WHERE kind = 'human'")
            agent_count = _scalar(connection, "SELECT COUNT(*) FROM agents")
            telegram_binding_count = _scalar(
                connection,
                "SELECT COUNT(*) FROM channel_bindings WHERE transport = ?",
                ("telegram",),
            )
            channel_membership_count = _scalar(connection, "SELECT COUNT(*) FROM channel_memberships")
            projection_count = _scalar(
                connection,
                "SELECT COUNT(*) FROM projection_checkpoints WHERE name = ?",
                ("canonical_conversations",),
            )
        finally:
            connection.close()
    except (OSError, sqlite3.Error) as exc:
        return f"Workshop bootstrap: NOT VERIFIED ({type(exc).__name__})"

    expected_memberships = 2 * (human_count if expected_humans is None else expected_humans)
    expected_state_present = (
        expected_humans is None or (human_count >= expected_humans and telegram_binding_count >= expected_humans)
    ) and channel_membership_count >= expected_memberships
    initialized = workshop_count >= 1 and agent_count >= 1 and projection_count == 1 and expected_state_present
    state = "initialized" if initialized else "pending"
    expectation = (
        "configured-human count unavailable" if expected_humans is None else f"expected humans={expected_humans}"
    )
    return (
        f"Workshop bootstrap: {state}; workshops={workshop_count}, humans={human_count}, "
        f"Telegram bindings={telegram_binding_count}, channel memberships={channel_membership_count}, "
        f"agents={agent_count}; {expectation}"
    )


def workshop_delivery_authority_status(db_path: Path) -> str:
    """Describe delivery-authority readiness using aggregate, non-secret state."""
    prefix = "Workshop delivery authority:"
    if not db_path.is_file():
        return f"{prefix} pending; authority schema unavailable"

    try:
        connection = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
        try:
            connection.execute("PRAGMA query_only=ON")
            connection.execute("BEGIN")
            tables = {
                str(row[0])
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
            }
            if not tables >= _DELIVERY_AUTHORITY_TABLES:
                return f"{prefix} pending; authority schema unavailable"

            epoch_count = _scalar(connection, "SELECT COUNT(*) FROM delivery_authority_epochs")
            active_rows = connection.execute(
                "SELECT id FROM delivery_authority_epochs WHERE status = 'active'"
            ).fetchall()
            active_count = len(active_rows)
            unclassified = _scalar(
                connection,
                "SELECT COUNT(*) FROM delivery_outbox "
                "WHERE purpose = 'conversation_reply' "
                "AND execution_contract = 'streaming_finalization' "
                "AND authority_epoch_id IS NULL",
            )
            active_epoch_id = str(active_rows[0][0]) if active_count == 1 else None
            prior_parameters: tuple[object, ...] = ()
            prior_filter = "authority_epoch_id IS NOT NULL"
            if active_epoch_id is not None:
                prior_filter += " AND authority_epoch_id != ?"
                prior_parameters = (active_epoch_id,)
            prior_nonterminal = _scalar(
                connection,
                "SELECT COUNT(*) FROM delivery_outbox WHERE "
                f"{prior_filter} AND status IN ('pending', 'leased', 'retry_wait')",
                prior_parameters,
            )
            prior_failed = _scalar(
                connection,
                f"SELECT COUNT(*) FROM delivery_outbox WHERE {prior_filter} AND status = 'failed'",
                prior_parameters,
            )
            prior_uncertain = _scalar(
                connection,
                "SELECT COUNT(DISTINCT o.id) FROM delivery_outbox o "
                "JOIN delivery_fragments f ON f.delivery_id = o.id WHERE "
                f"{prior_filter.replace('authority_epoch_id', 'o.authority_epoch_id')} "
                "AND o.status = 'failed' AND f.status = 'uncertain'",
                prior_parameters,
            )
            unacknowledged_epochs = _scalar(
                connection,
                "SELECT COUNT(*) FROM delivery_authority_epochs ae "
                "WHERE ae.status = 'deactivated' "
                "AND ae.terminal_failures_acknowledged_at IS NULL "
                "AND EXISTS (SELECT 1 FROM delivery_outbox o "
                "WHERE o.authority_epoch_id = ae.id AND o.status = 'failed')",
            )

            counts = {
                "pending": 0,
                "leased": 0,
                "retrying": 0,
                "succeeded": 0,
                "failed": 0,
                "uncertain": 0,
            }
            if active_epoch_id is not None:
                rows = connection.execute(
                    "SELECT o.status, EXISTS ("
                    "SELECT 1 FROM delivery_fragments f "
                    "WHERE f.delivery_id = o.id AND f.status = 'uncertain'"
                    ") AS uncertain FROM delivery_outbox o WHERE o.authority_epoch_id = ?",
                    (active_epoch_id,),
                ).fetchall()
                for status_value, uncertain_value in rows:
                    status = str(status_value)
                    if status == "failed" and int(uncertain_value):
                        counts["uncertain"] += 1
                    elif status == "retry_wait":
                        counts["retrying"] += 1
                    elif status in counts:
                        counts[status] += 1
        finally:
            connection.close()
    except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
        return f"{prefix} NOT VERIFIED ({type(exc).__name__})"

    if active_count > 1 or unclassified or prior_nonterminal or unacknowledged_epochs:
        state = "NOT READY"
    elif active_count == 1:
        state = "active"
    else:
        state = "inactive"
    return (
        f"{prefix} {state}; epochs={epoch_count}, unclassified={unclassified}, "
        f"prior nonterminal={prior_nonterminal}, prior failed={prior_failed}, "
        f"prior uncertain={prior_uncertain}, unacknowledged epochs={unacknowledged_epochs}, "
        f"active pending={counts['pending']}, "
        f"leased={counts['leased']}, retrying={counts['retrying']}, "
        f"succeeded={counts['succeeded']}, failed={counts['failed']}, "
        f"uncertain={counts['uncertain']}"
    )


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _event_from_row(row: sqlite3.Row) -> EventEnvelope:
    actor = row["actor_principal_id"]
    envelope = EventEnvelope(
        event_id=EventId(str(row["event_id"])),
        envelope_version=int(row["envelope_version"]),
        event_type=str(row["event_type"]),
        event_version=int(row["event_version"]),
        workshop_id=WorkshopId(str(row["workshop_id"])),
        aggregate_type=str(row["aggregate_type"]),
        aggregate_id=parse_opaque_id(str(row["aggregate_id"])),
        actor_principal_id=PrincipalId(str(actor)) if actor is not None else None,
        occurred_at=_parse_timestamp(str(row["occurred_at"])),
        idempotency_key=str(row["idempotency_key"]) if row["idempotency_key"] is not None else None,
        payload=json.loads(str(row["payload_json"])),
        metadata=json.loads(str(row["metadata_json"])),
    )
    if envelope.content_hash != row["content_hash"]:
        raise ValueError("Workshop event content hash mismatch")
    return envelope


def _required_payload_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Workshop event lacks {key}")
    return value


def _insert_replayed_fact(facts: dict[str, Any], fact_id: str, value: Any) -> None:
    if fact_id in facts:
        raise ValueError("Workshop event replay contains a duplicate aggregate")
    facts[fact_id] = value


def _replay_state(connection: sqlite3.Connection) -> _ReplayState:
    principal_kinds: dict[str, str] = {}
    channel_kinds: dict[str, str] = {}
    workshop_memberships: dict[str, tuple[str, str, str]] = {}
    channel_bindings: dict[str, tuple[str, str, str]] = {}
    channel_memberships: dict[str, tuple[str, str, str]] = {}
    replayed: list[_ReplayMessage] = []
    for row in connection.execute("SELECT * FROM event_log ORDER BY position"):
        envelope = _event_from_row(row)
        payload = envelope.payload
        aggregate_id = str(envelope.aggregate_id)
        if (
            envelope.event_type
            in {
                WorkshopEventType.PRINCIPAL_CREATED,
                WorkshopEventType.CHANNEL_CREATED,
                WorkshopEventType.WORKSHOP_MEMBER_ADDED,
                WorkshopEventType.CHANNEL_MEMBER_ADDED,
                WorkshopEventType.TRANSPORT_CHANNEL_BOUND,
                WorkshopEventType.MESSAGE_CREATED,
            }
            and envelope.event_version != 1
        ):
            raise ValueError("Workshop event replay encountered an unsupported event version")
        if envelope.event_type == WorkshopEventType.PRINCIPAL_CREATED:
            _insert_replayed_fact(
                principal_kinds,
                aggregate_id,
                _required_payload_text(payload, "kind"),
            )
            continue
        if envelope.event_type == WorkshopEventType.CHANNEL_CREATED:
            _insert_replayed_fact(
                channel_kinds,
                aggregate_id,
                _required_payload_text(payload, "kind"),
            )
            continue
        if envelope.event_type == WorkshopEventType.WORKSHOP_MEMBER_ADDED:
            _insert_replayed_fact(
                workshop_memberships,
                aggregate_id,
                (
                    str(envelope.workshop_id),
                    _required_payload_text(payload, "principal_id"),
                    _required_payload_text(payload, "role"),
                ),
            )
            continue
        if envelope.event_type == WorkshopEventType.TRANSPORT_CHANNEL_BOUND:
            _insert_replayed_fact(
                channel_bindings,
                aggregate_id,
                (
                    _required_payload_text(payload, "channel_id"),
                    _required_payload_text(payload, "transport"),
                    _required_payload_text(payload, "external_channel_id"),
                ),
            )
            continue
        if envelope.event_type == WorkshopEventType.CHANNEL_MEMBER_ADDED:
            _insert_replayed_fact(
                channel_memberships,
                aggregate_id,
                (
                    _required_payload_text(payload, "channel_id"),
                    _required_payload_text(payload, "principal_id"),
                    _required_payload_text(payload, "role"),
                ),
            )
            continue
        if envelope.event_type != WorkshopEventType.MESSAGE_CREATED:
            continue

        author_id = _required_payload_text(payload, "author_principal_id")
        kind = principal_kinds.get(author_id)
        if kind == "human":
            direction = "user"
        elif kind == "agent":
            direction = "assistant"
        else:
            raise ValueError("Workshop message author kind is not parity-comparable")
        reply_to = payload.get("reply_to_message_id")
        if reply_to is not None and not isinstance(reply_to, str):
            raise ValueError("Workshop message reply reference is invalid")
        replayed.append(
            _ReplayMessage(
                message_id=str(envelope.aggregate_id),
                channel_id=_required_payload_text(payload, "channel_id"),
                author_principal_id=author_id,
                reply_to_message_id=reply_to,
                body=_required_payload_text(payload, "body"),
                event_position=int(row["position"]),
                created_at=envelope.occurred_at.isoformat(),
                direction=direction,
            )
        )
    return _ReplayState(
        principal_kinds=principal_kinds,
        channel_kinds=channel_kinds,
        workshop_memberships=workshop_memberships,
        channel_bindings=channel_bindings,
        channel_memberships=channel_memberships,
        messages=tuple(replayed),
    )


def _mapping_mismatches(expected: dict[str, Any], actual: dict[str, Any]) -> int:
    return sum(expected.get(fact_id) != actual.get(fact_id) for fact_id in expected.keys() | actual.keys())


def _projection_mismatches(connection: sqlite3.Connection, replayed: _ReplayState) -> tuple[int, int]:
    expected = {
        message.message_id: (
            message.channel_id,
            message.author_principal_id,
            message.reply_to_message_id,
            message.body,
            message.event_position,
            message.created_at,
            message.direction,
        )
        for message in replayed.messages
    }
    actual = {
        str(row[0]): (
            str(row[1]),
            str(row[2]),
            str(row[3]) if row[3] is not None else None,
            str(row[4]),
            int(row[5]),
            str(row[6]),
            "user" if row[7] == "human" else "assistant" if row[7] == "agent" else "unknown",
        )
        for row in connection.execute(
            "SELECT m.id, m.channel_id, m.author_principal_id, m.reply_to_message_id, m.body, "
            "m.created_event_position, m.created_at, p.kind FROM messages m "
            "JOIN principals p ON p.id = m.author_principal_id"
        ).fetchall()
    }
    actual_principal_kinds = {
        str(row[0]): str(row[1]) for row in connection.execute("SELECT id, kind FROM principals").fetchall()
    }
    actual_channel_kinds = {
        str(row[0]): str(row[1]) for row in connection.execute("SELECT id, kind FROM channels").fetchall()
    }
    actual_bindings = {
        str(row[0]): (str(row[1]), str(row[2]), str(row[3]))
        for row in connection.execute(
            "SELECT id, channel_id, transport, external_channel_id FROM channel_bindings"
        ).fetchall()
    }
    actual_workshop_memberships = {
        str(row[0]): (str(row[1]), str(row[2]), str(row[3]))
        for row in connection.execute("SELECT id, workshop_id, principal_id, role FROM workshop_memberships").fetchall()
    }
    actual_channel_memberships = {
        str(row[0]): (str(row[1]), str(row[2]), str(row[3]))
        for row in connection.execute("SELECT id, channel_id, principal_id, role FROM channel_memberships").fetchall()
    }
    mismatches = (
        _mapping_mismatches(expected, actual)
        + _mapping_mismatches(replayed.principal_kinds, actual_principal_kinds)
        + _mapping_mismatches(replayed.channel_kinds, actual_channel_kinds)
        + _mapping_mismatches(replayed.workshop_memberships, actual_workshop_memberships)
        + _mapping_mismatches(replayed.channel_bindings, actual_bindings)
        + _mapping_mismatches(replayed.channel_memberships, actual_channel_memberships)
    )
    return len(actual), mismatches


def _read_legacy_keys(history_root: Path, external_channel_id: str) -> tuple[list[tuple[str, str]], int, int]:
    if not _TELEGRAM_SUBJECT_PATTERN.fullmatch(external_channel_id):
        return [], 0, 1
    if history_root.is_symlink():
        return [], 0, 1
    history_dir = history_root / external_channel_id
    if history_dir.is_symlink():
        return [], 0, 1
    if not history_dir.is_dir():
        return [], 0, 0

    keys: list[tuple[str, str]] = []
    pending_user_turns: deque[bool] = deque()
    malformed = 0
    unreadable = 0
    for path in sorted(history_dir.glob("*.jsonl")):
        if path.is_symlink():
            unreadable += 1
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            unreadable += 1
            continue
        for line in lines:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if not isinstance(record, dict):
                malformed += 1
                continue
            direction = record.get("dir")
            text = record.get("text")
            chat_id = record.get("chat_id")
            if direction not in {"user", "assistant"} or not isinstance(text, str):
                continue
            if str(chat_id) != external_channel_id:
                continue
            if direction == "user":
                media = record.get("media")
                is_shadowed_message = media is None or (
                    isinstance(media, dict)
                    and media.get("type") in {"photo", "document", "voice"}
                    and media.get("workshop_message_shadowed") is True
                )
                pending_user_turns.append(is_shadowed_message)
                if is_shadowed_message:
                    keys.append((direction, hashlib.sha256(text.encode("utf-8")).hexdigest()))
                continue
            if text.startswith(_SCHEDULED_ASSISTANT_PREFIXES):
                continue
            if not pending_user_turns:
                continue
            shadowed_turn = pending_user_turns.popleft()
            if not shadowed_turn or _SYNTHETIC_ASSISTANT_PATTERN.fullmatch(text):
                continue
            keys.append((direction, hashlib.sha256(text.encode("utf-8")).hexdigest()))
    return keys, malformed, unreadable


def _compare_legacy_suffix(
    canonical_keys: list[tuple[str, str]],
    legacy_keys: list[tuple[str, str]],
) -> tuple[int, int, int]:
    """Align the canonical sequence after an allowed pre-shadow legacy prefix."""
    if not canonical_keys:
        return 0, 0, 0

    candidates: list[tuple[int, int, int, int]] = []
    for start, legacy_key in enumerate(legacy_keys):
        if legacy_key != canonical_keys[0]:
            continue
        canonical_index = 0
        matched = 0
        for candidate in legacy_keys[start:]:
            if canonical_index < len(canonical_keys) and candidate == canonical_keys[canonical_index]:
                canonical_index += 1
                matched += 1
        missing = len(canonical_keys) - matched
        unmatched = len(legacy_keys) - start - matched
        candidates.append((matched, missing, unmatched, start))
    if not candidates:
        return 0, len(canonical_keys), 0
    matched, missing, unmatched, _ = min(
        candidates,
        key=lambda result: (-result[0], result[1] + result[2], -result[3]),
    )
    return matched, missing, unmatched


def _legacy_parity(
    replayed: _ReplayState,
    history_root: Path,
) -> tuple[int, int, int, int, int, int]:
    bindings: dict[str, str] = {}
    duplicate_bindings = 0
    for channel_id, transport, external_channel_id in replayed.channel_bindings.values():
        if transport != "telegram" or replayed.channel_kinds.get(channel_id) != "direct":
            continue
        if channel_id in bindings:
            duplicate_bindings += 1
        bindings[channel_id] = external_channel_id

    by_channel: dict[str, list[_ReplayMessage]] = {}
    for message in replayed.messages:
        if message.channel_id in bindings:
            by_channel.setdefault(message.channel_id, []).append(message)

    matched = 0
    missing = 0
    unmatched = 0
    malformed = 0
    unreadable = duplicate_bindings
    for channel_id, messages in by_channel.items():
        legacy_keys, channel_malformed, channel_unreadable = _read_legacy_keys(
            history_root,
            bindings[channel_id],
        )
        malformed += channel_malformed
        unreadable += channel_unreadable
        channel_matched, channel_missing, channel_unmatched = _compare_legacy_suffix(
            [message.legacy_key for message in messages],
            legacy_keys,
        )
        matched += channel_matched
        missing += channel_missing
        unmatched += channel_unmatched
    return matched, missing, unmatched, len(by_channel), malformed, unreadable


def workshop_message_parity_status(db_path: Path, history_root: Path) -> str:
    """Compare event-replayed messages with the projection and legacy JSONL.

    The deployed database is opened in SQLite read-only mode. Message bodies,
    principal identities, channel identities, filenames, and hashes never
    appear in the returned operator status.
    """
    if not db_path.is_file():
        return "Workshop message parity: pending; canonical message schema unavailable"

    try:
        connection = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA query_only=ON")
            connection.execute("BEGIN")
            tables = {
                str(row[0])
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
            }
            if not tables >= _PARITY_TABLES:
                return "Workshop message parity: pending; canonical message schema unavailable"
            replayed = _replay_state(connection)
            projected_count, replay_mismatches = _projection_mismatches(connection, replayed)
            matched, missing, unmatched, channel_count, malformed, unreadable = _legacy_parity(
                replayed,
                history_root,
            )
        finally:
            connection.close()
    except (OSError, sqlite3.Error, UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return f"Workshop message parity: NOT VERIFIED ({type(exc).__name__})"

    if not replayed.messages and replay_mismatches == 0:
        return "Workshop message parity: pending; canonical=0, projected=0; awaiting shadow messages"

    if malformed or unreadable:
        details: list[str] = []
        if malformed:
            details.append(f"malformed JSONL records={malformed}")
        if unreadable:
            details.append(f"unreadable history sources={unreadable}")
        return f"Workshop message parity: NOT VERIFIED; {', '.join(details)}"

    state = "clean" if replay_mismatches == 0 and missing == 0 and unmatched == 0 else "diverged"
    return (
        f"Workshop message parity: {state}; canonical={len(replayed.messages)}, projected={projected_count}, "
        f"replay mismatches={replay_mismatches}, JSONL matched={matched}, JSONL missing={missing}, "
        f"JSONL unmatched={unmatched}, Telegram channels={channel_count}"
    )
