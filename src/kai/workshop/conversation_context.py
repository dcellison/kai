"""Bounded backend context assembled from canonical Workshop messages."""

from __future__ import annotations

from dataclasses import dataclass

from kai.workshop.domain import MessageId
from kai.workshop.run_lifecycle import DurableRun
from kai.workshop.store import WorkshopEventStore

_MAX_CONTEXT_MESSAGES = 50
_MAX_CONTEXT_CHARACTERS = 24_000
_MAX_MESSAGE_CHARACTERS = 6_000
_MIN_PRIOR_PAIR_SCAN_MESSAGES = 20


@dataclass(frozen=True, slots=True)
class CanonicalConversationContext:
    text: str
    message_count: int
    through_event_position: int


def _render(author: str, kind: str, body: str) -> str:
    label = author.strip() or ("Agent" if kind == "agent" else "Human")
    bounded = body if len(body) <= _MAX_MESSAGE_CHARACTERS else body[:_MAX_MESSAGE_CHARACTERS] + "\n[…truncated]"
    return f"{label}:\n{bounded.strip()}"


async def assemble_canonical_conversation_context(
    store: WorkshopEventStore,
    run: DurableRun,
) -> CanonicalConversationContext:
    """Return recent canonical messages strictly before the run's prompt."""
    if not isinstance(run.inbound_message_id, MessageId):
        raise TypeError("run must identify a typed inbound message")
    async with store.connection.execute(
        "SELECT created_event_position FROM messages WHERE id = ? AND channel_id = ?",
        (run.inbound_message_id, run.channel_id),
    ) as cursor:
        inbound = await cursor.fetchone()
    if inbound is None:
        raise RuntimeError("Canonical inbound message no longer exists")
    inbound_position = int(inbound[0])

    async with store.connection.execute(
        "SELECT p.display_name, p.kind, m.body, m.created_event_position "
        "FROM messages m JOIN principals p ON p.id = m.author_principal_id "
        "WHERE m.channel_id = ? AND m.created_event_position < ? "
        "ORDER BY m.created_event_position DESC LIMIT ?",
        (run.channel_id, inbound_position, _MAX_CONTEXT_MESSAGES),
    ) as cursor:
        rows = list(await cursor.fetchall())

    selected: list[tuple[str, str, str, int]] = []
    characters = 0
    for row in rows:
        rendered = _render(str(row[0]), str(row[1]), str(row[2]))
        if selected and characters + len(rendered) > _MAX_CONTEXT_CHARACTERS:
            break
        selected.append((str(row[0]), str(row[1]), str(row[2]), int(row[3])))
        characters += len(rendered)
    selected.reverse()
    text = "\n\n".join(_render(name, kind, body) for name, kind, body, _ in selected)
    through = selected[-1][3] if selected else 0
    return CanonicalConversationContext(text, len(selected), through)


async def assemble_canonical_prior_pairs(
    store: WorkshopEventStore,
    run: DurableRun,
    *,
    limit: int,
) -> tuple[tuple[str, str], ...]:
    """Return completed human/agent pairs before this run's inbound message.

    This is the canonical counterpart to the compatibility JSONL episode
    window.  The current exchange is excluded by event position, so callers
    do not need the old ``+1`` fetch followed by dropping the newest pair.
    """
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
        raise ValueError("limit must be a non-negative integer")
    if limit == 0:
        return ()
    async with store.connection.execute(
        "SELECT created_event_position FROM messages WHERE id = ? AND channel_id = ?",
        (run.inbound_message_id, run.channel_id),
    ) as cursor:
        inbound = await cursor.fetchone()
    if inbound is None:
        raise RuntimeError("Canonical inbound message no longer exists")

    scan_limit = max(_MIN_PRIOR_PAIR_SCAN_MESSAGES, limit * 4)
    async with store.connection.execute(
        "SELECT p.kind, m.body FROM messages m "
        "JOIN principals p ON p.id = m.author_principal_id "
        "WHERE m.channel_id = ? AND m.created_event_position < ? "
        "AND p.kind IN ('human', 'agent') "
        "ORDER BY m.created_event_position DESC LIMIT ?",
        (run.channel_id, int(inbound[0]), scan_limit),
    ) as cursor:
        rows = list(await cursor.fetchall())

    pending_human: str | None = None
    pairs: list[tuple[str, str]] = []
    for row in reversed(rows):
        kind = str(row[0])
        body = str(row[1]).strip()
        if not body:
            continue
        if kind == "human":
            pending_human = body
        elif pending_human is not None:
            pairs.append((pending_human, body))
            pending_human = None
    return tuple(pairs[-limit:])
