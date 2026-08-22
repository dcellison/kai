"""Bounded backend context assembled from canonical Workshop messages."""

from __future__ import annotations

from dataclasses import dataclass

from kai.workshop.domain import MessageId
from kai.workshop.run_lifecycle import DurableRun
from kai.workshop.store import WorkshopEventStore

_MAX_CONTEXT_MESSAGES = 50
_MAX_CONTEXT_CHARACTERS = 24_000
_MAX_MESSAGE_CHARACTERS = 6_000


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
    """Return completed exchanges before ``run`` in its exact owner lane.

    Pairing follows durable run lineage rather than adjacent channel messages.
    That prevents another human, another agent, notifications, failures, or an
    interleaved run from becoming semantic-memory episode context.
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

    async with store.connection.execute(
        "SELECT source.body, result.body FROM runs prior "
        "JOIN messages source ON source.id = prior.inbound_message_id "
        "JOIN messages result ON result.id = prior.result_message_id "
        "WHERE prior.channel_id = ? AND prior.agent_id = ? "
        "AND prior.requested_by_principal_id = ? AND prior.status = 'completed' "
        "AND source.created_event_position < ? "
        "ORDER BY source.created_event_position DESC, prior.id DESC LIMIT ?",
        (
            run.channel_id,
            run.agent_id,
            run.requested_by_principal_id,
            int(inbound[0]),
            limit,
        ),
    ) as cursor:
        rows = list(await cursor.fetchall())
    return tuple((str(row[0]), str(row[1])) for row in reversed(rows))
