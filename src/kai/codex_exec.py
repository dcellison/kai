"""
Codex `exec --json` NDJSON parsing helpers shared by one-shot callers.

The codex CLI's one-shot mode (`codex exec --json`) emits NDJSON events on
stdout - one JSON object per line, each tagged by a top-level `type`
field from the ThreadEvent enum. One-shot callers (triage.py, review.py,
and any future codex-driven agent) all need to recover the final
agent_message text from that stream, so the parser lives here rather
than being duplicated per caller.

This module is intentionally tiny and dependency-free: no Kai-side
imports, no I/O, no logging. It exists so the two callers above can
share one definition site without one importing from the other (which
would couple unrelated agent surfaces; review.py has no reason to know
about triage.py and vice versa).

Distinct from `codex.py`, which manages the persistent codex app-server
subprocess for conversational use. That module speaks the JSON-RPC
`thread/turn/item` protocol; this one parses the dot-separated event
stream `codex exec --json` writes to stdout in one-shot mode. Same
underlying data model, different wire encodings, different callers,
different lifecycles - hence the separate file.
"""

import json


def extract_codex_text(stdout: str) -> str:
    """
    Walk codex's NDJSON event stream and return the agent message text.

    `codex exec --json` emits one JSON event per line. Each event has
    a top-level `type` tag from the ThreadEvent enum:
    `thread.started`, `turn.started`, `turn.completed`, `turn.failed`,
    `item.started`, `item.updated`, `item.completed`, `error`. (Note
    the DOT separator; the app-server protocol uses slashes
    instead. Same data model, different wire encoding.)

    Callers only care about the agent's final natural-language
    response. The `item.completed` event for an agent_message item
    carries the full consolidated text:

        {"type": "item.completed",
         "item": {"id": "...", "type": "agent_message", "text": "..."}}

    Schema reference: codex-rs/exec/src/exec_events.rs in the codex
    repo. The `ThreadItemDetails` enum is `#[serde(tag = "type",
    rename_all = "snake_case")]` so the discriminator is
    `"agent_message"` (snake_case), and `text` is a flat field on
    the item object (the inner enum is `#[serde(flatten)]`).

    A streaming run may emit `item.updated` events for the same
    agent_message id before its `item.completed`. We trust the
    completed event as authoritative; if no completed event arrived
    (e.g. truncated stream) we fall back to the latest updated text
    so the caller gets something rather than nothing.

    Schema-drift posture: a future codex release that adds new event
    types or item types must not break extraction. Unknown shapes
    are silently skipped. A `turn.failed` event short-circuits to an
    empty result so the caller raises a clearer error than a partial
    body would.

    Args:
        stdout: The full stdout from `codex exec --json`.

    Returns:
        The agent_message text from the last `item.completed`
        event, or the latest `item.updated` text as a fallback.
        Empty string if no agent_message was emitted.
    """
    completed_text: str | None = None
    latest_updated_text: str | None = None
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        event_type = obj.get("type")
        # `turn.failed` is a terminal failure - no body text to
        # extract; let the caller see an empty string and surface
        # a clearer error than a half-event would.
        if event_type == "turn.failed":
            return ""
        if event_type in ("item.completed", "item.updated"):
            text = _recover_agent_message_text(obj)
            if text is None:
                continue
            if event_type == "item.completed":
                completed_text = text
            else:
                latest_updated_text = text
    if completed_text is not None:
        return completed_text.strip()
    if latest_updated_text is not None:
        return latest_updated_text.strip()
    return ""


def _recover_agent_message_text(obj: dict) -> str | None:
    """
    Pull `item.text` from a codex exec event when the item is an
    agent_message; return None otherwise.

    The wire shape is `{..., "item": {"id": "...", "type": "agent_message", "text": "..."}}`
    because ThreadItemDetails is serde-flattened onto ThreadItem.
    """
    item = obj.get("item")
    if not isinstance(item, dict):
        return None
    if item.get("type") != "agent_message":
        return None
    text = item.get("text")
    if isinstance(text, str):
        return text
    return None
