"""
Per-chat concurrency primitives for serializing message handling.

Provides functionality to:
1. Allocate per-chat asyncio locks so only one message is processed at a time
2. Allocate per-chat stop events so /stop can interrupt in-flight responses
3. Bound memory usage by evicting the oldest entries when limits are reached

Both lock and stop-event pools are bounded dicts keyed by chat_id. The eviction
limit (_MAX_LOCKS) is generous for a single-user bot but prevents unbounded
growth if the bot were ever exposed to multiple chats.
"""

import asyncio

# Maximum number of per-chat locks/events to keep in memory.
# When exceeded, the oldest (least-recently-inserted) entry is evicted.
_MAX_LOCKS = 64

# chat_id → asyncio.Lock: ensures only one Claude interaction per chat at a time
_chat_locks: dict[int, asyncio.Lock] = {}

# chat_id → asyncio.Event: set when the user sends /stop to cancel a response
_stop_events: dict[int, asyncio.Event] = {}


def get_lock(chat_id: int) -> asyncio.Lock:
    """
    Get or create an asyncio lock for this chat.

    Used to serialize message handling — while Claude is processing one message,
    subsequent messages for the same chat wait on this lock rather than spawning
    concurrent Claude interactions.

    Args:
        chat_id: Telegram chat ID to get the lock for.

    Returns:
        An asyncio.Lock unique to this chat_id (created on first access).
    """
    lock = _chat_locks.get(chat_id)
    if lock is not None:
        return lock
    # Evict oldest entry if at capacity
    if len(_chat_locks) >= _MAX_LOCKS:
        oldest = next(iter(_chat_locks))
        del _chat_locks[oldest]
    lock = asyncio.Lock()
    _chat_locks[chat_id] = lock
    return lock


def get_stop_event(chat_id: int) -> asyncio.Event:
    """
    Get or create a stop event for this chat.

    The /stop command sets this event, and the streaming loop in
    _handle_response() checks it between stream chunks. When set,
    the response is aborted and the Claude process is killed.

    Args:
        chat_id: Telegram chat ID to get the stop event for.

    Returns:
        An asyncio.Event unique to this chat_id. Set = stop requested.
    """
    event = _stop_events.get(chat_id)
    if event is not None:
        return event
    # Evict oldest entry if at capacity
    if len(_stop_events) >= _MAX_LOCKS:
        oldest = next(iter(_stop_events))
        del _stop_events[oldest]
    event = asyncio.Event()
    _stop_events[chat_id] = event
    return event
