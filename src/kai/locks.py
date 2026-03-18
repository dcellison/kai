"""
Per-user concurrency primitives for serializing message handling.

Provides functionality to:
1. Allocate per-user asyncio locks so only one message is processed at a time
2. Allocate per-user stop events so /stop can interrupt in-flight responses
3. Bound memory usage by evicting the oldest entries when limits are reached

Both lock and stop-event pools are bounded dicts keyed by user_id. The eviction
limit (_MAX_LOCKS) is generous but prevents unbounded growth if many users are
configured.
"""

import asyncio

# Maximum number of per-user locks/events to keep in memory.
# When exceeded, the oldest (least-recently-inserted) entry is evicted.
_MAX_LOCKS = 64

# user_id -> asyncio.Lock: ensures only one Claude interaction per user at a time
_user_locks: dict[int, asyncio.Lock] = {}

# user_id -> asyncio.Event: set when the user sends /stop to cancel a response
_stop_events: dict[int, asyncio.Event] = {}


def get_lock(user_id: int) -> asyncio.Lock:
    """
    Get or create an asyncio lock for this user.

    Used to serialize message handling -- while Claude is processing one message,
    subsequent messages for the same user wait on this lock rather than spawning
    concurrent Claude interactions.

    Args:
        user_id: Telegram user ID to get the lock for.

    Returns:
        An asyncio.Lock unique to this user_id (created on first access).
    """
    lock = _user_locks.get(user_id)
    if lock is not None:
        return lock
    # Evict oldest entry if at capacity, but skip any lock that is currently
    # held - evicting an active lock would break the serialization guarantee
    # for that user (a new get_lock() call would create a different lock).
    if len(_user_locks) >= _MAX_LOCKS:
        for candidate in list(_user_locks):
            if not _user_locks[candidate].locked():
                del _user_locks[candidate]
                break
    lock = asyncio.Lock()
    _user_locks[user_id] = lock
    return lock


def get_stop_event(user_id: int) -> asyncio.Event:
    """
    Get or create a stop event for this user.

    The /stop command sets this event, and the streaming loop in
    _handle_response() checks it between stream chunks. When set,
    the response is aborted and the Claude process is killed.

    Args:
        user_id: Telegram user ID to get the stop event for.

    Returns:
        An asyncio.Event unique to this user_id. Set = stop requested.
    """
    event = _stop_events.get(user_id)
    if event is not None:
        return event
    # Evict oldest entry if at capacity, but skip any event that is currently
    # set - evicting an active stop event would cause /stop to create a new
    # event the in-flight streaming loop never sees.
    if len(_stop_events) >= _MAX_LOCKS:
        for candidate in list(_stop_events):
            if not _stop_events[candidate].is_set():
                del _stop_events[candidate]
                break
    event = asyncio.Event()
    _stop_events[user_id] = event
    return event
