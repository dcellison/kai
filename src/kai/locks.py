"""
Per-lane concurrency primitives for serializing agent execution.

Provides functionality to:
1. Allocate per-lane asyncio locks so only one message is processed at a time
2. Allocate per-lane stop events so cancellation can interrupt in-flight responses
3. Bound memory usage by evicting the oldest entries when limits are reached

Both lock and stop-event pools are bounded dicts keyed by a canonical execution
lane. Compatibility callers may still use an integer until their canonical
assignment is loaded. The eviction
limit (_MAX_LOCKS) is generous for a single-user bot but prevents unbounded
growth if the bot were ever exposed to multiple chats.
"""

import asyncio

# Maximum number of per-chat locks/events to keep in memory.
# When exceeded, the oldest (least-recently-inserted) entry is evicted.
_MAX_LOCKS = 64

ExecutionLaneKey = int | str

# lane → asyncio.Lock: ensures only one agent interaction per lane at a time
_chat_locks: dict[ExecutionLaneKey, asyncio.Lock] = {}

# lane → asyncio.Event: set when the user requests response cancellation
_stop_events: dict[ExecutionLaneKey, asyncio.Event] = {}


def get_lock(lane: ExecutionLaneKey) -> asyncio.Lock:
    """
    Get or create an asyncio lock for this execution lane.

    Used to serialize message handling while an agent is processing one message.
    Subsequent messages for the same lane wait instead of spawning concurrent
    interactions.

    Args:
        lane: Canonical channel-agent lane, or a compatibility key before cutover.

    Returns:
        An asyncio.Lock unique to this execution lane (created on first access).
    """
    lock = _chat_locks.get(lane)
    if lock is not None:
        return lock
    # Evict oldest entry if at capacity, but skip any lock that is currently
    # held - evicting an active lock would break the serialization guarantee
    # for that chat (a new get_lock() call would create a different lock).
    if len(_chat_locks) >= _MAX_LOCKS:
        for candidate in list(_chat_locks):
            if not _chat_locks[candidate].locked():
                del _chat_locks[candidate]
                break
    lock = asyncio.Lock()
    _chat_locks[lane] = lock
    return lock


def get_stop_event(lane: ExecutionLaneKey) -> asyncio.Event:
    """
    Get or create a stop event for this execution lane.

    Cancellation sets this event, and the compatibility streaming loop checks it
    between stream chunks. When set, the response is aborted and its agent
    process is killed.

    Args:
        lane: Canonical channel-agent lane, or a compatibility key before cutover.

    Returns:
        An asyncio.Event unique to this execution lane. Set = stop requested.
    """
    event = _stop_events.get(lane)
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
    _stop_events[lane] = event
    return event
