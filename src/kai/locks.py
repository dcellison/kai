import asyncio

# Per-chat locks to serialize messages within a conversation.
# Bounded to prevent unbounded growth; evicts least-recently-inserted
# entries when the limit is reached (unlikely for a single-user bot).
_MAX_LOCKS = 64
_chat_locks: dict[int, asyncio.Lock] = {}
_stop_events: dict[int, asyncio.Event] = {}


def get_lock(chat_id: int) -> asyncio.Lock:
    lock = _chat_locks.get(chat_id)
    if lock is not None:
        return lock
    if len(_chat_locks) >= _MAX_LOCKS:
        oldest = next(iter(_chat_locks))
        del _chat_locks[oldest]
    lock = asyncio.Lock()
    _chat_locks[chat_id] = lock
    return lock


def get_stop_event(chat_id: int) -> asyncio.Event:
    """Get or create a stop event for this chat. Set = stop requested."""
    event = _stop_events.get(chat_id)
    if event is not None:
        return event
    if len(_stop_events) >= _MAX_LOCKS:
        oldest = next(iter(_stop_events))
        del _stop_events[oldest]
    event = asyncio.Event()
    _stop_events[chat_id] = event
    return event
