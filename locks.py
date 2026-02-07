from __future__ import annotations

import asyncio

# Per-chat locks to serialize messages within a conversation
_chat_locks: dict[int, asyncio.Lock] = {}


def get_lock(chat_id: int) -> asyncio.Lock:
    return _chat_locks.setdefault(chat_id, asyncio.Lock())
