"""Tests for locks.py lock and stop event management."""

import asyncio

import pytest

from kai import locks
from kai.locks import get_lock, get_stop_event


@pytest.fixture(autouse=True)
def _clean_locks():
    """Reset internal dicts before and after each test."""
    locks._chat_locks.clear()
    locks._stop_events.clear()
    yield
    locks._chat_locks.clear()
    locks._stop_events.clear()


# ── get_lock ─────────────────────────────────────────────────────────


class TestGetLock:
    def test_creates_new_lock(self):
        lock = get_lock(1)
        assert isinstance(lock, asyncio.Lock)

    def test_returns_same_lock(self):
        lock1 = get_lock(1)
        lock2 = get_lock(1)
        assert lock1 is lock2

    def test_different_ids_different_locks(self):
        assert get_lock(1) is not get_lock(2)

    def test_eviction_at_max(self, monkeypatch):
        monkeypatch.setattr(locks, "_MAX_LOCKS", 3)
        get_lock(1)
        get_lock(2)
        get_lock(3)
        assert 1 in locks._chat_locks
        # Adding a 4th should evict chat_id=1 (oldest)
        get_lock(4)
        assert 1 not in locks._chat_locks
        assert 4 in locks._chat_locks
        assert len(locks._chat_locks) == 3


# ── get_stop_event ───────────────────────────────────────────────────


class TestGetStopEvent:
    def test_creates_new_event(self):
        event = get_stop_event(1)
        assert isinstance(event, asyncio.Event)

    def test_returns_same_event(self):
        event1 = get_stop_event(1)
        event2 = get_stop_event(1)
        assert event1 is event2

    def test_different_ids_different_events(self):
        assert get_stop_event(1) is not get_stop_event(2)

    def test_eviction_at_max(self, monkeypatch):
        monkeypatch.setattr(locks, "_MAX_LOCKS", 3)
        get_stop_event(10)
        get_stop_event(20)
        get_stop_event(30)
        # Adding a 4th should evict chat_id=10
        get_stop_event(40)
        assert 10 not in locks._stop_events
        assert 40 in locks._stop_events
        assert len(locks._stop_events) == 3
