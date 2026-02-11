"""Tests for cron.py pure functions."""

from datetime import UTC, datetime, timezone, timedelta

from kai.cron import _ensure_utc


class TestEnsureUtc:
    def test_naive_gets_utc(self):
        naive = datetime(2026, 1, 15, 12, 0, 0)
        assert naive.tzinfo is None
        result = _ensure_utc(naive)
        assert result.tzinfo is UTC
        assert result.year == 2026
        assert result.hour == 12

    def test_utc_aware_returned_as_is(self):
        aware = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)
        result = _ensure_utc(aware)
        assert result is aware

    def test_other_timezone_returned_as_is(self):
        eastern = timezone(timedelta(hours=-5))
        dt = datetime(2026, 1, 15, 12, 0, 0, tzinfo=eastern)
        result = _ensure_utc(dt)
        assert result is dt
        assert result.tzinfo is eastern
