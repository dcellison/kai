"""
Tests for src/kai/totp.py.

All subprocess.run calls are mocked so tests don't require root access or
actual /etc/kai/* files. The mock pattern replaces subprocess.run globally
within the totp module for each test.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from unittest.mock import MagicMock, patch

import pyotp
import pytest

import kai.totp
from kai.totp import (
    TotpStateError,
    get_failure_count,
    get_lockout_remaining,
    is_totp_configured,
    verify_code,
)


@pytest.fixture(autouse=True)
def _reset_totp_cache():
    """
    Reset the is_totp_configured module-level cache before and after each test.

    Without this, a test that calls is_totp_configured() and gets True would
    pollute the cache for subsequent tests in the same process run.
    """
    kai.totp._totp_is_configured = False
    yield
    kai.totp._totp_is_configured = False


@pytest.fixture(autouse=True)
def _protected_metadata_ok(monkeypatch):
    """Protected metadata checks are mocked unless a test overrides them."""
    monkeypatch.setattr("kai.totp.validate_protected_file_metadata", lambda *args, **kwargs: True)


# A stable base32 secret used across tests.
_TEST_SECRET = "JBSWY3DPEHPK3PXP"


def _secret_proc(secret: str = _TEST_SECRET) -> MagicMock:
    """Return a mock subprocess result that looks like a successful sudo cat of the secret file."""
    m = MagicMock()
    m.returncode = 0
    m.stdout = secret + "\n"
    return m


def _attempts_proc(failures: int = 0, lockout_until: float = 0, principal_id: int = 12345) -> MagicMock:
    """Return a mock subprocess result that looks like a successful sudo cat of the attempts file."""
    m = MagicMock()
    m.returncode = 0
    m.stdout = json.dumps(
        {
            "version": 2,
            "principals": {
                str(principal_id): {
                    "failures": failures,
                    "lockout_until": lockout_until,
                }
            },
        }
    )
    return m


def _failed_proc() -> MagicMock:
    """Return a mock subprocess result with a non-zero exit code (file doesn't exist, etc.)."""
    m = MagicMock()
    m.returncode = 1
    m.stdout = ""
    return m


def _tee_proc() -> MagicMock:
    """Return a mock subprocess result for a successful sudo tee write."""
    m = MagicMock()
    m.returncode = 0
    m.stdout = ""
    return m


# ── verify_code: valid and invalid codes ─────────────────────────────


def test_verify_code_rejects_malformed_input():
    """verify_code returns False immediately for non-6-digit input, with no subprocess calls."""
    with patch("kai.totp.subprocess.run") as mock_run:
        assert verify_code("12345", 12345) is False  # too short
        assert verify_code("1234567", 12345) is False  # too long
        assert verify_code("12345a", 12345) is False  # non-digit
        assert verify_code("", 12345) is False  # empty
        mock_run.assert_not_called()


def test_verify_code_valid():
    """verify_code returns True when a correct TOTP code is supplied."""
    valid_code = pyotp.TOTP(_TEST_SECRET).now()

    # subprocess.run is called in order: _read_attempts, _read_secret, _write_attempts
    with patch("kai.totp.subprocess.run") as mock_run:
        mock_run.side_effect = [
            _attempts_proc(),  # _read_attempts
            _secret_proc(),  # _read_secret
            _tee_proc(),  # _write_attempts (reset counter on success)
        ]
        result = verify_code(valid_code, 12345)

    assert result is True


def test_verify_code_invalid():
    """verify_code returns False for a wrong code and increments the failure counter."""
    with patch("kai.totp.subprocess.run") as mock_run:
        mock_run.side_effect = [
            _attempts_proc(),  # _read_attempts
            _secret_proc(),  # _read_secret
            _tee_proc(),  # _write_attempts (increment failures)
        ]
        result = verify_code("000000", 12345)

    assert result is False


# ── Rate limiting: failure counter ───────────────────────────────────


def test_failure_counter_increments():
    """Each failed attempt increments the stored failure count."""
    with patch("kai.totp.subprocess.run") as mock_run:
        mock_run.side_effect = [
            _attempts_proc(failures=0),  # _read_attempts
            _secret_proc(),  # _read_secret
            _tee_proc(),  # _write_attempts
        ]
        verify_code("000000", 12345)

        # Inspect what was written - it's the third call, with input= containing the state.
        write_call = mock_run.call_args_list[2]
        written = json.loads(write_call.kwargs["input"])

    assert written["principals"]["12345"]["failures"] == 1
    assert written["principals"]["12345"]["lockout_until"] == 0


def test_lockout_triggers_after_n_failures():
    """Reaching lockout_attempts consecutive failures sets a non-zero lockout_until timestamp."""
    # Simulate already at 2 failures (one below default lockout_attempts=3).
    with patch("kai.totp.subprocess.run") as mock_run:
        mock_run.side_effect = [
            _attempts_proc(failures=2),  # _read_attempts - already at 2
            _secret_proc(),  # _read_secret
            _tee_proc(),  # _write_attempts - should trigger lockout
        ]
        verify_code("000000", 12345)

        write_call = mock_run.call_args_list[2]
        written = json.loads(write_call.kwargs["input"])

    record = written["principals"]["12345"]
    assert record["failures"] == 3
    # lockout_until should be roughly now + 15 minutes
    assert record["lockout_until"] > time.time()
    assert record["lockout_until"] < time.time() + 15 * 60 + 5  # +5s tolerance


def test_verify_returns_false_during_lockout_even_with_valid_code():
    """A valid code is rejected when a lockout is active (no secret read attempted)."""
    valid_code = pyotp.TOTP(_TEST_SECRET).now()
    future_lockout = time.time() + 900  # 15 minutes from now

    with patch("kai.totp.subprocess.run") as mock_run:
        mock_run.side_effect = [
            _attempts_proc(failures=3, lockout_until=future_lockout),  # _read_attempts
        ]
        result = verify_code(valid_code, 12345)

    assert result is False
    # Only one subprocess call should have been made (reading attempts).
    # The secret file should NOT be read during lockout.
    assert mock_run.call_count == 1


def test_successful_code_resets_failure_counter():
    """A successful verification writes failures=0, lockout_until=0 to the attempts file."""
    valid_code = pyotp.TOTP(_TEST_SECRET).now()

    with patch("kai.totp.subprocess.run") as mock_run:
        mock_run.side_effect = [
            _attempts_proc(failures=2),  # _read_attempts - had 2 prior failures
            _secret_proc(),  # _read_secret
            _tee_proc(),  # _write_attempts
        ]
        result = verify_code(valid_code, 12345)

        write_call = mock_run.call_args_list[2]
        written = json.loads(write_call.kwargs["input"])

    assert result is True
    assert written["principals"]["12345"]["failures"] == 0
    assert written["principals"]["12345"]["lockout_until"] == 0


# ── is_totp_configured ───────────────────────────────────────────────


def test_is_totp_configured_true_when_readable():
    """Configured means both protected files exist, are readable, and validate."""
    with (
        patch("kai.totp._totp_files_present", return_value=(True, True)),
        patch("kai.totp.subprocess.run", side_effect=[_secret_proc(), _attempts_proc()]),
    ):
        assert is_totp_configured() is True


def test_is_totp_configured_false_only_when_both_files_absent():
    """A cleanly absent secret and attempts file is the sole disabled state."""
    with (
        patch("kai.totp._totp_files_present", return_value=(False, False)),
        patch("kai.totp.subprocess.run") as mock_run,
    ):
        assert is_totp_configured() is False
    mock_run.assert_not_called()


@pytest.mark.parametrize("presence", [(True, False), (False, True)])
def test_is_totp_configured_fails_closed_on_partial_state(presence):
    """Losing either protected file cannot silently disable configured TOTP."""
    with (
        patch("kai.totp._totp_files_present", return_value=presence),
        pytest.raises(TotpStateError, match="incomplete"),
    ):
        is_totp_configured()


def test_is_totp_configured_fails_closed_on_secret_read_error():
    """A sudo/permission failure is unavailable, not 'not configured'."""
    with (
        patch("kai.totp._totp_files_present", return_value=(True, True)),
        patch("kai.totp.subprocess.run", return_value=_failed_proc()),
        pytest.raises(TotpStateError, match="secret read failed"),
    ):
        is_totp_configured()


def test_totp_files_present_fails_closed_on_unsafe_metadata(monkeypatch):
    """Unsafe protected file metadata is unavailable, not disabled TOTP."""

    def _unsafe_metadata(*args, **kwargs):
        raise kai.totp.ProtectedConfigError("unsafe protected config")

    monkeypatch.setattr("kai.totp.validate_protected_file_metadata", _unsafe_metadata)

    with pytest.raises(TotpStateError, match="unsafe protected config"):
        kai.totp._totp_files_present()


# ── get_lockout_remaining ────────────────────────────────────────────


def test_get_lockout_remaining_zero_when_not_locked():
    """get_lockout_remaining returns 0 when lockout_until is 0 (no active lockout)."""
    with patch("kai.totp.subprocess.run", return_value=_attempts_proc(lockout_until=0)):
        assert get_lockout_remaining(12345) == 0


def test_get_lockout_remaining_positive_when_locked():
    """get_lockout_remaining returns a positive number of seconds when locked out."""
    future = time.time() + 300  # 5 minutes from now
    with patch("kai.totp.subprocess.run", return_value=_attempts_proc(lockout_until=future)):
        remaining = get_lockout_remaining(12345)

    # Should be close to 300, allow a few seconds of test execution slack.
    assert 295 <= remaining <= 300


# ── get_failure_count ────────────────────────────────────────────────


def test_get_failure_count_returns_failures_from_disk():
    """get_failure_count returns the current consecutive failure count from the attempts file."""
    with patch("kai.totp.subprocess.run", return_value=_attempts_proc(failures=2)):
        count = get_failure_count(12345)

    assert count == 2


def test_get_failure_count_returns_zero_on_clean_state():
    """get_failure_count returns 0 when there are no recorded failures."""
    with patch("kai.totp.subprocess.run", return_value=_attempts_proc(failures=0)):
        count = get_failure_count(12345)

    assert count == 0


# ── _read_attempts validation ────────────────────────────────────────


def test_corrupt_lockout_until_fails_closed():
    """A non-numeric lockout timestamp is rejected rather than reset.

    Fixes #36: the validation checked failures but not lockout_until,
    so a corrupted value like "abc" would pass validation and later
    crash on comparison with time.time().
    """
    corrupt = MagicMock()
    corrupt.returncode = 0
    corrupt.stdout = json.dumps({"failures": 0, "lockout_until": "abc"})
    with (
        patch("kai.totp.subprocess.run", return_value=corrupt),
        pytest.raises(TotpStateError, match="lockout timestamp"),
    ):
        get_lockout_remaining(12345)


def test_corrupt_failures_fails_closed():
    """A non-numeric failure count is rejected rather than reset."""
    corrupt = MagicMock()
    corrupt.returncode = 0
    corrupt.stdout = json.dumps({"failures": "xyz", "lockout_until": 0})
    with (
        patch("kai.totp.subprocess.run", return_value=corrupt),
        pytest.raises(TotpStateError, match="failure count"),
    ):
        get_failure_count(12345)


def test_legacy_global_state_is_preserved_during_migration():
    """The old global counter becomes a wildcard fallback, not a reset."""
    legacy = MagicMock(returncode=0, stdout=json.dumps({"failures": 2, "lockout_until": 0}))
    with patch("kai.totp.subprocess.run", return_value=legacy):
        assert get_failure_count(67890) == 2


def test_attempt_state_is_isolated_per_principal():
    """One Telegram user's failures do not consume another user's attempts."""
    document = {
        "version": 2,
        "principals": {
            "111": {"failures": 2, "lockout_until": 0},
            "222": {"failures": 0, "lockout_until": 0},
        },
    }
    proc = MagicMock(returncode=0, stdout=json.dumps(document))
    with patch("kai.totp.subprocess.run", return_value=proc):
        assert get_failure_count(111) == 2
        assert get_failure_count(222) == 0


def test_verification_updates_only_requesting_principal():
    """Writing one failure preserves every other principal's record."""
    document = {
        "version": 2,
        "principals": {
            "111": {"failures": 1, "lockout_until": 0},
            "222": {"failures": 2, "lockout_until": 0},
        },
    }
    attempts = MagicMock(returncode=0, stdout=json.dumps(document))
    with (
        patch("kai.totp.subprocess.run", side_effect=[attempts, _secret_proc(), _tee_proc()]) as run,
        patch("kai.totp.pyotp.TOTP.verify", return_value=False),
    ):
        assert verify_code("000000", 111) is False

    written = json.loads(run.call_args_list[2].kwargs["input"])
    assert written["principals"]["111"]["failures"] == 2
    assert written["principals"]["222"]["failures"] == 2


def test_failed_attempt_write_is_not_silently_ignored():
    """A tee failure raises, so a valid code cannot bypass durable state."""
    valid_code = pyotp.TOTP(_TEST_SECRET).now()
    with (
        patch(
            "kai.totp.subprocess.run",
            side_effect=[_attempts_proc(), _secret_proc(), _failed_proc()],
        ),
        pytest.raises(TotpStateError, match="write failed"),
    ):
        verify_code(valid_code, 12345)


def test_concurrent_failures_are_serialized():
    """Concurrent read-modify-write transactions cannot lose an increment."""
    stored = {"version": 2, "principals": {}}

    def read_attempts():
        return deepcopy(stored)

    def write_attempts(value):
        nonlocal stored
        # Widen the race window. The module lock must still preserve both writes.
        time.sleep(0.01)
        stored = deepcopy(value)

    with (
        patch("kai.totp._read_attempts", side_effect=read_attempts),
        patch("kai.totp._read_secret", return_value=_TEST_SECRET),
        patch("kai.totp._write_attempts", side_effect=write_attempts),
        patch("kai.totp.pyotp.TOTP.verify", return_value=False),
        ThreadPoolExecutor(max_workers=2) as executor,
    ):
        results = list(executor.map(lambda _: verify_code("000000", 12345), range(2)))

    assert results == [False, False]
    assert stored["principals"]["12345"]["failures"] == 2
