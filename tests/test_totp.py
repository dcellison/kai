"""
Tests for src/kai/totp.py.

All subprocess.run calls are mocked so tests don't require root access or
actual /etc/kai/* files. The mock pattern replaces subprocess.run globally
within the totp module for each test.
"""

import json
import time
from unittest.mock import MagicMock, patch

import pyotp
import pytest

import kai.totp
from kai.totp import (
    get_failure_count,
    get_lockout_remaining,
    is_totp_configured,
    verify_code,
)


@pytest.fixture(autouse=True)
def _reset_totp_cache():
    """Reset the is_totp_configured module-level cache before and after each test.

    Without this, a test that calls is_totp_configured() and gets True would
    pollute the cache for subsequent tests in the same process run.
    """
    kai.totp._totp_is_configured = False
    yield
    kai.totp._totp_is_configured = False


# A stable base32 secret used across tests.
_TEST_SECRET = "JBSWY3DPEHPK3PXP"


def _secret_proc(secret: str = _TEST_SECRET) -> MagicMock:
    """Return a mock subprocess result that looks like a successful sudo cat of the secret file."""
    m = MagicMock()
    m.returncode = 0
    m.stdout = secret + "\n"
    return m


def _attempts_proc(failures: int = 0, lockout_until: float = 0) -> MagicMock:
    """Return a mock subprocess result that looks like a successful sudo cat of the attempts file."""
    m = MagicMock()
    m.returncode = 0
    m.stdout = json.dumps({"failures": failures, "lockout_until": lockout_until})
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


# ---------------------------------------------------------------------------
# verify_code: valid and invalid codes
# ---------------------------------------------------------------------------


def test_verify_code_rejects_malformed_input():
    """verify_code returns False immediately for non-6-digit input, with no subprocess calls."""
    with patch("kai.totp.subprocess.run") as mock_run:
        assert verify_code("12345") is False  # too short
        assert verify_code("1234567") is False  # too long
        assert verify_code("12345a") is False  # non-digit
        assert verify_code("") is False  # empty
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
        result = verify_code(valid_code)

    assert result is True


def test_verify_code_invalid():
    """verify_code returns False for a wrong code and increments the failure counter."""
    with patch("kai.totp.subprocess.run") as mock_run:
        mock_run.side_effect = [
            _attempts_proc(),  # _read_attempts
            _secret_proc(),  # _read_secret
            _tee_proc(),  # _write_attempts (increment failures)
        ]
        result = verify_code("000000")

    assert result is False


# ---------------------------------------------------------------------------
# Rate limiting: failure counter
# ---------------------------------------------------------------------------


def test_failure_counter_increments():
    """Each failed attempt increments the stored failure count."""
    with patch("kai.totp.subprocess.run") as mock_run:
        mock_run.side_effect = [
            _attempts_proc(failures=0),  # _read_attempts
            _secret_proc(),  # _read_secret
            _tee_proc(),  # _write_attempts
        ]
        verify_code("000000")

        # Inspect what was written - it's the third call, with input= containing the state.
        write_call = mock_run.call_args_list[2]
        written = json.loads(write_call.kwargs["input"])

    assert written["failures"] == 1
    assert written["lockout_until"] == 0


def test_lockout_triggers_after_n_failures():
    """Reaching lockout_attempts consecutive failures sets a non-zero lockout_until timestamp."""
    # Simulate already at 2 failures (one below default lockout_attempts=3).
    with patch("kai.totp.subprocess.run") as mock_run:
        mock_run.side_effect = [
            _attempts_proc(failures=2),  # _read_attempts - already at 2
            _secret_proc(),  # _read_secret
            _tee_proc(),  # _write_attempts - should trigger lockout
        ]
        verify_code("000000")

        write_call = mock_run.call_args_list[2]
        written = json.loads(write_call.kwargs["input"])

    assert written["failures"] == 3
    # lockout_until should be roughly now + 15 minutes
    assert written["lockout_until"] > time.time()
    assert written["lockout_until"] < time.time() + 15 * 60 + 5  # +5s tolerance


def test_verify_returns_false_during_lockout_even_with_valid_code():
    """A valid code is rejected when a lockout is active (no secret read attempted)."""
    valid_code = pyotp.TOTP(_TEST_SECRET).now()
    future_lockout = time.time() + 900  # 15 minutes from now

    with patch("kai.totp.subprocess.run") as mock_run:
        mock_run.side_effect = [
            _attempts_proc(failures=3, lockout_until=future_lockout),  # _read_attempts
        ]
        result = verify_code(valid_code)

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
        result = verify_code(valid_code)

        write_call = mock_run.call_args_list[2]
        written = json.loads(write_call.kwargs["input"])

    assert result is True
    assert written["failures"] == 0
    assert written["lockout_until"] == 0


# ---------------------------------------------------------------------------
# is_totp_configured
# ---------------------------------------------------------------------------


def test_is_totp_configured_true_when_readable():
    """is_totp_configured returns True when the secret file is readable."""
    with patch("kai.totp.subprocess.run", return_value=_secret_proc()):
        assert is_totp_configured() is True


def test_is_totp_configured_false_when_file_missing():
    """is_totp_configured returns False when sudo cat exits non-zero (file missing, no sudoers rule, etc.)."""
    with patch("kai.totp.subprocess.run", return_value=_failed_proc()):
        assert is_totp_configured() is False


# ---------------------------------------------------------------------------
# get_lockout_remaining
# ---------------------------------------------------------------------------


def test_get_lockout_remaining_zero_when_not_locked():
    """get_lockout_remaining returns 0 when lockout_until is 0 (no active lockout)."""
    with patch("kai.totp.subprocess.run", return_value=_attempts_proc(lockout_until=0)):
        assert get_lockout_remaining() == 0


def test_get_lockout_remaining_positive_when_locked():
    """get_lockout_remaining returns a positive number of seconds when locked out."""
    future = time.time() + 300  # 5 minutes from now
    with patch("kai.totp.subprocess.run", return_value=_attempts_proc(lockout_until=future)):
        remaining = get_lockout_remaining()

    # Should be close to 300, allow a few seconds of test execution slack.
    assert 295 <= remaining <= 300


# ---------------------------------------------------------------------------
# get_failure_count
# ---------------------------------------------------------------------------


def test_get_failure_count_returns_failures_from_disk():
    """get_failure_count returns the current consecutive failure count from the attempts file."""
    with patch("kai.totp.subprocess.run", return_value=_attempts_proc(failures=2)):
        count = get_failure_count()

    assert count == 2


def test_get_failure_count_returns_zero_on_clean_state():
    """get_failure_count returns 0 when there are no recorded failures."""
    with patch("kai.totp.subprocess.run", return_value=_attempts_proc(failures=0)):
        count = get_failure_count()

    assert count == 0
