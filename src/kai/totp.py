"""
TOTP verification and rate-limiting for Kai session authentication.

The TOTP secret and attempt state live in root-owned files under /etc/kai/.
The bot accesses them via sudoers-authorized `sudo cat` and `sudo tee` calls.
This means the kai user (and any subprocess it spawns, including inner Claude)
cannot directly read or tamper with either file.

Rate limiting state is persisted to disk so lockouts survive bot restarts.
"""

import json
import subprocess
import time

import pyotp

# Root-owned files, mode 0600. Only accessible via sudo.
# The sudoers rule in /etc/sudoers.d/kai authorizes the bot process
# to run exactly these two commands on exactly these paths, NOPASSWD.
TOTP_SECRET_PATH = "/etc/kai/totp.secret"
TOTP_ATTEMPTS_PATH = "/etc/kai/totp.attempts"


def _read_secret() -> str | None:
    """
    Read the TOTP base32 secret from the root-owned secret file via sudo.

    Returns the secret string, or None if the file doesn't exist, the sudo
    rule isn't configured, or any subprocess error occurs.
    """
    try:
        result = subprocess.run(
            ["sudo", "cat", TOTP_SECRET_PATH],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip() or None
        return None
    except (subprocess.TimeoutExpired, OSError):
        return None


def _read_attempts() -> dict:
    """
    Read the rate-limiting state from the root-owned attempts file via sudo.

    Returns a dict with keys:
      - "failures": int, number of consecutive failed attempts
      - "lockout_until": float, Unix timestamp when lockout expires (0 = no lockout)

    Returns a clean default state if the file is missing, unreadable, or corrupt.
    """
    try:
        result = subprocess.run(
            ["sudo", "cat", TOTP_ATTEMPTS_PATH],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return json.loads(result.stdout.strip())
    except (subprocess.TimeoutExpired, OSError, json.JSONDecodeError):
        pass
    return {"failures": 0, "lockout_until": 0}


def _write_attempts(state: dict) -> None:
    """
    Write the rate-limiting state to the root-owned attempts file via sudo tee.

    Silently swallows errors - if the write fails, the lockout won't persist
    across restarts, which degrades gracefully (the in-memory state still applies
    for the current session).
    """
    try:
        subprocess.run(
            ["sudo", "tee", TOTP_ATTEMPTS_PATH],
            input=json.dumps(state),
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (subprocess.TimeoutExpired, OSError):
        pass


def is_totp_configured() -> bool:
    """
    Return True if the TOTP secret file exists and is readable via sudo.

    Used by the bot to decide whether to prompt for a code at session start.
    If this returns False, TOTP is disabled and the bot runs without it.
    """
    return _read_secret() is not None


def get_lockout_remaining() -> int:
    """
    Return the number of seconds remaining in the current lockout, or 0 if not locked out.

    Used to give the user a meaningful "try again in X minutes" message rather
    than a silent rejection.
    """
    state = _read_attempts()
    remaining = state.get("lockout_until", 0) - time.time()
    return max(0, int(remaining))


def verify_code(code: str, lockout_attempts: int = 3, lockout_minutes: int = 15) -> bool:
    """
    Verify a 6-digit TOTP code against the stored secret.

    Rate limiting is handled internally:
    - After `lockout_attempts` consecutive failures, further attempts are blocked
      for `lockout_minutes` minutes, even with a valid code.
    - A successful verification resets the failure counter.

    Returns True only if the code is valid and the account is not locked out.
    Returns False if: locked out, secret unavailable, or code invalid.
    """
    # Reject obviously malformed codes immediately, before any subprocess calls.
    if not code.isdigit() or len(code) != 6:
        return False

    # Check lockout before doing anything else - don't even read the secret
    # if we're in a lockout period, to avoid unnecessary sudo calls.
    state = _read_attempts()
    if state.get("lockout_until", 0) > time.time():
        return False

    # Read the secret and verify the code.
    secret = _read_secret()
    if not secret:
        return False

    totp = pyotp.TOTP(secret)
    if totp.verify(code):
        # Success - reset the failure counter so a future failure starts fresh.
        _write_attempts({"failures": 0, "lockout_until": 0})
        return True

    # Failed attempt - increment counter and trigger lockout if threshold is reached.
    failures = state.get("failures", 0) + 1
    if failures >= lockout_attempts:
        _write_attempts(
            {
                "failures": failures,
                "lockout_until": time.time() + lockout_minutes * 60,
            }
        )
    else:
        _write_attempts({"failures": failures, "lockout_until": 0})
    return False
