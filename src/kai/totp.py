"""
TOTP verification and rate-limiting for Kai session authentication.

The TOTP secret and attempt state live in root-owned files under /etc/kai/.
The bot accesses them via sudoers-authorized `sudo cat` and `sudo tee` calls.
This means the kai user (and any subprocess it spawns, including inner Claude)
cannot directly read or tamper with either file.

Rate limiting state is persisted to disk so lockouts survive bot restarts.

CLI usage (run as root or with sudo):
    python -m kai totp setup    # generate secret, show QR, confirm
    python -m kai totp status   # check whether TOTP is configured
    python -m kai totp reset    # remove secret and attempts files
"""

import json
import os
import shutil
import subprocess
import sys
import time

import pyotp

# Root-owned files, mode 0600. Only accessible via sudo.
# The sudoers rule in /etc/sudoers.d/kai authorizes the bot process
# to run exactly these two commands on exactly these paths, NOPASSWD.
TOTP_SECRET_PATH = "/etc/kai/totp.secret"
TOTP_ATTEMPTS_PATH = "/etc/kai/totp.attempts"

# Module-level cache for is_totp_configured().
# Once TOTP is confirmed configured, it stays configured for the lifetime of the
# process - the secret file can only be removed by `totp reset`, which requires
# root and would be followed by a bot restart anyway. We only cache True so that
# the False -> True transition (setting TOTP up while the bot is running) is
# picked up on the next message without a restart. False results are never cached.
_totp_is_configured: bool = False


def _read_secret() -> str | None:
    """
    Read the TOTP base32 secret from the root-owned secret file via sudo.

    Returns the secret string, or None if the file doesn't exist, the sudo
    rule isn't configured, or any subprocess error occurs.
    """
    try:
        result = subprocess.run(
            ["sudo", "-n", "cat", TOTP_SECRET_PATH],
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
            ["sudo", "-n", "cat", TOTP_ATTEMPTS_PATH],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout.strip())
            # Validate expected types to avoid TypeError on arithmetic later.
            # Both fields are used in numeric operations: failures gets + 1,
            # lockout_until gets compared to time.time(). A corrupted or
            # hand-edited file with string values would crash without this.
            if (
                isinstance(data, dict)
                and isinstance(data.get("failures"), (int, float))
                and isinstance(data.get("lockout_until"), (int, float))
            ):
                return data
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
            ["sudo", "-n", "tee", TOTP_ATTEMPTS_PATH],
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

    The True result is cached for the lifetime of the process to avoid spawning
    a subprocess on every incoming message. False is never cached so that
    enabling TOTP while the bot is running takes effect on the next message.
    """
    global _totp_is_configured
    if _totp_is_configured:
        return True
    result = _read_secret() is not None
    if result:
        _totp_is_configured = True
    return result


def get_lockout_remaining() -> int:
    """
    Return the number of seconds remaining in the current lockout, or 0 if not locked out.

    Used to give the user a meaningful "try again in X minutes" message rather
    than a silent rejection.
    """
    state = _read_attempts()
    remaining = state.get("lockout_until", 0) - time.time()
    return max(0, int(remaining))


def get_failure_count() -> int:
    """
    Return the number of consecutive failed verification attempts since the last success.

    Public wrapper around the private _read_attempts() so bot.py doesn't need to
    import or call a private function across module boundaries.
    """
    return _read_attempts().get("failures", 0)


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


# ── CLI entry point (python -m kai totp <subcommand>) ────────────────

# Resolve binary paths for the sudoers rule. shutil.which() finds the
# binary on the current PATH; fallbacks match platform conventions.
_CAT = shutil.which("cat") or ("/bin/cat" if sys.platform == "darwin" else "/usr/bin/cat")
_TEE = shutil.which("tee") or "/usr/bin/tee"


def _cmd_setup() -> None:
    """
    Generate a TOTP secret, write it to /etc/kai/totp.secret (root-owned, 0600),
    display a QR code and the raw secret, then confirm with a test code.

    Must be run as root (or via sudo python -m kai totp setup) since it creates
    files owned by root. Exits with a non-zero status on any failure.
    """
    if os.geteuid() != 0:
        print("'totp setup' must be run as root (try: sudo python -m kai totp setup)")
        sys.exit(1)

    # Create /etc/kai/ if it doesn't exist, owned by root.
    etc_kai = "/etc/kai"
    os.makedirs(etc_kai, mode=0o755, exist_ok=True)

    # Generate a cryptographically random base32 secret.
    secret = pyotp.random_base32()

    # Write secret file: root:root 0600. Uses os.open() with explicit mode
    # to avoid a TOCTOU window where the file briefly exists with default
    # umask permissions before chmod() tightens them.
    fd = os.open(TOTP_SECRET_PATH, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(secret)
    os.chown(TOTP_SECRET_PATH, 0, 0)

    # Create a clean attempts file: root:root 0600.
    fd = os.open(TOTP_ATTEMPTS_PATH, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps({"failures": 0, "lockout_until": 0}))
    os.chown(TOTP_ATTEMPTS_PATH, 0, 0)

    # Generate and print the QR code to the terminal.
    import qrcode  # type: ignore[import-untyped]

    uri = pyotp.TOTP(secret).provisioning_uri(name="Kai", issuer_name="Kai")
    qr = qrcode.QRCode()
    qr.add_data(uri)
    qr.make(fit=True)
    print("\nScan this QR code with your authenticator app:\n")
    qr.print_ascii(invert=True)

    # Also print the raw secret for manual entry.
    print(f"\nManual entry secret: {secret}")
    print("Account: Kai / Issuer: Kai")

    # Print the sudoers rule BEFORE asking for the confirmation code.
    # The bot process (running as the 'kai' user) needs these rules to verify
    # codes at runtime. Showing them first ensures the user adds them before
    # treating setup as complete - without sudoers, the bot can't read the
    # secret file and will silently behave as if TOTP is not configured.
    print("\nAdd the following lines to /etc/sudoers.d/kai (via visudo -f /etc/sudoers.d/kai):")
    print("(Complete this step before restarting the bot.)\n")
    print(f"  kai ALL=(root) NOPASSWD: {_CAT} {TOTP_SECRET_PATH}")
    print(f"  kai ALL=(root) NOPASSWD: {_CAT} {TOTP_ATTEMPTS_PATH}")
    print(f"  kai ALL=(root) NOPASSWD: {_TEE} {TOTP_ATTEMPTS_PATH}")

    # Confirm setup with a live code from the authenticator.
    # This runs as root (sudo python -m kai totp setup) so it doesn't depend
    # on the sudoers rules above - root can always sudo.
    code = input("\nEnter a 6-digit code to confirm setup: ").strip()
    if verify_code(code):
        print("TOTP setup complete.")
    else:
        print(
            "Code incorrect. Setup files written but verification failed.\n"
            "Run 'sudo python -m kai totp reset' and try again."
        )
        sys.exit(1)


def _cmd_status() -> None:
    """
    Report whether the TOTP secret file is present and readable via sudo.

    Does not require root - reads via the sudoers-authorized sudo call.
    """
    if is_totp_configured():
        print("TOTP is configured.")
    else:
        print("TOTP is not configured.")


def _cmd_reset() -> None:
    """
    Delete /etc/kai/totp.secret and /etc/kai/totp.attempts.

    Must be run as root. After reset, the bot will start without TOTP authentication.
    """
    if os.geteuid() != 0:
        print("'totp reset' must be run as root (try: sudo python -m kai totp reset)")
        sys.exit(1)

    removed = []
    for path in (TOTP_SECRET_PATH, TOTP_ATTEMPTS_PATH):
        try:
            os.remove(path)
            removed.append(path)
        except FileNotFoundError:
            pass  # already gone, that's fine

    if removed:
        print(f"Removed: {', '.join(removed)}")
    else:
        print("Nothing to remove (TOTP was not configured).")


def cli(args: list[str]) -> None:
    """
    Dispatch TOTP CLI subcommands.

    Usage:
        python -m kai totp setup    -- generate secret, show QR, confirm
        python -m kai totp status   -- check whether TOTP is configured
        python -m kai totp reset    -- remove secret and attempts files
    """
    subcommands = {"setup": _cmd_setup, "status": _cmd_status, "reset": _cmd_reset}

    if not args or args[0] not in subcommands:
        print("Usage: python -m kai totp {setup|status|reset}")
        sys.exit(1)

    subcommands[args[0]]()
