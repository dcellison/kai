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
import threading
import time

import pyotp

from kai.protected_config import ProtectedConfigError, validate_protected_file_metadata

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

# ``python-telegram-bot`` is configured with concurrent updates, so two code
# attempts for the same principal can reach worker threads at the same time.
# Hold this lock across the complete read/verify/write transaction.  An RLock
# also lets is_totp_configured() validate the same files through the private
# read helpers without creating a second locking protocol.
_totp_state_lock = threading.RLock()

_ATTEMPTS_VERSION = 2
_LEGACY_PRINCIPAL = "*"


class TotpStateError(RuntimeError):
    """TOTP is enabled but its protected state cannot be trusted."""


def _empty_attempts() -> dict:
    """Return a fresh versioned, per-principal attempt-state document."""
    return {"version": _ATTEMPTS_VERSION, "principals": {}}


def _validate_attempt_record(value: object) -> dict:
    """Validate and normalize one principal's persisted lockout record."""
    if not isinstance(value, dict):
        raise TotpStateError("TOTP attempt state contains a non-object principal record")
    failures = value.get("failures")
    lockout_until = value.get("lockout_until")
    if isinstance(failures, bool) or not isinstance(failures, int) or failures < 0:
        raise TotpStateError("TOTP attempt state contains an invalid failure count")
    if isinstance(lockout_until, bool) or not isinstance(lockout_until, (int, float)) or lockout_until < 0:
        raise TotpStateError("TOTP attempt state contains an invalid lockout timestamp")
    return {"failures": failures, "lockout_until": float(lockout_until)}


def _normalize_attempts(data: object) -> dict:
    """Normalize current and legacy attempt-state documents.

    The legacy document stored one global ``failures``/``lockout_until`` pair.
    Preserve it under the wildcard principal during migration so deploying the
    new schema cannot silently clear an active lockout.  A principal gets its
    own record on its next verification attempt.
    """
    if not isinstance(data, dict):
        raise TotpStateError("TOTP attempt state must be a JSON object")

    if "version" not in data and ("failures" in data or "lockout_until" in data):
        return {
            "version": _ATTEMPTS_VERSION,
            "principals": {_LEGACY_PRINCIPAL: _validate_attempt_record(data)},
        }

    if data.get("version") != _ATTEMPTS_VERSION:
        raise TotpStateError("TOTP attempt state has an unsupported schema version")
    principals = data.get("principals")
    if not isinstance(principals, dict):
        raise TotpStateError("TOTP attempt state is missing its principals mapping")

    normalized: dict[str, dict] = {}
    for principal, record in principals.items():
        if not isinstance(principal, str) or not principal:
            raise TotpStateError("TOTP attempt state contains an invalid principal key")
        normalized[principal] = _validate_attempt_record(record)
    return {"version": _ATTEMPTS_VERSION, "principals": normalized}


def _principal_key(principal_id: int) -> str:
    """Return the stable persisted key for a Telegram principal."""
    if isinstance(principal_id, bool) or not isinstance(principal_id, int) or principal_id <= 0:
        raise ValueError("principal_id must be a positive Telegram user ID")
    return str(principal_id)


def _principal_attempts(state: dict, principal: str) -> dict:
    """Read a principal record, inheriting a legacy global record once."""
    principals = state["principals"]
    record = principals.get(principal, principals.get(_LEGACY_PRINCIPAL))
    if record is None:
        return {"failures": 0, "lockout_until": 0.0}
    return dict(record)


def _totp_files_present() -> tuple[bool, bool]:
    """Return strict presence flags for the secret and attempt-state files.

    Stat does not read either root-owned file.  A permission or I/O error is
    distinct from a clean FileNotFoundError and must not disable a configured
    authentication boundary.
    """

    def _present(path: str) -> bool:
        try:
            return validate_protected_file_metadata(path, missing_ok=True)
        except FileNotFoundError:
            return False
        except ProtectedConfigError as exc:
            raise TotpStateError(str(exc)) from exc
        except OSError as exc:
            raise TotpStateError(f"Could not determine TOTP state for {path}: {exc}") from exc

    return _present(TOTP_SECRET_PATH), _present(TOTP_ATTEMPTS_PATH)


def _read_secret() -> str:
    """
    Read the TOTP base32 secret from the root-owned secret file via sudo.

    Every failure raises TotpStateError.  Callers decide whether TOTP is
    disabled by strictly checking that *both* protected files are absent
    before reaching this helper.
    """
    try:
        validate_protected_file_metadata(TOTP_SECRET_PATH)
    except (OSError, ProtectedConfigError) as exc:
        raise TotpStateError(str(exc)) from exc
    try:
        result = subprocess.run(
            ["sudo", "-n", "cat", TOTP_SECRET_PATH],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise TotpStateError("Could not read the protected TOTP secret") from exc
    if result.returncode != 0:
        raise TotpStateError(f"Protected TOTP secret read failed with exit status {result.returncode}")
    secret = result.stdout.strip()
    if not secret:
        raise TotpStateError("Protected TOTP secret is empty")
    try:
        # Construction alone does not decode base32; generating a value does.
        pyotp.TOTP(secret).at(0)
    except Exception as exc:
        raise TotpStateError("Protected TOTP secret is invalid") from exc
    return secret


def _read_attempts() -> dict:
    """
    Read the rate-limiting state from the root-owned attempts file via sudo.

    Returns a versioned document whose ``principals`` mapping contains each
    Telegram user's consecutive failure count and lockout expiry.

    Raises TotpStateError if the file is missing, unreadable, corrupt, or has
    an invalid schema.  Resetting failures on a read error would make lockout
    fail open.
    """
    try:
        validate_protected_file_metadata(TOTP_ATTEMPTS_PATH)
    except (OSError, ProtectedConfigError) as exc:
        raise TotpStateError(str(exc)) from exc
    try:
        result = subprocess.run(
            ["sudo", "-n", "cat", TOTP_ATTEMPTS_PATH],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise TotpStateError("Could not read protected TOTP attempt state") from exc
    if result.returncode != 0:
        raise TotpStateError(f"Protected TOTP attempt-state read failed with exit status {result.returncode}")
    if not result.stdout.strip():
        raise TotpStateError("Protected TOTP attempt state is empty")
    try:
        return _normalize_attempts(json.loads(result.stdout))
    except json.JSONDecodeError as exc:
        raise TotpStateError("Protected TOTP attempt state is not valid JSON") from exc


def _write_attempts(state: dict) -> None:
    """
    Write the rate-limiting state to the root-owned attempts file via sudo tee.

    Raises TotpStateError unless sudo tee confirms success.  Authentication is
    not granted when resetting the failure counter cannot be persisted.
    """
    try:
        validate_protected_file_metadata(TOTP_ATTEMPTS_PATH)
    except (OSError, ProtectedConfigError) as exc:
        raise TotpStateError(str(exc)) from exc
    try:
        result = subprocess.run(
            ["sudo", "-n", "tee", TOTP_ATTEMPTS_PATH],
            input=json.dumps(state),
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise TotpStateError("Could not write protected TOTP attempt state") from exc
    if result.returncode != 0:
        raise TotpStateError(f"Protected TOTP attempt-state write failed with exit status {result.returncode}")


def is_totp_configured() -> bool:
    """
    Return True if the TOTP secret file exists and is readable via sudo.

    False is returned only when both the secret and attempt-state files are
    cleanly absent.  If either file exists, both must be readable and valid;
    any partial or inaccessible state raises TotpStateError so the bot denies
    access instead of silently disabling TOTP.

    The True result is cached for the lifetime of the process to avoid spawning
    a subprocess on every incoming message. False is never cached so that
    enabling TOTP while the bot is running takes effect on the next message.
    """
    global _totp_is_configured
    with _totp_state_lock:
        if _totp_is_configured:
            return True
        secret_present, attempts_present = _totp_files_present()
        if not secret_present and not attempts_present:
            return False
        if not secret_present or not attempts_present:
            raise TotpStateError("TOTP is enabled but its protected files are incomplete")
        _read_secret()
        _read_attempts()
        _totp_is_configured = True
        return True


def get_lockout_remaining(principal_id: int) -> int:
    """
    Return the number of seconds remaining in the current lockout, or 0 if not locked out.

    Used to give the user a meaningful "try again in X minutes" message rather
    than a silent rejection.
    """
    principal = _principal_key(principal_id)
    with _totp_state_lock:
        state = _principal_attempts(_read_attempts(), principal)
        remaining = state["lockout_until"] - time.time()
        return max(0, int(remaining))


def get_failure_count(principal_id: int) -> int:
    """
    Return the number of consecutive failed verification attempts since the last success.

    Public wrapper around the private _read_attempts() so bot.py doesn't need to
    import or call a private function across module boundaries.
    """
    principal = _principal_key(principal_id)
    with _totp_state_lock:
        return _principal_attempts(_read_attempts(), principal)["failures"]


def verify_code(
    code: str,
    principal_id: int,
    lockout_attempts: int = 3,
    lockout_minutes: int = 15,
) -> bool:
    """
    Verify a 6-digit TOTP code against the stored secret.

    Rate limiting is handled internally:
    - After `lockout_attempts` consecutive failures, further attempts are blocked
      for `lockout_minutes` minutes, even with a valid code.
    - A successful verification resets the failure counter.

    Returns True only if the code is valid, the principal is not locked out,
    and the reset state was durably written. Returns False for a lockout or an
    invalid code. Protected-state failures raise TotpStateError.
    """
    # Reject obviously malformed codes immediately, before any subprocess calls.
    if not code.isdigit() or len(code) != 6:
        return False

    principal = _principal_key(principal_id)
    with _totp_state_lock:
        # The lock covers the entire read/verify/write transaction. Without
        # it, concurrent failures can both read N and persist N+1.
        document = _read_attempts()
        state = _principal_attempts(document, principal)
        if state["lockout_until"] > time.time():
            return False

        secret = _read_secret()
        if pyotp.TOTP(secret).verify(code):
            document["principals"][principal] = {"failures": 0, "lockout_until": 0.0}
            _write_attempts(document)
            return True

        failures = state["failures"] + 1
        lockout_until = time.time() + lockout_minutes * 60 if failures >= lockout_attempts else 0.0
        document["principals"][principal] = {
            "failures": failures,
            "lockout_until": lockout_until,
        }
        _write_attempts(document)
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
        f.write(json.dumps(_empty_attempts()))
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
    # secret file and will deny authentication because TOTP state is unavailable.
    print("\nAdd the following lines to /etc/sudoers.d/kai (via visudo -f /etc/sudoers.d/kai):")
    print("(Complete this step before restarting the bot.)\n")
    print(f"  kai ALL=(root) NOPASSWD: {_CAT} {TOTP_SECRET_PATH}")
    print(f"  kai ALL=(root) NOPASSWD: {_CAT} {TOTP_ATTEMPTS_PATH}")
    print(f"  kai ALL=(root) NOPASSWD: {_TEE} {TOTP_ATTEMPTS_PATH}")

    # Confirm setup with a live code from the authenticator.
    # This runs as root (sudo python -m kai totp setup) so it doesn't depend
    # on the sudoers rules above - root can always sudo.
    code = input("\nEnter a 6-digit code to confirm setup: ").strip()
    if pyotp.TOTP(secret).verify(code):
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
    try:
        if is_totp_configured():
            print("TOTP is configured.")
        else:
            print("TOTP is not configured.")
    except TotpStateError as exc:
        print(f"TOTP is configured but unavailable: {exc}")
        sys.exit(1)


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
