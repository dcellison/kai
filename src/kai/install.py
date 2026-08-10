"""
Protected installation tooling for deploying Kai to /opt/kai/.

Provides functionality to:
1. Interactively collect configuration values (config subcommand)
2. Apply configuration to create a protected installation (apply subcommand)
3. Report current installation state (status subcommand)

The two-step workflow separates privilege levels:
    python -m kai install config   -- interactive Q&A, writes install.conf (no sudo)
    sudo python -m kai install apply  -- reads install.conf, creates /opt layout (root)
    python -m kai install status   -- shows current state (no sudo)

A protected installation puts read-only source in /opt/kai/ (root-owned) and
writable runtime data in /var/lib/kai/ (service-user-owned). Secrets live in
/etc/kai/ (root-owned, mode 0600) and are read at startup via sudo cat with
NOPASSWD rules. This separation means the inner Claude process cannot read
secrets or modify the bot's source code.

The install.conf file bridges the two steps: config writes it, apply reads it.
It's a JSON file with a version field for forward compatibility.
"""

import grp
import hashlib
import json
import os
import pwd
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from collections.abc import Iterable
from pathlib import Path

import yaml

from kai.backend_registry import render_backend_registry
from kai.config import (
    _VALID_ROLES,
    BACKEND_PROVIDERS,
    BACKENDS_NEEDING_PROVIDER_PROMPT,
    CODEX_EFFORT_LEVELS,
    CODEX_MODELS,
    EFFORT_LEVELS,
    MODEL_REGISTRY,
    ONESHOT_REASONER_BACKENDS,
    PROJECT_ROOT,
    PROVIDER_KEY_VARS,
    PROVIDER_MODELS,
    VALID_BACKENDS,
    ModelRole,
    _read_protected_file,
    _resolve_renamed_key,
    canonicalize_model_for_backend,
    get_default_model_for_backend,
    models_for_backend,
    validate_model_for_backend,
)
from kai.user_isolation import validate_protected_user_isolation

# Config file written by `config`, read by `apply`.
# Anchored to PROJECT_ROOT so it resolves correctly regardless of CWD.
INSTALL_CONF = PROJECT_ROOT / "install.conf"

# Canonical post-install location of users.yaml (the secrets apply step
# deploys any staged copy here). The apply steps that read users.yaml
# resolve this attribute at call time as their users_yaml_path default,
# and the wizard reads it directly for its protected-mode path, so the
# test suite can patch one attribute and stay isolated from the host's
# real runtime config.
USERS_YAML = Path("/etc/kai/users.yaml")
BACKENDS_YAML = Path("/etc/kai/backends.yaml")

# Default installation paths
_DEFAULT_INSTALL_DIR = "/opt/kai"
_DEFAULT_DATA_DIR = "/var/lib/kai"
_DEFAULT_SERVICE_USER = "kai"

# Current install.conf schema version
_CONF_VERSION = 1

# Plist label for the launchd service
_LAUNCHD_LABEL = "com.syrinx.kai"


# Retry budget for `_start_service`. The bootstrap-then-verify cycle
# runs at most _SERVICE_START_MAX_ATTEMPTS times: a brief settle
# delay after each bootstrap call gives the service manager time to
# update its bookkeeping before the verify query, and a longer retry
# delay between attempts absorbs the transient launchd-domain-not-yet-
# released window that follows a bootout. Total worst-case wait is
# (settle + retry) * (attempts - 1) + settle = 7 seconds at the
# current values, which is generous enough to cover the failure
# pattern observed in the originating incident while staying short
# enough that an install run does not feel hung.
_SERVICE_START_MAX_ATTEMPTS = 3
_SERVICE_START_SETTLE_SECONDS = 1
_SERVICE_START_RETRY_SECONDS = 2

_PRIVATE_USER_ROOT_MODE = 0o711
_PRIVATE_USER_DIR_MODE = 0o700
_PRIVATE_USER_FILE_MODE = 0o600


class ServiceStartError(Exception):
    """Raised by `_start_service` when the post-condition verify
    confirms the service is not actually registered/running after the
    retry budget is exhausted.

    The platform service managers do not always report start failures
    reliably via exit code: launchctl bootstrap returns 5 ("Input/
    output error") for several distinct conditions including
    "actually succeeded" and "actually failed". The authoritative
    check is the verify query (`launchctl print system/<label>` on
    macOS, `systemctl is-active <unit>` on Linux). This exception
    distinguishes "tried, verified, definitely failed" from any
    other failure mode so the caller in `_cmd_apply` can decide
    whether to swallow (apply already failed; don't mask the
    original) or propagate (apply succeeded; the install has not
    produced a working system).
    """


# Files and directories to copy from source to the install location.
# Excludes __pycache__, .pyc, and other build artifacts.
_SOURCE_EXCLUDES = {"__pycache__", "*.pyc", "*.egg-info", ".git", ".venv", ".env"}
_INSTALL_CONSTRAINTS_REL = Path("requirements") / "constraints.txt"

# ── Input helpers ────────────────────────────────────────────────────


def _prompt(label: str, default: str = "", required: bool = False) -> str:
    """
    Prompt the user for input with an optional default value.

    Shows the default in brackets. If the user presses Enter without typing
    anything, the default is returned. Required fields reject empty input.

    Args:
        label: The prompt text shown to the user.
        default: Default value shown in brackets and returned on empty input.
        required: If True, empty input is rejected with a retry prompt.

    Returns:
        The user's input, or the default if input was empty.
    """
    suffix = f" [{default}]" if default else ""
    while True:
        value = input(f"{label}{suffix}: ").strip()
        if not value:
            if default:
                return default
            if required:
                print("  This field is required.")
                continue
            return ""
        return value


def _prompt_choice(label: str, choices: list[str], default: str = "") -> str:
    """
    Prompt the user to pick from a list of valid choices.

    Rejects input not in the choices list and re-prompts. If the caller
    supplies a `default` that is not in `choices`, the function behaves
    as if no default were supplied: the suffix is not shown, and empty
    input re-prompts rather than returning the invalid value. This guards
    against call sites that read `default` from operator-facing state
    (existing_env, install.conf) whose values can drift out of `choices`
    when a sibling setting changes.

    Membership is case-sensitive against `choices`. A `default` whose
    case or whitespace does not match a canonical entry is treated as
    out-of-list and re-prompted. This is consistent with the apply-time
    normalization in load_config and stops a hand-edited install.conf
    from re-selecting a near-miss value on Enter.

    Args:
        label: The prompt text shown to the user.
        choices: List of valid string values.
        default: Default value if the user presses Enter. Ignored if not
            in `choices`.

    Returns:
        The chosen value, guaranteed to be in `choices`.
    """
    choices_str = "/".join(choices)
    # An out-of-list default is treated as no default. The empty-input
    # path previously returned the value unchecked, violating the
    # function's "Returns a value in choices" contract. Recomputing here
    # keeps the suffix display and the empty-input branch consistent so
    # the prompt never advertises an option the function would not accept
    # via typed input.
    effective_default = default if default in choices else ""
    suffix = f" [{effective_default}]" if effective_default else ""
    while True:
        value = input(f"{label} ({choices_str}){suffix}: ").strip().lower()
        if not value and effective_default:
            return effective_default
        if value in choices:
            return value
        print(f"  Please choose one of: {choices_str}")


def _prompt_optional_choice(
    label: str,
    choices: list[str],
    default: str = "",
    empty_hint: str = "empty = default",
) -> str:
    """
    Prompt the user to pick from a list of valid choices, with empty
    input promoted to a first-class valid answer.

    Use for set-or-absent settings where absence has its own meaning:
    omitting the variable from install.conf is the signal to let the
    downstream consumer use its own default. The Codex effort prompt
    is the seed call site; absence means CodexBackend skips the
    `-c model_reasoning_effort` override and codex falls back to the
    per-OS-user `~/.codex/config.toml` or the model default.

    The prefill is normalized via `.strip().lower()` before the `in
    choices` check, mirroring the runtime config parser's tolerance
    for copy-paste-shaped values. A prefill that lands in `choices`
    after normalization round-trips on Enter and is displayed in its
    canonical lowercased form. A prefill that does not match after
    normalization is treated as no prefill: the inline hint advertises
    the empty-default path, and Enter returns "".

    Display:
        - With a usable prefill:    `label (a/b/c) [b]: `
        - Without a usable prefill: `label (a/b/c, empty = default): `

    Args:
        label: The prompt text shown to the user.
        choices: List of valid string values. Membership is checked
            case-sensitively after the caller's input is lowercased,
            so `choices` should contain canonical lowercase entries.
        default: Prefill candidate. Normalized via `.strip().lower()`
            before the membership check. An out-of-list value after
            normalization is treated as no default.
        empty_hint: Phrase shown inline next to the choices in the
            no-prefill display path, and repeated in the re-prompt on
            invalid input. Default "empty = default" is generic;
            callers should pass a site-specific phrase like
            "empty = codex default" so the operator sees the semantics
            in their own terms.

    Returns:
        The chosen value (guaranteed to be in `choices`), or "" when
        the operator explicitly leaves the answer empty AND there is
        no usable prefill.
    """
    choices_str = "/".join(choices)
    # Match the runtime config parser's tolerance for copy-paste-shaped
    # values (pinned by test_uppercase_and_whitespace_normalized in the
    # codex-effort config tests) so a re-run of `make config` with a
    # value like "  HIGH " sitting in /etc/kai/env keeps the existing
    # setting instead of silently treating it as out-of-list and
    # dropping it.
    effective_default = default.strip().lower()
    if effective_default in choices:
        prompt_text = f"{label} ({choices_str}) [{effective_default}]: "
    else:
        # No usable prefill: advertise the empty-default path inline
        # so a first-time operator does not have to guess that empty
        # is a valid answer. The empty hint never appears alongside
        # `[prefill]` so the prompt does not give two different
        # "what happens on Enter" answers at once.
        prompt_text = f"{label} ({choices_str}, {empty_hint}): "
        effective_default = ""
    while True:
        value = input(prompt_text).strip().lower()
        if not value:
            return effective_default
        if value in choices:
            return value
        # The recovery message has to match what Enter actually does
        # at this point in the loop. With a usable prefill, Enter
        # round-trips that prefill (NOT the downstream default), so
        # advertising the empty-default hint here would be a lie and
        # would mislead the operator into thinking they can clear the
        # override by pressing Enter on the re-prompt.
        if effective_default:
            print(f"  Must be one of {choices_str}, or empty to keep {effective_default}.")
        else:
            print(f"  Must be one of {choices_str}, or {empty_hint}.")


def _prompt_bool(label: str, default: bool = False) -> bool:
    """
    Prompt the user for a yes/no answer.

    Args:
        label: The prompt text shown to the user.
        default: Default boolean value.

    Returns:
        True for yes/true, False for no/false.
    """
    default_str = "true" if default else "false"
    value = _prompt_choice(label, ["true", "false"], default_str)
    return value == "true"


def _prompt_default_model(agent_backend: str, eff_provider: str, default_val: str) -> str:
    """
    Deprecated compatibility helper for the retired DEFAULT_MODEL prompt.

    The install wizard no longer calls this. The conversational default
    is MODEL_REGISTRY's ModelRole.AGENT row for the active
    backend/provider. Keep this helper temporarily so older tests and
    out-of-tree tooling that monkeypatch it fail softly rather than
    raising AttributeError.

    Three shapes of model surface need different prompts:
    - Codex backend ships its own curated CLI model list (CODEX_MODELS)
      which is fully separate from goose's openai-API surface. Wizard
      offers the codex-only set; never falls through to the OpenAI
      provider list.
    - Curated providers (those with an entry in PROVIDER_MODELS, currently
      anthropic, openai, google) ship a fixed model list; the operator
      picks from the list via _prompt_choice.
    - Open-ended providers (those in OPEN_ENDED_PROVIDERS, currently
      openrouter and ollama; also any provider in BACKEND_PROVIDERS
      without a PROVIDER_MODELS entry) accept arbitrary model
      identifiers; the operator types one via _prompt(required=True).

    The required=True on the open-ended branch forced the operator to
    commit to a concrete model string. Current installs should use the
    agent row in MODEL_REGISTRY instead.

    Args:
        agent_backend: The active backend ("claude", "codex", "goose").
            Codex routes to CODEX_MODELS; everything else delegates to
            the provider-based selection below.
        eff_provider: The effective provider for non-codex backends
            (anthropic for claude; the configured provider for
            goose; ignored when agent_backend == "codex").
        default_val: Prefill for the prompt. Callers should pass a value
            that is valid for the backend's surface (CODEX_DEFAULT_MODEL
            for codex, PROVIDER_DEFAULTS[eff_provider] for others) when
            re-prompting after rejecting an invalid existing value.

    Returns:
        The chosen model identifier. Always non-empty.
    """
    surface = models_for_backend(agent_backend, eff_provider)
    if surface:
        return _prompt_choice(
            "Default model",
            sorted(surface.keys()),
            default_val,
        )
    return _prompt(
        "Default model ID",
        default_val,
        required=True,
    )


def _validate_user_ids(value: str) -> bool:
    """Check that a comma-separated string contains only positive integers."""
    try:
        ids = [int(x.strip()) for x in value.split(",") if x.strip()]
        return len(ids) > 0 and all(i > 0 for i in ids)
    except ValueError:
        return False


def _validate_telegram_id(value: str) -> bool:
    """Check that a string is a single positive integer (Telegram user ID)."""
    try:
        return int(value.strip()) > 0
    except (ValueError, AttributeError):
        return False


# Letters, digits, spaces, hyphens, underscores. Prevents YAML structural
# characters (colons, hashes) in names; _yaml_scalar() separately handles
# YAML 1.1 boolean keyword quoting.
_DISPLAY_NAME_RE = re.compile(r"^[a-zA-Z0-9 _-]+$")


def _validate_display_name(value: str) -> bool:
    """Check that a display name contains only safe characters for YAML output."""
    return bool(value.strip()) and _DISPLAY_NAME_RE.match(value.strip()) is not None


# OS usernames: alphanumeric, dots, hyphens, underscores.
_OS_USER_RE = re.compile(r"^[a-zA-Z0-9._-]+$")


def _validate_os_user(value: str) -> bool:
    """Check that a string contains only valid OS username characters."""
    return bool(value.strip()) and _OS_USER_RE.match(value.strip()) is not None


def _validate_port(value: str) -> bool:
    """Check that a string is a valid port number (1-65535)."""
    try:
        port = int(value)
        return 1 <= port <= 65535
    except ValueError:
        return False


def _validate_positive_int(value: str) -> bool:
    """Check that a string is a positive integer."""
    try:
        return int(value) > 0
    except ValueError:
        return False


def _validate_non_negative_int(value: str) -> bool:
    """Check that a string is a non-negative integer (zero allowed).

    Used for fields where 0 is a meaningful kill-switch value, not an
    invalid input. Distinct from `_validate_positive_int` so the
    operator-facing error message can reflect the actual constraint.
    """
    try:
        return int(value) >= 0
    except ValueError:
        return False


def _validate_non_negative_float(value: str) -> bool:
    """Check that a string is a non-negative float (zero allowed).

    Sibling of `_validate_non_negative_int` for float-typed fields
    where 0 is a meaningful disable value (AGENT_MAX_SESSION_HOURS
    uses 0 for "no session-age limit").
    """
    try:
        return float(value) >= 0
    except ValueError:
        return False


def _validate_chat_id(value: str) -> bool:
    """Check that a string is a valid Telegram chat ID (any non-zero integer)."""
    try:
        return int(value) != 0
    except ValueError:
        return False


def _validate_claude_bin(value: str) -> bool:
    """Return True when the path is absolute, exists, and is executable.

    Required by the claude wizard prompt; the path is baked into both
    the runtime CLAUDE_BIN env var and the sudoers rule that allows
    cross-os_user claude spawns, so a non-existent path here would
    surface as a confusing 'a password is required' at first message.
    The path must also be absolute: a relative path resolves against
    the wizard's working directory at validation time but against the
    daemon's working directory at spawn time, and sudoers command
    matching expects absolute paths, so a relative value that passes
    here would still break both consumers.
    """
    if not value:
        return False
    p = Path(value)
    return p.is_absolute() and p.is_file() and os.access(p, os.X_OK)


def _validate_codex_bin(value: str) -> bool:
    """Return True when the path is absolute, exists, and is executable.

    Required by the codex wizard prompt; the path is baked into both
    the runtime CODEX_BIN env var and the sudoers rule that allows
    cross-os_user codex spawns, so a non-existent path here would
    surface as a confusing 'a password is required' at first message.
    The absoluteness requirement mirrors `_validate_claude_bin` above:
    relative paths resolve differently for the wizard and the daemon,
    and sudoers command matching expects absolute paths.
    """
    if not value:
        return False
    p = Path(value)
    return p.is_absolute() and p.is_file() and os.access(p, os.X_OK)


def _resolve_default_claude_bin(service_user: str) -> str:
    """Return the default Claude binary path used by sudoers.

    Native Claude installs land under the service user's home. The
    Homebrew cask installs under /opt/homebrew/bin/claude. Prefer the
    native service-user path when it exists, but recover the common
    macOS Homebrew install instead of pinning sudoers to a missing
    ~/.local/bin/claude.
    """
    svc_home = _user_home(service_user)
    native = Path(svc_home) / ".local" / "bin" / "claude"
    if _validate_claude_bin(str(native)):
        return str(native)
    homebrew = Path("/opt/homebrew/bin/claude")
    if svc_home.startswith("/Users/") and _validate_claude_bin(str(homebrew)):
        return str(homebrew)
    return str(native)


def _resolve_codex_bin_prompt_default(existing_env: dict[str, str]) -> str:
    """Return a usable default for every Codex binary-path prompt.

    Preserve a saved CODEX_BIN while it is executable. If an upgrade moved
    the binary, discard the stale value and use the same common-path
    resolver that sudoers generation uses.
    """
    saved = existing_env.get("CODEX_BIN", "")
    if _validate_codex_bin(saved):
        return saved
    return _resolve_default_codex_bin()


def _resolve_default_codex_bin() -> str:
    """Return the default Codex binary path used by sudoers.

    Prefer stable absolute install locations over the installer caller's
    PATH. The runtime uses the same precedence when CODEX_BIN is unset,
    so a users.yaml-only backend switch can be made effective with
    `make install` rather than an interactive `make config` pass.
    """
    for candidate in ("/usr/local/bin/codex", "/opt/homebrew/bin/codex"):
        if _validate_codex_bin(candidate):
            return candidate
    discovered = shutil.which("codex")
    if discovered and _validate_codex_bin(discovered):
        return discovered
    return "/usr/local/bin/codex"


def _validate_opencode_bin(value: str) -> bool:
    """Return True when the path is absolute, exists, and is executable.

    Required by the opencode wizard prompt; the path is baked into
    both the runtime OPENCODE_BIN env var and the sudoers rule that
    allows cross-os_user opencode spawns, so a non-existent path
    here would surface as a confusing 'a password is required' or
    'command not found' at the first one-shot call. Paired with
    `_validate_codex_bin` above so the two binary validators stay
    together; the body shape matches codex byte-for-byte (absolute
    plus is-file plus executable) because the underlying requirement
    is the same.
    """
    if not value:
        return False
    p = Path(value)
    return p.is_absolute() and p.is_file() and os.access(p, os.X_OK)


def _validate_goose_bin(value: str) -> bool:
    """Return True when the path is absolute, exists, and is executable.

    Required by the goose wizard prompt; same dual consumer as the
    codex and opencode validators above (runtime GOOSE_BIN env var
    plus the per-user sudoers rule), and the same absolute plus
    is-file plus executable body because the underlying requirement
    is the same.
    """
    if not value:
        return False
    p = Path(value)
    return p.is_absolute() and p.is_file() and os.access(p, os.X_OK)


def _install_staging_path(filename: str) -> Path:
    """Return the per-operator staging path for a first-time install file.

    `make config` runs as the unprivileged operator account; `sudo make
    install` runs as root. The staging file produced by config has to
    live somewhere both runs can reach without crossing the secret-
    discipline boundary at `/etc/kai/`. The operator's `${HOME}/.cache/
    kai-install/` directory satisfies both: the operator owns it (config
    can write without elevation) and root can read it during apply. We
    deliberately avoid the project tree because the spec for #557
    canonicalizes config locations outside the source checkout.

    The parent directory is created mode 0700 on every call so the
    staging file inherits a restrictive enclosing scope even before
    its own 0600 chmod lands. Idempotent: a pre-existing directory
    keeps its current mode.
    """
    cache_dir = Path.home() / ".cache" / "kai-install"
    cache_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    return cache_dir / filename


def _strip_install_conf_keys(*keys: str) -> None:
    """Remove the named top-level keys from `install.conf` and rewrite.

    Used by `_cmd_apply` after `apply_succeeded = True` to drop one-
    shot installer-metadata keys (currently just `users_yaml_staging_
    path`) so a subsequent re-run does not redo a handoff that already
    completed. Top-level only: keys inside the `env` dict belong to
    `/etc/kai/env` and are not touched by this helper.

    Preserves the 0600 mode because `install.conf` still carries
    secrets (bot token, webhook secret) until the operator explicitly
    deletes it. Missing keys are silently ignored so the helper is
    idempotent against repeated apply runs. A missing or unreadable
    `install.conf` is a no-op rather than an error because the helper
    is called during the success path of an apply that already
    validated the file at start.
    """
    if not INSTALL_CONF.exists():
        return
    try:
        conf = json.loads(INSTALL_CONF.read_text())
    except (json.JSONDecodeError, OSError):
        return
    mutated = False
    for key in keys:
        if key in conf:
            conf.pop(key)
            mutated = True
    if not mutated:
        return
    INSTALL_CONF.write_text(json.dumps(conf, indent=2) + "\n")
    os.chmod(INSTALL_CONF, 0o600)


def _xdg_users_yaml_path() -> Path:
    """Return the canonical single-user users.yaml location.

    Mirrors the runtime resolver in `config._resolve_users_yaml_path`
    for the non-protected branch: `${XDG_CONFIG_HOME:-$HOME/.config}/
    kai/users.yaml`. The wizard writes directly to this path in
    single-user mode (no staging handoff is needed because the
    operator owns the destination and there is no privilege boundary
    to cross on apply).
    """
    explicit = os.environ.get("XDG_CONFIG_HOME", "").strip()
    base = Path(explicit).expanduser() if explicit else Path.home() / ".config"
    return base / "kai" / "users.yaml"


def _read_users_yaml_text(path: Path) -> str | None:
    """Return the contents of a users.yaml file, falling back to sudo
    for root-owned protected copies.

    The wizard reads users.yaml as the operator account, not as root.
    `/etc/kai/users.yaml` is installed mode 0600 root-owned, so a
    direct `read_text()` raises `PermissionError`. The fallback uses
    interactive `sudo cat` (no `-n` flag) because the operator is at
    the terminal during `make config`; a password prompt mid-wizard
    is acceptable UX. Returns the file content on success, None on
    missing file, no sudo rights, or unreadable content.
    """
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except PermissionError:
        pass
    try:
        result = subprocess.run(
            ["sudo", "cat", str(path)],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def _users_yaml_entries(users_yaml_path: Path) -> list[dict]:
    """
    Read the canonical users.yaml and return its well-formed user
    entries for wizard-side scans.

    Reads via `_read_users_yaml_text` (sudo-tolerant on protected
    installs) because the wizard runs as the operator account. A
    missing, unreadable, or malformed file, a non-dict document, a
    missing or non-list `users` key, and non-dict list items all
    degrade to an empty result: wizard scans must never crash
    mid-flow on user-owned YAML, and the runtime's own users.yaml
    validation surfaces any real misconfiguration at startup.

    Deliberately NOT built on `_collect_backends_from_yaml`: that
    sibling serves `_apply_sudoers` (apply side), reads the file
    directly as root, and raises on malformed YAML so the install
    fails loudly - the opposite degrade posture from the wizard.
    """
    raw = _read_users_yaml_text(users_yaml_path)
    if raw is None:
        return []
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError:
        return []
    if not isinstance(data, dict):
        return []
    users = data.get("users")
    if not isinstance(users, list):
        return []
    return [entry for entry in users if isinstance(entry, dict)]


def _protected_user_assignments(users_yaml_path: Path) -> list[tuple[int, str, str | None]]:
    """Load the interactive principals relevant to protected OS isolation.

    This is an installer-side projection of config._load_user_configs: invalid
    entries are ignored, duplicate Telegram IDs use the first valid entry, and
    all valid entries retain only the identity fields needed by the isolation
    validator.  Unlike the wizard's informational scanners, this reader fails
    loudly on an unreadable or malformed file because apply must validate the
    effective users.yaml before stopping a working service.
    """
    raw = _read_users_yaml_text(users_yaml_path)
    if raw is None:
        raise ValueError(f"Protected users.yaml is missing or unreadable: {users_yaml_path}")
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ValueError(f"Protected users.yaml is malformed: {users_yaml_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Protected users.yaml must contain a YAML mapping: {users_yaml_path}")
    entries = data.get("users")
    if not isinstance(entries, list):
        raise ValueError(f"Protected users.yaml must contain a 'users' list: {users_yaml_path}")

    assignments: list[tuple[int, str, str | None]] = []
    seen_ids: set[int] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        raw_id = entry.get("telegram_id")
        try:
            if isinstance(raw_id, bool):
                raise ValueError
            telegram_id = int(raw_id)
            if telegram_id <= 0:
                raise ValueError
        except (TypeError, ValueError):
            continue
        name = str(entry.get("name") or "").strip()
        role = str(entry.get("role", "user")).strip().lower()
        if not name or role not in _VALID_ROLES or telegram_id in seen_ids:
            continue
        seen_ids.add(telegram_id)

        raw_os_user = entry.get("os_user")
        os_user = str(raw_os_user).strip() if raw_os_user is not None else ""
        os_user = os_user or None
        if os_user is not None and not _validate_os_user(os_user):
            raise ValueError(f"Invalid os_user {os_user!r} in {users_yaml_path}: must match {_OS_USER_RE.pattern}")
        assignments.append((telegram_id, name, os_user))
    return assignments


def _validate_protected_users_yaml(
    users_yaml_path: Path,
    service_user: str,
    *,
    require_existing_accounts: bool,
    service_uid: int | None = None,
) -> tuple[str, ...]:
    """Validate protected user isolation, optionally including passwd lookup."""
    assignments = _protected_user_assignments(users_yaml_path)
    return validate_protected_user_isolation(
        assignments,
        service_user,
        account_uid=(lambda name: pwd.getpwnam(name).pw_uid) if require_existing_accounts else None,
        service_uid=service_uid,
    )


def _entry_backend(entry: dict) -> object:
    """
    Read a users.yaml entry's backend, preferring the new `backend`
    key and falling back to the deprecated `default_backend` and then
    `agent_backend` keys for one release (the per-user key was renamed
    twice: agent_backend -> default_backend -> backend).

    Returns the raw value (or None) so callers apply their own
    normalization/validation, matching how they read the key today.
    No deprecation warning here: the daemon-side `_load_user_configs`
    emits the per-user warning; these installer scanners only need
    set membership.
    """
    for key in ("backend", "default_backend", "agent_backend"):
        value = entry.get(key)
        if value is not None:
            return value
    return None


def _entry_provider(entry: dict) -> object:
    """
    Read a users.yaml entry's provider, preferring the new `provider`
    key and falling back to the deprecated `llm_provider` key for one
    release. Mirrors `_entry_backend`.

    Returns the raw value (or None) so callers apply their own
    normalization, matching how they read the key today. No deprecation
    warning here: the daemon-side `_load_user_configs` emits the
    per-user warning; these installer scanners only need set membership.
    """
    for key in ("provider", "llm_provider"):
        value = entry.get(key)
        if value is not None:
            return value
    return None


def _users_yaml_goose_providers(users_yaml_path: Path, global_provider: str) -> list[str]:
    """
    Collect the distinct providers per-user goose entries need API
    keys for.

    Returns the sorted set of `provider` values across entries
    whose `backend` is "goose", falling back to `global_provider`
    for entries that omit the field - the same cascade the runtime
    applies. Goose is the only backend whose
    per-user auth rides the daemon environment: claude, codex, and
    opencode authenticate via per-OS-user on-disk state managed
    outside the wizard, so entries on those backends contribute
    nothing here. Parsing and degrade behavior live in
    `_users_yaml_entries` (empty result on any malformed input).
    """
    providers: set[str] = set()
    for entry in _users_yaml_entries(users_yaml_path):
        # Normalize the same way the runtime loader does
        # (str.strip().lower()): mixed-case `Goose` / `DeepSeek`
        # values are valid at runtime, so the scan must not let a
        # casing difference skip a key the daemon will demand.
        backend = _entry_backend(entry)
        if not isinstance(backend, str) or backend.strip().lower() != "goose":
            continue
        provider = _entry_provider(entry) or global_provider
        if isinstance(provider, str) and provider.strip():
            providers.add(provider.strip().lower())
    return sorted(providers)


def _users_yaml_agent_backends(users_yaml_path: Path) -> set[str]:
    """
    Collect the distinct per-user `backend` values for
    wizard-side binary collection.

    A users.yaml entry on a non-global backend makes that backend's
    binary load-bearing at runtime (the chat spawn and the per-user
    memory-extraction dispatch both route by each user's effective
    backend, and the fail-closed startup gate refuses to boot when
    an extraction-eligible user routes to an unresolvable binary),
    so the wizard must know which backends are in play beyond the
    global one. Parsing and degrade behavior live in
    `_users_yaml_entries` (empty result on any malformed input).
    """
    backends: set[str] = set()
    for entry in _users_yaml_entries(users_yaml_path):
        backend = _entry_backend(entry)
        # Normalize the same way the runtime loader does
        # (str.strip().lower()): mixed-case values like `Codex` are
        # valid at runtime and route the user, so the scan must
        # produce the canonical form the prompt gates compare
        # against. Non-string values are skipped here; the runtime
        # loader rejects them loudly at startup.
        if isinstance(backend, str) and backend.strip():
            backends.add(backend.strip().lower())
    return backends


# ── Config subcommand ────────────────────────────────────────────────


def _cmd_config() -> None:
    """
    Interactive Q&A that collects configuration values and writes install.conf.

    If install.conf already exists, its values are used as defaults so re-running
    only asks about changes. Auto-detects platform, generates distinct named
    webhook secrets, and offers an explicit safe-default choice to preserve or
    retire an existing legacy secret. Validates all inputs before writing.

    No sudo required - this runs as the current user.
    """
    print("Kai Installation - Configuration")
    print("=" * 45)
    print()

    # Load existing config as defaults if present
    existing: dict = {}
    if INSTALL_CONF.exists():
        try:
            existing = json.loads(INSTALL_CONF.read_text())
            print(f"Loaded existing {INSTALL_CONF} as defaults.\n")
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: could not read existing {INSTALL_CONF}: {e}\n")

    existing_env: dict = existing.get("env", {})
    legacy_webhook_secret = existing_env.get("WEBHOOK_SECRET", "")

    # Auto-detect platform
    if sys.platform == "darwin":
        detected_platform = "darwin"
    elif sys.platform.startswith("linux"):
        detected_platform = "linux"
    else:
        detected_platform = sys.platform

    # -- Deployment mode --
    # `protected`: root-owned install at /opt/kai with /etc/kai/ secrets,
    # the service-management apply flow, and `sudo make install` to
    # finalize. This is the existing multi-user / production shape.
    #
    # `single_user`: the operator runs Kai directly from the repo
    # without root. Secrets and per-user config live under the
    # operator's home (.env in PROJECT_ROOT, users.yaml under XDG
    # config home). `make config` writes everything and there is no
    # `sudo make install` step; `make run` (or `python -m kai`) starts
    # the daemon. This is the on-ramp for evaluators and the
    # documented dev workflow.
    #
    # Default to `protected` when install.conf has no mode so existing
    # operators on the long-standing path see no behavior change on a
    # re-run.
    print("-- Deployment mode --")
    deployment_mode = _prompt_choice(
        "Deployment mode",
        ["protected", "single_user"],
        existing.get("deployment_mode", "protected"),
    )
    print()

    # Migration safety: if the operator picks single_user but a
    # readable `/etc/kai/env` exists from a previous protected install,
    # the runtime predicate in `config._resolve_users_yaml_path` will
    # still see the protected env as authoritative and route at
    # `/etc/kai/users.yaml`, silently bypassing the single-user
    # artifacts we are about to write. The runtime cannot tell which
    # is the operator's intent; the wizard refuses up front so the
    # operator removes the protected leftovers before reaching that
    # ambiguous state. A protected install on a host whose operator
    # has the sudoers cat rule from a prior `sudo make install` is
    # the typical trigger.
    if deployment_mode == "single_user" and _read_protected_file("/etc/kai/env"):
        raise SystemExit(
            "single_user mode was selected, but /etc/kai/env is readable from a "
            "previous protected install. The runtime would still boot from the "
            "protected files, silently ignoring the single-user artifacts. "
            "Remove the protected install before retrying:\n"
            "    sudo rm -rf /etc/kai\n"
            "    sudo rm -f /etc/sudoers.d/kai\n"
            "Then re-run 'make config' and pick single_user."
        )

    # In single-user mode the install location, data directory,
    # service user, and platform service manager are not relevant:
    # Kai runs from the cloned repo under the operator's account.
    # We initialize these fields with stable defaults so downstream
    # install.conf shape stays consistent across modes; nothing reads
    # them in single-user mode but consumers that check for the keys
    # do not need a mode-aware lookup.
    if deployment_mode == "protected":
        # -- Installation paths --
        print("-- Installation paths --")
        install_dir = _prompt(
            "Install location",
            existing.get("install_dir", _DEFAULT_INSTALL_DIR),
        )
        if not os.path.isabs(install_dir):
            raise SystemExit("Install location must be an absolute path.")

        data_dir = _prompt(
            "Data directory",
            existing.get("data_dir", _DEFAULT_DATA_DIR),
        )
        if not os.path.isabs(data_dir):
            raise SystemExit("Data directory must be an absolute path.")

        service_user = _prompt(
            "Service user",
            existing.get("service_user", _DEFAULT_SERVICE_USER),
            required=True,
        )

        platform = _prompt_choice(
            "Platform",
            ["darwin", "linux"],
            existing.get("platform", detected_platform),
        )
        print()
    else:
        # Single-user defaults. install_dir points at PROJECT_ROOT so
        # any code that resolves paths through install.conf still
        # finds a real directory; data_dir defaults to PROJECT_ROOT
        # for the same reason. service_user is the current account.
        install_dir = str(PROJECT_ROOT)
        data_dir = str(PROJECT_ROOT)
        service_user = os.environ.get("USER", _DEFAULT_SERVICE_USER)
        platform = detected_platform

    # -- Telegram --
    print("-- Telegram --")
    bot_token = _prompt(
        "Telegram bot token",
        existing_env.get("TELEGRAM_BOT_TOKEN", ""),
        required=True,
    )

    # -- User setup --
    # `/etc/kai/users.yaml` is the only canonical source for the wizard.
    # The wizard reads it via `_read_users_yaml_text` which sudo-cats
    # the mode 0600 root-owned file. `Path.exists()` works through the
    # listable parent directory even though the file is unreadable to
    # the operator account, so the existence check does not require
    # elevation.
    #
    # The canonical users.yaml location depends on deployment mode:
    # - protected: `/etc/kai/users.yaml` (mode 0600 root-owned). The
    #   wizard reads it via `_read_users_yaml_text`'s sudo-cat path
    #   when it cannot read directly; the existence check goes through
    #   the listable parent directory and does not require elevation.
    # - single_user: `${XDG_CONFIG_HOME:-$HOME/.config}/kai/users.yaml`,
    #   owned by the operator. Direct read.
    #
    # If no canonical file exists, the first-time branch writes to the
    # appropriate location for the chosen mode (see below). A stray
    # `PROJECT_ROOT/users.yaml` from a previous install cycle is
    # ignored; a one-line deprecation warning tells the operator to
    # remove it or move it to the canonical location.
    stray_project_users_yaml = PROJECT_ROOT / "users.yaml"
    if deployment_mode == "protected":
        users_yaml_path = USERS_YAML
    else:
        users_yaml_path = _xdg_users_yaml_path()
    users_yaml_exists = users_yaml_path.exists()
    stray_note = (
        f"  Note: {stray_project_users_yaml} is no longer used. Move it to {users_yaml_path} or remove it."
        if stray_project_users_yaml.exists()
        else None
    )
    # The user-setup section emits output only when there is something to
    # do: a first-time install (the admin prompts below run) or a stray
    # leftover to warn about. When the canonical users.yaml already
    # exists the wizard leaves it untouched, so it prints no bare header.
    if not users_yaml_exists or stray_note:
        print("-- User setup --")
    if stray_note:
        print(stray_note)

    # Admin os_user captured by the advanced-mode prompt block below.
    # Carried in scope here so the codex-memory branch can reuse it
    # when re-prompting on a malformed admin entry.
    admin_os_user: str | None = None
    # Admin identity is captured in scope so the codex-memory branch
    # below can re-prompt for a valid os_user and rewrite the wizard-
    # owned users.yaml without re-collecting the static fields. Both
    # stay empty on the existing-users.yaml branch; the codex-memory
    # validation block is gated on `not users_yaml_exists` so it only
    # consults these when the wizard actually owns the file.
    admin_telegram_id: str = ""
    admin_name: str = ""
    admin_home_workspace: str | None = None
    # First-time wizard sets this to the absolute path of the staging
    # file written below; on every other branch it stays None. The
    # value is persisted as a top-level `install.conf` key (NOT inside
    # the env dict) so `_cmd_apply` can locate the staging file when
    # `sudo make install` runs under a different HOME than the wizard.
    # When the canonical `/etc/kai/users.yaml` already exists we
    # deliberately leave it None and DO NOT carry forward any prior
    # value from `existing`: a stale key would cause apply to overwrite
    # the canonical file from a no-longer-current staging artifact.
    users_yaml_staging_path: str | None = None

    # First-time install only: collect the admin identity and stage a
    # users.yaml. An existing canonical file is left untouched.
    if not users_yaml_exists:
        while True:
            admin_telegram_id = _prompt(
                "Admin Telegram ID",
                "",
                required=True,
            )
            admin_telegram_id = admin_telegram_id.strip()
            if _validate_telegram_id(admin_telegram_id):
                break
            print("  Must be a positive integer. Message @userinfobot on Telegram to find yours.")

        while True:
            admin_name = _prompt("Admin display name", "admin", required=True).strip()
            if _validate_display_name(admin_name):
                break
            print("  Name may only contain letters, numbers, spaces, hyphens, and underscores.")

        if deployment_mode == "protected":
            # The service account can sudo-cat exact protected Kai files.
            # A persistent agent running as that identity would inherit
            # those capabilities, so protected mode has no same-user path.
            print("  Protected mode requires a distinct OS account for each interactive user.")
            while True:
                admin_os_user = _prompt(
                    "OS user for subprocess isolation",
                    "",
                    required=True,
                ).strip()
                if not _validate_os_user(admin_os_user):
                    print("  Username may only contain letters, numbers, dots, hyphens, and underscores.")
                    continue
                if admin_os_user == service_user:
                    print(f"  Must differ from the Kai service account {service_user!r}.")
                    continue
                break
        else:
            # Single-user mode intentionally keeps the historic optional
            # same-user spawn shape. There is no protected service account
            # or cross-principal boundary in this deployment mode.
            advanced = _prompt_bool("Configure advanced user options", False)
            if advanced:
                default_os_user = os.environ.get("USER", "")
                while True:
                    admin_os_user = _prompt("OS user for subprocess isolation", default_os_user).strip() or None
                    if admin_os_user is None or _validate_os_user(admin_os_user):
                        break
                    print("  Username may only contain letters, numbers, dots, hyphens, and underscores.")

        if admin_os_user is not None:
            # No wizard prompt for home_workspace post-#353. The admin lands
            # in DATA_DIR/home/<chat_id>/ like any other user; the per-user
            # default is private to them. An admin who wants a path outside
            # DATA_DIR can add `home_workspace` to users.yaml by hand.
            admin_home_workspace = None

        # First-time write target depends on deployment mode:
        # - protected: stage at `${HOME}/.cache/kai-install/users.yaml`
        #   so `sudo make install` can copy it into `/etc/kai/users.yaml`
        #   without the operator running the wizard as root. The path
        #   is persisted as a top-level key in install.conf below;
        #   `_cmd_apply` unlinks the staging file and strips the conf
        #   key after `apply_succeeded`.
        # - single_user: write directly to the XDG users.yaml path
        #   (`${XDG_CONFIG_HOME:-$HOME/.config}/kai/users.yaml`). No
        #   staging handoff is needed because the operator owns the
        #   destination and there is no privilege boundary to cross
        #   on apply. The staging key stays unset so apply (if it ever
        #   runs by mistake) does not try to copy this file anywhere.
        users_yaml_content = _generate_users_yaml(
            admin_telegram_id,
            admin_name,
            os_user=admin_os_user,
            home_workspace=admin_home_workspace,
        )
        if deployment_mode == "protected":
            users_yaml_path = _install_staging_path("users.yaml")
            users_yaml_path.write_text(users_yaml_content)
            os.chmod(users_yaml_path, 0o600)
            users_yaml_staging_path = str(users_yaml_path)
        else:
            users_yaml_path = _xdg_users_yaml_path()
            users_yaml_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            users_yaml_path.write_text(users_yaml_content)
            os.chmod(users_yaml_path, 0o600)
        print(f"  Generated {users_yaml_path}")

    if deployment_mode == "protected":
        try:
            _validate_protected_users_yaml(
                users_yaml_path,
                service_user,
                require_existing_accounts=False,
            )
        except ValueError as exc:
            raise SystemExit(f"Protected user isolation validation failed before writing install.conf:\n{exc}") from exc
    # Blank separator only when the user-setup section above actually
    # printed something (first-time prompts or a stray-leftover note),
    # gated on the same condition as its header. On a re-run with an
    # existing users.yaml the section is silent, so this would otherwise
    # be an orphaned blank line before the transport prompt.
    if not users_yaml_exists or stray_note:
        print()

    transport = _prompt_choice(
        "Telegram transport",
        ["polling", "webhook"],
        existing_env.get("TELEGRAM_TRANSPORT", "polling"),
    )

    webhook_url = ""
    tg_webhook_secret = ""
    if transport == "webhook":
        webhook_url = _prompt(
            "Telegram webhook URL",
            existing_env.get("TELEGRAM_WEBHOOK_URL", ""),
            required=True,
        )
        tg_webhook_secret = _prompt(
            "Telegram webhook secret",
            existing_env.get("TELEGRAM_WEBHOOK_SECRET", ""),
        )
    print()

    # -- Default backend --
    print("-- Default backend --")
    # Prefill reads DEFAULT_BACKEND, falling back to the deprecated
    # AGENT_BACKEND key so a re-run against a legacy install.conf keeps
    # the operator's prior choice; the wizard writes DEFAULT_BACKEND only.
    agent_backend = _prompt_choice(
        "Default backend",
        sorted(VALID_BACKENDS),
        _resolve_renamed_key(
            existing_env.get,
            new_key="DEFAULT_BACKEND",
            legacy_keys=["AGENT_BACKEND"],
            context="install.conf",
            default="claude",
        )
        or "claude",
    )

    # Codex auth handling runs BEFORE the legacy provider/key block so
    # subscription mode is not gated on an OPENAI_API_KEY the operator
    # does not need. Codex sits outside BACKENDS_NEEDING_PROVIDER_PROMPT
    # (it is single-provider: always openai), so the provider/API-key
    # block below skips codex entirely; codex follows the same path
    # claude does for the provider prompt.
    #
    # Gate on the global backend selection only. Per-user
    # `backend: codex` overrides in users.yaml do NOT trigger
    # the auth-mode setup here; codex AUTH for those users stays
    # out-of-band (per-OS-user `codex login`). Their BINARY path is
    # collected by the per-user backend scan after the provider
    # block, which feeds this same codex_bin variable.
    codex_auth_mode = ""
    codex_api_key = ""
    codex_bin = ""
    # OpenCode binary path collected by the global-opencode block
    # below (and re-prompted by the memory-extraction defense-in-depth
    # gate if a future refactor exposes a path where the global block
    # did not run). Empty when opencode is not in play so the
    # persistence gate at the tail of this function skips emitting
    # OPENCODE_BIN to install.conf.
    opencode_bin = ""
    # Goose binary path: same collection and persistence shape as
    # opencode above (global-goose block, extraction defense-in-depth
    # re-prompt, empty-skip persistence gate for GOOSE_BIN).
    goose_bin = ""
    # Claude binary path: same collection and persistence shape as
    # opencode (global-claude block, extraction defense-in-depth
    # re-prompt, empty-skip persistence gate for CLAUDE_BIN).
    claude_bin = ""
    if agent_backend == "claude":
        # Claude binary path: wizard-prompted and persisted in
        # install.conf so `make install` writes both /etc/kai/env's
        # CLAUDE_BIN and the sudoers SETENV rule from the same source
        # of truth. Default suggestion uses `which claude` from the
        # operator's PATH; falls back to the empty string when which
        # finds nothing (claude has two canonical install locations,
        # the native installer's ~/.local/bin/claude and the Homebrew
        # cask's /opt/homebrew/bin/claude, so an empty default forces
        # the operator to type the path explicitly rather than accept
        # a wrong guess between the two). The _cmd_apply env
        # passthrough (`sudo CLAUDE_BIN=... make install`) still works
        # as the ad-hoc deploy escape hatch; the wizard path is the
        # canonical source of truth.
        while True:
            which_claude = shutil.which("claude") or ""
            claude_bin = _prompt(
                "Claude binary path",
                existing_env.get("CLAUDE_BIN", which_claude),
                required=True,
            )
            if _validate_claude_bin(claude_bin):
                break
            print(f"  Path '{claude_bin}' is not an absolute path to an existing executable.")
    if agent_backend == "codex":
        codex_auth_mode = _prompt_choice(
            "Codex auth mode",
            ["subscription", "api_key"],
            existing_env.get("CODEX_AUTH_MODE", "subscription"),
        )
        if codex_auth_mode == "api_key":
            codex_api_key = _prompt(
                "OPENAI_API_KEY",
                existing_env.get("OPENAI_API_KEY", ""),
                required=True,
            )
        # Codex binary path: wizard-prompted and persisted in install.conf
        # so `make install` writes both /etc/kai/env's CODEX_BIN and the
        # sudoers SETENV rule from the same source of truth. Default
        # suggestion keeps a valid saved path, otherwise uses `which codex`
        # from the operator's PATH. This recovers automatically when an
        # upgrade moves the binary. Falls back to Homebrew only when which
        # finds nothing; validation still prevents accepting a missing path.
        while True:
            codex_bin = _prompt(
                "Codex binary path",
                _resolve_codex_bin_prompt_default(existing_env),
                required=True,
            )
            if _validate_codex_bin(codex_bin):
                break
            print(f"  Path '{codex_bin}' is not an absolute path to an existing executable.")
        if codex_auth_mode == "subscription":
            print("  After install, log in to codex as the target os_user:")
            print("    <os_user> ~$ codex login")
            print("  Run as the os_user themselves, not via sudo from another account.")
            print("  If users.yaml has per-user `backend: codex` entries with")
            print("  different os_users, log in as each of those too.")

    # OpenCode setup: binary-path wizard prompt + post-install auth
    # reminder. OpenCode joins BACKENDS_NEEDING_PROVIDER_PROMPT so the
    # operator names the provider used by the (backend, provider, role)
    # registry; the API-key sub-prompt skips for opencode because
    # opencode auth lives in `~/.local/share/opencode/auth.json` and
    # is managed via `opencode auth login`, not by Kai. Model selection
    # routes through _prompt_default_model later in this function; since
    # models_for_backend("opencode", _) returns None, the operator gets
    # a free-text prompt for a full `provider/model` ID.
    #
    # Gate on the global backend selection only. Per-user
    # `backend: opencode` overrides in users.yaml do NOT
    # trigger the setup here; opencode AUTH for those users stays
    # out-of-band (`opencode auth login` per OS user). Their BINARY
    # path is collected by the per-user backend scan after the
    # provider block, which feeds this same opencode_bin variable.
    if agent_backend == "opencode":
        # OpenCode binary path: wizard-prompted and persisted in
        # install.conf so `make install` writes both /etc/kai/env's
        # OPENCODE_BIN and the sudoers SETENV rule from the same
        # source of truth. Default suggestion uses `which opencode`
        # from the operator's PATH; falls back to the empty string
        # when which finds nothing (opencode has no single canonical
        # install location across operator platforms, so an empty
        # default forces the operator to type the path explicitly
        # rather than accept a wrong default). The _cmd_apply env
        # passthrough (`sudo OPENCODE_BIN=... make install`) still
        # works as the ad-hoc deploy escape hatch; the wizard path is
        # the canonical source of truth.
        while True:
            which_opencode = shutil.which("opencode") or ""
            opencode_bin = _prompt(
                "OpenCode binary path",
                existing_env.get("OPENCODE_BIN", which_opencode),
                required=True,
            )
            if _validate_opencode_bin(opencode_bin):
                break
            print(f"  Path '{opencode_bin}' is not an absolute path to an existing executable.")
        print("  After install, authenticate OpenCode for at least one provider:")
        print("    <service_user> ~$ opencode auth login")
        print("  Kai writes the active model into OPENCODE_CONFIG_CONTENT at process spawn;")
        print("  OpenCode resolves it against the credentials in ~/.local/share/opencode/auth.json.")

    # Goose setup: binary-path wizard prompt. Mirrors the opencode
    # block above: the wizard-persisted path drives /etc/kai/env's
    # GOOSE_BIN and the per-user sudoers SETENV rule from one source
    # of truth, and resolve_oneshot_binary("goose") prefers it over
    # PATH discovery at run time. Default suggestion uses `which
    # goose` from the operator's PATH (Homebrew installs land there);
    # falls back to the empty string when which finds nothing so the
    # operator types the path explicitly rather than accepting a
    # wrong default. Provider auth needs no extra reminder here: the
    # provider prompt below collects the API key for key-based
    # providers, and keychain auth via `goose configure` is per-user
    # and out of band.
    if agent_backend == "goose":
        while True:
            which_goose = shutil.which("goose") or ""
            goose_bin = _prompt(
                "Goose binary path",
                existing_env.get("GOOSE_BIN", which_goose),
                required=True,
            )
            if _validate_goose_bin(goose_bin):
                break
            print(f"  Path '{goose_bin}' is not an absolute path to an existing executable.")

    # Multi-provider backends (opencode, goose): operator picks the
    # provider that drives the (backend, provider, role) registry
    # lookup at runtime. Single-provider backends (claude, codex)
    # bypass this block via BACKENDS_NEEDING_PROVIDER_PROMPT
    # membership; their provider is implicit (claude through
    # get_effective_provider, codex always openai). OpenCode also
    # skips the API-key sub-prompt because opencode's auth is
    # managed by `opencode auth login`, not by Kai; the wizard
    # captures only the provider name so the registry triple-key
    # can find a row.
    llm_provider = ""
    llm_api_key_var = ""
    llm_api_key = ""
    # Prefill the provider prompt from the existing env, preferring the
    # canonical DEFAULT_PROVIDER key with a one-release fallback to the
    # deprecated LLM_PROVIDER name.
    provider_prefill = (
        _resolve_renamed_key(
            existing_env.get,
            new_key="DEFAULT_PROVIDER",
            legacy_keys=["LLM_PROVIDER"],
            context="install.conf",
            default="",
        )
        or ""
    )
    valid_providers: tuple[str, ...] | None = (
        BACKEND_PROVIDERS.get(agent_backend) if agent_backend in BACKENDS_NEEDING_PROVIDER_PROMPT else None
    )
    if valid_providers is not None:
        llm_provider = _prompt_choice(
            "LLM provider",
            sorted(valid_providers),
            provider_prefill,
        )
        if agent_backend != "opencode":
            llm_api_key_var = PROVIDER_KEY_VARS.get(llm_provider, "")
            if llm_api_key_var:
                llm_api_key = _prompt(
                    llm_api_key_var,
                    existing_env.get(llm_api_key_var, ""),
                    required=True,
                )
            else:
                # Ollama and any other auth-less provider on the
                # goose path.
                print(f"  {llm_provider} does not require an API key.")
    print()

    # Per-user goose entries authenticate from the daemon environment
    # (their provider keys are forwarded through the per-backend env
    # allowlists and sudo preserve lists), and the wizard is the only
    # owner of that environment. This is why goose key collection is
    # per-user-aware while codex / opencode per-user overrides stay
    # out-of-band: those backends authenticate via per-OS-user files
    # (codex login, opencode auth login) the wizard does not manage,
    # but a goose entry has no equivalent the operator could use -
    # keys added to the env file by hand are wiped on regeneration.
    # Scan users.yaml for goose entries and prompt for any provider
    # key the global block above did not already collect; defaults
    # come from the existing env so a re-run keeps stored values.
    extra_provider_keys: dict[str, str] = {}
    for peruser_provider in _users_yaml_goose_providers(
        users_yaml_path,
        llm_provider or provider_prefill,
    ):
        peruser_key_var = PROVIDER_KEY_VARS.get(peruser_provider, "")
        if not peruser_key_var:
            # Ollama and any other auth-less provider.
            continue
        if peruser_key_var == llm_api_key_var and llm_api_key:
            # The global block already collected this exact var.
            continue
        print(f"  users.yaml has a goose entry on {peruser_provider}; the daemon env needs {peruser_key_var}.")
        extra_provider_keys[peruser_key_var] = _prompt(
            peruser_key_var,
            existing_env.get(peruser_key_var, ""),
            required=True,
        )
    if extra_provider_keys:
        print()

    # Per-user entries on a non-global backend make that backend's
    # binary load-bearing: the chat spawn and the per-user memory-
    # extraction dispatch both route by each user's effective
    # backend, and the fail-closed startup gate refuses to boot when
    # an extraction-eligible user routes to an unresolvable binary.
    # The blocks above collect a binary only for the global backend,
    # so collect here for every other backend per-user entries put
    # in play; the values flow into the same codex_bin / opencode_bin
    # / goose_bin variables, so the existing persistence (env var,
    # install.conf, sudoers SETENV rule) needs no new emission sites.
    # Prompt loops mirror the global blocks, including each backend's
    # default posture (codex falls back to Homebrew; claude, goose,
    # and opencode force an explicit path when `which` finds nothing).
    peruser_backends = _users_yaml_agent_backends(users_yaml_path)
    if "claude" in peruser_backends and not claude_bin:
        print("  users.yaml has a claude entry; the daemon needs a resolvable claude binary.")
        while True:
            which_claude = shutil.which("claude") or ""
            claude_bin = _prompt(
                "Claude binary path",
                existing_env.get("CLAUDE_BIN", which_claude),
                required=True,
            )
            if _validate_claude_bin(claude_bin):
                break
            print(f"  Path '{claude_bin}' is not an absolute path to an existing executable.")
    if "codex" in peruser_backends and not codex_bin:
        print("  users.yaml has a codex entry; the daemon needs a resolvable codex binary.")
        while True:
            codex_bin = _prompt(
                "Codex binary path",
                _resolve_codex_bin_prompt_default(existing_env),
                required=True,
            )
            if _validate_codex_bin(codex_bin):
                break
            print(f"  Path '{codex_bin}' is not an absolute path to an existing executable.")
    if "opencode" in peruser_backends and not opencode_bin:
        print("  users.yaml has an opencode entry; the daemon needs a resolvable opencode binary.")
        while True:
            which_opencode = shutil.which("opencode") or ""
            opencode_bin = _prompt(
                "OpenCode binary path",
                existing_env.get("OPENCODE_BIN", which_opencode),
                required=True,
            )
            if _validate_opencode_bin(opencode_bin):
                break
            print(f"  Path '{opencode_bin}' is not an absolute path to an existing executable.")
    if "goose" in peruser_backends and not goose_bin:
        print("  users.yaml has a goose entry; the daemon needs a resolvable goose binary.")
        while True:
            which_goose = shutil.which("goose") or ""
            goose_bin = _prompt(
                "Goose binary path",
                existing_env.get("GOOSE_BIN", which_goose),
                required=True,
            )
            if _validate_goose_bin(goose_bin):
                break
            print(f"  Path '{goose_bin}' is not an absolute path to an existing executable.")

    # -- Agent --
    # Determine the effective provider for model choices. Claude
    # backend always uses Anthropic; Codex always uses OpenAI; multi-
    # provider backends use the selected provider. The conversational
    # default model is no longer prompted here; it is the
    # MODEL_REGISTRY[ModelRole.AGENT] row for this backend/provider.
    # Existing DEFAULT_MODEL / CLAUDE_MODEL keys are intentionally not
    # re-emitted so re-running the wizard migrates installs back to the
    # registry default. Per-user `models.agent` remains the override
    # surface for individual users.
    eff_provider = "anthropic" if agent_backend == "claude" else llm_provider

    print("-- Agent --")

    # Per-role model customization. The conversational role (`agent`)
    # is not prompted; its default is marked in MODEL_REGISTRY. This
    # block offers the non-conversational roles (PR review, issue
    # triage, memory extraction, memory episode, behavioral judge,
    # behavioral gen) as an optional follow-up. Default-accept is one keystroke
    # ("no, use registry defaults"); operators who want per-role
    # control answer yes and walk the role list.
    #
    # The captured dict drops entries equal to the registry default
    # (delta-from-defaults discipline). The result writes to
    # DEFAULT_MODELS_JSON in /etc/kai/env so load_config sees it as
    # a global fallback below per-user `models:` and above the
    # MODEL_REGISTRY default.
    default_models_override: dict[str, str] = {}
    existing_default_models_raw = existing_env.get("DEFAULT_MODELS_JSON", "").strip()
    existing_default_models: dict[str, str] = {}
    if existing_default_models_raw:
        try:
            parsed_existing = json.loads(existing_default_models_raw)
            if isinstance(parsed_existing, dict):
                existing_default_models = {
                    str(k): canonicalize_model_for_backend(str(v), agent_backend) for k, v in parsed_existing.items()
                }
        except ValueError:
            # Invalid JSON in env: ignore; the wizard re-captures.
            pass

    customize_models = _prompt_bool(
        "Customize per-role models (PR review, triage, memory, eval)",
        default=bool(existing_default_models),
    )
    if customize_models:
        # `models_for_backend` returns a curated dict for the
        # claude / codex / goose-on-curated path, or None for opencode
        # and OPEN_ENDED_PROVIDERS (openrouter, ollama). Branch on
        # the return: curated path uses _prompt_choice, None path
        # uses free-text _prompt with the registry default as
        # suggestion.
        model_surface = models_for_backend(agent_backend, eff_provider)
        # Only prompt for roles the active backend can actually serve;
        # _build_registry guarantees a row exists for every (backend,
        # provider, role) triple in BACKEND_PROVIDERS, but the
        # behavioral roles are eval-only and rarely set per-user.
        # Walk every ModelRole regardless so the wizard surfaces the
        # full set; operators accept defaults to skip individual roles.
        for role in ModelRole:
            if role is ModelRole.AGENT:
                continue
            role_key = role.value
            try:
                role_default = MODEL_REGISTRY[(agent_backend, eff_provider, role)]
            except KeyError:
                # Registry has no row for this triple; skip the role
                # rather than offering an empty suggestion. Will not
                # happen in steady state because _check_model_registry_complete
                # asserts completeness at load_config time.
                continue
            existing_value = existing_default_models.get(role_key, role_default)
            if model_surface is None:
                # opencode + OPEN_ENDED_PROVIDERS: free-text prompt.
                # The operator's auth state determines validity; a
                # curated keyboard would mislead.
                value = _prompt(
                    f"  {role_key}",
                    default=existing_value,
                )
            else:
                # claude / codex / goose-on-curated: numbered-choice
                # list from the curated dict. The operator's last
                # selected value (or registry default) pre-fills as
                # the highlighted default.
                value = _prompt_choice(
                    f"  {role_key}",
                    sorted(model_surface.keys()),
                    existing_value,
                )
            if value and value != role_default:
                default_models_override[role_key] = value

    while True:
        # DEFAULT_TIMEOUT is canonical; legacy AGENT_TIMEOUT_SECONDS is
        # the one-release fallback for upgrades.
        timeout_default = existing_env.get("DEFAULT_TIMEOUT", existing_env.get("AGENT_TIMEOUT_SECONDS", "120"))
        timeout = _prompt(
            "Agent timeout (seconds)",
            timeout_default,
        )
        if _validate_positive_int(timeout):
            break
        print("  Must be a positive integer.")

    # Session lifecycle tunables. Both govern the subprocess pool for
    # every backend (recycle-by-age and idle eviction), hence the
    # AGENT_ prefix; the CLAUDE_-prefixed forms are legacy aliases
    # honored as prefill fallbacks here and popped from the env dict
    # below so a regenerated config carries only the canonical keys.
    while True:
        max_session_hours = _prompt(
            "Max session age in hours (0 = no limit)",
            existing_env.get("AGENT_MAX_SESSION_HOURS", existing_env.get("CLAUDE_MAX_SESSION_HOURS", "0")),
        )
        if _validate_non_negative_float(max_session_hours):
            break
        print("  Must be a non-negative number.")
    while True:
        idle_timeout = _prompt(
            "Idle eviction timeout in seconds (0 disables)",
            existing_env.get("AGENT_IDLE_TIMEOUT", existing_env.get("CLAUDE_IDLE_TIMEOUT", "1800")),
        )
        if _validate_non_negative_int(idle_timeout):
            break
        print("  Must be a non-negative integer.")

    # Autocompact threshold controls when Claude automatically compresses
    # conversation history. Lower values compact sooner, reducing token
    # usage at the cost of losing raw context earlier. Claude Code caps
    # this at ~83% regardless of what you set - values above that are
    # silently clamped. Claude-binary specific (CLAUDE_AUTOCOMPACT_PCT
    # is consumed only by ClaudeCodeBackend per claude.py); Goose has
    # no autocompact concept, so the prompt is suppressed when goose is
    # the selected backend. Initialize to "0" first (the dataclass
    # default) so the env-emission check `int(autocompact_pct) != 0`
    # below correctly skips writing the var for goose without needing
    # a parallel gate down at the emission site.
    autocompact_pct = "0"
    if agent_backend == "claude":
        while True:
            autocompact_pct = _prompt(
                "Autocompact threshold (% of context window, 0 = default ~83%)",
                existing_env.get("CLAUDE_AUTOCOMPACT_PCT", "80"),
            )
            try:
                val = int(autocompact_pct)
                if 0 <= val <= 100:
                    break
            except ValueError:
                pass
            print("  Must be 0-100.")
        print()

    # CLAUDE_EFFORT_LEVEL is consumed only by ClaudeCodeBackend's
    # `--effort` CLI flag (claude.py); Goose has no equivalent flag,
    # so the prompt is suppressed when goose is the selected backend.
    # Initialize to "high" first (matches Config.claude_effort_level
    # default) so the env-emission check `claude_effort_level != "high"`
    # below correctly skips writing the var for goose without needing
    # a parallel gate down at the emission site. _prompt_choice still
    # enforces the allow-list at wizard time for the claude path,
    # mirroring the runtime allow-list check in config._VALID_EFFORT_LEVELS
    # so the operator cannot persist an invalid value to install.conf
    # and have it fail later at service startup.
    claude_effort_level = "high"
    if agent_backend == "claude":
        # Use the canonical EFFORT_LEVELS tuple from config so the wizard
        # prompt and the runtime allow-list cannot drift out of sync. Cast
        # to list because _prompt_choice expects a list (it joins with "/"
        # for the display string and uses `in` for membership; tuple would
        # work for both but the type signature asks for list).
        claude_effort_level = _prompt_choice(
            "Inner Claude effort level",
            list(EFFORT_LEVELS),
            existing_env.get("CLAUDE_EFFORT_LEVEL", "high"),
        )
        print()

    # CODEX_EFFORT_LEVEL mirrors the claude effort prompt for the
    # codex backend with one contract difference: empty is a valid
    # answer (set-or-absent). Empty means CodexBackend passes no
    # `-c model_reasoning_effort` override and codex falls back to the
    # per-OS-user ~/.codex/config.toml or the model default, which is
    # the right posture because codex config is per-OS-user and
    # operator-owned, and xhigh availability is model-dependent.
    # _prompt_optional_choice exists to express that contract: it
    # validates non-empty input against the same allow-list config.py
    # enforces at load so install.conf cannot carry a value that fails
    # at service startup, while accepting "" as the no-override signal.
    codex_effort_level = ""
    if agent_backend == "codex":
        codex_effort_level = _prompt_optional_choice(
            "Codex reasoning effort",
            list(CODEX_EFFORT_LEVELS),
            existing_env.get("CODEX_EFFORT_LEVEL", ""),
            empty_hint="empty = codex default",
        )
        print()

    # -- Webhook server --
    print("-- Webhook server --")
    while True:
        port = _prompt(
            "Webhook port",
            existing_env.get("WEBHOOK_PORT", "8080"),
        )
        if _validate_port(port):
            break
        print("  Must be a valid port number (1-65535).")

    # Named credentials are the long-term configuration. Keep them distinct
    # from an existing legacy credential so the runtime can identify and log
    # callers that still depend on compatibility authentication.
    default_github_secret = existing_env.get("GITHUB_WEBHOOK_SECRET", "")
    reserved_webhook_secrets = {legacy_webhook_secret, tg_webhook_secret}
    while not default_github_secret or default_github_secret in reserved_webhook_secrets:
        default_github_secret = secrets.token_hex(32)
    while True:
        github_webhook_secret = _prompt(
            "GitHub webhook secret",
            default_github_secret,
            required=True,
        )
        if github_webhook_secret not in reserved_webhook_secrets:
            break
        print("  GitHub, Telegram, and legacy webhook secrets must use different values.")
        default_github_secret = secrets.token_hex(32)

    if tg_webhook_secret and tg_webhook_secret == legacy_webhook_secret:
        print("  Telegram and legacy webhook secrets must be different; Telegram will use a generated token.")
        tg_webhook_secret = ""

    generic_webhook_secret = existing_env.get("GENERIC_WEBHOOK_SECRET", "")
    reserved_webhook_secrets = {github_webhook_secret, legacy_webhook_secret, tg_webhook_secret}
    while not generic_webhook_secret or generic_webhook_secret in reserved_webhook_secrets:
        generic_webhook_secret = secrets.token_hex(32)

    # install.conf is generated output from this wizard. An upgraded install
    # must therefore retire the legacy fallback here rather than by editing or
    # post-processing that artifact. Preserve compatibility on Enter: callers
    # have to be migrated and tested before the operator explicitly chooses
    # false. Fresh installations never carry WEBHOOK_SECRET and skip the prompt.
    if legacy_webhook_secret:
        print()
        print("A deprecated WEBHOOK_SECRET compatibility fallback is present.")
        print("Retire it only after GitHub and generic webhook callers use their named secrets.")
        retain_legacy_webhook_secret = _prompt_bool(
            "Retain deprecated WEBHOOK_SECRET fallback",
            True,
        )
        if not retain_legacy_webhook_secret:
            legacy_webhook_secret = ""
            print("  WEBHOOK_SECRET will be omitted from the generated configuration.")
    print()

    # -- Workspaces --
    # WORKSPACE_BASE and ALLOWED_WORKSPACES are inheritable installation
    # defaults. users.yaml entries can override `workspace_base` per user
    # and add per-user allowed_workspaces, but the global env values
    # apply across every user without an override. Both prompts fire on
    # every wizard run.
    print("-- Workspaces --")
    workspace_base = _prompt(
        "Workspace base directory",
        existing_env.get("WORKSPACE_BASE", ""),
    )
    # Expand ~ for display but store as-is (load_config handles expansion)
    if workspace_base.startswith("~"):
        expanded = os.path.expanduser(workspace_base)
        print(f"  (expands to {expanded})")

    allowed_workspaces = _prompt(
        "Allowed workspaces (comma-separated paths, optional)",
        existing_env.get("ALLOWED_WORKSPACES", ""),
    )
    print()

    # -- PR review agent --
    # PR_REVIEW_COOLDOWN and PR_REVIEW_TIMEOUT_S are global resource
    # controls for the review subprocess: any review by any user
    # counts against the same cooldown and the same subprocess
    # timeout. Both fire on every wizard run. The per-user
    # `pr_review` toggle lives in users.yaml (or /github reviews).
    print("-- PR review agent --")

    # Global cooldown always prompts: any opted-in user can drive
    # reviews, so the cooldown must be configurable for any install
    # that has any opted-in user.
    while True:
        pr_review_cooldown = _prompt(
            "Review cooldown in seconds (prevents spam from rapid pushes)",
            existing_env.get("PR_REVIEW_COOLDOWN", "300"),
        )
        if _validate_positive_int(pr_review_cooldown):
            break
        print("  Must be a positive integer.")

    # Timeout for the review subprocess. Always collectable: it
    # applies to any review whether or not the global env flag is set.
    while True:
        pr_review_timeout_s = _prompt(
            "Review subprocess timeout in seconds",
            existing_env.get("PR_REVIEW_TIMEOUT_S", "900"),
        )
        if _validate_positive_int(pr_review_timeout_s):
            break
        print("  Must be a positive integer.")
    print()

    # -- Optional features --
    print("-- Optional features --")
    voice_enabled = _prompt_bool(
        "Voice transcription",
        existing_env.get("VOICE_ENABLED", "false").lower() in ("1", "true", "yes"),
    )
    tts_enabled = _prompt_bool(
        "Text-to-speech",
        existing_env.get("TTS_ENABLED", "false").lower() in ("1", "true", "yes"),
    )

    # Per-user OS isolation lives in users.yaml `os_user` (admin-set
    # baseline) and is no longer mirrored to a global env var. The
    # wizard surfaces this here only so an operator coming from an
    # older install knows where the setting moved.
    print("  Per-user OS isolation: set 'os_user' in users.yaml.")
    print()

    # -- Semantic memory --
    # MEMORY_ENABLED is a single global toggle for Mem0 + Qdrant. Per-user
    # partitioning happens at runtime via user_id = telegram_chat_id, so
    # there is no users.yaml branching here. Without these prompts the
    # wizard silently drops memory config on every reinstall (see #343).
    print("-- Semantic memory --")
    memory_enabled = _prompt_bool(
        "Enable semantic memory (Mem0 + Qdrant)",
        existing_env.get("MEMORY_ENABLED", "false").lower() in ("1", "true", "yes"),
    )
    # Every backend in ONESHOT_REASONER_BACKENDS can run semantic
    # memory with a backend-matched reasoner; extraction dispatches
    # per user at runtime via memory_extraction._build_memory_reasoner.
    # The historical "codex disables memory" guard was removed once
    # OneShotReasoner abstracted away the claude-only assumption in
    # memory_extraction.
    # Defaults match the dataclass values in config.py. Only non-defaults
    # (or memory_enabled=true itself) are written to the env dict below.
    memory_extraction_enabled = False
    memory_extraction_timeout_s = "10"
    memory_consolidation_candidates_n = "8"
    # Issue #392: episode classifier context window. Pre-init matches
    # the dataclass default so the emission gate `int(...) != 3`
    # correctly suppresses the env entry when an operator (a) skips
    # the prompt because of the non-claude branch or (b) reaches the
    # prompt and accepts the default. The prompt itself only fires
    # when the operator is on the claude backend AND extraction is
    # enabled - same gating pattern as the other extraction tunables.
    episode_classifier_context_turns = "3"
    # Stage-2 episode timeout pre-init (issue #385). The value matches
    # the corresponding wizard prompt default and the emission-gate
    # sentinel so the data flow reads consistently whether the
    # operator runs the episode block or skips it. The episode model
    # itself is resolved per-user from the registry at extraction
    # time; no wizard prompt or env var carries it.
    memory_episode_timeout_s = "120"
    memory_token_budget = "2000"
    memory_search_limit = "10"
    # Pre-init for the extraction-time paraphrase-dedup threshold.
    # Mirrors the other extraction tunables above: the value is set
    # only when the wizard reaches the prompt (claude backend AND
    # extraction enabled). Non-claude installs keep the dataclass
    # default at runtime via load_config, so the env entry stays
    # suppressed by the delta-from-default check below.
    memory_duplicate_threshold = "0.9"
    if memory_enabled:
        # Memory extraction is supported on agent backends that ship a
        # OneShotReasoner. Every shipped backend is in
        # ONESHOT_REASONER_BACKENDS, so the guard below always passes
        # today; it stays so a future backend added without a reasoner
        # degrades to retrieval-only (with the printed note in the
        # else branch) instead of silently failing extraction. The
        # runtime eligibility gate at `bot._ingest_memory` and
        # `config._compute_extraction_eligible_backends` read from the
        # same constant; all sites stay in lockstep through it.
        if agent_backend in ONESHOT_REASONER_BACKENDS:
            memory_extraction_enabled = _prompt_bool(
                "Enable memory extraction (proactive memory writes)",
                existing_env.get("MEMORY_EXTRACTION_ENABLED", "false").lower() in ("1", "true", "yes"),
            )
            if memory_extraction_enabled:
                # Per-user dispatch (issue #515). Memory reasoner and
                # model derive from each user's effective agent_backend
                # at extraction time via memory_extraction._build_
                # memory_reasoner + get_model_for(role, effective_
                # backend). There is no global reasoner or model
                # config; the wizard prompts that used to ask for both
                # were retired with the env vars they emitted.
                #
                # No codex-memory-specific os_user gate fires here.
                # Codex now follows claude's `resolve_claude_user`
                # symmetry: missing `os_user` spawns codex in-process
                # as the bot user, the same self-sudo-skip path
                # claude uses. The earlier global-codex block already
                # collected CODEX_BIN; nothing else specific to codex
                # memory needs setup at this point. CODEX_BIN must be
                # collected on every global-codex install; this second
                # prompt is a defense-in-depth no-op when the earlier
                # block already ran.
                if agent_backend == "codex" and not codex_bin:
                    while True:
                        codex_bin = _prompt(
                            "Codex binary path (required by codex memory reasoner)",
                            _resolve_codex_bin_prompt_default(existing_env),
                            required=True,
                        )
                        if _validate_codex_bin(codex_bin):
                            break
                        print(f"  Path '{codex_bin}' is not an absolute path to an existing executable.")
                # Symmetric defense-in-depth for the opencode global
                # backend. The global-opencode block above already
                # collected opencode_bin on the normal flow; the gate
                # here exists for the same future-refactor reason
                # codex documents above: a refactor that moved the
                # memory-extraction prompt before the global-backend
                # block could reach this point with opencode_bin
                # still empty. Default-suggestion value differs from
                # codex (empty fallback instead of Homebrew) because
                # opencode has no canonical install location across
                # platforms; an empty default forces the operator to
                # type the path explicitly rather than accept a wrong
                # default.
                if agent_backend == "opencode" and not opencode_bin:
                    while True:
                        which_opencode = shutil.which("opencode") or ""
                        opencode_bin = _prompt(
                            "OpenCode binary path (required by opencode memory reasoner)",
                            existing_env.get("OPENCODE_BIN", which_opencode),
                            required=True,
                        )
                        if _validate_opencode_bin(opencode_bin):
                            break
                        print(f"  Path '{opencode_bin}' is not an absolute path to an existing executable.")
                # Same defense-in-depth shape for the goose global
                # backend: the global-goose block above collects
                # goose_bin on the normal flow; this gate covers a
                # future refactor that reaches the extraction prompts
                # with goose_bin still empty. Defaults mirror the
                # global block (`which goose`, else empty so the
                # operator types the path explicitly).
                if agent_backend == "goose" and not goose_bin:
                    while True:
                        which_goose = shutil.which("goose") or ""
                        goose_bin = _prompt(
                            "Goose binary path (required by goose memory reasoner)",
                            existing_env.get("GOOSE_BIN", which_goose),
                            required=True,
                        )
                        if _validate_goose_bin(goose_bin):
                            break
                        print(f"  Path '{goose_bin}' is not an absolute path to an existing executable.")
                # Same defense-in-depth shape for the claude global
                # backend, completing the four-backend set: the
                # global-claude block above collects claude_bin on
                # the normal flow; this gate covers a future refactor
                # that reaches the extraction prompts with claude_bin
                # still empty. Defaults mirror the global block
                # (`which claude`, else empty so the operator types
                # the path explicitly).
                if agent_backend == "claude" and not claude_bin:
                    while True:
                        which_claude = shutil.which("claude") or ""
                        claude_bin = _prompt(
                            "Claude binary path (required by claude memory reasoner)",
                            existing_env.get("CLAUDE_BIN", which_claude),
                            required=True,
                        )
                        if _validate_claude_bin(claude_bin):
                            break
                        print(f"  Path '{claude_bin}' is not an absolute path to an existing executable.")
                # Extraction timeout is the LLM-call hard cap inside
                # memory_extraction.py:541. Default 10s is too aggressive
                # for production - real extractions routinely take 20-30s
                # and silently abort at the boundary (#345). Operators
                # MUST be able to raise this without hand-editing the
                # generated install.conf, which `sudo make install`
                # rebuilds wholesale on every reinstall.
                while True:
                    memory_extraction_timeout_s = _prompt(
                        "Per-extraction timeout in seconds",
                        existing_env.get("MEMORY_EXTRACTION_TIMEOUT_S", "10"),
                    )
                    if _validate_positive_int(memory_extraction_timeout_s):
                        break
                    print("  Must be a positive integer.")
                # Consolidation candidate count. Zero is a valid
                # kill-switch value (extractor falls back to all-`new`
                # output and the EXISTING FACTS data block is omitted),
                # so use the non-negative validator instead of the
                # positive-only one. Sits inside the extraction guard
                # because consolidation is only meaningful when the
                # Haiku extractor runs.
                while True:
                    memory_consolidation_candidates_n = _prompt(
                        "Consolidation candidate count (0 disables consolidation)",
                        existing_env.get("MEMORY_CONSOLIDATION_CANDIDATES_N", "8"),
                    )
                    if _validate_non_negative_int(memory_consolidation_candidates_n):
                        break
                    print("  Must be a non-negative integer (0 to disable consolidation).")
                # Episode-classifier context-turn count (issue #392).
                # Inline parse-and-range-check rather than calling
                # `_validate_non_negative_int` because that helper only
                # enforces `>= 0` with no upper bound; the 0-10 cap
                # exists so an operator typo (3000 instead of 3) cannot
                # produce a single payload with ~3001 pairs in the
                # PRIOR CONTEXT block that exceeds Haiku's per-call
                # token limit. The lower-bound-plus-ceiling shape is
                # handled inline rather than through a single-use
                # validator helper. The
                # cap is enforced again at load_config (config.py); the
                # wizard inline check exists only to give the operator
                # immediate feedback rather than a delayed SystemExit
                # at next daemon start.
                while True:
                    episode_classifier_context_turns = _prompt(
                        "Episode classifier context turns (number of prior exchanges shown to "
                        "the classifier as background; 0 disables windowing)",
                        existing_env.get("EPISODE_CLASSIFIER_CONTEXT_TURNS", "3"),
                    )
                    try:
                        val = int(episode_classifier_context_turns)
                        if 0 <= val <= 10:
                            break
                    except ValueError:
                        pass
                    print("  Must be 0-10.")
                # Stage-2 episode timeout (issue #385) sits inside the
                # extraction-enabled guard because episodes only fire
                # when stage 1 fires (the has_episode classifier comes
                # from stage 1's output).
                # The episode model is resolved per-user from the
                # registry at extraction time via get_model_for(
                # ModelRole.MEMORY_EPISODE, effective_backend); the
                # wizard prompt for the model was retired with the
                # MEMORY_EPISODE_MODEL env var.
                # Episode timeout has a 10s floor: Haiku's warm-up alone
                # routinely runs several seconds, so a sub-floor timeout
                # would surface as systematic timeouts that mask real
                # model failure as configuration error. Inline check
                # rather than a dedicated validator helper because this
                # is the only call site with a per-field minimum.
                while True:
                    memory_episode_timeout_s = _prompt(
                        "Per-episode timeout in seconds (minimum 10)",
                        existing_env.get("MEMORY_EPISODE_TIMEOUT_S", "120"),
                    )
                    try:
                        if int(memory_episode_timeout_s) >= 10:
                            break
                    except ValueError:
                        pass
                    print("  Must be an integer of at least 10.")
                # Write-time paraphrase-dedup threshold. Sits inside
                # the extraction-enabled guard because the gate only
                # fires in `_store_facts` on the extraction write
                # path. Inline range check rather than a dedicated
                # helper: the [0.3, 1.01] band is unique to this
                # field (lower bound = operator floor for cosine
                # similarity, upper bound permits 1.01 as an
                # unambiguous disable sentinel). The same bound is
                # re-validated in load_config so a hand-edited
                # install.conf still fails fast at next daemon start.
                while True:
                    memory_duplicate_threshold = _prompt(
                        "Paraphrase-dedup threshold (cosine; 1.0 effectively disables, 1.01 unambiguously)",
                        existing_env.get("MEMORY_DUPLICATE_THRESHOLD", "0.9"),
                    )
                    try:
                        val = float(memory_duplicate_threshold)
                        if 0.3 <= val <= 1.01:
                            break
                    except ValueError:
                        pass
                    print("  Must be a number between 0.3 and 1.01.")
        else:
            # Backends without a OneShotReasoner never see the
            # extraction prompts above, and without this line the
            # operator's only signal is silence: retrieval works but
            # facts never accumulate, and nothing says why. One
            # sentence at configure time names the limitation the
            # moment the operator opts into memory.
            print(
                f"  Note: memory extraction is not available on the {agent_backend} "
                "backend; memory will be retrieval-only (no facts are written from "
                "conversations)."
            )
        while True:
            memory_token_budget = _prompt(
                "Memory context token budget per turn",
                existing_env.get("MEMORY_TOKEN_BUDGET", "2000"),
            )
            if _validate_positive_int(memory_token_budget):
                break
            print("  Must be a positive integer.")
        # Search limit caps the retrieval set returned from Mem0/Qdrant
        # before the token-budget filter. Lower values reduce noise but
        # may miss relevant facts; higher values pay an embedding-search
        # cost per turn. Operator-tunable for the same reason as the
        # token budget: workload-specific, regression class #345.
        while True:
            memory_search_limit = _prompt(
                "Memory search result limit per query",
                existing_env.get("MEMORY_SEARCH_LIMIT", "10"),
            )
            if _validate_positive_int(memory_search_limit):
                break
            print("  Must be a positive integer.")
    print()

    # -- External services --
    print("-- External services --")
    perplexity_key = _prompt(
        "Perplexity API key (optional)",
        existing_env.get("PERPLEXITY_API_KEY", ""),
    )
    print()

    # Build the env dict (only include non-empty values).
    # Truly global vars are always written regardless of users.yaml.
    env: dict[str, str] = {
        "TELEGRAM_BOT_TOKEN": bot_token,
        "WEBHOOK_PORT": port,
        "GITHUB_WEBHOOK_SECRET": github_webhook_secret,
        "GENERIC_WEBHOOK_SECRET": generic_webhook_secret,
        "VOICE_ENABLED": str(voice_enabled).lower(),
        "TTS_ENABLED": str(tts_enabled).lower(),
    }
    if legacy_webhook_secret:
        env["WEBHOOK_SECRET"] = legacy_webhook_secret

    # DEFAULT_BACKEND is the global default backend; users.yaml entries
    # can override it per user. Only write non-default values to keep
    # the env file clean (the runtime defaults to claude when the key
    # is absent). The wizard writes the new name only; the runtime keeps
    # a one-release read fallback for a legacy AGENT_BACKEND key.
    if agent_backend != "claude":
        env["DEFAULT_BACKEND"] = agent_backend

    # LLM provider and API key. Written alongside the backend choice
    # so they survive into /etc/kai/env and are not wiped on reinstall.
    # The wizard writes the new DEFAULT_PROVIDER name only; the runtime
    # keeps a one-release read fallback for a legacy LLM_PROVIDER key.
    if llm_provider:
        env["DEFAULT_PROVIDER"] = llm_provider
    if llm_api_key_var and llm_api_key:
        env[llm_api_key_var] = llm_api_key

    # Codex auth mode + optional api_key. Only written on codex
    # installs; the load_config default ("subscription") fires
    # otherwise. OPENAI_API_KEY is emitted only in api_key mode and
    # only if the operator supplied one, mirroring the
    # llm_api_key_var emission above for goose.
    if agent_backend == "codex":
        if codex_auth_mode and codex_auth_mode != "subscription":
            env["CODEX_AUTH_MODE"] = codex_auth_mode
        if codex_api_key:
            env["OPENAI_API_KEY"] = codex_api_key

    # Keys collected for per-user goose entries. Required prompts, so
    # values are non-empty whenever the scan found a keyed provider.
    for peruser_key_var, peruser_key_value in sorted(extra_provider_keys.items()):
        if peruser_key_value:
            env[peruser_key_var] = peruser_key_value

    # Provider keys already stored in the env survive regeneration
    # even when this run's prompts did not fire, so a key collected
    # once is never silently dropped by an unrelated re-run (the env
    # dict is built fresh above; without this pass, only keys a
    # prompt produced this session would be emitted). The one
    # exception is OPENAI_API_KEY on a codex install: the codex auth
    # block owns that var's lifecycle (api_key mode collects it,
    # subscription mode deliberately sheds it), and preservation must
    # not resurrect a key the operator just chose to retire.
    for provider_key_var in sorted(set(PROVIDER_KEY_VARS.values())):
        if provider_key_var in env:
            continue
        if agent_backend == "codex" and provider_key_var == "OPENAI_API_KEY":
            continue
        existing_key_value = existing_env.get(provider_key_var, "")
        if existing_key_value:
            env[provider_key_var] = existing_key_value

    # Persist the wizard-collected codex binary path whenever any
    # codex surface (agent backend or memory reasoner) is in play.
    # Gating on `codex_bin` rather than `agent_backend == "codex"`
    # covers the supported claude+codex / goose+codex memory cases
    # where the codex binary is required solely because the memory
    # reasoner is codex; the second collection block at the memory
    # gate above guarantees `codex_bin` is set on those paths. The
    # same value drives /etc/kai/env's CODEX_BIN and the sudoers
    # SETENV rule so the two can never drift. _cmd_apply's env-var
    # override block keeps working for ad-hoc deploys (sudo
    # CODEX_BIN=... kai install apply) but the wizard path remains
    # the canonical source of truth.
    if codex_bin:
        env["CODEX_BIN"] = codex_bin
    # Same gating shape as codex above: the value persists only when
    # the wizard collected one (truthy opencode_bin), so a claude or
    # goose install does not pollute install.conf with an empty
    # OPENCODE_BIN= line. The single env emission drives both
    # /etc/kai/env's OPENCODE_BIN and the sudoers SETENV rule so the
    # two cannot drift.
    if opencode_bin:
        env["OPENCODE_BIN"] = opencode_bin
    # Same gating shape again for goose: persist only when the wizard
    # collected a value, so non-goose installs do not carry an empty
    # GOOSE_BIN= line. The single emission drives both /etc/kai/env's
    # GOOSE_BIN and the sudoers SETENV rule so the two cannot drift.
    if goose_bin:
        env["GOOSE_BIN"] = goose_bin
    # Same gating shape again for claude: persist only when the wizard
    # collected a value, so installs that never collected one carry no
    # CLAUDE_BIN= line and keep resolving claude via the service PATH.
    # The single emission drives both /etc/kai/env's CLAUDE_BIN and
    # the sudoers SETENV rule so the two cannot drift.
    if claude_bin:
        env["CLAUDE_BIN"] = claude_bin

    # Remove stale renamed keys if present - leaving both the old and
    # new key causes silent confusion (the deprecation warning is
    # suppressed when the new key exists). CLAUDE_MAX_SESSION_HOURS and
    # CLAUDE_IDLE_TIMEOUT are renamed to their AGENT_-prefixed forms;
    # pop the legacy keys on every regenerate so the next /etc/kai/env
    # carries only the canonical forms.
    env.pop("CLAUDE_MODEL", None)
    env.pop("CLAUDE_MAX_SESSION_HOURS", None)
    env.pop("CLAUDE_IDLE_TIMEOUT", None)
    # Retired keys (no canonical replacement); drop lingering values
    # so the regenerated env does not carry settings the runtime no
    # longer reads. Mirrors the `_retired_env_vars` warning map in
    # load_config.
    env.pop("CLAUDE_MAX_CONTEXT_WINDOW", None)
    env.pop("CLAUDE_MAX_BUDGET_USD", None)
    env.pop("BUDGET_CEILING", None)
    env.pop("PR_REVIEW_BUDGET_USD", None)
    env.pop("MEMORY_EXTRACTION_BUDGET_USD", None)
    env.pop("MEMORY_EPISODE_BUDGET_USD", None)
    env.pop("MEMORY_SCOPED_RECALL_ENABLED", None)
    env.pop("MEMORY_RECALL_SHADOW_ENABLED", None)
    # DEFAULT_MODEL / CLAUDE_MODEL are retired from the wizard output.
    # The conversational default now comes from MODEL_REGISTRY's
    # ModelRole.AGENT row for the active backend/provider. Existing
    # hand-edited DEFAULT_MODEL remains loadable, but re-running
    # `make config` migrates back to the registry default.
    env.pop("DEFAULT_MODEL", None)

    # DEFAULT_MODELS_JSON carries the per-role customization overrides
    # captured above (delta-from-defaults: only roles the operator
    # changed appear). load_config parses it as global fallback below
    # per-user `models:` and above the MODEL_REGISTRY default.
    # Suppress the key entirely when the operator accepted every
    # default so /etc/kai/env stays as a delta from the in-tree
    # canonical surface.
    if default_models_override:
        env["DEFAULT_MODELS_JSON"] = json.dumps(default_models_override, sort_keys=True)

    # DEFAULT_TIMEOUT is an inheritable installation default; per-user
    # timeouts in users.yaml override at runtime. Always emitted because
    # the prompt always fires. (The legacy AGENT_TIMEOUT_SECONDS key is
    # migrated to the new name at apply time.)
    env["DEFAULT_TIMEOUT"] = timeout

    # Session lifecycle keys are written delta-from-default (matching
    # the autocompact / effort treatment below) so a default-accepting
    # run keeps /etc/kai/env minimal; load_config supplies the same
    # defaults at runtime. Compare as parsed numbers so inputs like
    # "0.0" or "01800" that pass validation still count as defaults.
    if float(max_session_hours) != 0:
        env["AGENT_MAX_SESSION_HOURS"] = max_session_hours
    if int(idle_timeout) != 1800:
        env["AGENT_IDLE_TIMEOUT"] = idle_timeout

    if int(autocompact_pct) != 0:
        env["CLAUDE_AUTOCOMPACT_PCT"] = autocompact_pct

    # CLAUDE_EFFORT_LEVEL is global (not per-user), so it is considered
    # regardless of users.yaml. Only emit when the operator picked a
    # non-default value to keep install.conf as a delta from defaults
    # rather than a snapshot of every available knob - matches the
    # autocompact_pct treatment immediately above. The wizard's
    # _prompt_choice already validated the value against the same
    # allow-list config.py uses, so no re-validation here.
    if claude_effort_level != "high":
        env["CLAUDE_EFFORT_LEVEL"] = claude_effort_level

    # CODEX_EFFORT_LEVEL: same delta-from-defaults posture. The
    # default is empty (codex's own config rules), so any non-empty
    # value is operator intent worth persisting; empty stays out of
    # install.conf entirely.
    if codex_effort_level:
        env["CODEX_EFFORT_LEVEL"] = codex_effort_level

    # Conditionally add optional values
    if transport == "webhook":
        env["TELEGRAM_TRANSPORT"] = "webhook"
        if webhook_url:
            env["TELEGRAM_WEBHOOK_URL"] = webhook_url
        if tg_webhook_secret:
            env["TELEGRAM_WEBHOOK_SECRET"] = tg_webhook_secret
    if perplexity_key:
        env["PERPLEXITY_API_KEY"] = perplexity_key

    # WORKSPACE_BASE is an inheritable installation default; per-user
    # `workspace_base` in users.yaml overrides at runtime. Always
    # emitted when set, regardless of users.yaml presence.
    if workspace_base:
        env["WORKSPACE_BASE"] = workspace_base

    # PR_REVIEW_COOLDOWN is a global resource control. Always written
    # when non-default because any user can drive reviews via users.yaml
    # or /github reviews on|off.
    if pr_review_cooldown != "300":
        env["PR_REVIEW_COOLDOWN"] = pr_review_cooldown

    # ALLOWED_WORKSPACES is an inheritable installation default.
    if allowed_workspaces:
        env["ALLOWED_WORKSPACES"] = allowed_workspaces

    # Review subprocess resource limits. They apply globally to any
    # review, regardless of which user opted in.
    if pr_review_timeout_s != "900":
        env["PR_REVIEW_TIMEOUT_S"] = pr_review_timeout_s

    # Semantic memory: global env vars (per-user partitioning is runtime).
    # Toggling memory_enabled from true back to false correctly drops
    # MEMORY_* keys here, so the next /etc/kai/env reflects the new state.
    # Numeric comparisons (not string) so "0.010" or "2000.0" are not
    # treated as non-default and spuriously written.
    if memory_enabled:
        env["MEMORY_ENABLED"] = "true"
        if memory_extraction_enabled:
            env["MEMORY_EXTRACTION_ENABLED"] = "true"
            # MEMORY_REASONER_BACKEND, MEMORY_EXTRACTION_MODEL, and
            # MEMORY_EPISODE_MODEL are deprecated (issue #515). Memory
            # reasoner and model both derive per-user from each user's
            # effective agent_backend via the model registry; there is
            # no env-var surface to emit. Legacy values still parse
            # as deprecation warnings at load_config time so existing
            # installs survive one reboot before this wizard regenerates
            # /etc/kai/env without those keys.
            # Extraction-only tunable. Numeric compare so "10" vs "10 "
            # or "010" do not produce a spurious env entry equal to
            # the dataclass default.
            if int(memory_extraction_timeout_s) != 10:
                env["MEMORY_EXTRACTION_TIMEOUT_S"] = memory_extraction_timeout_s
            # Consolidation candidate count. Same gate as the other
            # extraction tunables because the field is only consulted
            # by the Haiku extraction path; numeric compare so "08" or
            # "8 " do not produce a spurious entry equal to the
            # dataclass default.
            if int(memory_consolidation_candidates_n) != 8:
                env["MEMORY_CONSOLIDATION_CANDIDATES_N"] = memory_consolidation_candidates_n
            # Episode-classifier context-turn count (issue #392). Same
            # delta-from-default discipline as the surrounding memory
            # tunables. Numeric compare so "03" or "3 " do not produce
            # a spurious entry equal to the dataclass default. No
            # backend gate needed here: the prompt is already nested
            # inside `if agent_backend == "claude":` above, so on
            # non-claude this value stays at its pre-init "3" and the
            # `!= 3` check suppresses the emission.
            if int(episode_classifier_context_turns) != 3:
                env["EPISODE_CLASSIFIER_CONTEXT_TURNS"] = episode_classifier_context_turns
            # Stage-2 episode tunables (issue #385). Same delta-from-default
            # discipline as the stage-1 keys above. MEMORY_EPISODE_MODEL
            # is no longer emitted: the episode model resolves per-user
            # from the registry at extraction time via get_model_for
            # (issue #515).
            if int(memory_episode_timeout_s) != 120:
                env["MEMORY_EPISODE_TIMEOUT_S"] = memory_episode_timeout_s
            # Paraphrase-dedup threshold. Same delta-from-default gate
            # as the other extraction tunables; numeric compare so
            # "0.90" or "0.9 " do not produce a spurious entry equal
            # to the dataclass default.
            if float(memory_duplicate_threshold) != 0.9:
                env["MEMORY_DUPLICATE_THRESHOLD"] = memory_duplicate_threshold
        if int(memory_token_budget) != 2000:
            env["MEMORY_TOKEN_BUDGET"] = memory_token_budget
        # Search limit applies to retrieval (read path), not extraction
        # (write path), so it sits outside the extraction guard but
        # inside the memory_enabled guard. Disabling memory entirely
        # naturally drops this key via the surrounding if.
        if int(memory_search_limit) != 10:
            env["MEMORY_SEARCH_LIMIT"] = memory_search_limit

    # Drop the three deprecated memory env vars on every wizard run
    # (issue #515). Reasoner and model both derive per-user from each
    # user's effective agent_backend via the registry; the env vars
    # are no longer parsed beyond a one-shot deprecation warning at
    # load_config time. Popping unconditionally keeps install.conf
    # from carrying the keys forward across reinstall regardless of
    # which agent backend the operator selected.
    env.pop("MEMORY_REASONER_BACKEND", None)
    env.pop("MEMORY_EXTRACTION_MODEL", None)
    env.pop("MEMORY_EPISODE_MODEL", None)

    # Drop stale extraction keys when the agent backend has no memory
    # reasoner (i.e. is not in ONESHOT_REASONER_BACKENDS). A user who
    # flips backend from a reasoner-capable backend to one without a
    # reasoner should not see lingering extraction config that would
    # be silently ignored at runtime. Mirrors the wizard-side
    # extraction-prompt gate above so a value prompted-and-accepted on
    # an eligible backend persists end to end; a value prompted on a
    # then-ineligible backend gets dropped here.
    if agent_backend not in ONESHOT_REASONER_BACKENDS:
        env.pop("MEMORY_EXTRACTION_ENABLED", None)
        env.pop("MEMORY_EXTRACTION_TIMEOUT_S", None)
        env.pop("MEMORY_CONSOLIDATION_CANDIDATES_N", None)
        # Episode-classifier window key (issue #392). Same lifecycle
        # as the other stage-1 extraction tunables: only consulted on
        # backends with a reasoner, so leaving a stale value here
        # after a supported→goose flip would be misleading.
        env.pop("EPISODE_CLASSIFIER_CONTEXT_TURNS", None)
        # Episode keys follow the same lifecycle: stage 2 only fires
        # when stage 1 fires, and stage 1 silently skips on backends
        # without a reasoner. Leaving these in the env file would
        # mislead an operator who flips backend.
        env.pop("MEMORY_EPISODE_TIMEOUT_S", None)
        # Paraphrase-dedup threshold is consulted by `_store_facts`,
        # which only runs under the Haiku extraction path. Same
        # lifecycle as the other extraction keys: dropped on a
        # claude→non-claude backend flip so a stale value does not
        # mislead an operator after the flip.
        env.pop("MEMORY_DUPLICATE_THRESHOLD", None)

    # Build and write install.conf
    conf = {
        "version": _CONF_VERSION,
        "deployment_mode": deployment_mode,
        "install_dir": install_dir,
        "data_dir": data_dir,
        "service_user": service_user,
        "platform": platform,
        "env": env,
    }

    # Pending first-install handoff: `_cmd_apply` reads this top-level
    # key to locate the staging file written above and copies it to
    # `/etc/kai/users.yaml`. The key sits alongside `version`/
    # `install_dir`/`env` (NOT inside `env`) because it is installer
    # metadata; routing it through `env` would surface it in
    # `/etc/kai/env` and pollute runtime daemon configuration. The
    # presence of this key is also the only signal apply uses: a stale
    # staging file without a matching key in install.conf is ignored.
    # Single-user mode never sets this key because the wizard writes
    # directly to the XDG users.yaml destination; the apply guard
    # below double-checks by requiring deployment_mode == "protected"
    # before honoring any recorded staging path.
    if users_yaml_staging_path:
        conf["users_yaml_staging_path"] = users_yaml_staging_path

    INSTALL_CONF.write_text(json.dumps(conf, indent=2) + "\n")
    # Restrict permissions since the file contains secrets (bot token, webhook secret)
    os.chmod(INSTALL_CONF, 0o600)
    print(f"Configuration written to {INSTALL_CONF}")
    if deployment_mode == "protected":
        print("Review the file, then run: sudo python -m kai install apply")
    else:
        # In single-user mode, also write the local .env so the
        # daemon can find runtime env config via `load_dotenv`
        # (protected mode reads `/etc/kai/env` via `_read_protected_file`
        # which is not available without root). `_generate_env_file`
        # produces the same KEY=VALUE shape `/etc/kai/env` would carry;
        # we restrict the mode to 0600 because the file holds the same
        # secrets (bot token, webhook secret).
        env_path = PROJECT_ROOT / ".env"
        env_path.write_text(_generate_env_file(env))
        os.chmod(env_path, 0o600)
        print(f"Wrote runtime env to {env_path}")
        print(f"users.yaml is at {users_yaml_path}")
        print()
        print("Single-user mode does not require 'sudo make install'.")
        print("Start the daemon with: make run")


# ── Apply subcommand ─────────────────────────────────────────────────


def _parse_workspaces(env: dict[str, str]) -> list[Path]:
    """Parse ALLOWED_WORKSPACES from an env dict into a list of Paths."""
    raw = env.get("ALLOWED_WORKSPACES", "")
    return [Path(ws.strip()) for ws in raw.split(",") if ws.strip()]


def _file_checksum(path: Path) -> str:
    """Return the SHA-256 hex digest of a file, or empty string if missing."""
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _src_checksum(src_dir: Path) -> str:
    """
    Return a SHA-256 digest representing the contents of all .py files under src_dir.

    Walks the directory tree in sorted order (for determinism), feeding each
    file's relative path and content into a rolling hash. Returns an empty
    string if the directory doesn't exist. Used alongside _file_checksum() on
    pyproject.toml to detect source-only changes that require a pip reinstall.
    """
    if not src_dir.is_dir():
        return ""
    h = hashlib.sha256()
    # Sort for deterministic ordering across platforms and filesystems
    for py_file in sorted(src_dir.rglob("*.py")):
        # Include the relative path so renames/moves change the hash
        h.update(str(py_file.relative_to(src_dir)).encode())
        h.update(py_file.read_bytes())
    return h.hexdigest()


def _set_ownership(path: Path, uid: int, gid: int, recursive: bool = False) -> None:
    """
    Set ownership of a path, optionally recursing into directories.

    Uses lchown for symlinks so ownership is set on the symlink inode
    itself, not the target file. Without this, chowning a symlink to
    root would silently chown its target to root too.

    Args:
        path: File or directory to chown.
        uid: User ID for the new owner.
        gid: Group ID for the new group.
        recursive: If True, walk the directory tree and chown everything.
    """
    # lchown for symlinks to avoid following them to their targets.
    if path.is_symlink():
        os.lchown(path, uid, gid)
    else:
        os.chown(path, uid, gid)
    if recursive and path.is_dir() and not path.is_symlink():
        for child in path.rglob("*"):
            if child.is_symlink():
                os.lchown(child, uid, gid)
            else:
                os.chown(child, uid, gid)


def _set_private_user_tree_modes(path: Path) -> None:
    """
    Make a user-owned Kai data subtree private without following symlinks.

    Directories become 0700; regular files become 0600. Symlinks are skipped:
    chmod(2) would affect the target on most platforms, which is not safe for
    a tree that may contain stale or operator-created entries.
    """
    if path.is_symlink():
        return
    if path.is_dir():
        os.chmod(path, _PRIVATE_USER_DIR_MODE)
        for root, dirs, files in os.walk(path, followlinks=False):
            root_path = Path(root)
            if not root_path.is_symlink():
                os.chmod(root_path, _PRIVATE_USER_DIR_MODE)
            for dirname in dirs:
                child = root_path / dirname
                if not child.is_symlink():
                    os.chmod(child, _PRIVATE_USER_DIR_MODE)
            for filename in files:
                child = root_path / filename
                if not child.is_symlink():
                    os.chmod(child, _PRIVATE_USER_FILE_MODE)
    elif path.exists():
        os.chmod(path, _PRIVATE_USER_FILE_MODE)


def _copy_tree(
    src: Path,
    dst: Path,
    excludes: set[str] | None = None,
    *,
    replace: bool = False,
) -> None:
    """
    Copy a directory tree, excluding patterns like __pycache__.

    By default, uses a merge-based approach: walks the source tree and copies
    each file individually, creating destination directories as needed. Files
    at the destination that don't exist in the source are left untouched.
    Callers for generated trees such as the installed Python source can pass
    ``replace=True`` so deleted source files do not survive an upgrade and get
    packaged into the venv as obsolete modules.

    The previous implementation used shutil.rmtree(dst) before copytree(),
    which destroyed ALL destination contents including runtime data that the
    excludes were meant to protect. See issue #143.

    Symlink handling: shutil.copy2 follows symlinks and copies content. There
    is no symlink-recreation branch; the previous one existed solely to
    preserve the legacy IDENTITY.md / CLAUDE.md symlink pair under the
    install tree during the source-to-install copy, and was removed when
    that layout was retired (issues #442, #447). Tracked symlinks under
    templates/ or src/ would now be silently dereferenced rather than
    recreated. This is intentional: cross-platform symlink tracking
    (Windows) is fraught, and the source tree no longer contains any
    symlinks. Do not "restore" the branch as an apparent bug without
    first establishing a use case that needs cross-platform tracked
    symlinks.

    Args:
        src: Source directory.
        dst: Destination directory (created if it doesn't exist).
        excludes: Set of glob patterns to exclude (e.g., {"__pycache__", "*.pyc"}).
        replace: Remove the destination tree before copying.
    """
    if replace:
        if dst.is_symlink():
            dst.unlink()
        elif dst.exists():
            shutil.rmtree(dst)

    ignore_fn = shutil.ignore_patterns(*(excludes or set()))

    for src_dir, dirs, files in os.walk(src):
        rel = Path(src_dir).relative_to(src)
        # Check which names should be excluded at this level
        ignored = set(ignore_fn(str(src_dir), dirs + files))
        # Filter directories so os.walk doesn't descend into excluded ones
        dirs[:] = [d for d in dirs if d not in ignored]

        dst_dir = dst / rel
        dst_dir.mkdir(parents=True, exist_ok=True)

        for f in files:
            if f not in ignored:
                src_file = Path(src_dir) / f
                dst_file = dst_dir / f
                shutil.copy2(src_file, dst_file)


def _generate_env_file(env: dict[str, str]) -> str:
    """
    Generate the contents of /etc/kai/env from the env dict.

    Produces a key=value file with one variable per line, suitable for
    parsing by _read_protected_file() in config.py.

    Args:
        env: Dict of environment variable names to values.

    Returns:
        The file contents as a string.
    """
    lines = ["# Kai environment - managed by 'python -m kai install apply'"]
    lines.append("# Do not edit manually; re-run install config + apply instead.")
    lines.append("")
    for key, value in sorted(env.items()):
        # Quote values to handle spaces and special characters. Escape
        # embedded backslashes and double quotes so the file parses correctly.
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'{key}="{escaped}"')
    lines.append("")
    return "\n".join(lines)


def _generate_users_yaml(
    telegram_id: str,
    name: str,
    os_user: str | None = None,
    home_workspace: str | None = None,
) -> str:
    """
    Generate a minimal users.yaml with a single admin entry.

    Uses string formatting rather than yaml.dump() to keep the output
    deterministic and human-readable (consistent indentation, field order,
    comment header). All embedded values are pre-validated: telegram_id is
    a positive integer string, name passes _validate_display_name(),
    os_user passes _validate_os_user(), and home_workspace is serialized
    via yaml.dump() to handle arbitrary path characters safely.

    Args:
        telegram_id: The admin user's Telegram ID (validated positive int string).
        name: Display name for the admin user.
        os_user: Optional OS account for subprocess isolation.
        home_workspace: Optional home workspace path (absolute).

    Returns:
        The YAML file contents as a string.
    """

    # yaml.dump() appends a document end marker ("\n...\n") after
    # plain scalars (e.g., "alice\n...\n") and just "\n" after quoted
    # ones (e.g., "'yes'\n"). Strip the marker and trailing newlines
    # so we get a bare scalar for embedding in a larger document.
    # The removesuffix("...") is safe because the marker is always
    # on its own line, separated by "\n" from any "..." in the value.
    def _yaml_scalar(value: str) -> str:
        return yaml.dump(value, default_flow_style=True).rstrip("\n").removesuffix("...").rstrip("\n")

    lines = [
        "# Kai user configuration - generated by 'python -m kai install config'",
        "# See README (Multi-User section) for all available fields.",
        "",
        "users:",
        f"  - telegram_id: {telegram_id}",
        # Use yaml.dump for string scalars to handle YAML 1.1 boolean
        # keywords (yes/no/true/false/on/off) and null, which would
        # round-trip as non-string types through yaml.safe_load.
        f"    name: {_yaml_scalar(name)}",
        "    role: admin",
        "    allowed_services: []",
    ]
    if os_user:
        lines.append(f"    os_user: {_yaml_scalar(os_user)}")
    if home_workspace:
        lines.append(f"    home_workspace: {_yaml_scalar(home_workspace)}")
    lines.append("")  # trailing newline
    return "\n".join(lines)


def _collect_os_users_from_yaml(users_yaml_path: str | Path) -> list[str]:
    """
    Read users.yaml and return distinct, validated os_user values.

    The runtime (pool.py / claude.py) spawns the inner Claude as each user's
    `os_user` via `sudo -u`. That requires a matching sudoers rule for every
    distinct os_user. Without this loader, additional users had to be
    hand-added with visudo only to be wiped by the next `make install`.

    The reader is intentionally lightweight: it does not validate roles,
    models, or providers (config._load_user_configs already does that at
    runtime). Install only needs the os_user strings.

    Behavior:
        - Missing file → []   (first install: users.yaml may not exist yet)
        - Empty / non-dict / no `users:` key → []
        - Malformed YAML → raises (yaml.YAMLError); install should fail loudly
          rather than silently emit no per-user rules
        - Non-string / empty / whitespace-only os_user values are skipped
        - Non-empty values that fail _validate_os_user (e.g. contain `)`,
          newline, whitespace) raise ValueError. Without this check, a
          crafted users.yaml could inject arbitrary sudoers directives —
          /etc/sudoers.d files are loaded directly and visudo only validates
          syntax, not authorial intent. Hard fail rather than silent skip
          so the operator sees the bad entry rather than getting a
          half-functional install.
        - Duplicate os_user values are deduplicated, preserving first-seen order

    Args:
        users_yaml_path: Path to users.yaml (typically /etc/kai/users.yaml).

    Returns:
        Ordered list of unique, validated os_user strings.

    Raises:
        ValueError: If any non-empty os_user value fails username validation.
        yaml.YAMLError: If the file exists but cannot be parsed.
    """
    path = Path(users_yaml_path)

    # try/except instead of exists() + read_text() avoids a TOCTOU race:
    # the file could be deleted between the two calls. Functionally
    # identical here (root-only install path), but the idiom is cleaner.
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []

    if not text.strip():
        return []

    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        return []

    users = data.get("users")
    if not isinstance(users, list):
        return []

    seen: set[str] = set()
    result: list[str] = []
    for entry in users:
        if not isinstance(entry, dict):
            continue
        os_user = entry.get("os_user")
        # PyYAML may parse "yes"/"no"/numeric os_user values as bool/int;
        # only strings are valid OS usernames here.
        if not isinstance(os_user, str):
            continue
        normalized = os_user.strip()
        # Empty / whitespace-only is treated as "no os_user set" (legitimate;
        # that user runs as the service user). Skip without raising.
        if not normalized:
            continue
        # Anything else must be a valid username before we let it near the
        # sudoers writer. See the docstring above for the security rationale.
        if not _validate_os_user(normalized):
            raise ValueError(f"Invalid os_user {normalized!r} in {path}: must match {_OS_USER_RE.pattern}")
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _collect_backends_from_yaml(users_yaml_path: str | Path) -> set[str]:
    """
    Read users.yaml and return the distinct per-user backend values.

    Sibling of `_collect_os_users_from_yaml` with the same lightweight
    posture: the installer only needs the backend names to scope the
    missing-binary backstop in `_apply_sudoers` to backends the install
    actually uses; full validation of backend values happens at runtime
    in config's users.yaml loader. Unknown or misspelled values pass
    through as-is and are ignored by the caller's intersection with the
    known-binary map, so a typo in users.yaml cannot break the install
    (the runtime surfaces it on the user's first message instead).

    Behavior mirrors the os_user reader: missing file, empty file,
    non-dict document, or missing `users:` list all yield an empty set;
    malformed YAML raises so the install fails loudly.
    """
    path = Path(users_yaml_path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return set()
    if not text.strip():
        return set()
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        return set()
    users = data.get("users")
    if not isinstance(users, list):
        return set()
    backends: set[str] = set()
    for entry in users:
        if not isinstance(entry, dict):
            continue
        backend = _entry_backend(entry)
        # PyYAML may parse unquoted scalars as non-strings; only string
        # values can name a backend.
        if not isinstance(backend, str):
            continue
        # Normalize with .strip().lower() to match the runtime loader
        # (`_load_user_configs` lowercases before validating). A mixed-
        # case `backend: Goose` routes the user to goose at
        # runtime; without lowering here, `_apply_goose_config`'s
        # `"goose" in <set>` membership check would miss it and skip
        # the per-user goose config deployment.
        normalized = backend.strip().lower()
        if normalized:
            backends.add(normalized)
    return backends


def _collect_goose_os_users_from_yaml(
    users_yaml_path: str | Path,
    agent_backend: str,
) -> list[str]:
    """
    Read users.yaml and return distinct, validated os_user values of
    goose-backed users.

    A user is goose-backed when their entry's `backend` is
    "goose", or when the entry carries no per-user backend and the
    install's global backend (`agent_backend`) is goose - the same
    inheritance contract the runtime applies. `_apply_goose_config`
    deploys the goose config template into each such user's home:
    the per-user `goose acp` spawn runs under `sudo -H`, so goose
    resolves `~/.config/goose/config.yaml` beneath the TARGET user's
    home, where the service-user deploy is invisible.

    Same lightweight posture and failure behavior as
    `_collect_os_users_from_yaml`: missing file, empty file, non-dict
    document, or missing `users:` list yield an empty list; malformed
    YAML raises; non-string / empty os_user values are skipped (those
    users run as the service user, which gets its own deploy
    unconditionally); os_user values failing username validation
    raise ValueError before any filesystem path is built from them.
    """
    path = Path(users_yaml_path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    if not text.strip():
        return []
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        return []
    users = data.get("users")
    if not isinstance(users, list):
        return []
    seen: set[str] = set()
    result: list[str] = []
    for entry in users:
        if not isinstance(entry, dict):
            continue
        backend = _entry_backend(entry)
        # Non-string / empty per-user values mean "inherit the global
        # backend" (PyYAML may parse unquoted scalars as non-strings;
        # the runtime loader rejects those separately). Lowercase the
        # per-user value to match the runtime loader so a mixed-case
        # `backend: Goose` entry's os_user still receives the
        # goose config deploy. The global `agent_backend` arg is
        # already normalized by the caller (_cmd_apply).
        if isinstance(backend, str) and backend.strip():
            effective = backend.strip().lower()
        else:
            effective = agent_backend
        if effective != "goose":
            continue
        os_user = entry.get("os_user")
        if not isinstance(os_user, str):
            continue
        normalized = os_user.strip()
        if not normalized:
            continue
        if not _validate_os_user(normalized):
            raise ValueError(f"Invalid os_user {normalized!r} in {path}: must match {_OS_USER_RE.pattern}")
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _build_codex_login_reminder(
    env: dict[str, str],
    service_user: str,
    users_yaml_path: str | Path | None = None,
) -> str | None:
    """Compose the post-install codex subscription-auth reminder.

    Codex CLI auth state lives under the spawning user's home directory
    (~<os_user>/.codex/auth.json). The codex reasoner spawns codex
    per-user via `sudo -u <os_user>`, so subscription auth must be
    completed AS each target user before the first call; a service-
    user-only login lands the token in the wrong home and every
    cross-user spawn fails.

    Returns a generic reminder string when the global backend is codex
    AND auth mode is subscription, else None. Does NOT enumerate
    per-user os_users from users.yaml: that path was a leaky abstraction
    where operator-managed users.yaml content drove install messaging.
    Mixed-backend installs (global non-codex, per-user codex via
    users.yaml) receive no reminder; the operator is responsible for
    installing and authenticating codex out-of-band for those users.

    `service_user` and `users_yaml_path` are retained on the signature
    for backwards compatibility with the existing caller / tests, but
    are no longer read.
    """
    default_backend = (
        _resolve_renamed_key(
            env.get, new_key="DEFAULT_BACKEND", legacy_keys=["AGENT_BACKEND"], context="install.conf", default="claude"
        )
        or "claude"
    )
    if default_backend != "codex":
        return None
    if env.get("CODEX_AUTH_MODE", "subscription") != "subscription":
        return None
    return (
        "Codex subscription auth required:\n"
        "  Log in to codex as the target os_user before the first message:\n"
        "    <os_user> ~$ codex login\n"
        "  Run as the os_user themselves, not via sudo from another account.\n"
        "  If users.yaml has per-user `backend: codex` entries with\n"
        "  different os_users, log in as each of those too."
    )


def _collect_user_memory_owners(users_yaml_path: str | Path) -> list[tuple[int, str | None]]:
    """
    Read users.yaml and return (telegram_id, os_user) tuples.

    Used by the install migration to know which `memory/<chat_id>/`
    subdirectories to pre-create and which OS user each one should be
    chowned to. An os_user of None means that user's inner Claude runs
    as the service user - their memory subdir stays service-owned.

    Mirrors the behavior of _collect_os_users_from_yaml: silently
    returns [] on missing / empty / malformed-top-level files, raises
    on true YAML parse errors, and hard-fails on a non-empty os_user
    that fails _validate_os_user (same sudoers-injection rationale).

    First-seen order is preserved so that callers who care about the
    "primary operator" can take tuples[0] deterministically. Entries
    with a non-int or missing telegram_id are skipped, not raised, to
    match the loose validation elsewhere in this module (a bad yaml
    surfaces later as a runtime failure with a clearer message).

    Args:
        users_yaml_path: Path to users.yaml (typically /etc/kai/users.yaml).

    Returns:
        List of (telegram_id, os_user_or_None) tuples in yaml order.

    Raises:
        ValueError: If any non-empty os_user value fails username validation.
        yaml.YAMLError: If the file exists but cannot be parsed.
    """
    path = Path(users_yaml_path)

    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []

    if not text.strip():
        return []

    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        return []

    users = data.get("users")
    if not isinstance(users, list):
        return []

    result: list[tuple[int, str | None]] = []
    for entry in users:
        if not isinstance(entry, dict):
            continue
        telegram_id = entry.get("telegram_id")
        if not isinstance(telegram_id, int):
            # Users without a valid telegram_id cannot map to a chat_id
            # directory. Skip silently - config._load_user_configs will
            # surface the real validation error at runtime.
            continue
        os_user_raw = entry.get("os_user")
        os_user: str | None
        if isinstance(os_user_raw, str) and os_user_raw.strip():
            os_user = os_user_raw.strip()
            if not _validate_os_user(os_user):
                raise ValueError(f"Invalid os_user {os_user!r} in {path}: must match {_OS_USER_RE.pattern}")
        else:
            os_user = None
        result.append((telegram_id, os_user))
    return result


def _collect_user_home_overrides(users_yaml_path: str | Path) -> dict[int, Path]:
    """
    Read users.yaml and return a chat_id -> home_workspace override mapping.

    Used by the install migration to decide which `home/<chat_id>/`
    subdirectories to skip pre-creating: a user who pinned an explicit
    home_workspace path is opting out of the Kai-managed per-user
    directory (#353), so the installer should not provision one for
    them. Their override path is the operator's responsibility.

    Only entries with a string `home_workspace` field are included.
    Missing / null / empty / non-string values are silently skipped -
    those users get the default per-user directory provisioned.

    Mirrors the loose validation of _collect_user_memory_owners:
    silent return on missing / empty / malformed-top-level files,
    no exceptions for malformed individual entries.

    Args:
        users_yaml_path: Path to users.yaml (typically /etc/kai/users.yaml).

    Returns:
        Mapping of telegram_id (chat_id) to absolute Path of the override.
        Empty dict if the file is missing, empty, or has no overrides.
    """
    path = Path(users_yaml_path)

    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}

    if not text.strip():
        return {}

    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        return {}

    users = data.get("users")
    if not isinstance(users, list):
        return {}

    result: dict[int, Path] = {}
    for entry in users:
        if not isinstance(entry, dict):
            continue
        telegram_id = entry.get("telegram_id")
        if not isinstance(telegram_id, int):
            continue
        home_ws_raw = entry.get("home_workspace")
        if not isinstance(home_ws_raw, str):
            continue
        home_ws = home_ws_raw.strip()
        if not home_ws:
            continue
        # expanduser on the install host is fine: paths in users.yaml
        # are admin-authored and are expected to refer to the install
        # host's filesystem. Same expansion as config.py's loader.
        result[telegram_id] = Path(home_ws).expanduser().resolve()
    return result


def _generate_sudoers(
    service_user: str,
    os_users: Iterable[str] = (),
    claude_bin: str | None = None,
    codex_bin: str | None = None,
    opencode_bin: str | None = None,
    goose_bin: str | None = None,
) -> str:
    """
    Generate sudoers rules for the service user to read protected config files.

    The rules allow passwordless `sudo cat` on specific files only. This is
    validated with `visudo -cf` before being written to /etc/sudoers.d/.

    Uses shutil.which() to resolve the actual paths of `cat` and `tee`,
    since they live at /bin/ on macOS but /usr/bin/ on many Linux distros.
    Falls back to /bin/cat and /usr/bin/tee if the binaries aren't found
    in the current PATH (e.g., when running in a minimal environment).

    Per-user `(target_user) SETENV: NOPASSWD:` rules for the claude, codex,
    opencode, and goose binaries (plus a `NOPASSWD: /bin/kill` rule for the
    cross-user kill escalation) are emitted for every distinct `os_user`
    value in users.yaml. Users matching `service_user` are skipped (the
    runtime detects self-sudo via resolve_claude_user() and spawns the
    agent directly without sudo). The claude rule defaults to the
    service user's `~/.local/bin/claude` (the native installer
    location); operators with a custom install location (e.g. the
    Homebrew cask) override via CLAUDE_BIN. The codex binary path
    defaults to the first executable common install location; unusual
    installs override with the CODEX_BIN env var at install time. The
    opencode rule uses the
    service user's `~/.local/bin/opencode` path (where the upstream
    installer drops the binary by default); operators with a custom
    install location can override via OPENCODE_BIN. The goose rule
    defaults to /opt/homebrew/bin/goose (the Homebrew cask location)
    with the same GOOSE_BIN override shape.

    Args:
        service_user: The OS username that runs the Kai service.
        os_users: Distinct os_user values from users.yaml. Self-sudo entries
            (matching service_user) and duplicates are dropped.

    Returns:
        The sudoers file contents as a string.
    """
    # Resolve actual binary paths. macOS: /bin/cat, /usr/bin/tee.
    # Many Linux distros: /usr/bin/cat, /usr/bin/tee.
    cat_path = shutil.which("cat") or "/bin/cat"
    tee_path = shutil.which("tee") or "/usr/bin/tee"

    rules = textwrap.dedent(f"""\
        # Kai - allow service user to read protected config files.
        # Managed by 'python -m kai install apply'. Do not edit manually.
        {service_user} ALL=(root) NOPASSWD: {cat_path} /etc/kai/env
        {service_user} ALL=(root) NOPASSWD: {cat_path} /etc/kai/services.yaml
        {service_user} ALL=(root) NOPASSWD: {cat_path} /etc/kai/users.yaml
        {service_user} ALL=(root) NOPASSWD: {cat_path} /etc/kai/workspaces.yaml
        {service_user} ALL=(root) NOPASSWD: {cat_path} /etc/kai/memory-projects.yaml
        {service_user} ALL=(root) NOPASSWD: {cat_path} /etc/kai/backends.yaml
        {service_user} ALL=(root) NOPASSWD: {cat_path} /etc/kai/totp.secret
        {service_user} ALL=(root) NOPASSWD: {cat_path} /etc/kai/totp.attempts
        {service_user} ALL=(root) NOPASSWD: {tee_path} /etc/kai/totp.attempts
    """)

    # Yaml-derived os_users only. Drop self-sudo entries: pool.py uses
    # resolve_claude_user() to skip the sudo wrapper entirely when the
    # target matches the service user, so a rule would be dead code
    # (and slightly misleading to future readers).
    #
    # Defense-in-depth: validate every candidate before interpolating
    # into the sudoers template. `_collect_os_users_from_yaml` already
    # validates, but the guard stays here so any future caller cannot
    # bypass it. A `)` or newline in an unvalidated username would let
    # an attacker with control of the source inject arbitrary sudoers
    # directives.
    target_users: list[str] = []
    seen: set[str] = set()
    for candidate in os_users:
        if not candidate or candidate == service_user or candidate in seen:
            continue
        if not _validate_os_user(candidate):
            raise ValueError(f"Invalid sudoers target user {candidate!r}: must match {_OS_USER_RE.pattern}")
        seen.add(candidate)
        target_users.append(candidate)

    if target_users:
        # Anchor the rule path to the wizard-collected CLAUDE_BIN when
        # present, else the default resolver's service-user native
        # install / Homebrew fallback.
        # The runtime spawn uses the same precedence (claude.py reads
        # CLAUDE_BIN, else tries the same native/Homebrew fallbacks
        # before spawning bare `claude`), so the rule references the
        # same binary the bot invokes. We deliberately do NOT call
        # shutil.which here: resolving against whatever
        # PATH root has when `sudo make install` runs can pick up any
        # user's `~/.local/bin/claude` that happens to be on PATH at
        # install time, baking the wrong path into the rule and
        # breaking the bot's sudo dispatch.
        svc_home = _user_home(service_user)
        claude_bin_resolved = claude_bin or _resolve_default_claude_bin(service_user)
        # Codex binary path is now threaded as an argument. The wizard
        # prompts for and persists the value in install.conf; _cmd_apply
        # passes it to _apply_sudoers which passes it here. When a user
        # later switches users.yaml to codex without re-running the
        # wizard, resolve a common absolute Codex install path that the
        # runtime uses too. That keeps sudoers and runtime in sync while
        # still letting CODEX_BIN override unusual installs explicitly.
        codex_bin_resolved = codex_bin or _resolve_default_codex_bin()
        # OpenCode binary path. Upstream's install script drops the
        # binary under the SERVICE user's ~/.local/bin/opencode by
        # default, so the same `svc_home` anchoring applies as the
        # claude rule above. Operators with a custom install location
        # override via OPENCODE_BIN (the wizard prompts for and
        # persists the value the same way CODEX_BIN is handled).
        opencode_bin_resolved = opencode_bin or f"{svc_home}/.local/bin/opencode"
        # Goose binary path. Homebrew's block-goose-cli cask drops the
        # binary under /opt/homebrew/bin/goose on macOS; the wizard
        # prompts for and persists GOOSE_BIN the same way CODEX_BIN
        # and OPENCODE_BIN are handled, so the fallback fires only on
        # a first-time install where the wizard has not run yet.
        goose_bin_resolved = goose_bin or "/opt/homebrew/bin/goose"
        # kill(1) for the cross-user kill escalation (#456). The bot
        # runs `sudo -n -u <target> /bin/kill -<sig> <pid>` against
        # the inner claude grandchild because POSIX signal permissions
        # prevent the service user from signaling a target-user
        # process directly. The runtime invocation pins the absolute
        # path /bin/kill so sudo's secure_path resolution cannot
        # silently pick a different binary than the rule names; the
        # sudoers rule must match the runtime path exactly. /bin/kill
        # is canonical on macOS and on Linux with merged-usr
        # (/bin -> /usr/bin); on older Linux without merged-usr,
        # /bin/kill is a real binary and behaves the same for the
        # `kill -<sig> <pid>` calls we use here. Same anchoring
        # rationale as the claude_bin fix in PR #455.
        kill_bin = "/bin/kill"
        # SETENV: allows the service user to pass env vars (e.g.,
        # KAI_WEBHOOK_SECRET, provider API keys) through sudo to the
        # agent binaries. Scoped to the agent rules; the kill rule
        # does not need SETENV (kill ignores env entirely), and the
        # cat/tee config-read rules above remain locked down.
        #
        # Scope note for the generated rules below (also surfaced
        # as an inline comment in /etc/sudoers.d/kai itself so
        # operators reading the file can see this is deliberate):
        # the kill rule allows the service user to run /bin/kill
        # as <target> with ANY arguments, which means it can
        # signal any <target>-owned process - not just the inner
        # claude/codex grandchild. A PID-locked rule would be
        # tighter but sudoers argument matching is not safe per
        # the sudo(8) manual (Defaults entries warn against it; a
        # crafted argument can bypass the pattern). In practice
        # the scope delta is zero: the claude/codex rules above
        # already grant arbitrary code execution as <target>, so
        # an attacker with the service user can already
        # `claude bash -c "kill -9 -1"`. The kill rule just adds
        # a faster path for the bot's normal cleanup flow.
        rules += textwrap.dedent("""\

            # Per-target sudoers rules for the cross-os-user inner agent spawn.
            # The claude, codex, opencode, and goose rules grant arbitrary code
            # execution as <target> (all four agents have shell/file tools); the
            # kill rule grants signal delivery to any <target>-owned process. The
            # kill rule's scope is broader than a PID-locked rule because sudoers
            # argument matching is not safe per sudo(8). The kill rule is a strict
            # subset of capabilities the agent rules already provide; the practical
            # delta is zero.
        """)
        for target in target_users:
            rules += f"{service_user} ALL=({target}) SETENV: NOPASSWD: {claude_bin_resolved}\n"
            rules += f"{service_user} ALL=({target}) SETENV: NOPASSWD: {codex_bin_resolved}\n"
            rules += f"{service_user} ALL=({target}) SETENV: NOPASSWD: {opencode_bin_resolved}\n"
            rules += f"{service_user} ALL=({target}) SETENV: NOPASSWD: {goose_bin_resolved}\n"
            rules += f"{service_user} ALL=({target}) NOPASSWD: {kill_bin}\n"

    return rules


def _user_home(username: str) -> str:
    """
    Resolve a user's home directory via pwd lookup.

    Falls back to a platform-appropriate default if the user doesn't exist
    on the current system (e.g., generating a plist for a user that will be
    created later, or during tests with fake usernames).
    """
    try:
        return pwd.getpwnam(username).pw_dir
    except KeyError:
        # User doesn't exist yet (pre-install) or is a test fixture.
        # Use the platform convention so the generated config is plausible.
        if sys.platform == "darwin":
            return f"/Users/{username}"
        return f"/home/{username}"


def _generate_launcher_script(install_dir: str, webhook_port: int = 8080) -> str:
    """
    Generate a launcher script for launchd.

    Homebrew Python's framework binary re-execs itself through Python.app,
    creating a new PID. This causes launchd to lose track of the service
    process. The launcher script stays as the parent process that launchd
    tracks, and forwards SIGTERM to the Python child for graceful shutdown.
    """
    return textwrap.dedent(f"""\
        #!/bin/bash
        # Launcher script for Kai launchd service.
        # Keeps bash as the tracked PID so launchd can manage the service
        # even when Homebrew Python re-execs through the framework bundle.
        #
        # Homebrew Python's framework binary fork+execs through Python.app,
        # creating a grandchild process with a new PID. Launchd tracks this
        # bash script instead, and we forward signals to the real Python.
        {install_dir}/venv/bin/python3 -m kai &
        CHILD_PID=$!

        cleanup() {{
            # Ignore further TERM/INT while tearing down so the group
            # signal below does not re-enter this handler.
            trap '' TERM INT
            # Signal the launcher's whole process group: launchd runs
            # this script as its own group leader and non-interactive
            # bash creates no job-control groups, so the python child,
            # its re-exec'd grandchild, and any helper sleeps all share
            # the group (fork and exec both preserve the pgid). The
            # group signal is the only handle that reaches a python
            # that has re-exec'd but not yet bound the port; in that
            # window it has no individually resolvable pid. Fall back
            # to the direct child if this shell is somehow not a group
            # leader.
            kill -TERM -- -$$ 2>/dev/null || kill -TERM "$CHILD_PID" 2>/dev/null
            # Wait for the webhook port to be released so the stop is
            # not reported complete while a dying python still holds
            # the port a successor will need.
            while [ -n "$(/usr/sbin/lsof -ti :{webhook_port} -sTCP:LISTEN 2>/dev/null)" ]; do sleep 0.5; done
            exit 0
        }}
        # Installed BEFORE the bind poll: a service stop during the
        # (up to 120s) startup window must still tear the agent down;
        # with the default TERM action bash would exit alone and
        # orphan the starting python.
        trap cleanup TERM INT

        # Find the actual Python process (the re-exec'd grandchild) by
        # its listen port. lsof lives at /usr/sbin/ which may not be in
        # the service PATH. Healthy startups bind in 15-25 seconds on
        # this stack (the memory subsystem loads its embedding model
        # before the webhook server starts), so poll for up to 120s.
        # The direct child exits on the framework re-exec in normal
        # operation, so child death alone is not a failure signal;
        # only the port answers whether the agent came up.
        REAL_PID=""
        for _ in $(seq 1 60); do
            REAL_PID=$(/usr/sbin/lsof -ti :{webhook_port} -sTCP:LISTEN 2>/dev/null)
            [ -n "$REAL_PID" ] && break
            sleep 2
        done

        if [ -z "$REAL_PID" ]; then
            # No listener after the window: startup is dead (a
            # fail-closed config gate, a crash, or a hang that is
            # indistinguishable from one). Kill the process group so a
            # python that would bind AFTER the window cannot linger as
            # an orphan holding the port, then exit non-zero so
            # launchd's KeepAlive restarts the service - a throttled,
            # visible retry loop instead of an eternal sleep that
            # reports state=running with no agent behind it.
            echo "kai launcher: no listener on :{webhook_port} after 120s; exiting so launchd restarts the service" >&2
            trap '' TERM INT
            kill -TERM -- -$$ 2>/dev/null || kill -TERM "$CHILD_PID" 2>/dev/null
            exit 1
        fi

        # Poll for the real Python process to exit.
        # kill -0 checks if PID exists without sending a signal.
        # This is macOS-compatible (no GNU tail --pid needed).
        while kill -0 "$REAL_PID" 2>/dev/null; do sleep 1; done
    """)


def _generate_launchd_plist(install_dir: str, data_dir: str, service_user: str) -> str:
    """
    Generate a launchd plist for macOS.

    The plist is installed as a LaunchDaemon (not a LaunchAgent) so the service
    runs under the system domain at boot, independent of any user login session.
    It runs the bot as the service user, sets KAI_DATA_DIR so runtime data goes
    to the writable directory, and includes PATH entries for common tool locations.
    The service user's ~/.local/bin is included in PATH so the inner Claude Code
    process can find the `claude` binary (installed via the native installer).

    ProgramArguments points to a launcher script instead of Python directly.
    Homebrew Python re-execs through Python.app (changing the PID), which causes
    launchd to lose track of the process. The launcher script stays as the
    tracked parent and forwards signals to Python.

    StandardErrorPath and StandardOutPath capture Python's stderr/stdout to
    files under {data_dir}/logs/ rather than the launchd default of /dev/null.
    Without these keys, an early-init crash (a missing env var, an unimportable
    module, a SystemExit before logging is configured) goes nowhere visible
    and the bash wrapper's tracked PID stays alive, so `launchctl print`
    reports `state = running` even when the actual Python process is dead.
    The {data_dir}/logs/ directory is created with service-user ownership
    earlier in the install path; launchd creates the files themselves on
    first write.

    Args:
        install_dir: Root of the protected installation (e.g., /opt/kai).
        data_dir: Writable data directory (e.g., /var/lib/kai).
        service_user: The OS username that runs the service.

    Returns:
        The plist XML as a string.
    """
    # Resolve the service user's home directory for ~/.local/bin PATH entry.
    # Claude Code's native installer places the binary there.
    user_home = _user_home(service_user)
    return textwrap.dedent(f"""\
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        <plist version="1.0">
        <dict>
            <key>Label</key>
            <string>{_LAUNCHD_LABEL}</string>

            <key>UserName</key>
            <string>{service_user}</string>

            <key>ProgramArguments</key>
            <array>
                <string>{install_dir}/run.sh</string>
            </array>

            <key>WorkingDirectory</key>
            <string>{install_dir}</string>

            <key>EnvironmentVariables</key>
            <dict>
                <key>PATH</key>
                <string>{user_home}/.local/bin:/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin</string>
                <key>KAI_DATA_DIR</key>
                <string>{data_dir}</string>
                <key>KAI_INSTALL_DIR</key>
                <string>{install_dir}</string>
            </dict>

            <key>StandardErrorPath</key>
            <string>{data_dir}/logs/kai.stderr.log</string>

            <key>StandardOutPath</key>
            <string>{data_dir}/logs/kai.stdout.log</string>

            <key>RunAtLoad</key>
            <true/>

            <key>KeepAlive</key>
            <true/>

            <key>ThrottleInterval</key>
            <integer>10</integer>

            <key>ProcessType</key>
            <string>Background</string>
        </dict>
        </plist>
    """)


def _generate_systemd_unit(install_dir: str, data_dir: str, service_user: str) -> str:
    """
    Generate a systemd service unit for Linux.

    The unit runs the bot as the service user with KAI_DATA_DIR pointing to the
    writable data directory. Waits for network-online.target to avoid DNS
    failures during boot. The service user's ~/.local/bin is included in PATH
    so the inner Claude Code process can find the `claude` binary.

    Args:
        install_dir: Root of the protected installation (e.g., /opt/kai).
        data_dir: Writable data directory (e.g., /var/lib/kai).
        service_user: The OS username that runs the service.

    Returns:
        The systemd unit file contents as a string.
    """
    # Resolve the service user's home directory for ~/.local/bin PATH entry.
    # Claude Code's native installer places the binary there.
    user_home = _user_home(service_user)
    return textwrap.dedent(f"""\
        [Unit]
        Description=Kai Telegram Bot
        After=network-online.target
        Wants=network-online.target

        [Service]
        Type=simple
        User={service_user}
        WorkingDirectory={install_dir}
        ExecStart={install_dir}/venv/bin/python -m kai
        Restart=always
        RestartSec=5
        Environment=PATH={user_home}/.local/bin:/usr/local/bin:/usr/bin:/bin
        Environment=KAI_DATA_DIR={data_dir}
        Environment=KAI_INSTALL_DIR={install_dir}

        [Install]
        WantedBy=multi-user.target
    """)


def _stop_service(platform: str, dry_run: bool, **_kwargs: object) -> None:
    """
    Stop the Kai service before applying changes.

    Best-effort: uses check=False since the service may not be running
    (first install) or may not exist yet. Failing to stop is not fatal.

    Args:
        platform: "darwin" or "linux".
        dry_run: If True, print the command without executing.
    """
    if platform == "darwin":
        # Boot out from the system domain (LaunchDaemon, not LaunchAgent)
        cmd = ["launchctl", "bootout", f"system/{_LAUNCHD_LABEL}"]
    elif platform == "linux":
        cmd = ["systemctl", "stop", "kai"]
    else:
        print(f"  Warning: cannot stop service on platform '{platform}'")
        return

    if dry_run:
        print(f"[DRY RUN] Would run: {' '.join(cmd)}")
    else:
        result = subprocess.run(cmd, check=False, capture_output=True)
        if result.returncode == 0:
            print(f"  Stopped service ({' '.join(cmd[:2])})")
            # Give launchd time to fully release the service domain.
            # Without this, a subsequent bootstrap can fail transiently
            # on KeepAlive daemons.
            if platform == "darwin":
                time.sleep(2)
        else:
            # Non-zero is expected on first install (service not yet registered)
            print(f"  Service not running ({' '.join(cmd[:2])})")


def _start_service(platform: str, dry_run: bool, **_kwargs: object) -> None:
    """
    Start the Kai service after applying changes and verify it actually
    registered.

    The platform start commands (`launchctl bootstrap` on macOS,
    `systemctl start` on Linux) do not always report failures
    reliably via exit code. launchctl bootstrap is the worst
    offender: it returns exit code 5 ("Input/output error") for
    several distinct conditions including "actually succeeded" and
    "actually failed". The previous implementation trusted that
    exit code, which let installs report a passing warning while
    the daemon was actually unregistered, leaving the operator
    with a silently broken bot.

    The bootstrap/start exit code is now treated as advisory; the
    authoritative check is the post-condition verify query
    (`launchctl print system/<label>` on macOS, `systemctl
    is-active <unit>` on Linux). The cycle is retried up to
    _SERVICE_START_MAX_ATTEMPTS times with a brief settle delay
    after each start and a longer retry delay between attempts to
    absorb transient launchd-domain-not-yet-released windows.

    Args:
        platform: "darwin" or "linux".
        dry_run: If True, print the command without executing.

    Raises:
        ServiceStartError: The verify query confirmed the service
            is not registered after every attempt in the retry
            budget. The caller in `_cmd_apply` decides whether to
            propagate (apply succeeded; the install has not produced
            a working system) or swallow (apply already failed; do
            not mask the original exception).
    """
    if platform == "darwin":
        plist_path = Path("/Library/LaunchDaemons") / f"{_LAUNCHD_LABEL}.plist"
        start_cmd = ["launchctl", "bootstrap", "system", str(plist_path)]
        verify_cmd = ["launchctl", "print", f"system/{_LAUNCHD_LABEL}"]
    elif platform == "linux":
        start_cmd = ["systemctl", "start", "kai"]
        verify_cmd = ["systemctl", "is-active", "kai"]
    else:
        print(f"  Warning: cannot start service on platform '{platform}'")
        return

    if dry_run:
        print(f"[DRY RUN] Would run: {' '.join(start_cmd)}")
        return

    last_start_stderr = ""
    last_verify_stderr = ""
    for attempt in range(1, _SERVICE_START_MAX_ATTEMPTS + 1):
        # Exit code from the start command is advisory only. The
        # authoritative check is the verify query below.
        start = subprocess.run(start_cmd, check=False, capture_output=True)
        # Brief settle window so the service manager finishes its
        # internal bookkeeping before we query state.
        time.sleep(_SERVICE_START_SETTLE_SECONDS)
        verify = subprocess.run(verify_cmd, check=False, capture_output=True)
        if verify.returncode == 0:
            print(f"  Started service ({' '.join(start_cmd[:2])})")
            return
        # Verify failed. Capture both stderrs for the retry hint and
        # the eventual exhaustion error. Guard with `or b""` so a
        # CompletedProcess with stderr=None (only reachable via test
        # mocks; production capture_output=True always populates the
        # field) does not raise AttributeError mid-error-handling.
        last_start_stderr = (start.stderr or b"").decode().strip()
        last_verify_stderr = (verify.stderr or b"").decode().strip()
        if attempt < _SERVICE_START_MAX_ATTEMPTS:
            # Hint at the cause when available; an empty start stderr
            # under a verify-failure is itself diagnostic (the start
            # command exited cleanly but the daemon still is not
            # registered, which is exactly the launchctl bootstrap
            # exit-code-5-but-actually-failed pattern).
            hint = last_start_stderr or "start command exited cleanly but verify did not confirm registration"
            print(
                f"  Service not yet registered after attempt {attempt}/{_SERVICE_START_MAX_ATTEMPTS} "
                f"({hint}); retrying in {_SERVICE_START_RETRY_SECONDS}s..."
            )
            time.sleep(_SERVICE_START_RETRY_SECONDS)

    # Retry budget exhausted. Surface both the start stderr and the
    # verify stderr so the operator sees the authoritative failure
    # state rather than only the unreliable start exit code.
    detail = (
        f"verify command ({' '.join(verify_cmd)}) reported the service is not registered "
        f"after {_SERVICE_START_MAX_ATTEMPTS} attempts"
    )
    if last_verify_stderr:
        detail += f"; verify stderr: {last_verify_stderr}"
    if last_start_stderr:
        detail += f"; last start stderr: {last_start_stderr}"
    raise ServiceStartError(detail)


def _secure_codex_turn_image_staging(data_path: Path, dry_run: bool) -> None:
    """Retire world-readable Codex turn images from the legacy layout.

    Older releases wrote ``turn-image-*`` files directly under the shared
    ``codex_turn_images`` directory with mode ``0644``. The service is stopped
    while install migration runs, so no legitimate turn can still be using
    those transient files. Remove only the exact legacy filename class, leave
    unrelated entries and new per-principal subdirectories untouched, and make
    the staging root traversable but non-listable for isolated OS users.
    """
    staging_root = data_path / "files" / "codex_turn_images"
    if staging_root.is_symlink():
        raise RuntimeError(f"Refusing unsafe Codex image staging path: {staging_root}")
    if not staging_root.exists():
        return
    if not staging_root.is_dir():
        raise RuntimeError(f"Refusing unsafe Codex image staging path: {staging_root}")

    legacy_files = sorted(
        (
            entry
            for entry in staging_root.iterdir()
            if entry.name.startswith("turn-image-") and (entry.is_file() or entry.is_symlink())
        ),
        key=lambda entry: entry.name,
    )
    current_mode = staging_root.stat().st_mode & 0o777

    if dry_run:
        if legacy_files:
            print(f"[DRY RUN] Would remove {len(legacy_files)} legacy Codex turn image file(s) from {staging_root}")
        if current_mode != 0o711:
            print(f"[DRY RUN] Would set mode 0711 on Codex image staging root: {staging_root}")
        return

    for path in legacy_files:
        path.unlink(missing_ok=True)
    os.chmod(staging_root, 0o711)
    if legacy_files:
        print(f"  Removed {len(legacy_files)} legacy Codex turn image file(s) from {staging_root}")
    if current_mode != 0o711:
        print(f"  Secured Codex image staging root {staging_root} (mode 0711)")


def _apply_migrate(
    data_path: Path,
    install_path: Path,
    svc_uid: int,
    svc_gid: int,
    dry_run: bool,
    users_yaml_path: Path | None = None,
) -> None:
    """
    Migrate runtime data from the development directory to the data directory.

    One-time migration of database, log, history, memory, and uploaded files.
    Safe to run multiple times: existing files at the destination are never
    overwritten, and source files are never deleted (they serve as backups).

    Args:
        data_path: Writable data directory (e.g., /var/lib/kai).
        install_path: Installation directory (e.g., /opt/kai) for locating
            uploaded files at the old home/files/ location.
        svc_uid: Numeric UID for file ownership.
        svc_gid: Numeric GID for file ownership.
        dry_run: If True, print actions without executing.
        users_yaml_path: Path to the installed users.yaml. None means
            the module-level USERS_YAML (the post-_apply_secrets
            location); tests redirect it by patching that attribute.
    """
    # None resolves to the module-level USERS_YAML at call time rather
    # than in the signature: a def-time default would bake the
    # production path in at import and bypass the patch the test suite
    # uses to keep the host's real config out of test outcomes.
    if users_yaml_path is None:
        users_yaml_path = USERS_YAML

    # Retire the old shared/listable Codex image layout before migrating or
    # reconciling other runtime data. The protected service is stopped for the
    # whole apply transaction, so every root-level turn image is stale.
    _secure_codex_turn_image_staging(data_path, dry_run)

    # -- Database migration --
    db_src = PROJECT_ROOT / "kai.db"
    db_dst = data_path / "kai.db"

    if db_src.exists() and not db_dst.exists():
        if dry_run:
            print(f"[DRY RUN] Would copy database: {db_src} -> {db_dst}")
            print(f"[DRY RUN] Would verify integrity: sqlite3 {db_dst} 'PRAGMA integrity_check;'")
            print(f"[DRY RUN] Would set ownership: {db_dst} ({svc_uid}:{svc_gid})")
        else:
            shutil.copy2(db_src, db_dst)
            print(f"  Copied database to {db_dst}")

            # Verify the copied database is intact
            result = subprocess.run(
                ["sqlite3", str(db_dst), "PRAGMA integrity_check;"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0 or "ok" not in result.stdout.lower():
                print(f"  Warning: database integrity check failed: {result.stderr.strip()}")
            else:
                print("  Database integrity check passed")

            os.chmod(db_dst, 0o600)
            os.chown(db_dst, svc_uid, svc_gid)
    elif db_dst.exists():
        print("  Database already exists at destination, skipping migration")
        if dry_run:
            print(f"[DRY RUN] Would secure database: {db_dst} (mode 0600, {svc_uid}:{svc_gid})")
        else:
            os.chmod(db_dst, 0o600)
            os.chown(db_dst, svc_uid, svc_gid)
    elif not db_src.exists():
        print("  No source database found, skipping migration")

    # -- Log migration --
    logs_src = PROJECT_ROOT / "logs"
    logs_dst = data_path / "logs"

    if logs_src.exists():
        # Collect all log files (daily rotation produces .log, .log.1, etc.)
        log_files = list(logs_src.glob("*.log*"))
        if log_files:
            if dry_run:
                for f in log_files:
                    print(f"[DRY RUN] Would copy log: {f} -> {logs_dst / f.name}")
                print(f"[DRY RUN] Would set ownership: {logs_dst} ({svc_uid}:{svc_gid})")
            else:
                for f in log_files:
                    dst = logs_dst / f.name
                    if not dst.exists():
                        shutil.copy2(f, dst)
                        print(f"  Copied log: {f.name}")
                # Set ownership on the entire logs directory
                _set_ownership(logs_dst, svc_uid, svc_gid, recursive=True)

    # -- MEMORY.md migration --
    # One-time: land personal memory at the per-user path
    # DATA_DIR/memory/<chat_id>/MEMORY.md. The "primary operator"
    # (first entry in users.yaml, typically an admin) gets any legacy
    # content from DATA_DIR/memory/MEMORY.md (pre-#347 global location).
    # legacy_global is moved (not copied) so the DATA_DIR global path
    # cannot later read as stale state. If no legacy source exists, the
    # per-user MEMORY.md is seeded from the templates/.claude/MEMORY.md
    # template so the inner Claude has a writable starting point on
    # first message.
    # Subsequent users in users.yaml get the template, not the legacy
    # content, because one user's memory never belongs to another.
    #
    # This migration deliberately runs with users.yaml loaded from
    # the resolved install path (post-_apply_secrets, default
    # /etc/kai/users.yaml) so that all known operators get a seeded
    # directory on the install where this code first lands.
    memory_owners = _collect_user_memory_owners(users_yaml_path)
    memory_root = data_path / "memory"
    legacy_global = memory_root / "MEMORY.md"
    template = PROJECT_ROOT / "templates" / ".claude" / "MEMORY.md"

    # Resolve every os_user against the host's passwd database BEFORE
    # touching disk. If users.yaml names an OS user that does not exist
    # on this host (typo, new user listed before useradd, copy/paste
    # from another machine), pwd.getpwnam raises bare KeyError(name)
    # with no chat_id, no path, no hint. Hoist validation here so the
    # install hard-fails with a clear message before the migration
    # block creates, copies, or moves any MEMORY.md files. Surfacing
    # the misconfiguration after disk mutation would leave operators
    # diagnosing a half-applied state.
    per_user_ids: dict[str, tuple[int, int]] = {}
    for chat_id, os_user in memory_owners:
        if os_user is None:
            continue
        try:
            pwd_entry = pwd.getpwnam(os_user)
        except KeyError as exc:
            raise ValueError(
                f"users.yaml entry chat_id={chat_id} names os_user "
                f"{os_user!r}, which does not exist on this host. Source: "
                f"{users_yaml_path}. Create the OS account or correct the "
                f"users.yaml entry, then re-run sudo make install."
            ) from exc
        per_user_ids[str(chat_id)] = (pwd_entry.pw_uid, pwd_entry.pw_gid)
    known_user_dir_names = {str(chat_id) for chat_id, _os_user in memory_owners if chat_id is not None}

    # Resolve the primary operator's chat_id (first yaml entry). When
    # users.yaml is absent or empty (first-ever install, single-user
    # dev), leave memory/MEMORY.md alone - runtime code falls back to
    # that legacy path when chat_id is None. This keeps local `make
    # run` workflows identical to pre-#347 behavior.
    primary_chat_id: int | None = memory_owners[0][0] if memory_owners else None

    if primary_chat_id is not None:
        primary_dir = memory_root / str(primary_chat_id)
        primary_dst = primary_dir / "MEMORY.md"

        if not primary_dst.exists():
            if dry_run:
                if legacy_global.is_file():
                    print(f"[DRY RUN] Would move {legacy_global} -> {primary_dst}")
                elif template.is_file():
                    print(f"[DRY RUN] Would seed {primary_dst} from {template}")
                else:
                    # Mirror the real branch's last-resort placeholder
                    # below: when none of the source files exist (minimal
                    # host, broken install tree), the real path writes
                    # `# Memory\n` and prints "Created empty ...". Without
                    # this `else` the dry-run prints nothing and the
                    # operator sees a silent gap they cannot diagnose.
                    print(f"[DRY RUN] Would create empty {primary_dst} (no template found)")
            else:
                primary_dir.mkdir(parents=True, exist_ok=True)
                if legacy_global.is_file():
                    # Move, not copy - the legacy global path must not
                    # survive this migration, or a stale file will shadow
                    # the per-user read once one user's subdir fills up.
                    shutil.move(str(legacy_global), str(primary_dst))
                    print(f"  Migrated MEMORY.md to {primary_dst}")
                elif template.is_file():
                    shutil.copy2(template, primary_dst)
                    print(f"  Seeded {primary_dst} from memory template")
                else:
                    # Last-resort placeholder so the file is writable.
                    primary_dst.write_text("# Memory\n")
                    print(f"  Created empty {primary_dst}")

        # Seed every other known user from the memory template. Skips
        # the primary (handled above) and any user whose subdir already
        # has a MEMORY.md (idempotent across reinstalls).
        for chat_id, _os_user in memory_owners[1:]:
            user_dir = memory_root / str(chat_id)
            user_dst = user_dir / "MEMORY.md"
            if user_dst.exists():
                continue
            if dry_run:
                # Match the real branch below: the template may not ship
                # with the install tree, in which case the real path
                # writes a placeholder. Printing "from memory template"
                # unconditionally misleads operators on hosts where the
                # template is missing.
                if template.is_file():
                    print(f"[DRY RUN] Would seed {user_dst} from memory template")
                else:
                    print(f"[DRY RUN] Would create empty {user_dst} (no memory template)")
                continue
            user_dir.mkdir(parents=True, exist_ok=True)
            if template.is_file():
                shutil.copy2(template, user_dst)
                print(f"  Seeded {user_dst} from memory template")
            else:
                user_dst.write_text("# Memory\n")
                print(f"  Created empty {user_dst}")

    # -- Uploaded files migration --
    # One-time: copy user-uploaded files from the install tree
    # (home/files/, pre-DATA_DIR location) to DATA_DIR/files/.
    # Walks the full directory tree to handle per-user subdirectories
    # (files/{user_id}/). Existing files at the destination are not
    # overwritten. Source files are preserved as backups.
    files_src = install_path / "home" / "files"
    files_dst = data_path / "files"

    if files_src.exists() and any(files_src.iterdir()):
        # Use os.walk with followlinks=False (the default) so symlinks
        # pointing outside the directory are not followed during migration.
        copied = 0
        for root, _dirs, fnames in os.walk(files_src):
            for fname in fnames:
                src_file = Path(root) / fname
                rel = src_file.relative_to(files_src)
                dst_file = files_dst / rel
                if dst_file.exists():
                    continue
                if dry_run:
                    print(f"[DRY RUN] Would copy file: {src_file} -> {dst_file}")
                else:
                    dst_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_file, dst_file)
                copied += 1
        if copied and not dry_run:
            # Set ownership on the entire files tree, not just newly copied
            # files. This ensures uniform ownership after a partial migration
            # (e.g., some files copied on a previous run, new ones added now).
            for root, _subdirs, fnames in os.walk(files_dst):
                os.chown(root, svc_uid, svc_gid)
                for fname in fnames:
                    os.chown(os.path.join(root, fname), svc_uid, svc_gid)
            print(f"  Migrated {copied} uploaded file(s) to {files_dst}")
        elif copied and dry_run:
            print(f"[DRY RUN] Would migrate {copied} uploaded file(s) to {files_dst}")
        elif not copied and not dry_run:
            print("  Uploaded files already migrated or no files to copy")

    # -- Memory directory ownership --
    # Two ownership tiers in a single tree:
    #   (a) Service-owned: memory/, memory/qdrant/, memory/mem0_history.db,
    #       memory/extractor_cwd/, and any other kai-written artifacts.
    #       Init_memory() creates qdrant/ and mem0_history.db at runtime
    #       in the main process (service user), so they must be writable
    #       by the service user.
    #   (b) Per-user-owned: memory/<chat_id>/ subdirectories whose chat_id
    #       maps to a users.yaml entry with an explicit os_user. The
    #       inner Claude subprocess for that user runs via sudo -H -u
    #       <os_user>, so MEMORY.md ownership must match or writes fail
    #       with the #347 regression.
    # Subdirectories whose chat_id does not appear in users.yaml (or whose
    # user has no os_user - i.e., runs as the service user) fall through
    # to the service-owned tier. Same for any stray file at the top level.
    memory_tree = data_path / "memory"
    if memory_tree.is_dir():
        # per_user_ids was built and validated at the top of this
        # function, before any disk mutation. Reusing it here means a
        # bad os_user has already aborted the install with a clear
        # ValueError; we cannot reach this block in that state.

        if dry_run:
            print(f"[DRY RUN] Would set ownership/modes on {memory_tree} (service + per-user)")
            for name, (uid, gid) in per_user_ids.items():
                print(f"[DRY RUN]   {memory_tree / name} -> ({uid}:{gid}, mode 0700)")
        else:
            # Top-level directory: service-owned so the service user
            # can create new per-user subdirs on add-user flows.
            _set_ownership(memory_tree, svc_uid, svc_gid, recursive=False)
            os.chmod(memory_tree, _PRIVATE_USER_ROOT_MODE)
            for entry in memory_tree.iterdir():
                if entry.is_dir() and entry.name in known_user_dir_names:
                    uid, gid = per_user_ids.get(entry.name, (svc_uid, svc_gid))
                    _set_ownership(entry, uid, gid, recursive=True)
                    _set_private_user_tree_modes(entry)
                else:
                    # qdrant/, mem0_history.db, extractor_cwd/, and any
                    # memory/<chat_id>/ whose user has no os_user set.
                    _set_ownership(entry, svc_uid, svc_gid, recursive=True)

    # -- PREFERENCES.md per-user pre-creation --
    # Mirror the MEMORY.md per-user pattern. For every users.yaml
    # entry we pre-create DATA_DIR/preferences/<chat_id>/ and seed
    # PREFERENCES.md from the example template if it does not exist.
    # Initial ownership is set by the ownership pass below, not here;
    # keeping the chown in a single block makes the os_user-change
    # idempotency case fall out naturally rather than requiring a
    # "remember to also chown the file" reminder. Lazy bootstrap at
    # runtime (backend.ensure_user_preferences) is the fallback for
    # chat_ids added between installs.
    preferences_tree = data_path / "preferences"
    preferences_template = PROJECT_ROOT / "templates" / ".claude" / "PREFERENCES.md"

    for chat_id, _os_user in memory_owners:
        if chat_id is None:
            continue
        pref_dir = preferences_tree / str(chat_id)
        pref_dst = pref_dir / "PREFERENCES.md"

        if dry_run:
            if not pref_dst.exists():
                if preferences_template.is_file():
                    print(f"[DRY RUN] Would seed {pref_dst} from {preferences_template}")
                else:
                    print(f"[DRY RUN] Would seed {pref_dst} with placeholder (template missing)")
            continue

        pref_dir.mkdir(parents=True, exist_ok=True)
        if not pref_dst.exists():
            if preferences_template.is_file():
                shutil.copy2(preferences_template, pref_dst)
                print(f"  Seeded {pref_dst} from preferences template")
            else:
                # Match ensure_user_preferences() / ensure_user_memory()
                # missing-template precedent so the subprocess always
                # has a writable file. Warn so the operator notices the
                # install tree gap, but continue.
                pref_dst.write_text("# Preferences\n")
                print(f"  WARNING: {preferences_template} not found; wrote placeholder to {pref_dst}")

    # -- PREFERENCES.md directory ownership --
    # Recursively chown DATA_DIR/preferences/ on every install, exactly
    # mirroring the MEMORY.md ownership pass above. The top directory
    # is service-owned so the service user can create new per-user
    # subdirs on add-user flows. Per-user subdirs whose chat_id appears
    # in users.yaml are chowned to that user's os_user. Stray entries
    # (subdirs whose user has no os_user, or files at the top level)
    # fall through to service ownership.
    #
    # This pass is what corrects ownership drift when os_user changes
    # between installs. Without it, the seeding block above would set
    # ownership on first-run only and stale files would persist.
    if preferences_tree.is_dir():
        if dry_run:
            print(f"[DRY RUN] Would set ownership/modes on {preferences_tree} (service + per-user)")
            for name, (uid, gid) in per_user_ids.items():
                print(f"[DRY RUN]   {preferences_tree / name} -> ({uid}:{gid}, mode 0700)")
        else:
            _set_ownership(preferences_tree, svc_uid, svc_gid, recursive=False)
            os.chmod(preferences_tree, _PRIVATE_USER_ROOT_MODE)
            for entry in preferences_tree.iterdir():
                if entry.is_dir() and entry.name in known_user_dir_names:
                    uid, gid = per_user_ids.get(entry.name, (svc_uid, svc_gid))
                    _set_ownership(entry, uid, gid, recursive=True)
                    _set_private_user_tree_modes(entry)
                else:
                    # Subdirs whose chat_id has no os_user, or any
                    # stray top-level entry, fall through to service
                    # ownership. Same fallback as the memory pass.
                    _set_ownership(entry, svc_uid, svc_gid, recursive=True)

    # -- Per-user home workspace pre-creation (#353) --
    # Mirror the per-user memory pattern above. For every users.yaml
    # entry we pre-create the directory the inner Claude subprocess
    # (sudo -H -u <os_user>) will write into on first message. Three
    # cases, all handled below:
    #   1. No override: create DATA_DIR/home/<chat_id>/.
    #   2. Override path under DATA_DIR (rare, but valid): create the
    #      override path itself, since it lives in our tree and
    #      resolve_home_workspace() will route the user there at
    #      runtime. Creating DATA_DIR/home/<chat_id>/ instead would
    #      leave the actual runtime directory un-provisioned and the
    #      user's first write would fail.
    #   3. Override path outside DATA_DIR: skip entirely - the override
    #      is operator-managed (a clone of the dev tree, a synced
    #      volume, etc.). Provisioning here would chown a directory we
    #      do not own.
    # Subdirectories whose chat_id has no os_user fall through to the
    # service-owned tier, matching the memory tier-(a) rule above.
    home_root = data_path / "home"
    home_overrides = _collect_user_home_overrides(users_yaml_path)
    # Defensive: `_apply_directories` already creates home_root with
    # the right ownership, but we cannot assume ordering (future
    # refactors could split these steps). A missing home_root here
    # would otherwise silently skip every per-user provisioning step
    # and leave the installer reporting success while subsequent
    # first-messages crash. mkdir(exist_ok=True) is cheap and
    # idempotent. We deliberately do NOT chown home_root in this
    # fallback path: if _apply_directories ran (the normal case)
    # ownership is already correct, and if it did not run a defensive
    # chown here would mask the real bug rather than fix it.
    if not dry_run:
        home_root.mkdir(parents=True, exist_ok=True, mode=_PRIVATE_USER_ROOT_MODE)
        # mkdir's mode= is masked by umask. Under the production
        # service umask of 0o027 the directory would otherwise end up
        # 0o710. Force traversal-only public bits so distinct os_user
        # subprocesses can reach their own private slots without
        # listing sibling chat IDs.
        os.chmod(home_root, _PRIVATE_USER_ROOT_MODE)
    # Resolve data_path once for the symlink-safe containment check
    # below. _collect_user_home_overrides canonicalizes each override
    # via Path.resolve(); comparing against an unresolved data_path
    # would mis-classify legitimately-internal overrides as external on
    # any host where DATA_DIR traverses a symlink (e.g., macOS where
    # /var/lib is a symlink to /private/var/lib). Resolving both sides
    # makes is_relative_to a true containment test rather than a
    # string-prefix comparison.
    data_path_resolved = data_path.resolve()
    for chat_id, _os_user in memory_owners:
        override = home_overrides.get(chat_id)
        # Case 3: override pinned outside DATA_DIR - operator-managed,
        # skip. is_relative_to is the cleanest way to test path
        # containment; on Python 3.9+ it returns bool without raising.
        if override is not None and not override.is_relative_to(data_path_resolved):
            continue
        # Case 2: override under DATA_DIR - provision THAT path, since
        # resolve_home_workspace returns it verbatim at runtime. Case
        # 1: no override - provision the per-user default slot.
        user_dir = override if override is not None else home_root / str(chat_id)
        user_dir_exists = user_dir.exists()
        if dry_run:
            if str(chat_id) in per_user_ids:
                uid, gid = per_user_ids[str(chat_id)]
                action = "set private mode on" if user_dir_exists else "create"
                print(f"[DRY RUN] Would {action} {user_dir} ({uid}:{gid}, mode 0700)")
            else:
                action = "set private mode on" if user_dir_exists else "create"
                print(f"[DRY RUN] Would {action} {user_dir} ({svc_uid}:{svc_gid}, mode 0700)")
            continue
        user_dir.mkdir(parents=True, exist_ok=True, mode=_PRIVATE_USER_DIR_MODE)
        # mkdir does not always honor mode (it is masked by umask).
        # Force the intended bits explicitly so the per-user dir
        # ends up private regardless of the install-time umask.
        os.chmod(user_dir, _PRIVATE_USER_DIR_MODE)
        if str(chat_id) in per_user_ids:
            uid, gid = per_user_ids[str(chat_id)]
            _set_ownership(user_dir, uid, gid, recursive=False)
        else:
            _set_ownership(user_dir, svc_uid, svc_gid, recursive=False)
        if not user_dir_exists:
            print(f"  Created {user_dir}")

    # -- CLAUDE.md per-user pre-creation --
    # Mirror the MEMORY.md / PREFERENCES.md per-user pattern above. For
    # every users.yaml entry we pre-create
    # <DATA_DIR>/home/<chat_id>/.claude/CLAUDE.md from the template, so
    # `backend.build_session_context`'s identity-injection path and
    # the inner Claude's native CLAUDE.md discovery in the home
    # workspace both have a real file to read. Without this seed, a
    # new user's home workspace is an empty directory and the bot
    # runs with no baseline identity for that chat_id - the gap that
    # PR #449 inadvertently exposed and #450 reverted.
    #
    # Lazy bootstrap at runtime (backend.ensure_user_home) is the
    # fallback for chat_ids added between installs; this eager pass
    # is for users present in users.yaml at install time so they get
    # the right ownership (chowned to their os_user via the
    # _set_ownership call) instead of service-owned-via-mkdir.
    home_template = PROJECT_ROOT / "templates" / ".claude" / "CLAUDE.md"
    # Cache the template existence check once; the path is fixed for
    # the duration of the loop and is_file() would otherwise stat the
    # same inode N times for N users.
    home_template_exists = home_template.is_file()

    for chat_id, _os_user in memory_owners:
        if chat_id is None:
            continue
        override = home_overrides.get(chat_id)
        # Case 3: override outside DATA_DIR is operator-managed; the
        # home-creation loop above skipped it for the same reason and
        # we follow suit here. Touching it from the installer would
        # write under a path the operator owns.
        if override is not None and not override.is_relative_to(data_path_resolved):
            continue
        user_home = override if override is not None else home_root / str(chat_id)
        claude_dir = user_home / ".claude"
        claude_dst = claude_dir / "CLAUDE.md"

        if dry_run:
            if not claude_dst.exists():
                if home_template_exists:
                    print(f"[DRY RUN] Would seed {claude_dst} from {home_template}")
                else:
                    print(f"[DRY RUN] Would seed {claude_dst} with placeholder (template missing)")
            else:
                # Migration preview for an existing per-user copy. The
                # seed branch above only fires when claude_dst is
                # missing; the migration preview fires when claude_dst
                # exists and the sentinel header may or may not be in
                # place. Inspecting claude_dst here, before the
                # `continue` below, ensures the dry-run preview reaches
                # parity with what the live branch will do; without
                # this branch, the dry-run output would silently omit
                # the migration step the live install would execute.
                #
                # The helper's three-state return is the structural
                # distinction that lets the preview print accurate text
                # per case. Truthy/falsy alone would conflate sentinel-
                # present (False) with the failure paths (None), producing
                # a misleading "already present" line when the helper had
                # actually warned about a different problem.
                if home_template_exists:
                    migration_result = _migrate_recalled_memory_section(claude_dst, home_template, dry_run=True)
                    if migration_result is True:
                        print(f"[DRY RUN] Would append Reading Recalled Memory section to {claude_dst}")
                    elif migration_result is False:
                        # Sentinel present; surface the no-op explicitly so
                        # the operator sees the migration was considered.
                        print(
                            f"[DRY RUN] {claude_dst}: Reading Recalled Memory "
                            f"section already present, no migration needed"
                        )
                    # else: migration_result is None; the helper already
                    # printed its own WARNING for the specific failure path.
                    # No additional preview line needed.
                else:
                    # Template missing entirely; the helper is never called
                    # because of the short-circuit above, so no warning has
                    # been printed in this iteration. Surface the gap so the
                    # dry-run preview is honest about what the install can
                    # and cannot do for this user.
                    print(f"[DRY RUN] {claude_dst}: cannot evaluate migration (template missing at {home_template})")
            continue

        # Idempotent: never overwrite an operator-customized destination.
        # `not exists()` is the same guard ensure_user_memory and
        # ensure_user_preferences use; matches the seed-step contract.
        claude_dir.mkdir(parents=True, exist_ok=True)
        # mkdir's mode= is masked by umask; force the private mode
        # explicitly. The user-owned home slot is now 0700, so no
        # sibling user needs traversal through this subdir.
        os.chmod(claude_dir, _PRIVATE_USER_DIR_MODE)
        if not claude_dst.exists():
            # Wrap the actual file write in try/except so an OSError
            # (broken symlink at claude_dst, mounted FS, permissions)
            # surfaces with a clear operator-readable line BEFORE the
            # traceback aborts the install. Matches the rmtree wrap
            # in `_retire_install_home_claude`; install-time policy
            # is to abort loudly on a real error, not log-and-continue
            # (that swallow is `backend.ensure_user_home`'s contract
            # for the runtime path where session-init cannot crash).
            try:
                if home_template_exists:
                    shutil.copy2(home_template, claude_dst)
                    print(f"  Seeded {claude_dst} from CLAUDE.md template")
                else:
                    # Last-resort placeholder so the inner Claude has
                    # something to read. Mirrors MEMORY.md / PREFERENCES.md
                    # missing-template precedent above.
                    claude_dst.write_text("# Identity\n")
                    print(f"  WARNING: {home_template} not found; wrote placeholder to {claude_dst}")
            except OSError as exc:
                print(f"  ERROR: could not seed {claude_dst}: {exc}")
                raise

        # Append the Reading Recalled Memory section to pre-existing
        # per-user CLAUDE.md copies that predate the section. Lands
        # AFTER the seed step (so a fresh-seeded file from the current
        # template carries the section already, and the helper's
        # sentinel check is a clean no-op) and BEFORE the
        # `_set_ownership` chown step below (so the migration's
        # atomic `Path.replace`, which produces a new inode owned by
        # the install runner, gets reconciled back to the per-user
        # `os_user` in the same iteration; the chown comment block
        # below names the #347 regression shape this prevents).
        if home_template_exists and _migrate_recalled_memory_section(claude_dst, home_template, dry_run=False) is True:
            print(f"  Appended Reading Recalled Memory section to {claude_dst}")

        # Recursive chown over the .claude/ subdir so both the
        # directory and the freshly-seeded CLAUDE.md land on the
        # per-user os_user. The home-creation loop above only chowns
        # user_dir non-recursively, so without this pass the .claude/
        # tree would inherit service-user ownership from mkdir and
        # the inner subprocess (sudo -H -u <os_user>) could read but
        # not write CLAUDE.md - the same #347 regression the per-user
        # memory pattern fixed.
        #
        # The chown runs UNCONDITIONALLY on every install, even when
        # the seed step skipped because CLAUDE.md already existed.
        # Functionally mirrors the MEMORY.md / PREFERENCES.md
        # ownership reconciliation: idempotent per-install reset
        # corrects os_user drift. (Structurally those blocks use a
        # separate iterdir-based pass over the whole tree; this block
        # inlines the chown into the per-user seed loop because the
        # tree we care about is one .claude/ subdir per chat_id,
        # which iterdir would walk anyway.) Without the unconditional
        # reset, a chat_id whose os_user changed in users.yaml between
        # installs would keep its old ownership and the new subprocess
        # could not write the identity file.
        if str(chat_id) in per_user_ids:
            uid, gid = per_user_ids[str(chat_id)]
            _set_ownership(claude_dir, uid, gid, recursive=True)
        else:
            _set_ownership(claude_dir, svc_uid, svc_gid, recursive=True)
        _set_private_user_tree_modes(claude_dir)

    # -- Per-os-user temp directories (#454) --
    # The inner Claude binary writes a content-hashed settings cache
    # at $TMPDIR/claude-settings-<hex>.json. Two distinct os_users
    # whose --settings JSON hashes to the same hex collide on the
    # default /tmp path; the first writer owns the file at mode
    # 0o644 and the second claude exits with EACCES on the
    # write-open. The cross-user spawn paths (claude.py, codex.py,
    # and the shared ACP layer in acp.py) anchor TMPDIR per-os_user under
    # <DATA_DIR>/tmp/<os_user>/ so each identity has its own temp
    # namespace. Create those dirs here, mode 0o700 chowned to the
    # target os_user (only their owner needs to read them). Parent
    # tmp/ is service-owned mode 0o755 so the per-user subdirs
    # are traversable for the sudo -u <os_user> subprocess.
    #
    # Distinct os_users are derived from memory_owners (same source
    # as the memory/preferences/home blocks above). per_user_ids
    # was built and pwd-validated at the top of this function; we
    # re-look up here keyed by os_user name to dedupe because
    # per_user_ids is keyed by str(chat_id), not os_user.
    tmp_root = data_path / "tmp"
    distinct_os_users: dict[str, tuple[int, int]] = {}
    for _chat_id, os_user_name in memory_owners:
        if os_user_name is None or os_user_name in distinct_os_users:
            continue
        # Already validated above in the per_user_ids build loop;
        # this lookup cannot raise KeyError.
        pwd_entry = pwd.getpwnam(os_user_name)
        distinct_os_users[os_user_name] = (pwd_entry.pw_uid, pwd_entry.pw_gid)

    if dry_run:
        if distinct_os_users:
            print(f"[DRY RUN] Would create {tmp_root} (mode 0o755, {svc_uid}:{svc_gid})")
            for name, (uid, gid) in distinct_os_users.items():
                print(f"[DRY RUN]   {tmp_root / name} (mode 0o700, {uid}:{gid})")
    elif distinct_os_users:
        # Parent dir: service-owned, 0o755 so sudo -u <target>
        # subprocesses can traverse into their own subdir.
        tmp_root.mkdir(parents=True, exist_ok=True, mode=0o755)
        os.chmod(tmp_root, 0o755)
        _set_ownership(tmp_root, svc_uid, svc_gid, recursive=False)
        for name, (uid, gid) in distinct_os_users.items():
            user_tmp = tmp_root / name
            new_dir = not user_tmp.exists()
            user_tmp.mkdir(parents=True, exist_ok=True, mode=0o700)
            # mkdir mode is masked by umask; force the explicit bits
            # to correct drift across reinstalls and override any
            # weaker bits a previously-run install left behind.
            os.chmod(user_tmp, 0o700)
            _set_ownership(user_tmp, uid, gid, recursive=False)
            if new_dir:
                print(f"  Created {user_tmp}")


def _cmd_apply() -> None:
    """
    Read install.conf and perform the installation. Requires root.

    First-time installation creates the directory structure, copies source,
    creates a venv, writes secrets, configures sudoers, migrates data,
    and generates a service definition. The service is stopped before
    changes begin and started after everything completes. Updates detect
    existing installations and only change what's needed.

    When DRY_RUN=1 is set in the environment, every action is printed
    without being executed.
    """
    # -- Validate preconditions --
    if os.geteuid() != 0:
        raise SystemExit("'install apply' must be run as root (try: sudo python -m kai install apply)")

    if not INSTALL_CONF.exists():
        raise SystemExit(f"{INSTALL_CONF} not found. Run 'python -m kai install config' first.")

    try:
        conf = json.loads(INSTALL_CONF.read_text())
    except (json.JSONDecodeError, OSError) as e:
        raise SystemExit(f"Could not read {INSTALL_CONF}: {e}") from e

    # Deployment mode gate. Apply is meaningful only for protected
    # installs: it copies the staged users.yaml into /etc/kai/, writes
    # /etc/kai/env, installs the systemd / launchd service, and
    # configures sudoers. None of that applies to a single-user
    # install, where `make config` already wrote everything the
    # daemon reads (PROJECT_ROOT/.env, XDG users.yaml) and `make run`
    # starts the daemon from the repo.
    #
    # Default missing `deployment_mode` to `protected` so a legacy
    # install.conf written before this key existed continues to apply
    # cleanly. Single-user installs always carry the key because the
    # wizard writes it.
    deployment_mode = conf.get("deployment_mode", "protected")
    if deployment_mode == "single_user":
        raise SystemExit(
            "single_user mode is already applied by 'make config'; "
            "run 'make run' (or 'python -m kai') from the repo. "
            "'make install' is a no-op in this mode."
        )

    # Validate required fields
    install_dir = conf.get("install_dir")
    data_dir = conf.get("data_dir")
    service_user = conf.get("service_user")
    platform = conf.get("platform")
    env = conf.get("env", {})
    # Migrate a legacy AGENT_BACKEND key in the install.conf env dict to
    # DEFAULT_BACKEND once, here, before any gate reads it or
    # _apply_secrets writes /etc/kai/env. Apply consumes the env dict
    # directly (it never goes through load_config), so without this an
    # apply driven by a legacy install.conf would either skip
    # backend-specific setup or, for codex/goose, validate the default
    # model against the wrong (claude/anthropic) surface and fail. After
    # this block every downstream `env.get("DEFAULT_BACKEND")` sees the
    # resolved value and /etc/kai/env is written with the new name only.
    if "AGENT_BACKEND" in env and "DEFAULT_BACKEND" not in env:
        env["DEFAULT_BACKEND"] = env.pop("AGENT_BACKEND")
    elif "AGENT_BACKEND" in env:
        # Both present: the new key wins; drop the legacy one so it does
        # not linger in the written env file.
        del env["AGENT_BACKEND"]
    # Same migration for the provider key, renamed LLM_PROVIDER ->
    # DEFAULT_PROVIDER. Apply reads the provider when validating the
    # default model against a multi-provider backend's surface, so the
    # legacy name must be resolved here too, before any gate reads it.
    if "LLM_PROVIDER" in env and "DEFAULT_PROVIDER" not in env:
        env["DEFAULT_PROVIDER"] = env.pop("LLM_PROVIDER")
    elif "LLM_PROVIDER" in env:
        del env["LLM_PROVIDER"]
    # Top-level installer metadata: present only when the wizard
    # staged a first-time users.yaml that has not yet been applied.
    # The empty-string -> None coercion treats a hand-edited conf
    # with an empty value the same as a missing key (apply skips
    # the copy). `_apply_secrets` defends against a missing file
    # behind a non-empty path so a wrong value silently skips
    # rather than failing the apply.
    #
    # The staging path is consumed only in protected mode (the gate
    # above already refused single-user). A stale top-level key
    # carried forward across a mode switch would point at a path
    # under the operator's home that this apply has no business
    # touching; the protected-mode gate ensures we never reach here
    # with that combination.
    users_yaml_staging_path = conf.get("users_yaml_staging_path", "") or None

    if not all([install_dir, data_dir, service_user, platform]):
        raise SystemExit("install.conf is missing required fields.")

    # Validate service user exists
    try:
        user_info = pwd.getpwnam(service_user)
        svc_uid = user_info.pw_uid
        svc_gid = user_info.pw_gid
    except KeyError:
        raise SystemExit(f"Service user '{service_user}' does not exist on this system.") from None

    # Resolve and validate the users.yaml that this apply would leave active
    # before stopping the running service or touching disk. A readable staging
    # file wins on first install; otherwise the canonical protected copy stays
    # authoritative, matching _apply_secrets' copy/skip behavior.
    staged_users_yaml = Path(users_yaml_staging_path) if users_yaml_staging_path else None
    effective_users_yaml = (
        staged_users_yaml if staged_users_yaml is not None and staged_users_yaml.is_file() else USERS_YAML
    )
    try:
        _validate_protected_users_yaml(
            effective_users_yaml,
            service_user,
            require_existing_accounts=True,
            service_uid=svc_uid,
        )
    except ValueError as exc:
        raise SystemExit(
            f"Protected user isolation preflight failed; no installation changes were made:\n{exc}"
        ) from exc

    # Defensive validation: refuse to apply an install.conf whose
    # explicit DEFAULT_MODEL would fail load_config's startup check.
    # DEFAULT_MODEL itself is no longer required; when absent,
    # load_config uses MODEL_REGISTRY's ModelRole.AGENT row for the
    # active backend/provider.
    #
    # Normalize the same way load_config does (config.py reads both env
    # vars through .strip().lower() before validating). A hand-edited
    # install.conf with "DEFAULT_BACKEND": "Claude" or trailing whitespace
    # would otherwise compute the wrong effective provider here, pass
    # the apply gate, and then fail at startup. The legacy AGENT_BACKEND
    # key was already migrated to DEFAULT_BACKEND at the top of apply.
    #
    default_model_raw = env.get("DEFAULT_MODEL", "")
    agent_backend_raw = env.get("DEFAULT_BACKEND", "claude").strip().lower()
    # Write the normalized value back so the downstream goose-config and
    # sudoers gates (which read env.get("DEFAULT_BACKEND") raw) and the
    # written /etc/kai/env all see the canonical lowercase form. Only
    # when the key is present, to preserve the "claude omits the key"
    # contract. Without this, a hand-edited install.conf with
    # DEFAULT_BACKEND="Goose" passes validation here but skips goose
    # config deployment below.
    if "DEFAULT_BACKEND" in env:
        env["DEFAULT_BACKEND"] = agent_backend_raw
    # Reads the canonical DEFAULT_PROVIDER; a legacy LLM_PROVIDER key was
    # already migrated to it at the top of apply.
    provider_raw = env.get("DEFAULT_PROVIDER", "").strip().lower()
    eff_provider_for_check = "anthropic" if agent_backend_raw == "claude" else provider_raw
    try:
        registry_default_model = get_default_model_for_backend(agent_backend_raw, eff_provider_for_check)
    except LookupError as exc:
        raise SystemExit(
            f"MODEL_REGISTRY has no agent default for backend '{agent_backend_raw}' "
            f"and provider '{eff_provider_for_check}'. No usable default model is installed; "
            "fix src/kai/config.py or set an explicit compatible DEFAULT_MODEL before applying."
        ) from exc
    resolved_model = canonicalize_model_for_backend(default_model_raw or registry_default_model, agent_backend_raw)
    if default_model_raw and resolved_model != default_model_raw:
        env["DEFAULT_MODEL"] = resolved_model
    # Backend-aware validation: codex installs validate against
    # CODEX_MODELS only - no fallback to PROVIDER_MODELS["openai"].
    # That closes the regression where install.conf with
    # DEFAULT_BACKEND=codex and DEFAULT_MODEL=opus survived apply
    # because validate_model_for_provider accepted unknown effective-
    # provider "" unchecked, or where DEFAULT_MODEL=gpt-5.4-nano
    # (goose-valid) survived because the wizard mapped codex onto
    # openai's surface.
    if not validate_model_for_backend(resolved_model, agent_backend_raw, eff_provider_for_check):
        if agent_backend_raw == "codex":
            valid_models = sorted(CODEX_MODELS.keys())
            surface_label = "codex"
        else:
            valid_models = sorted(PROVIDER_MODELS.get(eff_provider_for_check, {}).keys())
            surface_label = f"provider '{eff_provider_for_check}'"
        if default_model_raw:
            msg = f"install.conf has DEFAULT_MODEL='{default_model_raw}' which is not valid for {surface_label}."
            remedy = "Remove DEFAULT_MODEL to use the registry default, or set a compatible model."
        else:
            msg = (
                f"MODEL_REGISTRY agent default '{registry_default_model}' is not valid for {surface_label}; "
                "no usable default model is installed."
            )
            remedy = "Fix MODEL_REGISTRY's agent row or set an explicit compatible DEFAULT_MODEL."
        valid_list = ", ".join(valid_models) or "(no curated list)"
        raise SystemExit(f"{msg} {remedy} Valid models: {valid_list}")

    dry_run = os.environ.get("DRY_RUN", "").strip() in ("1", "true", "yes")
    if dry_run:
        print("[DRY RUN] No changes will be made.\n")

    install_path = Path(install_dir)
    data_path = Path(data_dir)
    is_update = install_path.exists()

    if is_update:
        print(f"Updating existing installation at {install_dir}")
    else:
        print(f"Creating new installation at {install_dir}")
    print()

    # -- Stop service before making changes --
    _stop_service(platform, dry_run)

    # apply_succeeded is initialized BEFORE the try block so the
    # finally handler can always read it. Several apply-time
    # failures raise SystemExit (a BaseException, not an Exception)
    # rather than going through the `except Exception` handler;
    # without the pre-initialization, those paths would propagate
    # SystemExit out of the try, the finally would read
    # apply_succeeded, and Python would raise UnboundLocalError -
    # replacing the actionable apply failure with an internal
    # control-flow error. The default value of False is correct
    # because the flag flips True only after the last apply step
    # returns cleanly.
    apply_succeeded = False
    try:
        # -- Step 1: Create directories --
        # Resolve WORKSPACE_BASE, expanding ~ relative to the service user's home
        # (not root's, since we're running under sudo).
        ws_base_raw = env.get("WORKSPACE_BASE", "")
        ws_base: Path | None = None
        if ws_base_raw:
            if ws_base_raw.startswith("~"):
                svc_home = _user_home(service_user)
                # Strip ~ or ~/ prefix, then join with the service user's home.
                # Bare "~" produces an empty suffix, which resolves to svc_home itself.
                suffix = ws_base_raw.removeprefix("~").lstrip("/")
                ws_base = Path(svc_home) / suffix if suffix else Path(svc_home)
            else:
                ws_base = Path(ws_base_raw)
        _apply_directories(install_path, data_path, svc_uid, svc_gid, dry_run, ws_base)

        # Warn about traversal issues for workspace paths. These are non-fatal
        # since the user may fix permissions separately after install.
        ws_paths: list[Path] = []
        if ws_base:
            ws_paths.append(ws_base)
        ws_paths.extend(_parse_workspaces(env))
        for ws_path in ws_paths:
            warning = _check_traversal(ws_path, service_user)
            if warning:
                print(f"  WARNING: {warning}")

        # -- Step 2: Copy source --
        _apply_source(install_path, svc_uid, svc_gid, dry_run)

        # -- Step 3: Create/update venv --
        _apply_venv(install_path, is_update, dry_run)

        # -- Step 4: Copy models (if they exist in source) --
        _apply_models(install_path, dry_run)

        # -- Step 5: Write secrets --
        # Inject CODEX_BIN from the apply-time env so the operator's
        # explicit `sudo CODEX_BIN=... kai install apply` is honored
        # without round-tripping through the wizard. Apply-time env
        # wins over any value already in install.conf.
        #
        # Gate the apply-time shell env pass-through on the global
        # DEFAULT_BACKEND only: it is an ad-hoc escape hatch for global
        # backend installs (`sudo CODEX_BIN=... kai install apply`).
        # Per-user backend entries do NOT need it; their binary paths
        # are collected by the `make config` per-user backend scan
        # and arrive here through install.conf, which `_apply_sudoers`
        # consumes below. Codex AUTH for per-user entries stays
        # out-of-band (per-OS-user `codex login`).
        env_codex_bin = os.environ.get("CODEX_BIN")
        if env_codex_bin and env.get("DEFAULT_BACKEND") == "codex":
            env["CODEX_BIN"] = env_codex_bin
        # Mirror the same env pass-through for OPENCODE_BIN so an
        # `sudo OPENCODE_BIN=... kai install apply` invocation pins
        # the same path the running bot will resolve via
        # resolve_oneshot_binary("opencode"). Same scoping as
        # CODEX_BIN above: global installs only; per-user opencode
        # entries get their path from the wizard scan via
        # install.conf, with auth out-of-band (`opencode auth login`).
        env_opencode_bin = os.environ.get("OPENCODE_BIN")
        if env_opencode_bin and env.get("DEFAULT_BACKEND") == "opencode":
            env["OPENCODE_BIN"] = env_opencode_bin
        # And for GOOSE_BIN, completing the per-backend trio. Same
        # split as the two above: shell env override for global goose
        # installs, install.conf from the wizard scan for per-user
        # entries.
        env_goose_bin = os.environ.get("GOOSE_BIN")
        if env_goose_bin and env.get("DEFAULT_BACKEND") == "goose":
            env["GOOSE_BIN"] = env_goose_bin
        # And for CLAUDE_BIN, completing the per-backend set. The
        # DEFAULT_BACKEND gate compares against the runtime default
        # because the wizard omits the key on claude installs.
        env_claude_bin = os.environ.get("CLAUDE_BIN")
        if env_claude_bin and env.get("DEFAULT_BACKEND", "claude") == "claude":
            env["CLAUDE_BIN"] = env_claude_bin
        # DEFAULT_TIMEOUT migration. Operators upgrading without
        # re-running the wizard carry the legacy AGENT_TIMEOUT_SECONDS
        # key in install.conf; rewrite to the canonical name at apply
        # time so /etc/kai/env ends up clean and load_config does not
        # need the legacy fallback after the next start.
        if "DEFAULT_TIMEOUT" not in env and "AGENT_TIMEOUT_SECONDS" in env:
            env["DEFAULT_TIMEOUT"] = env["AGENT_TIMEOUT_SECONDS"]
        env.pop("AGENT_TIMEOUT_SECONDS", None)
        # Same migration for the session lifecycle keys (renamed from
        # the CLAUDE_-prefixed forms; they govern every backend's pool).
        if "AGENT_MAX_SESSION_HOURS" not in env and "CLAUDE_MAX_SESSION_HOURS" in env:
            env["AGENT_MAX_SESSION_HOURS"] = env["CLAUDE_MAX_SESSION_HOURS"]
        env.pop("CLAUDE_MAX_SESSION_HOURS", None)
        if "AGENT_IDLE_TIMEOUT" not in env and "CLAUDE_IDLE_TIMEOUT" in env:
            env["AGENT_IDLE_TIMEOUT"] = env["CLAUDE_IDLE_TIMEOUT"]
        env.pop("CLAUDE_IDLE_TIMEOUT", None)
        # Retired keys (no canonical replacement); drop them at apply
        # time so /etc/kai/env does not carry settings the runtime no
        # longer reads. Mirrors the `_retired_env_vars` warning map in
        # load_config.
        env.pop("CLAUDE_MAX_CONTEXT_WINDOW", None)
        env.pop("CLAUDE_MAX_BUDGET_USD", None)
        env.pop("BUDGET_CEILING", None)
        env.pop("PR_REVIEW_BUDGET_USD", None)
        env.pop("MEMORY_EXTRACTION_BUDGET_USD", None)
        env.pop("MEMORY_EPISODE_BUDGET_USD", None)
        env.pop("MEMORY_SCOPED_RECALL_ENABLED", None)
        env.pop("MEMORY_RECALL_SHADOW_ENABLED", None)
        _apply_secrets(env, dry_run, users_yaml_staging_path=users_yaml_staging_path)

        # -- Step 6: Deploy installed backend registry --
        _apply_backend_registry(service_user, env, dry_run)

        # -- Step 7: Deploy Goose config (if any goose-backed user) --
        # The function gates itself: global DEFAULT_BACKEND=goose or a
        # per-user backend override in users.yaml both mean some
        # session spawns `goose acp`; otherwise it no-ops.
        agent_backend = env.get("DEFAULT_BACKEND", "claude")
        _apply_goose_config(
            service_user,
            install_path,
            svc_uid,
            svc_gid,
            dry_run,
            agent_backend=agent_backend,
        )

        # -- Step 8: Configure sudoers --
        _apply_sudoers(
            service_user,
            dry_run,
            claude_bin=env.get("CLAUDE_BIN"),
            codex_bin=env.get("CODEX_BIN"),
            opencode_bin=env.get("OPENCODE_BIN"),
            goose_bin=env.get("GOOSE_BIN"),
            agent_backend=agent_backend,
        )

        # -- Step 9: Migrate runtime data --
        _apply_migrate(data_path, install_path, svc_uid, svc_gid, dry_run)

        # -- Step 10: Generate service definition --
        webhook_port = int(env.get("WEBHOOK_PORT", "8080"))
        _apply_service(install_dir, data_dir, service_user, platform, dry_run, webhook_port)
        # Apply path completed without exception. Flipping the flag
        # at the bottom of the try block (rather than in the
        # except branch) keeps the apply_succeeded state local and
        # unambiguous: True iff every step above ran cleanly. The
        # finally block reads it to decide how to handle a service
        # start failure.
        apply_succeeded = True

        # First-install staging handoff cleanup. Gated on `not dry_run`
        # because the dry-run contract is "no changes will be made";
        # unguarded cleanup would consume the one-shot staging file
        # during an inspection run and leave the next real apply with
        # nothing to copy. Cleanup runs inside the try block (rather
        # than after the finally) so it cannot delete the staging file
        # before service-start results are known - cleanup completes
        # before the service-start retry budget begins, and a
        # service-start failure after this point does not invalidate
        # the apply itself.
        if users_yaml_staging_path:
            if dry_run:
                print(f"[DRY RUN] Would unlink staging file: {users_yaml_staging_path}")
                print("[DRY RUN] Would strip users_yaml_staging_path from install.conf")
            else:
                Path(users_yaml_staging_path).unlink(missing_ok=True)
                _strip_install_conf_keys("users_yaml_staging_path")
    except Exception:
        print("\nInstallation failed. See error above.")
        print("The installation may be in a partial state.")
        print("Fix the issue and re-run: sudo python -m kai install apply")
        raise
    finally:
        # Always restart the service, even after failure. A partially updated
        # installation is better than an offline bot. Most steps are idempotent,
        # so re-running apply after fixing the cause will complete the update.
        try:
            _start_service(platform, dry_run)
        except ServiceStartError as e:
            # Verify confirmed the service is not registered after the
            # retry budget. Two paths:
            #
            # - apply already failed: emit a recovery hint and SWALLOW
            #   the start error. Python replaces the propagating
            #   exception if the finally block raises, and the original
            #   apply failure is the more important one to surface.
            #
            # - apply succeeded: re-raise. Without this propagation
            #   the install would print "Installed" and exit 0 with
            #   the daemon unregistered, which is the originating
            #   silent-failure mode this rewrite closes.
            print(f"ERROR: service failed to start after apply: {e}")
            if platform == "darwin":
                print("Manual recovery: sudo launchctl kickstart system/com.syrinx.kai")
                print("Then verify: sudo launchctl print system/com.syrinx.kai")
            else:
                print("Manual recovery: sudo systemctl start kai")
                print("Then verify: systemctl is-active kai")
            if apply_succeeded:
                raise
        except Exception:
            # Any other exception from _start_service is an unexpected
            # bug, not the verified-failure path. Same masking rule:
            # propagate only when apply succeeded, so a pre-existing
            # apply failure is not hidden.
            print("ERROR: unexpected failure while starting service.")
            if platform == "darwin":
                print("Manual recovery: sudo launchctl kickstart system/com.syrinx.kai")
            else:
                print("Manual recovery: sudo systemctl start kai")
            if apply_succeeded:
                raise

    # -- Summary --
    print()
    action = "Updated" if is_update else "Installed"
    if dry_run:
        print("[DRY RUN] No changes were made.")
    else:
        print(f"{action} successfully.")
        print(f"  Source:  {install_dir}")
        print(f"  Data:    {data_dir}")
        print("  Secrets: /etc/kai/env")
        print(f"  User:    {service_user}")
        # Remind the user that install.conf contains secrets and can be
        # cleaned up. Don't auto-delete - the user may want to re-run
        # apply or adjust config.
        if INSTALL_CONF.exists():
            print(
                f"\nNote: {INSTALL_CONF} contains secrets (bot token, webhook secret)."
                "\nYou can safely delete it now that secrets are in /etc/kai/env."
                "\nTo reconfigure later, re-run: python -m kai install config"
            )

        # Codex subscription auth reminder. Factored out so the gate
        # ("codex runs anywhere": agent backend OR memory reasoner is
        # codex) and the per-user enumeration can be unit-tested
        # without mocking the entire apply pipeline. Returns None when
        # no reminder applies.
        reminder = _build_codex_login_reminder(env, service_user)
        if reminder:
            print("\n" + reminder)


def _apply_directories(
    install_path: Path,
    data_path: Path,
    svc_uid: int,
    svc_gid: int,
    dry_run: bool,
    workspace_base: Path | None = None,
) -> None:
    """
    Create the directory structure for the installation.

    Builds a list of (path, uid, gid, mode) tuples for all required
    directories and creates any that don't already exist. The install
    tree is root-owned except for the workspace and data directories,
    which must be writable by the service user.

    Args:
        install_path: Root of the install tree (e.g., /opt/kai).
        data_path: Writable data directory (e.g., /var/lib/kai).
        svc_uid: UID of the service user.
        svc_gid: GID of the service user.
        dry_run: If True, print what would be created without doing it.
        workspace_base: Optional base directory for workspace name resolution.
    """
    # The home dir under the install path must be writable by the service
    # user so skills/ and other runtime dirs can be created inside it. The rest
    # of the install tree stays root-owned and read-only.
    home_path = install_path / "home"
    dirs: list[tuple[Path, int, int, int]] = [
        (install_path, 0, 0, 0o755),  # root-owned install dir
        (home_path, svc_uid, svc_gid, 0o755),  # user-writable home workspace
        (data_path, svc_uid, svc_gid, 0o755),  # user-owned data dir
        (data_path / "logs", svc_uid, svc_gid, 0o755),
        (data_path / "files", svc_uid, svc_gid, _PRIVATE_USER_ROOT_MODE),
        (data_path / "history", svc_uid, svc_gid, 0o755),
        (data_path / "memory", svc_uid, svc_gid, _PRIVATE_USER_ROOT_MODE),
        # Per-user preferences root (#400). Top-level dir is service-
        # owned so the bot can lazily create new preferences/<chat_id>/
        # subdirs at first message; per-user subdirs are pre-created
        # and chowned in _apply_migrate when the user has a distinct
        # os_user.
        (data_path / "preferences", svc_uid, svc_gid, _PRIVATE_USER_ROOT_MODE),
        # Per-user home root (#353). Top-level dir is service-owned so
        # the bot can lazily create new home/<chat_id>/ subdirs at first
        # message; per-user subdirs are pre-created and chowned in
        # _apply_migrate when the user has a distinct os_user.
        (data_path / "home", svc_uid, svc_gid, _PRIVATE_USER_ROOT_MODE),
        (Path("/etc/kai"), 0, 0, 0o755),
    ]

    # Create WORKSPACE_BASE if configured. The bot validates this directory
    # exists at startup, and on a fresh install it won't exist yet.
    if workspace_base:
        dirs.append((workspace_base, svc_uid, svc_gid, 0o755))

    for path, uid, gid, mode in dirs:
        if path.exists():
            continue
        if dry_run:
            owner = f"{uid}:{gid}"
            print(f"[DRY RUN] Would create directory: {path} ({owner} {oct(mode)})")
        else:
            path.mkdir(parents=True, exist_ok=True)
            os.chmod(path, mode)
            os.chown(path, uid, gid)
            print(f"  Created {path}")


def _migrate_identity_to_claude_md(
    install_path: Path,
    svc_uid: int,
    svc_gid: int,
    dry_run: bool,
) -> None:
    """
    One-time migration: convert the legacy IDENTITY.md plus CLAUDE.md symlink
    layout into a single regular CLAUDE.md file at home/.claude/CLAUDE.md.

    Pre-existing installs from before this change shipped a regular
    home/IDENTITY.md (operator-customized) plus a home/.claude/CLAUDE.md
    symlink with target ../IDENTITY.md. This helper detects that layout
    and converts it in place WITHOUT discarding the operator's
    customizations: rows 1 and 2 use Path.replace (atomic rename), so
    there is no partial-write window where the destination could be
    half-populated and the source could then be deleted on the next
    install pass. See the per-row comments below for full failure-mode
    analysis.

    Six pre-states (full 2x3 cross of file presence and CLAUDE.md type):

      1. IDENTITY.md regular file + CLAUDE.md is a symlink (target
         normally ../IDENTITY.md, but the guard catches any symlink
         target; the migration treats every symlink-with-IDENTITY.md
         pre-state as the legacy layout deliberately, since the
         operator's IDENTITY.md content is the canonical identity to
         preserve regardless of where the symlink originally pointed):
         replace the symlink with a regular CLAUDE.md holding the
         IDENTITY.md content; delete IDENTITY.md.
      2. IDENTITY.md regular file + CLAUDE.md missing: move IDENTITY.md
         to CLAUDE.md (preserves content, removes the source).
      3. IDENTITY.md regular file + CLAUDE.md regular file (inconsistent):
         keep CLAUDE.md (canonical going forward), delete IDENTITY.md,
         warn the operator so they can sanity-check which copy survived.
      4. IDENTITY.md missing + CLAUDE.md is a symlink (broken or
         otherwise): unlink the symlink so the seed step can populate
         a regular file cleanly.
      5. IDENTITY.md missing + CLAUDE.md regular file: already migrated.
         Emit a positive "already migrated; no action" log on every
         reinstall so operators see a non-silent confirmation.
      6. Neither exists: genuine fresh install. No-op here; the seed
         step in `_apply_source` (the conditional CLAUDE.md copy at the
         end) emits its own log when it copies the template.

    Args:
        install_path: Root of the install tree (e.g. /opt/kai).
        svc_uid: Service user UID. The migrated CLAUDE.md is owned by
            the service user so inner Claude can edit it from Telegram.
        svc_gid: Service group GID.
        dry_run: If True, log the actions that would happen without doing them.
    """
    identity_dst = install_path / "home" / "IDENTITY.md"
    claude_md_dst = install_path / "home" / ".claude" / "CLAUDE.md"
    # is_file() follows symlinks, which would conflate the symlink case
    # with the regular-file case. Pair it with `not is_symlink()` so
    # `identity_exists` and `claude_md_is_regular` are strictly regular
    # files, never symlinks.
    identity_exists = identity_dst.is_file() and not identity_dst.is_symlink()
    claude_md_is_symlink = claude_md_dst.is_symlink()
    claude_md_is_regular = claude_md_dst.is_file() and not claude_md_dst.is_symlink()

    if identity_exists and claude_md_is_symlink:
        # Row 1: replace the symlink with a regular CLAUDE.md holding
        # the IDENTITY.md content. Done in a single atomic step via
        # Path.replace, which on POSIX is rename(2): it operates on
        # directory entries, not content, so there is no partial-write
        # window where the destination is half-populated. The symlink
        # at claude_md_dst is replaced atomically (POSIX rename on a
        # symlink target operates on the symlink directory entry, not
        # the target file), and identity_dst's directory entry is
        # removed in the same call. Mode bits and ownership both ride
        # along with the inode, so no chmod/copystat dance is needed
        # here - only the chown to the service user.
        #
        # Partial-failure shape:
        #   - rename fails: symlink intact, IDENTITY.md intact -> row 1
        #     fires again on re-run.
        #   - chown fails: dst is correct content + correct mode but
        #     wrong ownership, IDENTITY.md is gone -> re-run enters
        #     row 5 (already migrated, no-op). The broader install
        #     ownership pass will reconcile ownership later in the
        #     same install run.
        # The prior shutil.copy2 implementation had a partial-write
        # window where copy2 could leave the dst in row-3 shape (both
        # files regular, dst partial), at which point the next install
        # would silently delete IDENTITY.md per the row-3 rule. Switching
        # to rename closes that data-loss path.
        if dry_run:
            print(f"[DRY RUN] Would migrate: replace symlink {claude_md_dst} with {identity_dst} (atomic rename)")
        else:
            identity_dst.replace(claude_md_dst)
            os.chown(claude_md_dst, svc_uid, svc_gid)
            print(f"  Migrated {identity_dst} content into regular file at {claude_md_dst}; removed {identity_dst}")
    elif identity_exists and not claude_md_dst.exists():
        # Row 2: rename IDENTITY.md to CLAUDE.md. Path.replace is atomic
        # within the same filesystem (the install tree always is), so
        # there is no transient state where neither file exists.
        if dry_run:
            print(f"[DRY RUN] Would move {identity_dst} -> {claude_md_dst}")
        else:
            claude_md_dst.parent.mkdir(parents=True, exist_ok=True)
            identity_dst.replace(claude_md_dst)
            os.chown(claude_md_dst, svc_uid, svc_gid)
            print(f"  Moved {identity_dst} -> {claude_md_dst}")
    elif identity_exists and claude_md_is_regular:
        # Row 3: inconsistent state. Two regular files - typically the
        # remnant of an interrupted prior install. Keep CLAUDE.md (the
        # canonical destination) and delete IDENTITY.md so reinstalls
        # converge on a single source of truth. WARN so an operator who
        # had real content only in IDENTITY.md notices the divergence.
        if dry_run:
            print(f"[DRY RUN] Would delete {identity_dst} ({claude_md_dst} regular file already in place)")
        else:
            identity_dst.unlink()
            print(
                f"  WARNING: both {identity_dst} and {claude_md_dst} "
                f"(regular file) existed; kept {claude_md_dst}, "
                f"deleted {identity_dst}"
            )
    elif claude_md_is_symlink and not identity_exists:
        # Row 4: CLAUDE.md is a symlink and IDENTITY.md is absent. Two
        # subcases land here. Either the symlink target is missing (the
        # legacy ../IDENTITY.md target was removed manually) or the
        # symlink points at some other valid path (an exotic post-merge
        # tarball-restore state). Both subcases need an unlink so the
        # seed step in _apply_source produces a clean regular file:
        #   - Broken target: Path.exists() returns False, so the seed
        #     step would proceed and shutil.copy2 would follow the
        #     symlink, writing the template content to the target path
        #     (e.g. <install>/home/IDENTITY.md for the legacy
        #     `../IDENTITY.md` target) rather than to the intended
        #     <install>/home/.claude/CLAUDE.md. The install would
        #     resurrect the legacy IDENTITY.md plus symlink layout
        #     instead of retiring it. (A "fails outright" path is
        #     possible if the target's parent dir is missing, but the
        #     home/ parent always exists on a real install, so the
        #     write-to-wrong-path outcome is the common one.)
        #   - Valid non-IDENTITY target: Path.exists() returns True, so
        #     the seed step would skip, and the install would keep a
        #     symlink pointing at unrelated content as the operator's
        #     identity. Wrong by definition.
        # Unlinking covers both subcases identically.
        if dry_run:
            print(f"[DRY RUN] Would remove symlink at {claude_md_dst} (no IDENTITY.md present)")
        else:
            claude_md_dst.unlink()
            print(f"  Removed symlink at {claude_md_dst} (no IDENTITY.md present)")
    elif claude_md_is_regular and not identity_exists:
        # Row 5: already migrated. Emit a positive log line on every
        # reinstall so operators get visible confirmation rather than
        # silent inaction. Mirrors the existing log discipline (see the
        # MEMORY.md migration above).
        if dry_run:
            print("[DRY RUN] Identity surface already migrated; no action")
        else:
            print("  Identity surface already migrated; no action")
    # else: row 6 (neither exists). Genuine fresh install. Silent at
    # this step; the conditional seed at the end of _apply_source emits
    # "Seeded CLAUDE.md from template" when it copies the template into
    # the destination.


# Sentinel header used by `_migrate_recalled_memory_section` to detect
# whether a per-user CLAUDE.md already carries the Reading Recalled
# Memory section. Module-level constant so the migration helper, the
# dry-run preview branch in the per-user CLAUDE.md seed loop, and the
# sibling install tests all read the same literal. A drift in this
# string would let the migration re-append on every install (sentinel
# always missing) and silently double-write the section.
_RECALLED_MEMORY_SECTION_HEADER = "## Reading Recalled Memory"


def _migrate_recalled_memory_section(
    claude_dst: Path,
    template_path: Path,
    *,
    dry_run: bool,
) -> bool | None:
    """
    Idempotently append the Reading Recalled Memory section from the
    tracked CLAUDE.md template to an existing per-user CLAUDE.md copy.

    Three return states so the dry-run preview can distinguish the
    legitimate no-op from any failure path:

    - True: the section would be (or was) appended. Sentinel absent in
      claude_dst, template present and well-formed.
    - False: the section is already present in claude_dst (sentinel
      header matched line-strip-equal). Nothing to do; not an error.
    - None: helper could not proceed. Failure paths include claude_dst
      not a regular file, claude_dst unreadable, template_path
      unreadable, and template missing the section header. Each path
      prints its own operator-facing warning before returning None, so
      the caller does not need to surface additional output.

    The three-state return is the structural distinction that lets the
    dry-run preview branch in `_apply_migrate` print accurate operator-
    facing text per case without re-checking the sentinel separately;
    truthiness alone (the prior bool return) conflated sentinel-present
    with the failure paths and produced a misleading "section already
    present" line when the helper had actually warned about a different
    problem.

    The section is identified by its header line
    (`## Reading Recalled Memory`, the module-level
    `_RECALLED_MEMORY_SECTION_HEADER` constant). If the header already
    appears as a line in the per-user copy (operator added it manually,
    a previous migration ran, or the seed step just copied a current
    template), the function returns False. Otherwise the section text
    is extracted from the template by header-bounded scan (header line
    through the next `## ` line or EOF) and appended verbatim with a
    single blank-line separator.

    Atomic via Path.replace on a temp file in the same directory. The
    rename closes the partial-write window where a crash or signal
    mid-write could leave the per-user copy half-populated. The temp
    file lives in claude_dst's parent so the rename stays within one
    filesystem; `Path.replace` delegates to `os.replace`, which calls
    `rename(2)` and surfaces `EXDEV` as `OSError` for cross-filesystem
    targets. No silent fallback to copy+unlink (that is `shutil.move`'s
    behavior, not `Path.replace`'s); the colocation here is a hard
    requirement for the rename to succeed at all.

    Ownership reconciliation rides on the caller. Path.replace is
    rename(2); the post-rename file inherits the temp file's ownership
    (the install runner, typically root via `sudo make install`), not
    the per-user `os_user` the inner Claude subprocess writes as. The
    per-user CLAUDE.md seed loop calls this helper BEFORE the
    `_set_ownership` chown step, so the appended file's ownership is
    reconciled in the same iteration; the comment block at the chown
    site documents the #347 regression shape this prevents.

    No in-source cleanup window. The helper is idempotent on every
    install (sentinel check returns False on a no-op iteration with
    no file IO past the existence + read), and is short enough that
    keeping it indefinitely is cheaper than tracking a removal date.
    Removal is a follow-up PR when the operator decides the population
    of pre-migration per-user copies is empirically zero.

    Args:
        claude_dst: The per-user CLAUDE.md to append to. Path returned
            by the per-user seed loop's resolution of
            `<DATA_DIR>/home/<chat_id>/.claude/CLAUDE.md`.
        template_path: The tracked template
            (`templates/.claude/CLAUDE.md`). Section text is extracted
            from here on every call so a future revision to the
            section's wording in the template propagates to the next
            install's migration run.
        dry_run: If True, return the would-be modification status
            without writing anything; the caller's dry-run preview
            branch consumes the three-state return to print the
            appropriate preview line.
    """
    # Skip if the per-user copy does not exist. The seed step earlier
    # in the loop just wrote a fresh copy from the current template if
    # claude_dst was missing, so by the time live execution reaches
    # this helper, claude_dst should exist; the explicit check guards
    # the dry-run branch (where the seed is a preview-only print) and
    # the operator-managed-override branch (where the seed is skipped
    # entirely).
    if not claude_dst.is_file():
        return None

    # Read the per-user copy and check for the sentinel header on a
    # full-line basis. Substring match against the file body would
    # false-positive on a future section like "## Reading Recalled
    # Memory Notes" that starts with the same prefix; line-strip-equal
    # is the precise check that matches what the migration's own
    # append produces.
    try:
        existing = claude_dst.read_text()
    except OSError as exc:
        # Broken symlink, permissions, mount issue. Match the
        # ensure_user_home swallow pattern: surface to operator log,
        # continue the install rather than abort. The next install
        # retries; the lazy seed path will also retry on first message.
        print(f"  WARNING: could not read {claude_dst} for migration: {exc}")
        return None

    for line in existing.splitlines():
        if line.strip() == _RECALLED_MEMORY_SECTION_HEADER:
            return False  # Sentinel present; legitimate no-op.

    # Read the template and extract the section by header-bounded scan.
    # Reading template_path on every call (rather than caching at module
    # import) keeps the helper testable with arbitrary template paths
    # via tmp_path fixtures.
    try:
        template_text = template_path.read_text()
    except OSError as exc:
        print(f"  WARNING: could not read template {template_path}: {exc}")
        return None

    template_lines = template_text.splitlines(keepends=True)
    section_start = None
    for i, line in enumerate(template_lines):
        if line.strip() == _RECALLED_MEMORY_SECTION_HEADER:
            section_start = i
            break

    if section_start is None:
        # Template predates the migration or has been edited to remove
        # the section. The migration cannot reconstruct content that is
        # not in the template; warn so the operator notices and either
        # restores the template or removes this helper from the install
        # flow. Mirrors the placeholder-warning shape the per-user
        # CLAUDE.md seed loop uses ("WARNING: <template> not found").
        print(
            f"  WARNING: {template_path} is missing the "
            f"{_RECALLED_MEMORY_SECTION_HEADER!r} section; "
            f"cannot migrate {claude_dst}"
        )
        return None

    # Find the section terminator: the next top-level `## ` line after
    # the header, or EOF. The header itself is excluded from the
    # `## ` scan via the `section_start + 1` start so it does not match
    # itself as its own terminator. lstrip().startswith matches the
    # whitespace-tolerant strip-equal check the sentinel scan uses,
    # so a future indented top-level header still terminates the section.
    section_end = len(template_lines)
    for i in range(section_start + 1, len(template_lines)):
        if template_lines[i].lstrip().startswith("## "):
            section_end = i
            break

    # Strip trailing blank lines from the extracted section so the
    # append produces exactly one trailing newline, matching the
    # tracked template's overall file shape (no trailing blank lines).
    section_text = "".join(template_lines[section_start:section_end]).rstrip() + "\n"

    if dry_run:
        # Caller's dry-run preview branch prints the operator-facing
        # line; this helper only signals "would modify" by returning
        # True. Returning early before the temp-file write means a
        # dry-run pass has zero filesystem side effects.
        return True

    # Append with a single blank-line separator. The rstrip() drops any
    # trailing whitespace from the existing per-user copy so the
    # separator is always exactly one blank line regardless of whether
    # the source file ended in zero, one, or multiple newlines.
    new_content = existing.rstrip() + "\n\n" + section_text

    # Atomic write via temp file + Path.replace in the same directory.
    # The temp file's name pairs the destination's name with a `.tmp`
    # suffix; deterministic enough that a crash-interrupted earlier run
    # leaves a recoverable artifact at a known path rather than a random
    # mkstemp name. The try/except cleans up the temp file when
    # write_text fails partway, so a partial-write does not leave debris
    # in the .claude/ directory.
    temp_path = claude_dst.parent / (claude_dst.name + ".tmp")
    try:
        temp_path.write_text(new_content)
        temp_path.replace(claude_dst)
    except OSError as exc:
        # Clean up the partial temp file before propagating so a retry
        # is not blocked by a stale .tmp file. unlink(missing_ok=True)
        # handles the case where write_text never created the file.
        temp_path.unlink(missing_ok=True)
        print(f"  ERROR: could not migrate {claude_dst}: {exc}")
        raise

    return True


def _retire_install_home_claude(install_path: Path, dry_run: bool) -> None:
    """
    Remove the dead `<install>/home/.claude/` subtree and any legacy
    `<install>/home/IDENTITY.md` left behind from the pre-#442 layout.

    Both paths predate the per-user `home_workspace` migration in #353.
    Since #353 every session resolves identity from the per-user
    `<DATA_DIR>/home/<chat_id>/.claude/CLAUDE.md`; nothing in the
    runtime reads either of the paths this helper deletes. Issue #447
    retires the install-tree scaffolding entirely. The per-user
    `CLAUDE.md` is seeded by `_apply_migrate`'s home block (eager,
    install-time) and `backend.ensure_user_home` (lazy, first-message
    fallback) from the still-tracked `templates/.claude/CLAUDE.md`.

    Behavior is idempotent: pre-checks (`is_dir()` for the subtree,
    `is_symlink() or is_file()` for IDENTITY.md) make a missing path
    a silent no-op. In dry-run mode every removal that WOULD happen
    is announced with a `[DRY RUN] Would remove ...` line so the
    dry-run preview matches the live install line-for-line.

    Every removed FILE is logged with its byte size before deletion so
    an operator who customized the dead `CLAUDE.md` content (and finds
    out post-install) can recover from the printed log if they kept a
    backup of `<install_path>/`. Empty directories under `.claude/`
    are not enumerated individually; `shutil.rmtree` removes them with
    the rest of the subtree, but they carry no operator content so
    omitting them from the log surfaces no recovery information.
    """
    claude_dir = install_path / "home" / ".claude"
    identity_md = install_path / "home" / "IDENTITY.md"

    if claude_dir.is_dir():
        # Enumerate before deleting so the log captures every retired
        # path plus its byte size. The lstat() (not stat()) below avoids
        # tracebacking on a broken file-symlink target - the row-1
        # pre-state has CLAUDE.md as a symlink that may point at a
        # removed IDENTITY.md. The contents of <install>/home/.claude/
        # in any pre-state we expect (regular files, file symlinks,
        # `skills/` and `history/` subdirs) do not include directory
        # symlinks, so rglob's directory-symlink-following behavior
        # (which varies across Python versions) does not affect this
        # helper in practice; if a future operator drops a directory
        # symlink under .claude/ they will see its target's files in
        # the log but `shutil.rmtree` removes the symlink itself
        # without descending, so the target is left intact.
        #
        # Verb tense: the live log uses "Removing" because the lines
        # are printed BEFORE shutil.rmtree runs (the enumeration needs
        # to lstat each file while it still exists). If rmtree raises
        # OSError - permissions, mounted filesystem, etc. - the
        # "Removing" lines accurately describe what was attempted, and
        # the trailing "Removed dead {dir}" summary will be absent so
        # the operator can see the deletion did not complete. A
        # past-tense per-file line here would lie on rmtree failure.
        for path in sorted(claude_dir.rglob("*")):
            if path.is_file() or path.is_symlink():
                try:
                    size = path.lstat().st_size
                except OSError:
                    size = 0
                if dry_run:
                    print(f"[DRY RUN] Would remove {path} ({size} bytes)")
                else:
                    print(f"  Removing {path} ({size} bytes)")
        if dry_run:
            print(f"[DRY RUN] Would remove dead {claude_dir}; nothing reads this path post-#447")
        else:
            # No ignore_errors: the is_dir() pre-check above makes the
            # call safe, and surfacing a real OSError (permissions,
            # mounted FS, etc.) is preferred to silent failure. The
            # past-tense "Removed dead" summary is correct because it
            # fires only after rmtree returns successfully. Catch the
            # OSError just long enough to print a clear operator-readable
            # error line so the traceback is not the first signal, then
            # re-raise to abort the install (a partial cleanup that
            # leaves stale content visible is worse than aborting).
            try:
                shutil.rmtree(claude_dir)
            except OSError as exc:
                print(f"  ERROR: could not remove {claude_dir}: {exc}")
                raise
            print(f"  Removed dead {claude_dir}; nothing reads this path post-#447")

    # The IDENTITY.md path is retired wholesale by #447. Whatever is
    # at the path - regular file (legacy operator content), valid
    # symlink, or broken symlink - is removed. `is_symlink() or
    # is_file()` covers all three: is_symlink() is True for both
    # valid and broken symlinks; is_file() follows symlinks and is
    # True only for regular files or symlinks pointing at regular
    # files (already caught by the is_symlink() arm). The pair
    # excludes directories, which would be a bizarre malformed state
    # at this path and is left for an explicit operator question
    # rather than silent rmtree. lstat avoids tracebacking on a
    # broken symlink target.
    if identity_md.is_symlink() or identity_md.is_file():
        try:
            size = identity_md.lstat().st_size
        except OSError:
            size = 0
        if dry_run:
            print(f"[DRY RUN] Would remove legacy {identity_md} ({size} bytes); content was identity baseline only")
        else:
            # Same try/except surface as the rmtree above: surface an
            # operator-readable error line BEFORE the traceback aborts
            # the install. Likelihood is low (install runs as root and
            # the pre-check is a short window), but the asymmetry would
            # be surprising given rmtree's wrap.
            try:
                identity_md.unlink()
            except OSError as exc:
                print(f"  ERROR: could not remove {identity_md}: {exc}")
                raise
            print(f"  Removed legacy {identity_md} ({size} bytes); content was identity baseline only")


def _retire_install_home_dir(install_path: Path, dry_run: bool) -> None:
    """
    Remove the `<install>/home/` directory if (and only if) it contains
    nothing more than an orphan `config/goose-config.yaml`.

    `<install>/home/` previously held the `.claude/` subtree, a legacy
    `IDENTITY.md`, a `config/goose-config.yaml` template, and a
    `files/` tree used by the pre-DATA_DIR uploaded-files migration.
    `_retire_install_home_claude` (called earlier in `_apply_source`)
    removes the first two; the goose-config template now lives at
    `<install>/config/goose-config.yaml`. The `files/` tree is still
    read by `_apply_migrate`'s legacy uploaded-files block as a
    backup source on hosts that have not yet migrated.

    The helper enforces the conservative guardrail from the originating
    issue: remove only when the directory contains nothing more than
    the orphan `config/` subdir (which may hold at most
    `goose-config.yaml`). If anything else is present - a leftover
    `files/` backup tree, an operator-placed file, a future code
    change leaving fresh content - the helper logs the unexpected
    paths and refuses to remove. The dir persists, the operator can
    investigate, and no data is destroyed.

    The cleanup catches three end states:
      - Clean install: the directory never existed; silent no-op.
      - Existing-install upgrade with only orphan `home/config/goose-config.yaml`:
        remove the orphan and the empty `home/` parent.
      - Existing-install upgrade with extra content (e.g., un-migrated
        `home/files/`): refuse, log the unexpected paths, return.

    Logging every removed file with its byte size before rmtree is
    symmetric with `_retire_install_home_claude` and matches the
    install-time loud-removal contract.
    """
    home_dir = install_path / "home"
    if not home_dir.exists():
        return

    # Per the issue contract: only `config/` is an allowed top-level
    # entry, and only `goose-config.yaml` is allowed inside it.
    # Anything else means refuse-and-log; do not destroy.
    top_unexpected = sorted(entry.name for entry in home_dir.iterdir() if entry.name != "config")
    config_subdir = home_dir / "config"
    config_unexpected: list[str] = []
    if config_subdir.is_dir():
        config_unexpected = sorted(entry.name for entry in config_subdir.iterdir() if entry.name != "goose-config.yaml")

    if top_unexpected or config_unexpected:
        # Single refusal line names everything unexpected so the
        # operator can locate it without re-running the install with
        # extra logging. Dry-run and live both emit the same message
        # because no deletion happens either way.
        prefix = "[DRY RUN] " if dry_run else "  "
        unexpected_paths: list[str] = []
        unexpected_paths.extend(str(home_dir / name) for name in top_unexpected)
        unexpected_paths.extend(str(config_subdir / name) for name in config_unexpected)
        print(
            f"{prefix}Refusing to remove {home_dir}: unexpected content present "
            f"({', '.join(unexpected_paths)}). Investigate and remove manually if intended."
        )
        return

    # Safe to remove: directory is either empty or contains exactly
    # `config/[goose-config.yaml]`. Enumerate any surviving file (the
    # orphan goose-config) first so the log captures its byte size
    # before rmtree runs.
    for path in sorted(home_dir.rglob("*")):
        if path.is_file() or path.is_symlink():
            try:
                size = path.lstat().st_size
            except OSError:
                size = 0
            if dry_run:
                print(f"[DRY RUN] Would remove {path} ({size} bytes)")
            else:
                print(f"  Removing {path} ({size} bytes)")

    if dry_run:
        print(f"[DRY RUN] Would remove retired {home_dir}")
    else:
        # Same try/except discipline as the rmtree in
        # `_retire_install_home_claude`: surface an operator-readable
        # error line BEFORE the traceback aborts the install.
        try:
            shutil.rmtree(home_dir)
        except OSError as exc:
            print(f"  ERROR: could not remove {home_dir}: {exc}")
            raise
        print(f"  Removed retired {home_dir}")


def _apply_source(install_path: Path, svc_uid: int, svc_gid: int, dry_run: bool) -> None:
    """Copy source tree, package metadata, optional constraints, and home config into the install tree."""
    src_src = PROJECT_ROOT / "src"
    src_dst = install_path / "src"
    pyproject_src = PROJECT_ROOT / "pyproject.toml"
    pyproject_dst = install_path / "pyproject.toml"
    constraints_src = PROJECT_ROOT / _INSTALL_CONSTRAINTS_REL
    constraints_dst = install_path / _INSTALL_CONSTRAINTS_REL
    # Config templates (e.g. goose-config.yaml) referenced by later
    # install steps like _apply_goose_config(). Root-owned since
    # these are static templates, not runtime data. Per-user runtime
    # CLAUDE.md is seeded by `_apply_migrate`'s home block (eager)
    # and `backend.ensure_user_home` (lazy, first-message fallback);
    # neither touches the install tree.
    config_src = PROJECT_ROOT / "templates" / "config"
    config_dst = install_path / "config"

    # One-time: rename workspace/ to home/ at the install location.
    # The directory was renamed in the source tree; this migrates the
    # production install so runtime content (skills, files, notes) is
    # preserved rather than orphaned.
    old_ws = install_path / "workspace"
    new_ws = install_path / "home"
    if old_ws.is_dir() and not new_ws.exists():
        if dry_run:
            print(f"[DRY RUN] Would rename: {old_ws} -> {new_ws}")
        else:
            old_ws.rename(new_ws)
            print(f"  Renamed {old_ws} -> {new_ws}")

    # Retire the dead <install>/home/.claude/ subtree (issue #447). This
    # path predates #353's per-user `home_workspace` migration; nothing
    # reads it at runtime since then. The helper also removes any legacy
    # <install>/home/IDENTITY.md left behind from the pre-#442 layout.
    # Runs before the rest of _apply_source so subsequent steps work
    # against a clean state; no remaining install-step depends on the
    # directory existing. Dry-run emits matching `[DRY RUN] Would remove`
    # lines so the preview matches the live behavior.
    _retire_install_home_claude(install_path, dry_run)

    if dry_run:
        print(f"[DRY RUN] Would copy: {src_src} -> {src_dst}")
        print(f"[DRY RUN] Would copy: {pyproject_src} -> {pyproject_dst}")
        if constraints_src.is_file():
            print(f"[DRY RUN] Would copy: {constraints_src} -> {constraints_dst}")
        elif constraints_dst.exists():
            print(f"[DRY RUN] Would remove stale install constraints: {constraints_dst}")
        if config_src.is_dir():
            print(f"[DRY RUN] Would copy: {config_src} -> {config_dst}")
        # Dry-run preview for the empty-home cleanup. Matches the
        # live cleanup at the end of this function so the operator
        # sees what would happen on an existing-install upgrade
        # where `<install>/home/config/` still exists.
        _retire_install_home_dir(install_path, dry_run=dry_run)
        return

    # The install source is generated, root-owned state with no runtime data.
    # Replace it exactly so removed modules cannot remain importable or be
    # included by the subsequent non-editable pip install.
    _copy_tree(src_src, src_dst, _SOURCE_EXCLUDES, replace=True)
    _set_ownership(src_dst, 0, 0, recursive=True)
    print(f"  Copied source to {src_dst}")

    shutil.copy2(pyproject_src, pyproject_dst)
    os.chown(pyproject_dst, 0, 0)
    print(f"  Copied {pyproject_dst}")

    if constraints_src.is_file():
        constraints_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(constraints_src, constraints_dst)
        os.chown(constraints_dst, 0, 0)
        print(f"  Copied {constraints_dst}")
    elif constraints_dst.exists():
        constraints_dst.unlink()
        print(f"  Removed stale install constraints {constraints_dst}")

    # Copy templates/config/ -> install/config/ (config templates like
    # goose-config.yaml). These are static templates referenced by
    # later install steps - e.g. _apply_goose_config() reads the Goose
    # extension config from here. Root-owned since they're installer
    # input, not runtime output.
    if config_src.is_dir():
        # `_copy_tree` creates `config_dst` itself (`install_path` is
        # guaranteed to exist by this point in the install flow, so
        # no parent-creation step is needed here).
        _copy_tree(config_src, config_dst)
        _set_ownership(config_dst, 0, 0, recursive=True)
        print(f"  Copied config templates to {config_dst}")

    # Retire the now-vestigial `<install>/home/` directory. On a clean
    # install nothing under it exists; on an existing-install upgrade
    # the cleanup above has already removed `.claude/` and `IDENTITY.md`,
    # and the goose-config relocation left an orphan `home/config/`.
    # The helper handles both end states. `dry_run` is propagated
    # rather than hardcoded so a future restructure of the early
    # return above cannot silently suppress dry-run output here.
    _retire_install_home_dir(install_path, dry_run=dry_run)


def _resolve_venv_base_python() -> str:
    """Resolve and validate the Python interpreter used to create/repair a venv."""
    python = shutil.which("python3.13")
    if python is None and sys.version_info >= (3, 13):
        # `sudo` may reset PATH and hide Homebrew even though the installer is
        # already running under a valid project-venv interpreter. Python's venv
        # module resolves that interpreter's base executable when creating or
        # upgrading the target environment.
        python = sys.executable
    python = python or shutil.which("python3") or "python3"
    result = subprocess.run(
        [python, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        parts = result.stdout.strip().split(".")
        major, minor = parts[0], parts[1]
        if (int(major), int(minor)) < (3, 13):
            raise SystemExit(
                f"Python >= 3.13 required, but {python} is {result.stdout.strip()}. "
                f"Install Python 3.13+ and ensure it is on PATH."
            )
    return python


def _optional_file_checksum(path: Path) -> str:
    """Return a file checksum, or an empty string when the optional file is absent."""
    return _file_checksum(path) if path.is_file() else ""


def _apply_venv(install_path: Path, is_update: bool, dry_run: bool) -> None:
    """
    Create or update the virtual environment in the install location.

    On a fresh install, creates a venv with the system Python and pip-installs
    the package with optional extras (totp, tts). On update, compares both the
    pyproject.toml and src/ checksums to detect changes and only reinstalls if
    needed. Both checks are required because the install is non-editable; pip
    copies the package into site-packages, so source changes at the install
    path are not reflected in the venv without a reinstall. Rejects Python
    versions below 3.13.

    Args:
        install_path: Root of the install tree containing src/ and pyproject.toml.
        is_update: True if updating an existing installation (vs fresh install).
        dry_run: If True, print what would be done without doing it.
    """
    venv_path = install_path / "venv"
    venv_python = venv_path / "bin" / "python"
    pyproject_dst = install_path / "pyproject.toml"
    constraints_dst = install_path / _INSTALL_CONSTRAINTS_REL
    src_dst = install_path / "src"
    build_path = install_path / "build"

    if is_update and venv_path.exists():
        # Homebrew removes the old Cellar path when Python is upgraded. A venv
        # created with symlinks can therefore remain on disk while its Python
        # interpreter becomes a dangling symlink. Detect that before the
        # checksum fast-path or a no-source-change update would leave a service
        # that cannot restart.
        venv_needs_repair = not (venv_python.is_file() and os.access(venv_python, os.X_OK))

        # Check if pyproject.toml or source files changed. Both are needed
        # because the install is non-editable: pip copies code into the venv's
        # site-packages, so updating src/ at the install path alone does
        # nothing. A pyproject.toml change means dependencies may have changed;
        # a source change means the installed package code is stale.
        pyproject_checksum_file = install_path / ".pyproject.sha256"
        old_pyproject = ""
        if pyproject_checksum_file.exists():
            old_pyproject = pyproject_checksum_file.read_text().strip()
        # During a real apply, _apply_source has already copied the incoming
        # files to the install tree. During a dry run that copy is deliberately
        # skipped, so compare the repository inputs directly or the preview
        # will incorrectly report an unchanged venv for every source-only
        # update.
        incoming_pyproject = PROJECT_ROOT / "pyproject.toml" if dry_run else pyproject_dst
        new_pyproject = _file_checksum(incoming_pyproject)

        constraints_checksum_file = install_path / ".constraints.sha256"
        old_constraints = ""
        if constraints_checksum_file.exists():
            old_constraints = constraints_checksum_file.read_text().strip()
        incoming_constraints = PROJECT_ROOT / _INSTALL_CONSTRAINTS_REL if dry_run else constraints_dst
        new_constraints = _optional_file_checksum(incoming_constraints)

        src_checksum_file = install_path / ".src.sha256"
        old_src = ""
        if src_checksum_file.exists():
            old_src = src_checksum_file.read_text().strip()
        incoming_src = PROJECT_ROOT / "src" if dry_run else src_dst
        new_src = _src_checksum(incoming_src)

        if (
            old_pyproject == new_pyproject
            and old_constraints == new_constraints
            and old_src == new_src
            and not venv_needs_repair
        ):
            print("  Venv unchanged (pyproject.toml, constraints, and source checksums match)")
            return

        # Report what changed for operator visibility
        changed: list[str] = []
        if old_pyproject != new_pyproject:
            changed.append("pyproject.toml")
        if old_constraints != new_constraints:
            changed.append("constraints")
        if old_src != new_src:
            changed.append("source")

        if dry_run:
            if venv_needs_repair:
                print("[DRY RUN] Would repair venv (Python interpreter missing or unusable)")
            if changed:
                print(f"[DRY RUN] Would update venv ({' and '.join(changed)} changed)")
            if build_path.exists():
                print(f"[DRY RUN] Would remove stale package build artifacts: {build_path}")
            return

        if venv_needs_repair:
            python = _resolve_venv_base_python()
            print(f"  Repairing venv with {python}...")
            # `venv --upgrade` does not overwrite dangling interpreter
            # symlinks. Remove only those generated links first; installed
            # packages and every usable executable remain untouched.
            removed_links = 0
            for candidate in sorted((venv_path / "bin").glob("python*")):
                if candidate.is_symlink() and not candidate.exists():
                    candidate.unlink()
                    removed_links += 1
            if removed_links:
                print(f"  Removed {removed_links} dangling venv interpreter symlink(s)")
            subprocess.run(
                [python, "-m", "venv", "--upgrade", str(venv_path)],
                check=True,
            )
            if not (venv_python.is_file() and os.access(venv_python, os.X_OK)):
                raise RuntimeError(f"Venv repair did not create a usable interpreter: {venv_python}")
            print("  Repaired venv interpreter")

        if changed:
            print(f"  Updating venv ({' and '.join(changed)} changed)...")
    else:
        if dry_run:
            print(f"[DRY RUN] Would create venv: {venv_path}")
            if build_path.exists():
                print(f"[DRY RUN] Would remove stale package build artifacts: {build_path}")
            print("[DRY RUN] Would install package into venv")
            return

        print(f"  Creating venv at {venv_path}...")
        python = _resolve_venv_base_python()

        subprocess.run(
            [python, "-m", "venv", str(venv_path)],
            check=True,
        )

    # setuptools can reuse build/lib from an earlier wheel build and include
    # Python modules that were subsequently deleted from src/. Always remove
    # this generated tree before pip builds the non-editable package.
    if build_path.exists():
        shutil.rmtree(build_path)
        print(f"  Removed stale package build artifacts: {build_path}")

    # Install the package with optional dependencies.
    # Uses a non-editable install (not -e) so the venv is self-contained
    # and doesn't depend on the source directory being writable.
    extras = "memory,totp,tts"
    install_spec = f"{install_path}[{extras}]"
    pip_install_cmd = [str(venv_python), "-m", "pip", "install"]
    if constraints_dst.is_file():
        pip_install_cmd.extend(["--constraint", str(constraints_dst)])
    pip_install_cmd.append(install_spec)
    subprocess.run(pip_install_cmd, check=True)
    print("  Installed package into venv")

    # Save checksums for future update detection. Both are written after a
    # successful install so that a partial failure (e.g., pip crash mid-install)
    # leaves stale checksums and triggers a retry on the next run.
    (install_path / ".pyproject.sha256").write_text(_file_checksum(pyproject_dst) + "\n")
    (install_path / ".constraints.sha256").write_text(_optional_file_checksum(constraints_dst) + "\n")
    (install_path / ".src.sha256").write_text(_src_checksum(src_dst) + "\n")

    # Set venv ownership to root (read-only for service user)
    _set_ownership(venv_path, 0, 0, recursive=True)


def _apply_models(install_path: Path, dry_run: bool) -> None:
    """Copy model files from source if they exist."""
    models_src = PROJECT_ROOT / "models"
    models_dst = install_path / "models"

    if not models_src.exists() or not any(models_src.iterdir()):
        return

    if dry_run:
        print(f"[DRY RUN] Would copy: {models_src} -> {models_dst}")
        return

    _copy_tree(models_src, models_dst)
    _set_ownership(models_dst, 0, 0, recursive=True)
    print(f"  Copied models to {models_dst}")


def _apply_secrets(
    env: dict[str, str],
    dry_run: bool,
    users_yaml_staging_path: str | None = None,
) -> None:
    """Write the /etc/kai/env file from install.conf environment values.

    `users_yaml_staging_path` is the absolute path of the first-time
    install staging file as recorded by the wizard at the top level of
    `install.conf`. It is passed explicitly (not read from `env`)
    because routing it through `env` would surface it in `/etc/kai/env`
    as runtime daemon configuration; the staging path is installer
    metadata and must never leak there. The caller in `_cmd_apply`
    handles post-success cleanup (unlink + strip the conf key); this
    function only performs the copy.
    """
    etc_kai = Path("/etc/kai")
    env_path = etc_kai / "env"
    env_content = _generate_env_file(env)

    # Resolve users.yaml staging once so the dry-run preview and the
    # real-apply branch share the same precedence rule. The presence
    # of a non-empty path AND a readable file on disk is the entire
    # signal: a stale staging file with no matching install.conf key
    # never reaches this function (the caller passes None), and a
    # recorded key that points at a missing file silently skips the
    # copy rather than failing the apply.
    users_yaml_src: Path | None = None
    if users_yaml_staging_path:
        candidate = Path(users_yaml_staging_path)
        if candidate.is_file():
            users_yaml_src = candidate

    if dry_run:
        print(f"[DRY RUN] Would write: {env_path} (mode 0600)")
        if users_yaml_src is not None:
            print(f"[DRY RUN] Would copy: {users_yaml_src} -> {etc_kai / 'users.yaml'} (mode 0600)")
        # TODO: services.yaml and workspaces.yaml still copy from
        # PROJECT_ROOT pending follow-up issues that mirror the
        # users.yaml staging flow for those files. The asymmetry is
        # intentional and tracked; users.yaml moves first because it
        # is auth-bearing.
        for yaml_name in ("services.yaml", "workspaces.yaml"):
            if (PROJECT_ROOT / yaml_name).exists():
                print(f"[DRY RUN] Would copy: {etc_kai / yaml_name} (mode 0600)")
        return

    env_path.write_text(env_content)
    os.chmod(env_path, 0o600)
    os.chown(env_path, 0, 0)
    print(f"  Wrote {env_path}")

    # users.yaml is handled separately from the other YAML files. Its
    # source is the per-operator staging file recorded in install.conf
    # by the wizard, not PROJECT_ROOT/users.yaml; the spec for #557
    # canonicalizes users.yaml on /etc/kai/users.yaml and removes the
    # project-tree path. Cleanup of the staging file and the conf key
    # happens in `_cmd_apply` after `apply_succeeded = True` so a
    # failed apply preserves both for a clean retry.
    if users_yaml_src is not None:
        users_yaml_dst = etc_kai / "users.yaml"
        shutil.copy2(users_yaml_src, users_yaml_dst)
        os.chmod(users_yaml_dst, 0o600)
        os.chown(users_yaml_dst, 0, 0)
        print(f"  Copied {users_yaml_dst}")

    # Copy optional YAML config files to /etc/kai/ if they exist in the
    # source directory. All get root-only permissions (mode 0600) since
    # they may contain sensitive configuration (API keys in services.yaml).
    # TODO: services.yaml and workspaces.yaml still read from
    # PROJECT_ROOT pending follow-up issues that mirror the users.yaml
    # staging flow. The asymmetry is documented in-code so it is
    # visible to anyone touching this function.
    for yaml_name in ("services.yaml", "workspaces.yaml"):
        yaml_src = PROJECT_ROOT / yaml_name
        yaml_dst = etc_kai / yaml_name
        if yaml_src.exists():
            shutil.copy2(yaml_src, yaml_dst)
            os.chmod(yaml_dst, 0o600)
            os.chown(yaml_dst, 0, 0)
            print(f"  Copied {yaml_dst}")


def _build_backend_registry(service_user: str, env: dict[str, str]) -> str:
    """Build the installed backend registry from admin-owned install state."""
    svc_home = _user_home(service_user)
    entries = {
        "claude": {
            "driver": "claude",
            "runtime": "local_process",
            "command": env.get("CLAUDE_BIN") or _resolve_default_claude_bin(service_user),
            "allowed_models": sorted(PROVIDER_MODELS["anthropic"].keys()),
        },
        "codex": {
            "driver": "codex",
            "runtime": "local_process",
            "command": env.get("CODEX_BIN") or _resolve_default_codex_bin(),
            "allowed_models": sorted(CODEX_MODELS.keys()),
        },
        "goose": {
            "driver": "goose",
            "runtime": "local_process",
            "command": env.get("GOOSE_BIN") or "/opt/homebrew/bin/goose",
        },
        "opencode": {
            "driver": "opencode",
            "runtime": "local_process",
            "command": env.get("OPENCODE_BIN") or f"{svc_home}/.local/bin/opencode",
        },
    }
    return render_backend_registry(entries)


def _apply_backend_registry(service_user: str, env: dict[str, str], dry_run: bool) -> None:
    """Write the non-secret installed backend registry."""
    content = _build_backend_registry(service_user, env)
    if dry_run:
        print(f"[DRY RUN] Would write: {BACKENDS_YAML} (mode 0644)")
        return
    BACKENDS_YAML.write_text(content)
    os.chmod(BACKENDS_YAML, 0o644)
    os.chown(BACKENDS_YAML, 0, 0)
    print(f"  Wrote {BACKENDS_YAML}")


def _apply_goose_config(
    service_user: str,
    install_path: Path,
    svc_uid: int,
    svc_gid: int,
    dry_run: bool,
    users_yaml_path: str | Path | None = None,
    agent_backend: str = "claude",
) -> None:
    """
    Deploy the Goose extension config to every home goose runs from.

    Copies config/goose-config.yaml from the install tree to
    `~/.config/goose/config.yaml` for the service user AND for each
    distinct goose-backed `os_user` in users.yaml, so `goose acp`
    picks up the right extension settings wherever it spawns. The
    service-user copy covers goose-backed users with no os_user (they
    run as the service user); the per-os_user copies cover isolated
    users, whose spawn runs under `sudo -H` and therefore resolves
    config beneath the target user's home. Directories are created if
    missing, with each user's tree owned by that user.

    `agent_backend` is the install's global backend; users.yaml
    entries without a per-user override inherit it (same contract as
    `_apply_sudoers`). No-ops when nothing in the install is
    goose-backed - neither the global backend nor any users.yaml
    override - so the apply pipeline can call it unconditionally and
    a claude-only install is never blocked on the goose template.
    """
    # None resolves to the module-level USERS_YAML at call time rather
    # than in the signature: a def-time default would bake the
    # production path in at import and bypass the patch the test suite
    # uses to keep the host's real config out of test outcomes.
    if users_yaml_path is None:
        users_yaml_path = USERS_YAML

    # Gate before the template check: a goose template is only a
    # requirement when some session will spawn `goose acp`. users.yaml
    # is canonical at /etc/kai by this step (the secrets step deploys
    # any staged copy first).
    if agent_backend != "goose" and "goose" not in _collect_backends_from_yaml(users_yaml_path):
        return

    src = install_path / "config" / "goose-config.yaml"

    # Check before dry_run so a missing template is caught during
    # pre-validation, not only on the real install run.
    if not src.exists():
        raise SystemExit(f"Goose config template not found at {src}")

    # Resolve every target home and its ownership BEFORE touching
    # disk, so a users.yaml entry naming an os_user that does not
    # exist on this host fails the whole step with a clear message
    # instead of leaving a half-deployed set of homes (mirrors the
    # pre-validation posture of _apply_migrate). The service user is
    # always a target; its uid/gid arrive pre-resolved from the
    # caller and its home resolves through _user_home like the rest
    # of the installer's service-user paths.
    targets: list[tuple[Path, int, int]] = [
        (Path(_user_home(service_user)), svc_uid, svc_gid),
    ]
    for name in _collect_goose_os_users_from_yaml(users_yaml_path, agent_backend):
        # An os_user matching the service user is already covered by
        # the unconditional service-user deploy (the runtime self-
        # sudo-skips that case anyway).
        if name == service_user:
            continue
        try:
            pwd_entry = pwd.getpwnam(name)
        except KeyError as exc:
            raise ValueError(
                f"users.yaml names os_user {name!r}, which does not exist "
                f"on this host. Source: {users_yaml_path}. Create the OS "
                "account or correct the users.yaml entry, then re-run "
                "sudo make install."
            ) from exc
        targets.append((Path(pwd_entry.pw_dir), pwd_entry.pw_uid, pwd_entry.pw_gid))

    if dry_run:
        for home, _uid, _gid in targets:
            goose_dir = home / ".config" / "goose"
            print(f"[DRY RUN] Would create: {goose_dir}")
            print(f"[DRY RUN] Would copy: {src} -> {goose_dir / 'config.yaml'}")
        return

    # Warn if the goose binary isn't on PATH. Not fatal because the
    # user may install it after running make install, but a clear
    # message now saves debugging an opaque runtime error later.
    # Placed after the dry-run guard so dry runs don't warn about
    # runtime dependencies.
    if not shutil.which("goose"):
        print("  WARNING: 'goose' binary not found on PATH.")
        print("  Kai will fail to start the Goose backend until goose is installed.")
        print("  See https://github.com/block/goose for installation instructions.")

    for home, uid, gid in targets:
        goose_dir = home / ".config" / "goose"
        dst = goose_dir / "config.yaml"

        # Track whether we're creating .config for the first time so
        # we can set ownership on it below. mkdir(parents=True)
        # creates both .config/ and .config/goose/ if needed.
        config_dir = home / ".config"
        config_dir_is_new = not config_dir.exists()

        goose_dir.mkdir(parents=True, exist_ok=True)
        # Own the .config/goose tree by the home's user so Goose can
        # write runtime state (session logs, etc.) alongside the
        # config.
        _set_ownership(goose_dir, uid, gid)
        # Only chown .config itself if we just created it. An existing
        # .config may be shared with other tools and should keep its
        # current ownership.
        if config_dir_is_new:
            _set_ownership(config_dir, uid, gid)

        shutil.copy2(src, dst)
        os.chmod(dst, 0o644)
        _set_ownership(dst, uid, gid)
        print(f"  Deployed Goose config to {dst}")


def _apply_sudoers(
    service_user: str,
    dry_run: bool,
    users_yaml_path: str | Path | None = None,
    claude_bin: str | None = None,
    codex_bin: str | None = None,
    opencode_bin: str | None = None,
    goose_bin: str | None = None,
    agent_backend: str = "claude",
) -> None:
    """
    Write sudoers rules for the service user to read protected config.

    Loads `users_yaml_path` (None means the module-level USERS_YAML,
    resolved at call time) to discover every distinct `os_user` the
    runtime may target via `sudo -u`, so each gets a matching
    SETENV: NOPASSWD: rule. Without this, hand-added per-user rules
    were silently wiped on every `sudo make install`.

    `claude_bin`, `codex_bin`, `opencode_bin`, and `goose_bin` are
    threaded from `_cmd_apply`'s env dict (which sources them from
    install.conf, after the apply-time env-var override block) so the
    SETENV rules pin the same absolute paths the running bot will
    invoke. `claude_bin` falls back to the service user's
    `~/.local/bin/claude` (the native installer location);
    `codex_bin` falls back to a common absolute codex install path;
    `opencode_bin`
    falls back to the service user's `~/.local/bin/opencode`;
    `goose_bin` falls back to /opt/homebrew/bin/goose. The fallbacks
    fire only on installs where the wizard has not collected a value.

    `agent_backend` is the install's global backend (the env dict's
    DEFAULT_BACKEND, defaulting to claude when the key is absent, which
    matches the runtime default). Together with the per-user
    backend overrides in users.yaml it scopes the missing-binary
    backstop below to backends the install actually uses.
    """
    # None resolves to the module-level USERS_YAML at call time rather
    # than in the signature: a def-time default would bake the
    # production path in at import and bypass the patch the test suite
    # uses to keep the host's real config out of test outcomes.
    if users_yaml_path is None:
        users_yaml_path = USERS_YAML

    sudoers_path = Path("/etc/sudoers.d/kai")
    # Load and validate users.yaml *before* the dry_run gate. Intentional:
    # a malformed YAML file or invalid os_user value should abort even a
    # dry run, since the operator's next step is `sudo make install` which
    # would hit the same error with worse blast radius (partial install).
    os_users = _collect_os_users_from_yaml(users_yaml_path)
    sudoers_content = _generate_sudoers(
        service_user,
        os_users,
        claude_bin=claude_bin,
        codex_bin=codex_bin,
        opencode_bin=opencode_bin,
        goose_bin=goose_bin,
    )

    # Backstop check: each per-user rule pins a backend binary to a
    # fixed absolute path; a rule that points at a nonexistent binary
    # makes the bot's sudo dispatch fail at runtime with no obvious
    # cause. Warn loudly but do not abort - the warning catches the
    # simple "wrong path" case; the operator still owns the symlink,
    # reinstall, or wizard re-run that makes the paths agree.
    #
    # The check is scoped to backends the install actually uses (the
    # global backend plus any per-user backend overrides
    # in users.yaml): telling an opencode-only operator to install
    # claude would manufacture a requirement that does not exist and
    # train operators to ignore the warning. The rules themselves are
    # still emitted for all four binaries: a rule pointing at an
    # absent path is inert, and unconditional emission means a later
    # backend switch cannot strand a user without a rule.
    if os_users:
        backends_in_use = {agent_backend} | _collect_backends_from_yaml(users_yaml_path)
        svc_home = _user_home(service_user)
        # Each entry's path must resolve to exactly what _generate_sudoers
        # pins for that backend's rule (including the fallbacks); a new
        # backend with a sudoers rule needs an entry in both places.
        expected_bins: dict[str, tuple[Path, str]] = {
            "claude": (
                Path(claude_bin or _resolve_default_claude_bin(service_user)),
                "Run 'make install' after installing claude globally, or re-run "
                "'make config' and point CLAUDE_BIN at the actual install location.",
            ),
            "codex": (
                Path(codex_bin or _resolve_default_codex_bin()),
                "Run 'make install' after installing codex globally, or re-run "
                "'make config' and point CODEX_BIN at the actual install location.",
            ),
            "opencode": (
                Path(opencode_bin or f"{svc_home}/.local/bin/opencode"),
                "Run 'make install' after installing opencode globally, or re-run "
                "'make config' and point OPENCODE_BIN at the actual install location.",
            ),
            "goose": (
                Path(goose_bin or "/opt/homebrew/bin/goose"),
                "Run 'make install' after installing goose globally, or re-run "
                "'make config' and point GOOSE_BIN at the actual install location.",
            ),
        }
        for backend in sorted(backends_in_use & expected_bins.keys()):
            expected_bin, remedy = expected_bins[backend]
            if not expected_bin.exists():
                print(
                    f"Warning: {expected_bin} not found; the {backend} sudoers "
                    f"rule may point at a nonexistent binary. {remedy}",
                    file=sys.stderr,
                )

    if dry_run:
        print(f"[DRY RUN] Would write: {sudoers_path} (mode 0440)")
        print("[DRY RUN] Would validate with visudo -cf")
        return

    # Write to a secure temp file first, validate, then move into place.
    # Uses mkstemp (random name, restrictive permissions) instead of a
    # predictable path in /tmp to prevent symlink attacks when running as root.
    fd, tmp_name = tempfile.mkstemp(prefix="kai-sudoers-", suffix=".tmp")
    try:
        os.write(fd, sudoers_content.encode())
        os.close(fd)

        result = subprocess.run(
            ["visudo", "-cf", tmp_name],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise SystemExit(
                f"Sudoers validation failed: {result.stderr.strip()}\n"
                "  Sudoers file was NOT written. Fix the issue and re-run."
            )

        shutil.move(tmp_name, str(sudoers_path))
    finally:
        # Clean up the temp file if it still exists (move succeeded or error)
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
    os.chmod(sudoers_path, 0o440)
    os.chown(sudoers_path, 0, 0)
    print(f"  Wrote {sudoers_path}")


def _apply_service(
    install_dir: str, data_dir: str, service_user: str, platform: str, dry_run: bool, webhook_port: int = 8080
) -> None:
    """
    Generate and install the platform-specific service definition.

    On macOS, writes a LaunchDaemon plist and a launcher shell script
    (the script keeps bash as the tracked PID so launchd can manage the
    service even when Homebrew Python re-execs). On Linux, writes a
    systemd unit file.

    Args:
        install_dir: Root of the install tree (e.g., /opt/kai).
        data_dir: Writable data directory (e.g., /var/lib/kai).
        service_user: OS username the service runs as.
        platform: "darwin" or "linux".
        dry_run: If True, print what would be written without doing it.
        webhook_port: Port for the webhook/API server (passed to launcher).
    """
    if platform == "darwin":
        # LaunchDaemons (not LaunchAgents) so the service runs under the
        # system domain at boot, independent of any user login session.
        plist_dir = Path("/Library/LaunchDaemons")
        plist_path = plist_dir / f"{_LAUNCHD_LABEL}.plist"
        plist_content = _generate_launchd_plist(install_dir, data_dir, service_user)

        # Launcher script keeps bash as the tracked PID so launchd can
        # manage the service even when Homebrew Python re-execs.
        launcher_path = Path(install_dir) / "run.sh"
        launcher_content = _generate_launcher_script(install_dir, webhook_port)

        if dry_run:
            print(f"[DRY RUN] Would write: {launcher_path}")
            print(f"[DRY RUN] Would write: {plist_path}")
            return

        launcher_path.write_text(launcher_content)
        os.chmod(launcher_path, 0o755)
        os.chown(launcher_path, 0, 0)
        print(f"  Wrote {launcher_path}")

        plist_dir.mkdir(parents=True, exist_ok=True)
        plist_path.write_text(plist_content)
        # LaunchDaemons must be owned by root:wheel
        os.chown(plist_path, 0, 0)
        os.chmod(plist_path, 0o644)
        print(f"  Wrote {plist_path}")

    elif platform == "linux":
        unit_path = Path("/etc/systemd/system/kai.service")
        content = _generate_systemd_unit(install_dir, data_dir, service_user)

        if dry_run:
            print(f"[DRY RUN] Would write: {unit_path}")
            return

        unit_path.write_text(content)
        os.chmod(unit_path, 0o644)
        os.chown(unit_path, 0, 0)
        print(f"  Wrote {unit_path}")

        # Reload systemd so it picks up the new/changed unit
        subprocess.run(["systemctl", "daemon-reload"], check=False)
    else:
        print(f"  Warning: no service definition for platform '{platform}'")


# ── Status subcommand ────────────────────────────────────────────────


def _check_path(path: Path, label: str) -> str:
    """Check if a path exists and report its ownership."""
    try:
        exists = path.exists()
    except PermissionError:
        return f"{label}: {path} (permission denied)"
    if not exists:
        return f"{label}: {path} (not found)"

    stat = path.stat()
    try:
        owner = pwd.getpwuid(stat.st_uid).pw_name
    except KeyError:
        owner = str(stat.st_uid)
    try:
        group = grp.getgrgid(stat.st_gid).gr_name
    except KeyError:
        group = str(stat.st_gid)

    return f"{label}: {path} (exists, {owner}:{group})"


def _check_traversal(path: Path, service_user: str) -> str | None:
    """
    Check if every component of path is traversable by the service user.

    Walks from the root down to path, checking execute permission on each
    directory. Returns a warning string if any parent lacks traverse
    permission for the service user, or None if fully traversable.

    Args:
        path: The directory path to check.
        service_user: The OS username that needs to traverse the path.

    Returns:
        A warning string naming the blocking directory, or None.
    """
    try:
        user_info = pwd.getpwnam(service_user)
    except KeyError:
        return f"User '{service_user}' does not exist; cannot check traversal"

    svc_uid = user_info.pw_uid
    svc_gid = user_info.pw_gid
    try:
        svc_groups = set(os.getgrouplist(service_user, svc_gid))
    except KeyError:
        svc_groups = {svc_gid}

    # Walk each component from root to the target path
    for parent in reversed(path.resolve().parents):
        if not parent.exists():
            continue
        st = parent.stat()
        mode = st.st_mode

        # Check execute bit for the appropriate permission class
        if st.st_uid == svc_uid:
            has_x = bool(mode & 0o100)
        elif st.st_gid in svc_groups:
            has_x = bool(mode & 0o010)
        else:
            has_x = bool(mode & 0o001)

        if not has_x:
            # Suggest the correct chmod class based on which check failed
            if st.st_uid == svc_uid:
                fix = f"chmod u+x {parent}"
            elif st.st_gid in svc_groups:
                fix = f"chmod g+x {parent}"
            else:
                fix = f"chmod o+x {parent}"
            return f"{parent} lacks execute permission for {service_user}. Fix: {fix}"

    return None


def _check_service_status(platform: str) -> str:
    """Check if the Kai service is running on the current platform."""
    if platform == "darwin":
        # Check the system domain (LaunchDaemon, not per-user LaunchAgent)
        result = subprocess.run(
            ["launchctl", "print", f"system/{_LAUNCHD_LABEL}"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return f"Service: {_LAUNCHD_LABEL} (loaded)"
        return f"Service: {_LAUNCHD_LABEL} (not loaded)"

    elif platform == "linux":
        result = subprocess.run(
            ["systemctl", "is-active", "kai"],
            capture_output=True,
            text=True,
        )
        status = result.stdout.strip()
        return f"Service: kai.service ({status})"

    return f"Service: unknown platform '{platform}'"


def _cmd_status() -> None:
    """
    Report the current installation state. No sudo required.

    Checks for the existence of installation directories, config files,
    and service status. Reports ownership for security verification.
    """
    # Load install.conf once for platform, install_dir, and data_dir.
    # Falls back to auto-detected platform and default paths if missing.
    platform = "darwin" if sys.platform == "darwin" else "linux"
    install_dir = _DEFAULT_INSTALL_DIR
    data_dir = _DEFAULT_DATA_DIR
    if INSTALL_CONF.exists():
        try:
            conf = json.loads(INSTALL_CONF.read_text())
            platform = conf.get("platform", platform)
            install_dir = conf.get("install_dir", install_dir)
            data_dir = conf.get("data_dir", data_dir)
        except (json.JSONDecodeError, OSError):
            pass

    print("Kai Installation Status")
    print("=" * 30)
    print(_check_path(Path(install_dir), "Installation"))
    print(_check_path(Path(data_dir), "Data"))
    print(_check_path(Path("/etc/kai/env"), "Secrets"))
    print(_check_path(Path("/etc/kai/services.yaml"), "Services"))
    print(_check_path(Path("/etc/sudoers.d/kai"), "Sudoers"))
    print(_check_service_status(platform))

    # Check workspace path traversal if install.conf has a service user
    if INSTALL_CONF.exists():
        try:
            conf = json.loads(INSTALL_CONF.read_text())
            svc_user = conf.get("service_user", "")
            env = conf.get("env", {})
            if svc_user:
                ws_paths: list[Path] = []
                ws_base = env.get("WORKSPACE_BASE", "")
                if ws_base:
                    ws_paths.append(Path(ws_base))
                ws_paths.extend(_parse_workspaces(env))
                for ws_path in ws_paths:
                    warning = _check_traversal(ws_path, svc_user)
                    if warning:
                        print(f"WARNING: {warning}")
        except (json.JSONDecodeError, OSError):
            pass

    # Show version if installed
    init_path = Path(install_dir) / "src" / "kai" / "__init__.py"
    if init_path.exists():
        for line in init_path.read_text().splitlines():
            if line.startswith("__version__"):
                version = line.split("=")[1].strip().strip('"').strip("'")
                print(f"Version: {version}")
                break


# ── CLI dispatch ─────────────────────────────────────────────────────


def cli(args: list[str]) -> None:
    """
    Dispatch install CLI subcommands.

    Usage:
        python -m kai install config              -- interactive Q&A, writes install.conf
        python -m kai install apply [--dry-run]    -- reads install.conf, creates /opt layout
        python -m kai install status               -- shows current installation state
    """
    subcommands = {
        "config": _cmd_config,
        "apply": _cmd_apply,
        "status": _cmd_status,
    }

    if not args or args[0] not in subcommands:
        raise SystemExit("Usage: python -m kai install {config|apply|status}")

    subcmd = args[0]
    remaining = args[1:]

    # The apply subcommand accepts --dry-run as an alternative to DRY_RUN=1
    if subcmd == "apply" and "--dry-run" in remaining:
        os.environ["DRY_RUN"] = "1"

    subcommands[subcmd]()
