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
from pathlib import Path

import yaml

from kai.config import MAX_CONTEXT_CEILING, PROJECT_ROOT, VALID_MODELS

# Config file written by `config`, read by `apply`.
# Anchored to PROJECT_ROOT so it resolves correctly regardless of CWD.
INSTALL_CONF = PROJECT_ROOT / "install.conf"

# Default installation paths
_DEFAULT_INSTALL_DIR = "/opt/kai"
_DEFAULT_DATA_DIR = "/var/lib/kai"
_DEFAULT_SERVICE_USER = "kai"

# Current install.conf schema version
_CONF_VERSION = 1

# Plist label for the launchd service
_LAUNCHD_LABEL = "com.syrinx.kai"

# Files and directories to copy from source to the install location.
# Excludes __pycache__, .pyc, and other build artifacts.
_SOURCE_EXCLUDES = {"__pycache__", "*.pyc", "*.egg-info", ".git", ".venv", ".env"}

# Excludes for home/.claude/ copy. These are runtime-generated or
# personal data that should not be part of a clean install:
#   history/    - conversation logs written by history.py at runtime
#   MEMORY.md   - personal data (gitignored), user creates from .example
#   skills/     - downloaded skills, environment-specific
# History and MEMORY.md now live in DATA_DIR, outside the install tree.
# Both are still excluded because stale files may remain at the source
# after migration (source files are preserved as backups, not deleted).
_HOME_CLAUDE_EXCLUDES = {"history", "MEMORY.md", "skills", "__pycache__"}


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

    Rejects input not in the choices list and re-prompts.

    Args:
        label: The prompt text shown to the user.
        choices: List of valid string values.
        default: Default value if the user presses Enter.

    Returns:
        The chosen value (guaranteed to be in choices).
    """
    choices_str = "/".join(choices)
    suffix = f" [{default}]" if default else ""
    while True:
        value = input(f"{label} ({choices_str}){suffix}: ").strip().lower()
        if not value and default:
            return default
        if value in choices:
            return value
        print(f"  Please choose one of: {choices_str}")


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


def _validate_positive_float(value: str) -> bool:
    """Check that a string is a positive float."""
    try:
        return float(value) > 0
    except ValueError:
        return False


def _validate_positive_int(value: str) -> bool:
    """Check that a string is a positive integer."""
    try:
        return int(value) > 0
    except ValueError:
        return False


def _validate_chat_id(value: str) -> bool:
    """Check that a string is a valid Telegram chat ID (any non-zero integer)."""
    try:
        return int(value) != 0
    except ValueError:
        return False


# ── Config subcommand ────────────────────────────────────────────────


def _cmd_config() -> None:
    """
    Interactive Q&A that collects configuration values and writes install.conf.

    If install.conf already exists, its values are used as defaults so re-running
    only asks about changes. Auto-detects platform and generates a webhook secret
    if one isn't already set. Validates all inputs before writing.

    No sudo required - this runs as the current user.
    """
    print("Kai Protected Installation - Configuration")
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

    # Auto-detect platform
    if sys.platform == "darwin":
        detected_platform = "darwin"
    elif sys.platform.startswith("linux"):
        detected_platform = "linux"
    else:
        detected_platform = sys.platform

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

    # -- Telegram --
    print("-- Telegram --")
    bot_token = _prompt(
        "Telegram bot token",
        existing_env.get("TELEGRAM_BOT_TOKEN", ""),
        required=True,
    )

    # -- User setup --
    # Check for an existing users.yaml. Try local project root first,
    # then the deployed copy at /etc/kai/ (for protected installs).
    # If either exists, skip the user prompt to avoid overwriting
    # manual edits.
    print("-- User setup --")
    users_yaml_path = PROJECT_ROOT / "users.yaml"
    users_yaml_exists = users_yaml_path.exists()

    if not users_yaml_exists:
        # /etc/kai/users.yaml is mode 0600 owned by root. Whether
        # Path.exists() works depends on the parent directory permissions;
        # it may return True even if the file isn't readable. Either way,
        # the worst case is re-prompting and the next 'make install'
        # overwrites the deployed copy with the new one.
        etc_users = Path("/etc/kai/users.yaml")
        if etc_users.exists():
            users_yaml_exists = True

    # Track whether advanced mode set os_user, so we can skip the
    # CLAUDE_USER prompt later (section 8). Needs to be in scope
    # regardless of which branch we take.
    admin_os_user: str | None = None

    if users_yaml_exists:
        # Summarize the existing config without modifying it.
        # Note: if only the /etc/kai/ copy exists (not the local one),
        # users_yaml_path still points to PROJECT_ROOT / "users.yaml"
        # which doesn't exist. Guard with .exists() so we skip the
        # summary gracefully rather than relying on the bare except.
        summary = ""
        if users_yaml_path.exists():
            try:
                data = yaml.safe_load(users_yaml_path.read_text())
                if isinstance(data, dict) and isinstance(data.get("users"), list):
                    entries = data["users"]
                    n_users = len(entries)
                    n_admins = sum(
                        1 for e in entries if isinstance(e, dict) and str(e.get("role", "")).lower() == "admin"
                    )
                    summary = (
                        f" ({n_users} user{'s' if n_users != 1 else ''}, "
                        f"{n_admins} admin{'s' if n_admins != 1 else ''})"
                    )
            except Exception:
                pass  # Malformed YAML - skip the summary
        print(f"  users.yaml already configured{summary}.")
        print("  To modify users, edit users.yaml directly or use Telegram commands.")
    else:
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

        # Advanced options: os_user and home_workspace.
        advanced = _prompt_bool("Configure advanced user options", False)
        admin_home_workspace: str | None = None

        if advanced:
            # Default os_user to CLAUDE_USER if previously set, else $USER.
            # Note: on machines where the service user and the current user
            # are the same (e.g., "kai" on the Mac mini), accepting the
            # default means os_user matches the bot process user. This is
            # fine - PR #192 handles the self-sudo skip for this case.
            default_os_user = existing_env.get("CLAUDE_USER", "") or os.environ.get("USER", "")
            while True:
                admin_os_user = _prompt("OS user for subprocess isolation", default_os_user).strip() or None
                if admin_os_user is None or _validate_os_user(admin_os_user):
                    break
                print("  Username may only contain letters, numbers, dots, hyphens, and underscores.")

            # Default home_workspace to PROJECT_ROOT.
            default_home = str(PROJECT_ROOT)
            admin_home_workspace = _prompt("Home workspace", default_home) or None
            if admin_home_workspace:
                # Resolve to absolute path for consistency with config.py parsing.
                admin_home_workspace = str(Path(admin_home_workspace).expanduser().resolve())

        # Write users.yaml to project root. _apply_secrets() copies it
        # to /etc/kai/users.yaml during 'make install'.
        users_yaml_content = _generate_users_yaml(
            admin_telegram_id,
            admin_name,
            os_user=admin_os_user,
            home_workspace=admin_home_workspace,
        )
        users_yaml_path.write_text(users_yaml_content)
        os.chmod(users_yaml_path, 0o600)
        print(f"  Generated {users_yaml_path}")
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

    # -- Claude --
    print("-- Claude --")
    model = _prompt_choice(
        "Claude model",
        sorted(VALID_MODELS),
        existing_env.get("CLAUDE_MODEL", "sonnet"),
    )

    while True:
        timeout = _prompt(
            "Claude timeout (seconds)",
            existing_env.get("CLAUDE_TIMEOUT_SECONDS", "120"),
        )
        if _validate_positive_int(timeout):
            break
        print("  Must be a positive integer.")

    while True:
        budget = _prompt(
            "Claude budget (USD)",
            existing_env.get("CLAUDE_MAX_BUDGET_USD", "10.0"),
        )
        if _validate_positive_float(budget):
            break
        print("  Must be a positive number.")

    # Context window tuning - smaller windows reduce token usage and
    # cache invalidation pressure on the inner Claude process.
    while True:
        max_context_window = _prompt(
            "Max context window (tokens, 0 = default 1M)",
            existing_env.get("CLAUDE_MAX_CONTEXT_WINDOW", "200000"),
        )
        try:
            val = int(max_context_window)
            if 0 <= val <= MAX_CONTEXT_CEILING:
                break
        except ValueError:
            pass
        print(f"  Must be 0-{MAX_CONTEXT_CEILING} (0 = use default).")

    # Autocompact threshold controls when Claude automatically compresses
    # conversation history. Lower values compact sooner, reducing token
    # usage at the cost of losing raw context earlier. Claude Code caps
    # this at ~83% regardless of what you set - values above that are
    # silently clamped.
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

    # Auto-generate webhook secret if not already set
    default_secret = existing_env.get("WEBHOOK_SECRET", "")
    if not default_secret:
        default_secret = secrets.token_hex(32)
    webhook_secret = _prompt("Webhook secret", default_secret, required=True)
    print()

    # -- Workspaces --
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
    print("-- PR review agent --")
    pr_review_enabled = _prompt_bool(
        "Enable PR review agent",
        existing_env.get("PR_REVIEW_ENABLED", "false").lower() in ("1", "true", "yes"),
    )
    pr_review_cooldown = "300"
    if pr_review_enabled:
        while True:
            pr_review_cooldown = _prompt(
                "Review cooldown in seconds (prevents spam from rapid pushes)",
                existing_env.get("PR_REVIEW_COOLDOWN", "300"),
            )
            if _validate_positive_int(pr_review_cooldown):
                break
            print("  Must be a positive integer.")
    print()

    # -- Issue triage agent --
    # Independent from PR review - you might want one without the other.
    print("-- Issue triage agent --")
    issue_triage_enabled = _prompt_bool(
        "Enable issue triage agent",
        existing_env.get("ISSUE_TRIAGE_ENABLED", "false").lower() in ("1", "true", "yes"),
    )
    print()

    # -- GitHub notifications --
    print("-- GitHub notifications --")
    github_notify_chat_id = ""
    while True:
        github_notify_chat_id = _prompt(
            "GitHub notification chat ID (optional)",
            existing_env.get("GITHUB_NOTIFY_CHAT_ID", ""),
        )
        # Empty is valid (feature disabled)
        if not github_notify_chat_id or _validate_chat_id(github_notify_chat_id):
            break
        print("  Must be a valid Telegram chat ID (integer).")
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

    # Skip if the user already set os_user via advanced user options.
    # CLAUDE_USER is the global fallback; os_user in users.yaml takes
    # precedence per-user at runtime.
    if admin_os_user:
        claude_user = ""
    else:
        claude_user = _prompt(
            "Claude subprocess user (optional, for process isolation)",
            existing_env.get("CLAUDE_USER", ""),
        )
    print()

    # -- External services --
    print("-- External services --")
    perplexity_key = _prompt(
        "Perplexity API key (optional)",
        existing_env.get("PERPLEXITY_API_KEY", ""),
    )
    print()

    # Build the env dict (only include non-empty values)
    env: dict[str, str] = {
        "TELEGRAM_BOT_TOKEN": bot_token,
        "CLAUDE_MODEL": model,
        "CLAUDE_TIMEOUT_SECONDS": timeout,
        "CLAUDE_MAX_BUDGET_USD": budget,
        "WEBHOOK_PORT": port,
        "WEBHOOK_SECRET": webhook_secret,
        "VOICE_ENABLED": str(voice_enabled).lower(),
        "TTS_ENABLED": str(tts_enabled).lower(),
    }

    # Context window tuning - only include if non-default.
    # Compare as int to handle inputs like "000" that pass validation.
    if int(max_context_window) != 0:
        env["CLAUDE_MAX_CONTEXT_WINDOW"] = max_context_window
    if int(autocompact_pct) != 0:
        env["CLAUDE_AUTOCOMPACT_PCT"] = autocompact_pct

    # Conditionally add optional values
    if transport == "webhook":
        env["TELEGRAM_TRANSPORT"] = "webhook"
        if webhook_url:
            env["TELEGRAM_WEBHOOK_URL"] = webhook_url
        if tg_webhook_secret:
            env["TELEGRAM_WEBHOOK_SECRET"] = tg_webhook_secret
    if workspace_base:
        env["WORKSPACE_BASE"] = workspace_base
    if allowed_workspaces:
        env["ALLOWED_WORKSPACES"] = allowed_workspaces
    if claude_user:
        env["CLAUDE_USER"] = claude_user
    if perplexity_key:
        env["PERPLEXITY_API_KEY"] = perplexity_key
    if pr_review_enabled:
        env["PR_REVIEW_ENABLED"] = "true"
        if pr_review_cooldown != "300":
            env["PR_REVIEW_COOLDOWN"] = pr_review_cooldown
    if issue_triage_enabled:
        env["ISSUE_TRIAGE_ENABLED"] = "true"
    if github_notify_chat_id:
        env["GITHUB_NOTIFY_CHAT_ID"] = github_notify_chat_id

    # Build and write install.conf
    conf = {
        "version": _CONF_VERSION,
        "install_dir": install_dir,
        "data_dir": data_dir,
        "service_user": service_user,
        "platform": platform,
        "env": env,
    }

    INSTALL_CONF.write_text(json.dumps(conf, indent=2) + "\n")
    # Restrict permissions since the file contains secrets (bot token, webhook secret)
    os.chmod(INSTALL_CONF, 0o600)
    print(f"Configuration written to {INSTALL_CONF}")
    print("Review the file, then run: sudo python -m kai install apply")


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

    Args:
        path: File or directory to chown.
        uid: User ID for the new owner.
        gid: Group ID for the new group.
        recursive: If True, walk the directory tree and chown everything.
    """
    os.chown(path, uid, gid)
    if recursive and path.is_dir():
        for child in path.rglob("*"):
            os.chown(child, uid, gid)


def _copy_tree(src: Path, dst: Path, excludes: set[str] | None = None) -> None:
    """
    Copy a directory tree, excluding patterns like __pycache__.

    Uses a merge-based approach: walks the source tree and copies each file
    individually, creating destination directories as needed. Files at the
    destination that don't exist in the source are left untouched. This is
    critical for workspace/.claude/ where runtime-created content (skills,
    Claude Code state files) must survive installs.

    The previous implementation used shutil.rmtree(dst) before copytree(),
    which destroyed ALL destination contents including runtime data that the
    excludes were meant to protect. See issue #143.

    Args:
        src: Source directory.
        dst: Destination directory (created if it doesn't exist).
        excludes: Set of glob patterns to exclude (e.g., {"__pycache__", "*.pyc"}).
    """
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
                shutil.copy2(Path(src_dir) / f, dst_dir / f)


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
    ]
    if os_user:
        lines.append(f"    os_user: {_yaml_scalar(os_user)}")
    if home_workspace:
        lines.append(f"    home_workspace: {_yaml_scalar(home_workspace)}")
    lines.append("")  # trailing newline
    return "\n".join(lines)


def _generate_sudoers(service_user: str, claude_user: str | None = None) -> str:
    """
    Generate sudoers rules for the service user to read protected config files.

    The rules allow passwordless `sudo cat` on specific files only. This is
    validated with `visudo -cf` before being written to /etc/sudoers.d/.

    Uses shutil.which() to resolve the actual paths of `cat` and `tee`,
    since they live at /bin/ on macOS but /usr/bin/ on many Linux distros.
    Falls back to /bin/cat and /usr/bin/tee if the binaries aren't found
    in the current PATH (e.g., when running in a minimal environment).

    When claude_user is set, an additional rule allows the service user to
    run the claude binary as that user (for process isolation via sudo -u).

    Args:
        service_user: The OS username that runs the Kai service.
        claude_user: Optional OS username for the inner Claude process.

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
        {service_user} ALL=(root) NOPASSWD: {cat_path} /etc/kai/totp.secret
        {service_user} ALL=(root) NOPASSWD: {cat_path} /etc/kai/totp.attempts
        {service_user} ALL=(root) NOPASSWD: {tee_path} /etc/kai/totp.attempts
    """)

    # Allow running the claude binary as the CLAUDE_USER for process isolation.
    # Resolve the actual binary location; fall back to the native installer's
    # default path under the service user's home if claude is not on PATH
    # (e.g., when running under sudo with a stripped environment).
    if claude_user:
        svc_home = _user_home(service_user)
        claude_bin = shutil.which("claude") or f"{svc_home}/.local/bin/claude"
        # SETENV: allows the service user to pass env vars (e.g.,
        # KAI_WEBHOOK_SECRET) through sudo to the claude process.
        # Scoped to this rule only; cat/tee rules remain locked down.
        rules += f"{service_user} ALL=({claude_user}) SETENV: NOPASSWD: {claude_bin}\n"

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

        # Wait for Python to re-exec and start listening
        sleep 2

        # Find the actual Python process (the re-exec'd grandchild).
        # lsof lives at /usr/sbin/ which may not be in the service PATH.
        REAL_PID=$(/usr/sbin/lsof -ti :{webhook_port} -sTCP:LISTEN 2>/dev/null)
        if [ -z "$REAL_PID" ]; then
            # Hasn't bound yet; wait a bit more
            sleep 3
            REAL_PID=$(/usr/sbin/lsof -ti :{webhook_port} -sTCP:LISTEN 2>/dev/null)
        fi

        cleanup() {{
            kill -TERM "$REAL_PID" 2>/dev/null
            # Poll until the process is gone (can't use wait on non-children)
            while kill -0 "$REAL_PID" 2>/dev/null; do sleep 0.5; done
        }}
        trap cleanup TERM INT

        # Poll for the real Python process to exit.
        # kill -0 checks if PID exists without sending a signal.
        # This is macOS-compatible (no GNU tail --pid needed).
        if [ -n "$REAL_PID" ]; then
            while kill -0 "$REAL_PID" 2>/dev/null; do sleep 1; done
        else
            # Could not find the process; wait indefinitely.
            # BSD sleep doesn't support "infinity", so loop with a long sleep.
            while true; do sleep 86400; done
        fi
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
    Start the Kai service after applying changes.

    Best-effort: uses check=False since launchctl/systemctl may report
    warnings that aren't actually failures (e.g., service already running).

    On macOS, launchctl bootstrap can fail transiently after a bootout
    if launchd hasn't fully released the service domain (common with
    KeepAlive daemons). We retry once after a brief delay to handle this.

    Args:
        platform: "darwin" or "linux".
        dry_run: If True, print the command without executing.
    """
    if platform == "darwin":
        plist_path = Path("/Library/LaunchDaemons") / f"{_LAUNCHD_LABEL}.plist"
        cmd = ["launchctl", "bootstrap", "system", str(plist_path)]
    elif platform == "linux":
        cmd = ["systemctl", "start", "kai"]
    else:
        print(f"  Warning: cannot start service on platform '{platform}'")
        return

    if dry_run:
        print(f"[DRY RUN] Would run: {' '.join(cmd)}")
        return

    result = subprocess.run(cmd, check=False, capture_output=True)
    if result.returncode == 0:
        print(f"  Started service ({' '.join(cmd[:2])})")
        return

    # On macOS, bootstrap can fail transiently after bootout.
    # Wait briefly for launchd to finish tearing down, then retry.
    if platform == "darwin":
        stderr_msg = result.stderr.decode().strip()
        print(f"  Bootstrap failed ({stderr_msg or 'unknown'}), retrying...")
        time.sleep(2)
        result = subprocess.run(cmd, check=False, capture_output=True)
        if result.returncode == 0:
            print(f"  Started service ({' '.join(cmd[:2])})")
            return

    # Show the actual error so the user knows what went wrong
    stderr_text = result.stderr.decode().strip()
    hint = f": {stderr_text}" if stderr_text else ""
    print(f"  Warning: service start failed ({' '.join(cmd[:2])}){hint}")


def _apply_migrate(data_path: Path, install_path: Path, svc_uid: int, svc_gid: int, dry_run: bool) -> None:
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
    """
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

            os.chown(db_dst, svc_uid, svc_gid)
    elif db_dst.exists():
        print("  Database already exists at destination, skipping migration")
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

    # -- History migration --
    # One-time: copy JSONL conversation logs from the source tree
    # (home/.claude/history/, pre-DATA_DIR location) to DATA_DIR/history/.
    # Safe on repeated runs: only copies files that
    # don't already exist at the destination. Source files are preserved
    # as backups (same pattern as the database and log migrations above).
    history_src = PROJECT_ROOT / "home" / ".claude" / "history"
    history_dst = data_path / "history"

    if history_src.is_dir():
        copied = 0
        for f in sorted(history_src.glob("*.jsonl")):
            dest = history_dst / f.name
            if dest.exists():
                continue
            if dry_run:
                print(f"[DRY RUN] Would copy history: {f} -> {dest}")
            else:
                history_dst.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dest)
                os.chown(dest, svc_uid, svc_gid)
                copied += 1
        if copied and not dry_run:
            print(f"  Migrated {copied} history file(s) to {history_dst}")
        elif not copied and not dry_run:
            print("  History already migrated or no files to copy")

    # -- MEMORY.md migration --
    # One-time: copy personal memory from the source tree
    # (home/.claude/MEMORY.md, pre-DATA_DIR location) to DATA_DIR/memory/.
    # If the file doesn't exist at the source location
    # (common - it was never created on most installs), _bootstrap_memory()
    # in main.py handles creation from the example template at startup.
    memory_src = PROJECT_ROOT / "home" / ".claude" / "MEMORY.md"
    memory_dst = data_path / "memory" / "MEMORY.md"
    if memory_src.is_file() and not memory_dst.exists():
        if dry_run:
            print(f"[DRY RUN] Would copy MEMORY.md: {memory_src} -> {memory_dst}")
        else:
            (data_path / "memory").mkdir(parents=True, exist_ok=True)
            shutil.copy2(memory_src, memory_dst)
            os.chown(memory_dst, svc_uid, svc_gid)
            print(f"  Migrated MEMORY.md to {memory_dst}")

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

    # Validate required fields
    install_dir = conf.get("install_dir")
    data_dir = conf.get("data_dir")
    service_user = conf.get("service_user")
    platform = conf.get("platform")
    env = conf.get("env", {})

    if not all([install_dir, data_dir, service_user, platform]):
        raise SystemExit("install.conf is missing required fields.")

    # Validate service user exists
    try:
        user_info = pwd.getpwnam(service_user)
        svc_uid = user_info.pw_uid
        svc_gid = user_info.pw_gid
    except KeyError:
        raise SystemExit(f"Service user '{service_user}' does not exist on this system.") from None

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
        _apply_secrets(env, dry_run)

        # -- Step 6: Configure sudoers --
        claude_user = env.get("CLAUDE_USER") or None
        _apply_sudoers(service_user, dry_run, claude_user)

        # -- Step 7: Migrate runtime data --
        _apply_migrate(data_path, install_path, svc_uid, svc_gid, dry_run)

        # -- Step 8: Generate service definition --
        webhook_port = int(env.get("WEBHOOK_PORT", "8080"))
        _apply_service(install_dir, data_dir, service_user, platform, dry_run, webhook_port)
    except Exception:
        print("\nInstallation failed. See error above.")
        print("The installation may be in a partial state.")
        print("Fix the issue and re-run: sudo python -m kai install apply")
        raise
    finally:
        # Always restart the service, even after failure. A partially updated
        # installation is better than an offline bot. Most steps are idempotent,
        # so re-running apply after fixing the cause will complete the update.
        # Wrapped in its own try/except so a start failure does not mask the
        # original exception (Python replaces the propagating exception if
        # finally raises).
        try:
            _start_service(platform, dry_run)
        except Exception:
            print("WARNING: Failed to restart service after apply.")
            if platform == "darwin":
                print("Manually restart with: sudo launchctl kickstart system/com.syrinx.kai")
            else:
                print("Manually restart with: sudo systemctl start kai")

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
        (data_path / "files", svc_uid, svc_gid, 0o755),
        (data_path / "history", svc_uid, svc_gid, 0o755),
        (data_path / "memory", svc_uid, svc_gid, 0o755),
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


def _apply_source(install_path: Path, svc_uid: int, svc_gid: int, dry_run: bool) -> None:
    """Copy source tree and home config from PROJECT_ROOT to the install location."""
    src_src = PROJECT_ROOT / "src"
    src_dst = install_path / "src"
    pyproject_src = PROJECT_ROOT / "pyproject.toml"
    pyproject_dst = install_path / "pyproject.toml"
    ws_claude_src = PROJECT_ROOT / "home" / ".claude"
    ws_claude_dst = install_path / "home" / ".claude"

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

    if dry_run:
        print(f"[DRY RUN] Would copy: {src_src} -> {src_dst}")
        print(f"[DRY RUN] Would copy: {pyproject_src} -> {pyproject_dst}")
        if ws_claude_src.is_dir():
            print(f"[DRY RUN] Would copy: {ws_claude_src} -> {ws_claude_dst}")
        return

    _copy_tree(src_src, src_dst, _SOURCE_EXCLUDES)
    _set_ownership(src_dst, 0, 0, recursive=True)
    print(f"  Copied source to {src_dst}")

    shutil.copy2(pyproject_src, pyproject_dst)
    os.chown(pyproject_dst, 0, 0)
    print(f"  Copied {pyproject_dst}")

    # Copy home/.claude/ (bot identity, memory template) excluding
    # runtime data. Without CLAUDE.md, the bot has no identity in the
    # home workspace and nothing to inject into foreign workspace sessions.
    # Files inside are root-owned (read-only config), but the directory
    # itself is service-user-owned so skills/ and other runtime dirs can
    # be created inside it.
    if ws_claude_src.is_dir():
        ws_claude_dst.parent.mkdir(parents=True, exist_ok=True)
        _copy_tree(ws_claude_src, ws_claude_dst, _HOME_CLAUDE_EXCLUDES)
        _set_ownership(ws_claude_dst, 0, 0, recursive=True)
        os.chown(ws_claude_dst, svc_uid, svc_gid)
        print(f"  Copied home config to {ws_claude_dst}")


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
    pyproject_dst = install_path / "pyproject.toml"
    src_dst = install_path / "src"

    if is_update and venv_path.exists():
        # Check if pyproject.toml or source files changed. Both are needed
        # because the install is non-editable: pip copies code into the venv's
        # site-packages, so updating src/ at the install path alone does
        # nothing. A pyproject.toml change means dependencies may have changed;
        # a source change means the installed package code is stale.
        pyproject_checksum_file = install_path / ".pyproject.sha256"
        old_pyproject = ""
        if pyproject_checksum_file.exists():
            old_pyproject = pyproject_checksum_file.read_text().strip()
        new_pyproject = _file_checksum(pyproject_dst)

        src_checksum_file = install_path / ".src.sha256"
        old_src = ""
        if src_checksum_file.exists():
            old_src = src_checksum_file.read_text().strip()
        new_src = _src_checksum(src_dst)

        if old_pyproject == new_pyproject and old_src == new_src:
            print("  Venv unchanged (pyproject.toml and source checksums match)")
            return

        # Report what changed for operator visibility
        changed: list[str] = []
        if old_pyproject != new_pyproject:
            changed.append("pyproject.toml")
        if old_src != new_src:
            changed.append("source")

        if dry_run:
            print(f"[DRY RUN] Would update venv ({' and '.join(changed)} changed)")
            return

        print(f"  Updating venv ({' and '.join(changed)} changed)...")
    else:
        if dry_run:
            print(f"[DRY RUN] Would create venv: {venv_path}")
            print("[DRY RUN] Would install package into venv")
            return

        print(f"  Creating venv at {venv_path}...")
        # Find Python 3.13+
        python = shutil.which("python3.13") or shutil.which("python3") or "python3"

        # Verify the resolved Python meets the minimum version before creating
        # the venv. Without this, a 3.12 venv gets built successfully but pip
        # install fails later with a confusing requires-python error.
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

        subprocess.run(
            [python, "-m", "venv", str(venv_path)],
            check=True,
        )

    # Install the package with optional dependencies.
    # Uses a non-editable install (not -e) so the venv is self-contained
    # and doesn't depend on the source directory being writable.
    pip = str(venv_path / "bin" / "pip")
    extras = "totp,tts"
    install_spec = f"{install_path}[{extras}]"
    subprocess.run(
        [pip, "install", install_spec],
        check=True,
    )
    print("  Installed package into venv")

    # Save checksums for future update detection. Both are written after a
    # successful install so that a partial failure (e.g., pip crash mid-install)
    # leaves stale checksums and triggers a retry on the next run.
    (install_path / ".pyproject.sha256").write_text(_file_checksum(pyproject_dst) + "\n")
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


def _apply_secrets(env: dict[str, str], dry_run: bool) -> None:
    """Write the /etc/kai/env file from install.conf environment values."""
    etc_kai = Path("/etc/kai")
    env_path = etc_kai / "env"
    env_content = _generate_env_file(env)

    if dry_run:
        print(f"[DRY RUN] Would write: {env_path} (mode 0600)")
        for yaml_name in ("services.yaml", "users.yaml", "workspaces.yaml"):
            if (PROJECT_ROOT / yaml_name).exists():
                print(f"[DRY RUN] Would copy: {etc_kai / yaml_name} (mode 0600)")
        return

    env_path.write_text(env_content)
    os.chmod(env_path, 0o600)
    os.chown(env_path, 0, 0)
    print(f"  Wrote {env_path}")

    # Copy optional YAML config files to /etc/kai/ if they exist in the
    # source directory. All get root-only permissions (mode 0600) since
    # they may contain sensitive configuration (API keys in services.yaml,
    # user IDs in users.yaml).
    for yaml_name in ("services.yaml", "users.yaml", "workspaces.yaml"):
        yaml_src = PROJECT_ROOT / yaml_name
        yaml_dst = etc_kai / yaml_name
        if yaml_src.exists():
            shutil.copy2(yaml_src, yaml_dst)
            os.chmod(yaml_dst, 0o600)
            os.chown(yaml_dst, 0, 0)
            print(f"  Copied {yaml_dst}")


def _apply_sudoers(service_user: str, dry_run: bool, claude_user: str | None = None) -> None:
    """Write sudoers rules for the service user to read protected config."""
    sudoers_path = Path("/etc/sudoers.d/kai")
    sudoers_content = _generate_sudoers(service_user, claude_user)

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
