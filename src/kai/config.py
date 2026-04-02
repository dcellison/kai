"""
Application configuration loaded from environment variables.

Provides functionality to:
1. Define the Config dataclass with all application settings
2. Load and validate configuration from .env file
3. Resolve filesystem paths relative to the project root
4. Fail fast with clear error messages on misconfiguration

The main interface is through load_config(), which returns a frozen Config instance.
All paths are resolved relative to PROJECT_ROOT (the repository root), which is
derived from this file's location in the source tree: src/kai/config.py -> project root.
"""

import logging
import os
import pwd
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

log = logging.getLogger(__name__)


# ── Module-level paths and constants ─────────────────────────────────

# Derive project root from file location: src/kai/config.py -> src/kai -> src -> project root.
# In a pip-installed deployment (e.g., /opt/kai/venv/lib/.../site-packages/kai/), this
# resolves to site-packages/ instead of the install root. KAI_INSTALL_DIR overrides it.
_FILE_ROOT = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = Path(os.environ.get("KAI_INSTALL_DIR") or str(_FILE_ROOT))

# Writable data directory for runtime artifacts (database, logs, crash flag).
# Defaults to PROJECT_ROOT for development. In a protected installation where
# source lives in read-only /opt/kai/, this points to user-owned /var/lib/kai/.
# Uses `or` so that an empty string also falls back (same pattern as CLAUDE_USER).
DATA_DIR = Path(os.environ.get("KAI_DATA_DIR") or str(PROJECT_ROOT))

# Single source of truth for valid Claude model names.
# install.py and bot.py both reference this instead of maintaining their own lists.
VALID_MODELS = {"haiku", "sonnet", "opus"}

# Image file extensions that Telegram renders inline as photos.
# Shared between bot.py (inbound document handling) and webhook.py (send-file API).
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


# ── Self-sudo resolution ─────────────────────────────────────────────


def resolve_claude_user(claude_user: str | None) -> str | None:
    """
    Return None if claude_user matches the current process user.

    When the bot process runs as the same OS user specified in
    claude_user, sudo -u is both unnecessary and likely to fail
    (sudoers typically disallows self-sudo). This function detects
    that case and returns None so callers skip the sudo wrapper.

    Returns claude_user unchanged when it differs from the current
    user, or when the current user cannot be determined (e.g.,
    containers with unmapped UIDs).
    """
    if not claude_user:
        return None
    try:
        current_user = pwd.getpwuid(os.getuid()).pw_name
    except KeyError:
        # UID has no passwd entry (e.g., containers with --user <uid>).
        # Can't determine if it's a self-sudo, so fall through to sudo.
        return claude_user
    if claude_user == current_user:
        log.warning(
            "claude_user %r matches bot process user; skipping sudo",
            claude_user,
        )
        return None
    return claude_user


# ── Per-workspace configuration ──────────────────────────────────────


@dataclass(frozen=True)
class WorkspaceConfig:
    """
    Per-workspace configuration loaded from workspaces.yaml.

    All fields except path are optional. When None, the global default
    from Config is used instead. This lets workspaces override only
    the settings they care about.

    Attributes:
        path: Canonical resolved workspace directory.
        model: Claude model override (haiku/sonnet/opus).
        budget: Per-session spending cap in USD.
        timeout: Per-readline timeout in seconds.
        env: Inline environment variables for the Claude subprocess.
        env_file: Path to a KEY=VALUE file to load as environment vars.
        system_prompt: Inline system prompt text.
        system_prompt_file: Path to a file containing the system prompt.
    """

    path: Path
    model: str | None = None
    budget: float | None = None
    timeout: int | None = None
    env: dict[str, str] | None = None
    env_file: Path | None = None
    system_prompt: str | None = None
    system_prompt_file: Path | None = None


# ── Per-user configuration ──────────────────────────────────────────


@dataclass(frozen=True)
class UserConfig:
    """
    Per-user configuration loaded from users.yaml.

    Defines a user's identity, authorization, and resource limits.
    Preferences that the user controls (active model, working budget)
    live in the settings table, not here.

    Attributes:
        telegram_id: Telegram user ID (authorization key).
        name: Display name for logs and notifications.
        role: "admin" or "user". Admins receive unattributed webhooks.
        github: GitHub username for webhook actor routing.
        os_user: OS username for subprocess isolation (Phase 3).
        home_workspace: Per-user home workspace directory.
        max_budget: Budget ceiling in USD (user cannot exceed via /budget).
    """

    telegram_id: int
    name: str
    role: str = "user"
    github: str | None = None
    os_user: str | None = None
    home_workspace: Path | None = None
    max_budget: float | None = None


# ── Config dataclass ─────────────────────────────────────────────────


@dataclass(frozen=True)
class Config:
    """
    Immutable application configuration populated from environment variables.

    All fields map to environment variables defined in .env (see .env.example for
    reference). Required fields raise SystemExit with descriptive messages if missing.
    Optional fields have sensible defaults for single-user local deployment.

    Attributes:
        telegram_bot_token: Bot token from @BotFather (required)
        telegram_webhook_url: Public URL where Telegram pushes updates via webhook.
            When set, Kai runs in webhook mode (Telegram POSTs updates here).
            When None, Kai falls back to long-polling (Kai pulls updates from Telegram).
        telegram_webhook_secret: Secret token for validating incoming Telegram updates.
            Sent by Telegram as X-Telegram-Bot-Api-Secret-Token header on each update.
            Only required in webhook mode. Defaults to WEBHOOK_SECRET if not explicitly set.
        allowed_user_ids: Set of Telegram user IDs permitted to interact with the bot (required)
        claude_model: Model name passed to the inner Claude Code process (haiku/sonnet/opus)
        claude_timeout_seconds: Seconds before a Claude response is considered timed out
        claude_max_budget_usd: Per-session spending cap in USD
        claude_max_session_hours: Hours before the inner Claude process is recycled. Prevents
            unbounded V8 memory growth that can trigger macOS Jetsam kernel panics. 0 = no limit.
        claude_workspace: Working directory for the inner Claude Code process
        session_db_path: Path to the SQLite database for sessions, jobs, and settings
        webhook_port: Port for the local aiohttp server (webhooks + scheduling API)
        webhook_secret: HMAC secret for verifying incoming webhook payloads
        voice_enabled: Whether to transcribe Telegram voice notes via whisper-cpp
        whisper_model_path: Path to the whisper-cpp GGML model file
        tts_enabled: Whether to enable Piper text-to-speech for voice responses
        piper_model_dir: Directory containing Piper voice model files
        workspace_base: Base directory for workspace name resolution (/workspace <name>)
        allowed_workspaces: Additional workspace directories accessible by name, from config only.
            These appear as pinned workspaces in /workspaces and are reachable via /workspace <name>
            without being under WORKSPACE_BASE. Non-existent paths are skipped at startup.
        claude_user: OS user for the inner Claude process. When set, Claude is spawned
            via 'sudo -u <user>' for OS-level isolation. Must be a non-admin user.
            When None, Claude runs as the same user as the bot (development default).
    """

    # Required fields - no defaults, must be provided
    telegram_bot_token: str
    allowed_user_ids: set[int]

    # Telegram transport mode: set telegram_webhook_url to use webhook mode,
    # leave as None to fall back to long-polling. The secret is only needed
    # in webhook mode to authenticate incoming updates from Telegram.
    telegram_webhook_url: str | None = None
    telegram_webhook_secret: str | None = None

    # Claude Code process configuration
    claude_model: str = "sonnet"
    claude_timeout_seconds: int = 120
    claude_max_budget_usd: float = 10.0
    claude_max_session_hours: float = 0  # 0 = no limit
    claude_idle_timeout: int = 1800  # seconds before idle subprocess eviction; 0 = no eviction
    claude_workspace: Path = field(default_factory=lambda: PROJECT_ROOT / "home")

    # Context window tuning. Both settings help control token usage and
    # reduce cache invalidation pressure on the inner Claude process.
    # 0 = use Claude Code defaults (1M window, ~83% autocompact threshold).
    claude_max_context_window: int = 0  # tokens; passed as --max-context-window CLI flag
    claude_autocompact_pct: int = 0  # 1-100; passed as CLAUDE_AUTOCOMPACT_PCT_OVERRIDE env var

    # Database - uses DATA_DIR so the db lands in the writable data directory
    session_db_path: Path = field(default_factory=lambda: DATA_DIR / "kai.db")

    # Webhook server
    webhook_port: int = 8080
    webhook_secret: str = ""

    # Voice input (speech-to-text via whisper-cpp)
    voice_enabled: bool = False
    whisper_model_path: Path = field(default_factory=lambda: PROJECT_ROOT / "models" / "ggml-base.en.bin")

    # Voice output (text-to-speech via Piper)
    tts_enabled: bool = False
    piper_model_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "models" / "piper")

    # Workspace switching
    workspace_base: Path | None = None
    allowed_workspaces: list[Path] = field(default_factory=list)

    # Per-workspace configuration from workspaces.yaml. Keyed by
    # canonical resolved path. Empty dict if no config file exists.
    workspace_configs: dict[Path, WorkspaceConfig] = field(default_factory=dict)

    # User separation: run Claude as a different OS user for process isolation.
    # When set, the bot spawns Claude via 'sudo -u <user> claude ...'.
    # The user must exist on the system and must NOT have admin/sudo privileges.
    # When unset, Claude runs as the same user as the bot (development default).
    claude_user: str | None = None

    # PR review agent: automatically review PRs when GitHub webhooks fire.
    # Disabled by default so existing users are not surprised by automatic reviews.
    pr_review_enabled: bool = False
    # Minimum seconds between reviews of the same PR. Absorbs force-push bursts
    # so rapid pushes to an open PR don't trigger a review for each one.
    pr_review_cooldown: int = 300
    # Deprecated: review agent now resolves repos via workspace config.
    # Kept for backwards compatibility with existing .env files; the value
    # is parsed but no longer used by webhook.py.
    github_repo: str = ""
    # Directory (relative to repo root) where spec files live for
    # branch-name matching. Does not affect body marker resolution,
    # which accepts any path relative to the repo root.
    spec_dir: str = "specs"

    # Issue triage agent: automatically triage new issues when webhooks fire.
    # Disabled by default so existing users are not surprised by automatic triage.
    issue_triage_enabled: bool = False

    # File retention: delete uploaded files older than this many days.
    # 0 = no cleanup (default). Cleanup runs once every 24 hours.
    file_retention_days: int = 0

    # Per-user configuration from users.yaml. Keyed by telegram_id.
    # None means users.yaml does not exist (fall back to allowed_user_ids).
    # Empty dict means users.yaml exists but has no valid entries.
    user_configs: dict[int, UserConfig] | None = None

    # TOTP two-factor authentication timing (only relevant when TOTP is enabled)
    totp_session_minutes: int = 30
    totp_challenge_seconds: int = 120
    totp_lockout_attempts: int = 3
    totp_lockout_minutes: int = 15

    def get_workspace_config(self, workspace: Path) -> WorkspaceConfig | None:
        """
        Get per-workspace config for a path, or None for global defaults.

        Resolves the path before lookup to handle symlinks and relative
        paths consistently.
        """
        return self.workspace_configs.get(workspace.resolve())

    def get_user_config(self, user_id: int) -> UserConfig | None:
        """Get per-user config by Telegram user ID, or None if not configured."""
        if self.user_configs is None:
            return None
        return self.user_configs.get(user_id)

    def get_user_by_github(self, github_login: str) -> UserConfig | None:
        """Look up a user by GitHub username. Used for webhook actor routing."""
        if self.user_configs is None:
            return None
        for uc in self.user_configs.values():
            if uc.github and uc.github.lower() == github_login.lower():
                return uc
        return None

    def get_admins(self) -> list[UserConfig]:
        """Get all admin users. Used for unattributed webhook fallback routing."""
        if self.user_configs is None:
            return []
        return [uc for uc in self.user_configs.values() if uc.role == "admin"]


# ── Config loading ───────────────────────────────────────────────────


def _read_protected_file(path: str) -> str | None:
    """
    Read a root-owned file via sudo cat.

    Used to load config from /etc/kai/ in a protected installation where the
    bot process runs as an unprivileged user. The -n flag ensures sudo fails
    immediately if no NOPASSWD rule exists (avoids blocking on a password prompt).

    Returns:
        File contents as a string, or None on any failure (missing file,
        no sudoers rule, timeout, etc.).
    """
    try:
        result = subprocess.run(
            ["sudo", "-n", "cat", path],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout if result.returncode == 0 else None
    except (subprocess.TimeoutExpired, OSError):
        return None


# Sentinel returned by _read_protected_yaml when the file exists but
# contains invalid YAML. Distinct from None (file absent) so callers
# can stop on malformed config rather than falling through to a local file.
_YAML_MALFORMED = object()


def _read_protected_yaml(filename: str) -> dict | object | None:
    """
    Read a YAML file from /etc/kai/ via sudo.

    Returns:
        Parsed dict on success, None if the file does not exist or cannot
        be read, or the _YAML_MALFORMED sentinel if the file exists but
        is invalid. Callers must check ``is _YAML_MALFORMED`` before use.
    """
    content = _read_protected_file(f"/etc/kai/{filename}")
    if content is None:
        return None
    try:
        result = yaml.safe_load(content)
        if isinstance(result, dict):
            return result
        log.warning("/etc/kai/%s: expected a YAML dict, got %s", filename, type(result).__name__)
        return _YAML_MALFORMED
    except yaml.YAMLError as e:
        log.error("Invalid YAML in /etc/kai/%s: %s", filename, e)
        return _YAML_MALFORMED


def _strip_quotes(value: str) -> str:
    """
    Remove matched surrounding quotes from a value string.

    Only strips when the first and last characters are the same quote
    type (' or "). This avoids corrupting values that contain the
    opposite quote type internally (e.g., 'he said "hello"').
    """
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
        return value[1:-1]
    return value


def parse_env_file(path: Path) -> dict[str, str]:
    """
    Parse a KEY=VALUE file into a dict.

    Handles:
    - Lines with KEY=VALUE or KEY="VALUE" or KEY='VALUE'
    - Lines starting with 'export ' (stripped)
    - Comments (lines starting with #) and blank lines (skipped)
    - Surrounding quotes on values (stripped as matched pairs only -
      inner quotes of the opposite type are preserved)

    Same parsing logic as _read_protected_file() uses for /etc/kai/env.
    Re-reads the file each time to pick up changes without restart.
    """
    env: dict[str, str] = {}
    try:
        # utf-8-sig transparently strips a BOM if present. Windows
        # editors (Notepad, etc.) often save UTF-8 with a BOM, which
        # would silently prepend \ufeff to the first key name.
        text = path.read_text(encoding="utf-8-sig")
    except OSError as e:
        log.warning("Cannot read env file %s: %s", path, e)
        return env
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        # Handle `export KEY=VALUE` lines (common in shell-sourced env files)
        line = line.removeprefix("export ")
        key, _, value = line.partition("=")
        env[key.strip()] = _strip_quotes(value.strip())
    return env


def _load_workspace_configs() -> dict[Path, WorkspaceConfig]:
    """
    Load per-workspace configs from workspaces.yaml.

    Tries /etc/kai/workspaces.yaml first (protected installation),
    falls back to PROJECT_ROOT/workspaces.yaml (development). Returns
    an empty dict if neither file exists.

    Returns a dict keyed by canonical resolved path for O(1) lookup.
    """
    # Try protected file first, fall back to local. A malformed
    # protected file stops loading entirely rather than silently
    # falling through to a local file (which could contain stale
    # or dev config on a production system).
    data = _read_protected_yaml("workspaces.yaml")
    if data is _YAML_MALFORMED:
        log.warning("Skipping workspace config: /etc/kai/workspaces.yaml is malformed or empty")
        return {}
    if data is None:
        local_path = PROJECT_ROOT / "workspaces.yaml"
        if not local_path.exists():
            return {}
        try:
            with open(local_path) as f:
                data = yaml.safe_load(f)
        except (yaml.YAMLError, OSError) as e:
            log.error("Cannot load %s: %s", local_path, e)
            return {}
        if not isinstance(data, dict):
            log.warning("%s: expected a YAML dict, got %s", local_path, type(data).__name__)
            return {}

    entries = data.get("workspaces")
    if not isinstance(entries, list):
        if entries is not None:
            log.warning("workspaces.yaml: 'workspaces' must be a list, got %s", type(entries).__name__)
        return {}

    # Helper for coercing YAML env values to strings. Defined once
    # outside the loop rather than re-created per workspace entry.
    def _coerce_env_value(v: object) -> str:
        if v is None:
            return ""
        if isinstance(v, bool):
            return str(v).lower()
        return str(v)

    configs: dict[Path, WorkspaceConfig] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            log.warning("workspaces.yaml: skipping non-dict entry: %s", entry)
            continue

        # Validate required path field
        raw_path = entry.get("path")
        if not raw_path:
            log.warning("workspaces.yaml: skipping entry without path")
            continue
        path = Path(str(raw_path)).expanduser().resolve()
        if not path.is_dir():
            log.warning("workspaces.yaml: skipping non-existent path: %s", path)
            continue

        # Duplicate check: first wins
        if path in configs:
            log.warning("workspaces.yaml: duplicate path %s; using first entry", path)
            continue

        # Parse the optional claude: section
        claude_section = entry.get("claude") or {}
        if not isinstance(claude_section, dict):
            log.warning("workspaces.yaml: invalid claude section for %s", path)
            continue

        # Validate model
        model = claude_section.get("model")
        if model is not None:
            model = str(model)
            if model not in VALID_MODELS:
                log.warning(
                    "workspaces.yaml: invalid model '%s' for %s (must be one of %s); skipping entry",
                    model,
                    path,
                    VALID_MODELS,
                )
                continue

        # Validate budget (same bool guard as timeout - float(True) is 1.0)
        budget = claude_section.get("budget")
        if budget is not None:
            try:
                if isinstance(budget, bool):
                    raise ValueError("must be a number, not a boolean")
                budget = float(budget)
                if budget <= 0:
                    raise ValueError("must be positive")
            except (TypeError, ValueError) as e:
                log.warning("workspaces.yaml: invalid budget for %s: %s; skipping entry", path, e)
                continue

        # Validate timeout (must be a positive integer, not a float or bool).
        # bool is a subclass of int in Python, so `timeout: true` would
        # silently become 1 without an explicit check.
        timeout = claude_section.get("timeout")
        if timeout is not None:
            try:
                if isinstance(timeout, bool):
                    raise ValueError("must be an integer, not a boolean")
                if isinstance(timeout, float) and not timeout.is_integer():
                    raise ValueError("must be an integer, not a float")
                timeout = int(timeout)
                if timeout <= 0:
                    raise ValueError("must be positive")
            except (TypeError, ValueError) as e:
                log.warning("workspaces.yaml: invalid timeout for %s: %s; skipping entry", path, e)
                continue

        # Parse env vars (inline dict)
        env = claude_section.get("env")
        if env is not None:
            if not isinstance(env, dict):
                log.warning("workspaces.yaml: invalid env for %s; skipping entry", path)
                continue

            env = {str(k): _coerce_env_value(v) for k, v in env.items()}

        # Validate env_file
        env_file = claude_section.get("env_file")
        if env_file is not None:
            env_file = Path(str(env_file)).expanduser().resolve()
            if not env_file.is_file():
                log.warning("workspaces.yaml: env_file not found for %s: %s; skipping entry", path, env_file)
                continue

        # Validate system_prompt / system_prompt_file mutual exclusion
        system_prompt = claude_section.get("system_prompt")
        system_prompt_file = claude_section.get("system_prompt_file")
        if system_prompt is not None and system_prompt_file is not None:
            log.error(
                "workspaces.yaml: both system_prompt and system_prompt_file set for %s; skipping entry",
                path,
            )
            continue
        if system_prompt is not None:
            system_prompt = str(system_prompt)
        if system_prompt_file is not None:
            system_prompt_file = Path(str(system_prompt_file)).expanduser().resolve()
            if not system_prompt_file.is_file():
                log.warning(
                    "workspaces.yaml: system_prompt_file not found for %s: %s; skipping entry",
                    path,
                    system_prompt_file,
                )
                continue

        configs[path] = WorkspaceConfig(
            path=path,
            model=model,
            budget=budget,
            timeout=timeout,
            env=env,
            env_file=env_file,
            system_prompt=system_prompt,
            system_prompt_file=system_prompt_file,
        )

    return configs


# Valid user roles for users.yaml
_VALID_ROLES = {"admin", "user"}


def _load_user_configs() -> dict[int, UserConfig] | None:
    """
    Load per-user configs from users.yaml.

    Tries /etc/kai/users.yaml first (protected), falls back to
    PROJECT_ROOT/users.yaml (dev). Returns None if neither exists
    (signals the caller to fall back to ALLOWED_USER_IDS).

    Returns a dict keyed by telegram_id for O(1) lookup.
    """
    # Same dual-mode loading pattern as _load_workspace_configs.
    data = _read_protected_yaml("users.yaml")
    if data is _YAML_MALFORMED:
        log.warning("Skipping user config: /etc/kai/users.yaml is malformed or empty")
        return None
    if data is None:
        local_path = PROJECT_ROOT / "users.yaml"
        if not local_path.exists():
            return None
        try:
            with open(local_path) as f:
                data = yaml.safe_load(f)
        except (yaml.YAMLError, OSError) as e:
            log.error("Cannot load %s: %s", local_path, e)
            return None
        if not isinstance(data, dict):
            log.warning("%s: expected a YAML dict, got %s", local_path, type(data).__name__)
            return None

    # After the _YAML_MALFORMED and None checks, data should be a dict
    # (protected path returns _YAML_MALFORMED for non-dicts, local path
    # checks isinstance explicitly). Guard defensively rather than assert
    # since assertions are stripped under Python -O.
    if not isinstance(data, dict):
        log.warning("users.yaml: expected a YAML dict, got %s", type(data).__name__)
        return None

    entries = data.get("users")
    if not isinstance(entries, list):
        # Warn for any non-list value, including None (missing key).
        # A users.yaml file without a 'users' key is almost certainly
        # a typo (e.g., 'user:' instead of 'users:').
        if entries is not None:
            log.warning("users.yaml: 'users' must be a list, got %s", type(entries).__name__)
        else:
            log.warning("users.yaml: no 'users' key found; check for typos")
        return None

    configs: dict[int, UserConfig] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            log.warning("users.yaml: skipping non-dict entry: %s", entry)
            continue

        # Validate required telegram_id (must be a positive integer, not a bool)
        raw_id = entry.get("telegram_id")
        if raw_id is None:
            log.warning("users.yaml: skipping entry without telegram_id")
            continue
        try:
            if isinstance(raw_id, bool):
                raise ValueError("must be an integer, not a boolean")
            telegram_id = int(raw_id)
            if telegram_id <= 0:
                raise ValueError("must be positive")
        except (TypeError, ValueError) as e:
            log.warning("users.yaml: invalid telegram_id %s: %s; skipping entry", raw_id, e)
            continue

        # Validate required name (strip first so whitespace-only is rejected)
        name = str(entry.get("name") or "").strip()
        if not name:
            log.warning("users.yaml: skipping entry for telegram_id %d without name", telegram_id)
            continue

        # Duplicate check: first wins
        if telegram_id in configs:
            log.warning("users.yaml: duplicate telegram_id %d; using first entry", telegram_id)
            continue

        # Validate role
        role = str(entry.get("role", "user")).strip().lower()
        if role not in _VALID_ROLES:
            log.warning(
                "users.yaml: invalid role '%s' for %s (must be one of %s); skipping entry",
                role,
                name,
                _VALID_ROLES,
            )
            continue

        # Optional fields
        github = entry.get("github")
        if github is not None:
            github = str(github).strip() or None

        os_user = entry.get("os_user")
        if os_user is not None:
            os_user = str(os_user).strip() or None

        # Validate home_workspace. Warn but don't skip the user if
        # the directory doesn't exist - it may be on an unmounted drive
        # or not yet created. The user keeps access; the workspace
        # falls back to the global default at runtime.
        home_workspace = entry.get("home_workspace")
        if home_workspace is not None:
            # Guard against empty strings: Path("").resolve() silently
            # returns CWD, which would give the user unintended access.
            home_workspace_str = str(home_workspace).strip()
            if not home_workspace_str:
                home_workspace = None
            else:
                home_workspace = Path(home_workspace_str).expanduser().resolve()
                if not home_workspace.is_dir():
                    log.warning(
                        "users.yaml: home_workspace not found for %s: %s; using global default",
                        name,
                        home_workspace,
                    )
                    home_workspace = None

        # Validate max_budget (same bool guard as workspace budget)
        max_budget = entry.get("max_budget")
        if max_budget is not None:
            try:
                if isinstance(max_budget, bool):
                    raise ValueError("must be a number, not a boolean")
                max_budget = float(max_budget)
                if max_budget <= 0:
                    raise ValueError("must be positive")
            except (TypeError, ValueError) as e:
                log.warning("users.yaml: invalid max_budget for %s: %s; skipping entry", name, e)
                continue

        configs[telegram_id] = UserConfig(
            telegram_id=telegram_id,
            name=name,
            role=role,
            github=github,
            os_user=os_user,
            home_workspace=home_workspace,
            max_budget=max_budget,
        )

    # Warn if no admin is defined - external webhooks will route to
    # an arbitrary user, which may be surprising.
    if configs and not any(uc.role == "admin" for uc in configs.values()):
        log.warning(
            "users.yaml: no admin users defined. External webhook notifications "
            "(GitHub, generic) will route to an arbitrary user."
        )

    return configs


def load_config() -> Config:
    """
    Load application configuration from environment variables.

    Reads from the .env file at the project root via python-dotenv, validates
    required fields, and returns a frozen Config instance. Calls SystemExit
    with descriptive messages on any misconfiguration so the bot fails fast
    at startup rather than encountering cryptic errors later.

    Returns:
        A frozen Config instance with all settings populated.

    Raises:
        SystemExit: If required environment variables are missing or invalid.
    """
    # Try protected config first (/etc/kai/env, root-owned). In a protected
    # installation, secrets live here instead of .env. Falls back to local
    # .env for development. Uses setdefault so explicitly set env vars
    # (e.g., from the launchd plist) take precedence - same as load_dotenv().
    protected_env = _read_protected_file("/etc/kai/env")
    if protected_env:
        for line in protected_env.splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                # Handle `export KEY=VALUE` lines (common in shell-sourced env files)
                line = line.removeprefix("export ")
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), _strip_quotes(value.strip()))
    else:
        load_dotenv(PROJECT_ROOT / ".env")

    # Validate required: Telegram bot token
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("TELEGRAM_BOT_TOKEN is required in .env")

    # Telegram transport mode: if TELEGRAM_WEBHOOK_URL is set, use webhook mode
    # (Telegram POSTs updates to this URL). If unset, fall back to long-polling
    # (Kai pulls updates from Telegram). This lets users without a tunnel/proxy
    # run Kai out of the box.
    telegram_webhook_url: str | None = None
    telegram_webhook_secret: str | None = None
    raw_webhook_url = os.environ.get("TELEGRAM_WEBHOOK_URL", "").strip()
    if raw_webhook_url:
        telegram_webhook_url = raw_webhook_url

        # Webhook secret: validates incoming updates from Telegram. Falls back to
        # WEBHOOK_SECRET so existing installs work without a config change. Must be
        # non-empty in webhook mode; an empty secret would let anyone POST fake
        # updates to /webhook/telegram.
        telegram_webhook_secret = os.environ.get("TELEGRAM_WEBHOOK_SECRET", "").strip()
        if not telegram_webhook_secret:
            telegram_webhook_secret = os.environ.get("WEBHOOK_SECRET", "").strip()
        if not telegram_webhook_secret:
            raise SystemExit(
                "TELEGRAM_WEBHOOK_SECRET (or WEBHOOK_SECRET as fallback) is required in .env "
                "when TELEGRAM_WEBHOOK_URL is set. Without it, anyone could POST fake updates "
                "to the Telegram webhook endpoint."
            )
        log.info("Telegram transport: webhook (%s)", telegram_webhook_url)
    else:
        log.info("Telegram transport: polling (TELEGRAM_WEBHOOK_URL not set)")

    # Parse allowed user IDs. users.yaml is the primary authorization
    # source. ALLOWED_USER_IDS is a legacy fallback for installations
    # that haven't migrated yet.
    # Defer the ValueError until after users.yaml is checked so that
    # a malformed env var doesn't block startup when users.yaml is
    # authoritative and would make the env var irrelevant.
    raw_ids = os.environ.get("ALLOWED_USER_IDS", "")
    allowed_ids: set[int] = set()
    allowed_ids_error: str | None = None
    try:
        allowed_ids = {int(uid.strip()) for uid in raw_ids.split(",") if uid.strip()}
    except ValueError:
        allowed_ids_error = (
            "ALLOWED_USER_IDS contains non-numeric values. "
            "Consider migrating to users.yaml (run 'make config' or "
            "see README for the schema). If using ALLOWED_USER_IDS, values must be "
            "numeric Telegram user IDs - message @userinfobot to find yours."
        )

    # Validate optional: workspace base directory (must exist if provided)
    workspace_base = None
    raw_base = os.environ.get("WORKSPACE_BASE", "").strip()
    if raw_base:
        workspace_base = Path(raw_base).expanduser().resolve()
        if not workspace_base.is_dir():
            raise SystemExit(f"WORKSPACE_BASE is not an existing directory: {workspace_base}")

    # Parse optional: allowed workspaces (comma-separated absolute paths).
    # Paths are resolved to canonical form so /a/b and /a/../a/b deduplicate
    # to one entry. Non-existent paths are skipped with a warning rather than
    # crashing, so a stale entry (e.g. an unmounted drive) doesn't block startup.
    allowed_workspaces: list[Path] = []
    seen_allowed: set[Path] = set()
    raw_allowed = os.environ.get("ALLOWED_WORKSPACES", "").strip()
    if raw_allowed:
        for raw_path in raw_allowed.split(","):
            p = Path(raw_path.strip()).expanduser().resolve()
            if p in seen_allowed:
                continue
            seen_allowed.add(p)
            if p.is_dir():
                allowed_workspaces.append(p)
            else:
                log.warning("ALLOWED_WORKSPACES: skipping non-existent path: %s", p)

    # Validate numeric config - fail fast with clear messages rather than
    # cryptic ValueError tracebacks from int()/float() on bad input
    try:
        claude_timeout_seconds = int(os.environ.get("CLAUDE_TIMEOUT_SECONDS", "120"))
    except ValueError:
        raise SystemExit("CLAUDE_TIMEOUT_SECONDS must be an integer") from None
    try:
        claude_max_budget_usd = float(os.environ.get("CLAUDE_MAX_BUDGET_USD", "10.0"))
    except ValueError:
        raise SystemExit("CLAUDE_MAX_BUDGET_USD must be a number") from None
    try:
        claude_max_session_hours = float(os.environ.get("CLAUDE_MAX_SESSION_HOURS", "0"))
    except ValueError:
        raise SystemExit("CLAUDE_MAX_SESSION_HOURS must be a number") from None
    try:
        claude_idle_timeout = int(os.environ.get("CLAUDE_IDLE_TIMEOUT", "1800"))
    except ValueError:
        raise SystemExit("CLAUDE_IDLE_TIMEOUT must be an integer") from None
    try:
        webhook_port = int(os.environ.get("WEBHOOK_PORT", "8080"))
    except ValueError:
        raise SystemExit("WEBHOOK_PORT must be an integer") from None
    try:
        file_retention_days = int(os.environ.get("FILE_RETENTION_DAYS", "0"))
    except ValueError:
        raise SystemExit("FILE_RETENTION_DAYS must be an integer") from None

    # Context window tuning - 0 means "use Claude Code defaults"
    _MAX_CONTEXT_CEILING = 1_000_000  # 1M tokens; Claude's maximum
    try:
        claude_max_context_window = int(os.environ.get("CLAUDE_MAX_CONTEXT_WINDOW", "0"))
        if claude_max_context_window < 0 or claude_max_context_window > _MAX_CONTEXT_CEILING:
            raise SystemExit(f"CLAUDE_MAX_CONTEXT_WINDOW must be 0-{_MAX_CONTEXT_CEILING} (0 = use default)")
    except ValueError:
        raise SystemExit("CLAUDE_MAX_CONTEXT_WINDOW must be an integer") from None
    try:
        claude_autocompact_pct = int(os.environ.get("CLAUDE_AUTOCOMPACT_PCT", "0"))
        if claude_autocompact_pct < 0 or claude_autocompact_pct > 100:
            raise SystemExit("CLAUDE_AUTOCOMPACT_PCT must be 0-100 (0 = use default)")
    except ValueError:
        raise SystemExit("CLAUDE_AUTOCOMPACT_PCT must be an integer") from None

    # PR review agent config
    pr_review_enabled = os.environ.get("PR_REVIEW_ENABLED", "").lower() in ("1", "true", "yes")
    try:
        pr_review_cooldown = int(os.environ.get("PR_REVIEW_COOLDOWN", "300"))
    except ValueError:
        raise SystemExit("PR_REVIEW_COOLDOWN must be an integer") from None

    # Issue triage agent config
    issue_triage_enabled = os.environ.get("ISSUE_TRIAGE_ENABLED", "").lower() in ("1", "true", "yes")

    try:
        totp_session_minutes = int(os.environ.get("TOTP_SESSION_MINUTES", "30"))
    except ValueError:
        raise SystemExit("TOTP_SESSION_MINUTES must be an integer") from None
    try:
        totp_challenge_seconds = int(os.environ.get("TOTP_CHALLENGE_SECONDS", "120"))
    except ValueError:
        raise SystemExit("TOTP_CHALLENGE_SECONDS must be an integer") from None
    try:
        totp_lockout_attempts = int(os.environ.get("TOTP_LOCKOUT_ATTEMPTS", "3"))
    except ValueError:
        raise SystemExit("TOTP_LOCKOUT_ATTEMPTS must be an integer") from None
    try:
        totp_lockout_minutes = int(os.environ.get("TOTP_LOCKOUT_MINUTES", "15"))
    except ValueError:
        raise SystemExit("TOTP_LOCKOUT_MINUTES must be an integer") from None

    # Per-workspace configuration. Loaded after ALLOWED_WORKSPACES so
    # YAML-defined workspaces can be merged into the allowed set.
    workspace_configs = _load_workspace_configs()

    # Merge YAML workspace paths into allowed_workspaces. Workspaces
    # defined in the config file are implicitly allowed.
    for p in workspace_configs:
        if p not in seen_allowed:
            seen_allowed.add(p)
            allowed_workspaces.append(p)

    # Per-user configuration. If users.yaml exists, it is authoritative;
    # ALLOWED_USER_IDS is ignored. If users.yaml does not exist,
    # ALLOWED_USER_IDS works as before (backward-compatible).
    user_configs = _load_user_configs()
    if user_configs is not None:
        if raw_ids:
            log.warning("ALLOWED_USER_IDS is set but users.yaml exists; using users.yaml")
        # Users in the YAML replace the ALLOWED_USER_IDS set.
        # Any ALLOWED_USER_IDS parse error is irrelevant since users.yaml is authoritative.
        allowed_ids = set(user_configs.keys())
        if not allowed_ids:
            raise SystemExit("users.yaml exists but contains no valid user entries")
    elif allowed_ids_error:
        # No users.yaml and ALLOWED_USER_IDS had a parse error
        raise SystemExit(allowed_ids_error)
    elif not allowed_ids:
        # No users.yaml and no ALLOWED_USER_IDS - can't start
        raise SystemExit(
            "No user authorization configured. "
            "Run 'make config' to generate users.yaml, "
            "or create one manually (see README for the schema)."
        )
    else:
        # ALLOWED_USER_IDS is valid and no users.yaml exists. This works
        # but is the legacy path - nudge toward users.yaml.
        log.info("Using ALLOWED_USER_IDS from env (legacy). Run 'make config' to migrate to users.yaml.")

    # Validate CLAUDE_MODEL against the same VALID_MODELS set used
    # for workspace config. Catches typos at startup instead of
    # letting them propagate to a confusing runtime failure.
    claude_model = os.environ.get("CLAUDE_MODEL", "sonnet")
    if claude_model not in VALID_MODELS:
        raise SystemExit(
            f"CLAUDE_MODEL '{claude_model}' is not valid (must be one of: {', '.join(sorted(VALID_MODELS))})"
        )

    return Config(
        telegram_bot_token=token,
        telegram_webhook_url=telegram_webhook_url,
        telegram_webhook_secret=telegram_webhook_secret,
        allowed_user_ids=allowed_ids,
        claude_model=claude_model,
        claude_timeout_seconds=claude_timeout_seconds,
        claude_max_budget_usd=claude_max_budget_usd,
        claude_max_session_hours=claude_max_session_hours,
        claude_idle_timeout=claude_idle_timeout,
        claude_max_context_window=claude_max_context_window,
        claude_autocompact_pct=claude_autocompact_pct,
        webhook_port=webhook_port,
        webhook_secret=os.environ.get("WEBHOOK_SECRET", ""),
        voice_enabled=os.environ.get("VOICE_ENABLED", "").lower() in ("1", "true", "yes"),
        tts_enabled=os.environ.get("TTS_ENABLED", "").lower() in ("1", "true", "yes"),
        workspace_base=workspace_base,
        allowed_workspaces=allowed_workspaces,
        workspace_configs=workspace_configs,
        claude_user=os.environ.get("CLAUDE_USER") or None,
        pr_review_enabled=pr_review_enabled,
        pr_review_cooldown=pr_review_cooldown,
        github_repo=os.getenv("GITHUB_REPO", ""),
        spec_dir=os.getenv("SPEC_DIR", "specs"),
        issue_triage_enabled=issue_triage_enabled,
        file_retention_days=file_retention_days,
        user_configs=user_configs,
        totp_session_minutes=totp_session_minutes,
        totp_challenge_seconds=totp_challenge_seconds,
        totp_lockout_attempts=totp_lockout_attempts,
        totp_lockout_minutes=totp_lockout_minutes,
    )
