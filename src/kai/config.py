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
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

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
    claude_workspace: Path = field(default_factory=lambda: PROJECT_ROOT / "workspace")

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

    # TOTP two-factor authentication timing (only relevant when TOTP is enabled)
    totp_session_minutes: int = 30
    totp_challenge_seconds: int = 120
    totp_lockout_attempts: int = 3
    totp_lockout_minutes: int = 15


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
                os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
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

    # Validate required: allowed user IDs (comma-separated numeric Telegram user IDs)
    raw_ids = os.environ.get("ALLOWED_USER_IDS", "")
    if not raw_ids:
        raise SystemExit("ALLOWED_USER_IDS is required in .env")
    try:
        allowed_ids = {int(uid.strip()) for uid in raw_ids.split(",") if uid.strip()}
    except ValueError as e:
        raise SystemExit(
            "ALLOWED_USER_IDS must be numeric Telegram user IDs (not usernames). "
            "Message @userinfobot on Telegram to find yours."
        ) from e

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
        webhook_port = int(os.environ.get("WEBHOOK_PORT", "8080"))
    except ValueError:
        raise SystemExit("WEBHOOK_PORT must be an integer") from None

    # PR review agent config
    pr_review_enabled = os.environ.get("PR_REVIEW_ENABLED", "").lower() in ("1", "true", "yes")
    try:
        pr_review_cooldown = int(os.environ.get("PR_REVIEW_COOLDOWN", "300"))
    except ValueError:
        raise SystemExit("PR_REVIEW_COOLDOWN must be an integer") from None

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

    return Config(
        telegram_bot_token=token,
        telegram_webhook_url=telegram_webhook_url,
        telegram_webhook_secret=telegram_webhook_secret,
        allowed_user_ids=allowed_ids,
        claude_model=os.environ.get("CLAUDE_MODEL", "sonnet"),
        claude_timeout_seconds=claude_timeout_seconds,
        claude_max_budget_usd=claude_max_budget_usd,
        claude_max_session_hours=claude_max_session_hours,
        webhook_port=webhook_port,
        webhook_secret=os.environ.get("WEBHOOK_SECRET", ""),
        voice_enabled=os.environ.get("VOICE_ENABLED", "").lower() in ("1", "true", "yes"),
        tts_enabled=os.environ.get("TTS_ENABLED", "").lower() in ("1", "true", "yes"),
        workspace_base=workspace_base,
        allowed_workspaces=allowed_workspaces,
        claude_user=os.environ.get("CLAUDE_USER") or None,
        pr_review_enabled=pr_review_enabled,
        pr_review_cooldown=pr_review_cooldown,
        totp_session_minutes=totp_session_minutes,
        totp_challenge_seconds=totp_challenge_seconds,
        totp_lockout_attempts=totp_lockout_attempts,
        totp_lockout_minutes=totp_lockout_minutes,
    )
