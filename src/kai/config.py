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

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Derive project root from file location: src/kai/config.py -> src/kai -> src -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass(frozen=True)
class Config:
    """
    Immutable application configuration populated from environment variables.

    All fields map to environment variables defined in .env (see .env.example for
    reference). Required fields raise SystemExit with descriptive messages if missing.
    Optional fields have sensible defaults for single-user local deployment.

    Attributes:
        telegram_bot_token: Bot token from @BotFather (required)
        allowed_user_ids: Set of Telegram user IDs permitted to interact with the bot (required)
        claude_model: Model name passed to the inner Claude Code process (haiku/sonnet/opus)
        claude_timeout_seconds: Seconds before a Claude response is considered timed out
        claude_max_budget_usd: Per-session spending cap in USD
        claude_workspace: Working directory for the inner Claude Code process
        session_db_path: Path to the SQLite database for sessions, jobs, and settings
        webhook_port: Port for the local aiohttp server (webhooks + scheduling API)
        webhook_secret: HMAC secret for verifying incoming webhook payloads
        voice_enabled: Whether to transcribe Telegram voice notes via whisper-cpp
        whisper_model_path: Path to the whisper-cpp GGML model file
        tts_enabled: Whether to enable Piper text-to-speech for voice responses
        piper_model_dir: Directory containing Piper voice model files
        workspace_base: Base directory for workspace name resolution (/workspace <name>)
    """

    # Required fields — no defaults, must be provided
    telegram_bot_token: str
    allowed_user_ids: set[int]

    # Claude Code process configuration
    claude_model: str = "sonnet"
    claude_timeout_seconds: int = 120
    claude_max_budget_usd: float = 10.0
    claude_workspace: Path = field(default_factory=lambda: PROJECT_ROOT / "workspace")

    # Database
    session_db_path: Path = field(default_factory=lambda: PROJECT_ROOT / "kai.db")

    # Webhook server
    webhook_port: int = 8080
    webhook_secret: str = ""

    # Voice input (speech-to-text via whisper-cpp)
    voice_enabled: bool = False
    whisper_model_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "models" / "ggml-base.en.bin"
    )

    # Voice output (text-to-speech via Piper)
    tts_enabled: bool = False
    piper_model_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "models" / "piper")

    # Workspace switching
    workspace_base: Path | None = None


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
    load_dotenv(PROJECT_ROOT / ".env")

    # Validate required: Telegram bot token
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("TELEGRAM_BOT_TOKEN is required in .env")

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

    return Config(
        telegram_bot_token=token,
        allowed_user_ids=allowed_ids,
        claude_model=os.environ.get("CLAUDE_MODEL", "sonnet"),
        claude_timeout_seconds=int(os.environ.get("CLAUDE_TIMEOUT_SECONDS", "120")),
        claude_max_budget_usd=float(os.environ.get("CLAUDE_MAX_BUDGET_USD", "10.0")),
        webhook_port=int(os.environ.get("WEBHOOK_PORT", "8080")),
        webhook_secret=os.environ.get("WEBHOOK_SECRET", ""),
        voice_enabled=os.environ.get("VOICE_ENABLED", "").lower() in ("1", "true", "yes"),
        tts_enabled=os.environ.get("TTS_ENABLED", "").lower() in ("1", "true", "yes"),
        workspace_base=workspace_base,
    )
