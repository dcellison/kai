import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# src/kai/config.py -> src/kai -> src -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass(frozen=True)
class Config:
    telegram_bot_token: str
    allowed_user_ids: set[int]
    claude_model: str = "sonnet"
    claude_timeout_seconds: int = 120
    claude_max_budget_usd: float = 1.0
    claude_workspace: Path = field(default_factory=lambda: PROJECT_ROOT / "workspace")
    session_db_path: Path = field(default_factory=lambda: PROJECT_ROOT / "sessions.db")
    webhook_port: int = 8080
    webhook_secret: str = ""


def load_config() -> Config:
    load_dotenv(PROJECT_ROOT / ".env")

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("TELEGRAM_BOT_TOKEN is required in .env")

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

    return Config(
        telegram_bot_token=token,
        allowed_user_ids=allowed_ids,
        claude_model=os.environ.get("CLAUDE_MODEL", "sonnet"),
        claude_timeout_seconds=int(os.environ.get("CLAUDE_TIMEOUT_SECONDS", "120")),
        claude_max_budget_usd=float(os.environ.get("CLAUDE_MAX_BUDGET_USD", "1.0")),
        webhook_port=int(os.environ.get("WEBHOOK_PORT", "8080")),
        webhook_secret=os.environ.get("WEBHOOK_SECRET", ""),
    )
