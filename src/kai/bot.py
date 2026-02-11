import asyncio
import base64
import functools
import json
import logging
import shutil
import time
from pathlib import Path

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Message, Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from kai import sessions, webhook
from kai.claude import PersistentClaude
from kai.config import PROJECT_ROOT, Config
from kai.history import log_message
from kai.locks import get_lock, get_stop_event
from kai.transcribe import TranscriptionError, transcribe_voice
from kai.tts import DEFAULT_VOICE, VOICES, TTSError, synthesize_speech

log = logging.getLogger(__name__)

# Minimum interval between Telegram message edits (seconds)
EDIT_INTERVAL = 2.0

# Flag file to track in-flight responses
_RESPONDING_FLAG = PROJECT_ROOT / ".responding_to"



def _set_responding(chat_id: int) -> None:
    _RESPONDING_FLAG.write_text(str(chat_id))


def _clear_responding() -> None:
    _RESPONDING_FLAG.unlink(missing_ok=True)


def _is_authorized(config: Config, user_id: int) -> bool:
    return user_id in config.allowed_user_ids


def _require_auth(func):
    """Decorator to check authorization before running a handler."""

    @functools.wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        config: Config = context.bot_data["config"]
        if not _is_authorized(config, update.effective_user.id):
            return
        return await func(update, context)

    return wrapper


def _truncate_for_telegram(text: str, max_len: int = 4096) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 4] + "\n..."


async def _reply_safe(msg: Message, text: str) -> Message:
    """Reply with markdown, falling back to plain text on parse failure."""
    try:
        return await msg.reply_text(text, parse_mode=ParseMode.MARKDOWN)
    except Exception:
        return await msg.reply_text(text)


async def _edit_message_safe(msg: Message, text: str) -> None:
    truncated = _truncate_for_telegram(text)
    try:
        await msg.edit_text(truncated, parse_mode=ParseMode.MARKDOWN)
    except Exception:
        try:
            await msg.edit_text(truncated)
        except Exception:
            pass


def _chunk_text(text: str, max_len: int = 4096) -> list[str]:
    chunks = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        split_at = text.rfind("\n\n", 0, max_len)
        if split_at == -1:
            split_at = text.rfind("\n", 0, max_len)
        if split_at == -1:
            split_at = max_len
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    return chunks


async def _send_response(update: Update, text: str) -> None:
    for chunk in _chunk_text(text):
        await _reply_safe(update.message, chunk)


def _get_claude(context: ContextTypes.DEFAULT_TYPE) -> PersistentClaude:
    return context.bot_data["claude"]


@_require_auth
async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Kai is ready. Send me a message.")


@_require_auth
async def handle_new(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    claude = _get_claude(context)
    await claude.restart()
    await sessions.clear_session(update.effective_chat.id)
    await update.message.reply_text("Session cleared. Starting fresh.")


_AVAILABLE_MODELS = {
    "opus": "\U0001f9e0 Claude Opus 4.6",
    "sonnet": "\u26a1 Claude Sonnet 4.5",
    "haiku": "\U0001fab6 Claude Haiku 4.5",
}


def _models_keyboard(current: str) -> InlineKeyboardMarkup:
    buttons = []
    for key, name in _AVAILABLE_MODELS.items():
        label = f"{name} \U0001f7e2" if key == current else name
        buttons.append([InlineKeyboardButton(label, callback_data=f"model:{key}")])
    return InlineKeyboardMarkup(buttons)


@_require_auth
async def handle_models(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    claude = _get_claude(context)
    await update.message.reply_text(
        "Choose a model:",
        reply_markup=_models_keyboard(claude.model),
    )


async def _switch_model(context: ContextTypes.DEFAULT_TYPE, chat_id: int, model: str) -> None:
    """Switch model: update Claude, restart process, clear session."""
    claude = _get_claude(context)
    claude.model = model
    await claude.restart()
    await sessions.clear_session(chat_id)


async def handle_model_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    config: Config = context.bot_data["config"]
    if not _is_authorized(config, update.effective_user.id):
        await query.answer("Not authorized.")
        return

    model = query.data.removeprefix("model:")
    if model not in _AVAILABLE_MODELS:
        await query.answer("Invalid model.")
        return

    claude = _get_claude(context)
    if model == claude.model:
        await query.answer()
        await query.edit_message_text("No change.", reply_markup=InlineKeyboardMarkup([]))
        return

    await query.answer()
    await _switch_model(context, update.effective_chat.id, model)
    await query.edit_message_text(
        f"Switched to {_AVAILABLE_MODELS[model]}. Session restarted.",
        reply_markup=InlineKeyboardMarkup([]),
    )


@_require_auth
async def handle_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Usage: /model <opus|sonnet|haiku>")
        return
    model = context.args[0].lower()
    if model not in _AVAILABLE_MODELS:
        await update.message.reply_text("Choose: opus, sonnet, or haiku")
        return
    await _switch_model(context, update.effective_chat.id, model)
    await update.message.reply_text(f"Model set to {_AVAILABLE_MODELS[model]}. Session restarted.")


# ── Voice TTS ────────────────────────────────────────────────────────


def _voices_keyboard(current: str) -> InlineKeyboardMarkup:
    buttons = []
    for key, name in VOICES.items():
        label = f"{name} \U0001f7e2" if key == current else name
        buttons.append([InlineKeyboardButton(label, callback_data=f"voice:{key}")])
    return InlineKeyboardMarkup(buttons)


_VOICE_MODES = {"off", "on", "only"}
_VOICE_MODE_LABELS = {"off": "OFF", "on": "ON (text + voice)", "only": "ONLY (voice only)"}


@_require_auth
async def handle_voice_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Toggle voice mode or set voice: /voice [on|only|off|<name>]."""
    config: Config = context.bot_data["config"]
    if not config.tts_enabled:
        await update.message.reply_text("TTS is not enabled. Set TTS_ENABLED=true in .env")
        return

    chat_id = update.effective_chat.id
    current_mode = await sessions.get_setting(f"voice_mode:{chat_id}") or "off"
    current_voice = await sessions.get_setting(f"voice_name:{chat_id}") or DEFAULT_VOICE

    if context.args:
        arg = context.args[0].lower()
        if arg in _VOICE_MODES:
            # /voice on|only|off — set mode directly
            await sessions.set_setting(f"voice_mode:{chat_id}", arg)
            await update.message.reply_text(
                f"Voice mode: {_VOICE_MODE_LABELS[arg]} (voice: {VOICES[current_voice]})"
            )
        elif arg in VOICES:
            # /voice <name> — set voice (enable in current mode, or default to "only")
            await sessions.set_setting(f"voice_name:{chat_id}", arg)
            if current_mode == "off":
                await sessions.set_setting(f"voice_mode:{chat_id}", "only")
                current_mode = "only"
            await update.message.reply_text(
                f"Voice set to {VOICES[arg]}. Voice mode: {_VOICE_MODE_LABELS[current_mode]}"
            )
        else:
            names = ", ".join(VOICES.keys())
            await update.message.reply_text(
                f"Unknown voice or mode. Usage:\n"
                f"/voice on — text + voice\n"
                f"/voice only — voice only\n"
                f"/voice off — text only\n"
                f"/voice <name> — set voice\n\n"
                f"Voices: {names}"
            )
    else:
        # /voice — toggle: off → only → off
        new_mode = "off" if current_mode != "off" else "only"
        await sessions.set_setting(f"voice_mode:{chat_id}", new_mode)
        await update.message.reply_text(
            f"Voice mode: {_VOICE_MODE_LABELS[new_mode]} (voice: {VOICES[current_voice]})"
        )


@_require_auth
async def handle_voices(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show inline keyboard of available voices."""
    config: Config = context.bot_data["config"]
    if not config.tts_enabled:
        await update.message.reply_text("TTS is not enabled. Set TTS_ENABLED=true in .env")
        return

    chat_id = update.effective_chat.id
    current_voice = await sessions.get_setting(f"voice_name:{chat_id}") or DEFAULT_VOICE
    await update.message.reply_text(
        "Choose a voice:",
        reply_markup=_voices_keyboard(current_voice),
    )


async def handle_voice_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle inline keyboard voice selection."""
    query = update.callback_query
    config: Config = context.bot_data["config"]
    if not _is_authorized(config, update.effective_user.id):
        await query.answer("Not authorized.")
        return

    voice = query.data.removeprefix("voice:")
    if voice not in VOICES:
        await query.answer("Invalid voice.")
        return

    chat_id = update.effective_chat.id
    current_voice = await sessions.get_setting(f"voice_name:{chat_id}") or DEFAULT_VOICE

    if voice == current_voice:
        await query.answer()
        await query.edit_message_text("No change.", reply_markup=InlineKeyboardMarkup([]))
        return

    current_mode = await sessions.get_setting(f"voice_mode:{chat_id}") or "off"
    await sessions.set_setting(f"voice_name:{chat_id}", voice)
    if current_mode == "off":
        await sessions.set_setting(f"voice_mode:{chat_id}", "only")
        current_mode = "only"
    await query.answer()
    await query.edit_message_text(
        f"Voice set to {VOICES[voice]}. Voice mode: {_VOICE_MODE_LABELS[current_mode]}",
        reply_markup=InlineKeyboardMarkup([]),
    )


@_require_auth
async def handle_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    claude = _get_claude(context)
    stats = await sessions.get_stats(update.effective_chat.id)
    alive = claude.is_alive
    if not stats:
        await update.message.reply_text(f"No active session.\nProcess alive: {alive}")
        return
    await update.message.reply_text(
        f"Session: {stats['session_id'][:8]}...\n"
        f"Model: {stats['model']}\n"
        f"Started: {stats['created_at']}\n"
        f"Last used: {stats['last_used_at']}\n"
        f"Total cost: ${stats['total_cost_usd']:.4f}\n"
        f"Process alive: {alive}"
    )


@_require_auth
async def handle_jobs(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    jobs = await sessions.get_jobs(update.effective_chat.id)
    if not jobs:
        await update.message.reply_text("No active scheduled jobs.")
        return
    lines = []
    for j in jobs:
        sched = j["schedule_type"]
        if sched == "once":
            data = json.loads(j["schedule_data"])
            detail = f"once at {data.get('run_at', '?')}"
        elif sched == "interval":
            data = json.loads(j["schedule_data"])
            secs = data.get("seconds", 0)
            if secs >= 3600:
                detail = f"every {secs // 3600}h"
            elif secs >= 60:
                detail = f"every {secs // 60}m"
            else:
                detail = f"every {secs}s"
        elif sched == "daily":
            data = json.loads(j["schedule_data"])
            detail = f"daily at {data.get('time', '?')} UTC"
        else:
            detail = sched
        type_tag = "\U0001f514" if j["job_type"] == "reminder" else "\U0001f916"
        lines.append(f"{type_tag} #{j['id']} {j['name']} ({detail})")
    await update.message.reply_text("Active jobs:\n" + "\n".join(lines))


@_require_auth
async def handle_canceljob(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Usage: /canceljob <id>")
        return
    try:
        job_id = int(context.args[0])
    except ValueError:
        await update.message.reply_text("Job ID must be a number.")
        return
    deleted = await sessions.delete_job(job_id)
    if not deleted:
        await update.message.reply_text(f"Job #{job_id} not found.")
        return
    # Remove from scheduler
    jq = context.application.job_queue
    current = jq.get_jobs_by_name(f"cron_{job_id}")
    for j in current:
        j.schedule_removal()
    await update.message.reply_text(f"Job #{job_id} cancelled.")


@_require_auth
async def handle_stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    claude = _get_claude(context)
    stop_event = get_stop_event(chat_id)
    stop_event.set()
    claude.force_kill()
    await update.message.reply_text("Stopping...")



def _resolve_workspace_path(target: str, base: Path | None) -> Path | None:
    """Resolve a workspace name to an absolute path under base.

    Only relative names are allowed. Returns None if no base is set.
    """
    if not base:
        return None
    resolved = (base / target).resolve()
    # Prevent traversal outside the base directory
    if not str(resolved).startswith(str(base) + "/") and resolved != base:
        return None
    return resolved


def _short_workspace_name(path: str, base: Path | None) -> str:
    """Shorten a workspace path for display."""
    base_str = str(base) if base else None
    if base_str and path.startswith(base_str.rstrip("/") + "/"):
        return path[len(base_str.rstrip("/")) + 1 :]
    return Path(path).name


async def _do_switch_workspace(context: ContextTypes.DEFAULT_TYPE, chat_id: int, path: Path) -> None:
    """Switch workspace: update Claude, clear session, persist setting."""
    claude = _get_claude(context)
    config: Config = context.bot_data["config"]
    home = config.claude_workspace

    await claude.change_workspace(path)
    await sessions.clear_session(chat_id)

    if path == home:
        await sessions.delete_setting("workspace")
    else:
        await sessions.set_setting("workspace", str(path))
        await sessions.upsert_workspace_history(str(path))


async def _switch_workspace(update: Update, context: ContextTypes.DEFAULT_TYPE, path: Path) -> None:
    """Switch to a workspace path and confirm via reply."""
    claude = _get_claude(context)
    config: Config = context.bot_data["config"]
    home = config.claude_workspace

    if path == claude.workspace:
        await update.message.reply_text("Already in that workspace.")
        return

    await _do_switch_workspace(context, update.effective_chat.id, path)

    if path == home:
        await update.message.reply_text("Switched to home workspace. Session cleared.")
    else:
        notes = []
        if (path / ".git").is_dir():
            notes.append("Git repo")
        if (path / ".claude" / "CLAUDE.md").exists():
            notes.append("Has CLAUDE.md")
        suffix = f" ({', '.join(notes)})" if notes else ""
        await update.message.reply_text(f"Workspace: {path}{suffix}\nSession cleared.")


async def _workspaces_keyboard(
    history: list[dict],
    current_path: str,
    home_path: str,
    base: Path | None,
) -> InlineKeyboardMarkup:
    """Build inline keyboard for workspace switching."""
    buttons = []
    # Home button
    home_label = "\U0001f3e0 Home"
    if current_path == home_path:
        home_label += " \U0001f7e2"
    buttons.append([InlineKeyboardButton(home_label, callback_data="ws:home")])
    # History entries
    for i, entry in enumerate(history):
        p = entry["path"]
        if p == home_path:
            continue  # already shown as Home
        short = _short_workspace_name(p, base)
        label = short
        if p == current_path:
            label += " \U0001f7e2"
        buttons.append([InlineKeyboardButton(label, callback_data=f"ws:{i}")])
    return InlineKeyboardMarkup(buttons)


@_require_auth
async def handle_workspaces(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    history = await sessions.get_workspace_history()
    claude = _get_claude(context)
    config: Config = context.bot_data["config"]
    current = str(claude.workspace)
    home = str(config.claude_workspace)

    if not history and current == home:
        await update.message.reply_text("No workspace history yet.\nUse /workspace new <name> to create one.")
        return

    keyboard = await _workspaces_keyboard(history, current, home, config.workspace_base)
    await update.message.reply_text("Workspaces:", reply_markup=keyboard)


async def handle_workspace_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    config: Config = context.bot_data["config"]
    if not _is_authorized(config, update.effective_user.id):
        await query.answer("Not authorized.")
        return

    data = query.data.removeprefix("ws:")
    claude = _get_claude(context)
    home = config.claude_workspace
    base = config.workspace_base

    # Resolve target path
    if data == "home":
        path = home
        label = "Home"
    else:
        try:
            idx = int(data)
        except ValueError:
            await query.answer("Invalid selection.")
            await query.edit_message_text("No change.", reply_markup=InlineKeyboardMarkup([]))
            return
        history = await sessions.get_workspace_history()
        if idx < 0 or idx >= len(history):
            await query.answer("Workspace no longer in history.")
            await query.edit_message_text("No change.", reply_markup=InlineKeyboardMarkup([]))
            return
        path = Path(history[idx]["path"])
        if not path.is_dir():
            await sessions.delete_workspace_history(str(path))
            await query.answer("That workspace no longer exists.")
            history = await sessions.get_workspace_history()
            keyboard = await _workspaces_keyboard(history, str(claude.workspace), str(home), base)
            await query.edit_message_reply_markup(reply_markup=keyboard)
            return
        label = _short_workspace_name(str(path), base)

    # Already there — dismiss
    if path == claude.workspace:
        await query.answer()
        await query.edit_message_text("No change.", reply_markup=InlineKeyboardMarkup([]))
        return

    # Switch and confirm
    await query.answer()
    await _do_switch_workspace(context, update.effective_chat.id, path)
    await query.edit_message_text(
        f"Switched to {label}. Session cleared.",
        reply_markup=InlineKeyboardMarkup([]),
    )


_NO_BASE_MSG = "WORKSPACE_BASE is not set. Add it to .env and restart."


@_require_auth
async def handle_workspace(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    claude = _get_claude(context)
    config: Config = context.bot_data["config"]
    home = config.claude_workspace
    base = config.workspace_base

    # No args: show current workspace
    if not context.args:
        current = claude.workspace
        short = _short_workspace_name(str(current), base)
        if current == home:
            short = "Home"
        await update.message.reply_text(f"Workspace: {short}\n{current}")
        return

    target = " ".join(context.args)

    # "home" keyword: always allowed
    if target.lower() == "home":
        await _switch_workspace(update, context, home)
        return

    # Reject absolute paths and ~ expansion
    if target.startswith("/") or target.startswith("~"):
        await update.message.reply_text("Absolute paths are not allowed. Use a workspace name.")
        return

    # "new" keyword: create a new workspace
    if target.lower().startswith("new"):
        parts = target.split(None, 1)
        if len(parts) < 2:
            await update.message.reply_text("Usage: /workspace new <name>")
            return
        if not base:
            await update.message.reply_text(_NO_BASE_MSG)
            return
        name = parts[1]
        resolved = _resolve_workspace_path(name, base)
        if resolved is None:
            await update.message.reply_text("Invalid workspace name.")
            return
        if resolved.exists():
            await update.message.reply_text(f"Already exists:\n{resolved}")
            return
        resolved.mkdir(parents=True)
        proc = await asyncio.create_subprocess_exec(
            "git",
            "init",
            cwd=str(resolved),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
        await _switch_workspace(update, context, resolved)
        return

    # Resolve via base directory
    if not base:
        await update.message.reply_text(_NO_BASE_MSG)
        return

    resolved = _resolve_workspace_path(target, base)
    if resolved is None:
        await update.message.reply_text("Invalid workspace name.")
        return

    if not resolved.exists():
        await update.message.reply_text(f"Path does not exist:\n{resolved}")
        return
    if not resolved.is_dir():
        await update.message.reply_text(f"Not a directory:\n{resolved}")
        return

    await _switch_workspace(update, context, resolved)


@_require_auth
async def handle_webhooks(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    config: Config = context.bot_data["config"]
    running = webhook.is_running()
    status = "running" if running else "not running"
    has_secret = bool(config.webhook_secret)
    lines = [
        f"Webhook server: {status}",
        f"Port: {config.webhook_port}",
        "",
        "Endpoints:",
        "  GET  /health          (health check)",
    ]
    if has_secret:
        lines += [
            "  POST /webhook/github  (GitHub events)",
            "  POST /webhook         (generic)",
            "  POST /api/schedule    (scheduling API)",
        ]
    else:
        lines += [
            "",
            "WEBHOOK_SECRET not set — only /health is active.",
            "Set WEBHOOK_SECRET in .env to enable webhooks and scheduling.",
        ]
    if running and has_secret:
        lines += [
            "",
            "GitHub setup:",
            "1. Set Payload URL to https://your-host/webhook/github",
            "2. Content type: application/json",
            "3. Set the secret to match WEBHOOK_SECRET",
            "4. Choose events: Pushes, Pull requests, Issues, Comments",
        ]
    await update.message.reply_text("\n".join(lines))


@_require_auth
async def handle_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "/stop - Interrupt current response\n"
        "/new - Start a fresh session\n"
        "/workspace - Show current workspace\n"
        "/workspace <name> - Switch by name\n"
        "/workspace new <name> - Create + git init + switch\n"
        "/workspace home - Return to default\n"
        "/workspaces - Switch workspace (inline buttons)\n"
        "/models - Choose a model\n"
        "/model <name> - Switch model directly\n"
        "/voice - Toggle voice on/off\n"
        "/voice only - Voice only (no text)\n"
        "/voice on - Text + voice\n"
        "/voice <name> - Set voice\n"
        "/voices - Choose a voice (inline buttons)\n"
        "/stats - Show session info and cost\n"
        "/jobs - List scheduled jobs\n"
        "/canceljob <id> - Cancel a job\n"
        "/webhooks - Show webhook server status\n"
        "/help - This message"
    )


@_require_auth
async def handle_unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        f"Unknown command: {update.message.text.split()[0]}\nTry /help for available commands."
    )


@_require_auth
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.photo:
        return

    chat_id = update.effective_chat.id
    claude = _get_claude(context)
    model = claude.model

    # Download the largest available resolution
    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)
    data = await file.download_as_bytearray()
    b64 = base64.b64encode(bytes(data)).decode()

    caption = update.message.caption or "What's in this image?"
    log_message(direction="user", chat_id=chat_id, text=caption, media={"type": "photo"})
    content = [
        {"type": "text", "text": caption},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
    ]

    async with get_lock(chat_id):
        _set_responding(chat_id)
        try:
            await _handle_response(update, context, chat_id, content, claude, model)
        finally:
            _clear_responding()


# File extensions treated as readable text
_TEXT_EXTENSIONS = {
    ".txt",
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".json",
    ".csv",
    ".tsv",
    ".md",
    ".rst",
    ".xml",
    ".html",
    ".htm",
    ".css",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
    ".sh",
    ".bash",
    ".zsh",
    ".fish",
    ".sql",
    ".log",
    ".env",
    ".gitignore",
    ".dockerfile",
    ".makefile",
    ".rb",
    ".go",
    ".rs",
    ".java",
    ".kt",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".swift",
    ".r",
    ".lua",
    ".pl",
    ".php",
    ".ex",
    ".exs",
    ".erl",
}

# Image extensions that can be sent as documents
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}

# Media type mapping for images
_IMAGE_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


@_require_auth
async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.document:
        return

    doc = update.message.document
    file_name = doc.file_name or "unknown"
    suffix = Path(file_name).suffix.lower()
    caption = update.message.caption or ""

    chat_id = update.effective_chat.id
    claude = _get_claude(context)
    model = claude.model

    if suffix in _IMAGE_EXTENSIONS:
        # Handle images sent as documents (uncompressed)
        file = await context.bot.get_file(doc.file_id)
        data = await file.download_as_bytearray()
        b64 = base64.b64encode(bytes(data)).decode()
        media_type = _IMAGE_MEDIA_TYPES[suffix]
        log_message(
            direction="user",
            chat_id=chat_id,
            text=caption or file_name,
            media={"type": "document", "filename": file_name},
        )
        content = [
            {"type": "text", "text": caption or f"What's in this image ({file_name})?"},
            {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
        ]
    elif suffix in _TEXT_EXTENSIONS or (doc.mime_type and doc.mime_type.startswith("text/")):
        file = await context.bot.get_file(doc.file_id)
        data = await file.download_as_bytearray()
        try:
            text_content = bytes(data).decode("utf-8")
        except UnicodeDecodeError:
            await update.message.reply_text(f"Couldn't decode {file_name} as text.")
            return
        header = f"File: {file_name}\n```\n{text_content}\n```"
        log_message(
            direction="user",
            chat_id=chat_id,
            text=caption or f"[file: {file_name}]",
            media={"type": "document", "filename": file_name},
        )
        if caption:
            content = f"{caption}\n\n{header}"
        else:
            content = header
    else:
        await update.message.reply_text(
            f"I can't process {suffix or 'this'} files yet. I support text files and images."
        )
        return

    async with get_lock(chat_id):
        _set_responding(chat_id)
        try:
            await _handle_response(update, context, chat_id, content, claude, model)
        finally:
            _clear_responding()


@_require_auth
async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.voice:
        return

    chat_id = update.effective_chat.id
    claude = _get_claude(context)
    config: Config = context.bot_data["config"]

    if not config.voice_enabled:
        await update.message.reply_text("Voice messages are not enabled.")
        return

    missing = []
    if not shutil.which("ffmpeg"):
        missing.append("ffmpeg")
    if not shutil.which("whisper-cli"):
        missing.append("whisper-cpp")
    if not config.whisper_model_path.exists():
        missing.append("whisper model")
    if missing:
        await update.message.reply_text(
            f"Voice is enabled but dependencies are missing: {', '.join(missing)}. "
            "See the wiki for setup instructions: Voice-Message-Setup"
        )
        return

    voice = update.message.voice
    file = await context.bot.get_file(voice.file_id)
    audio_data = bytes(await file.download_as_bytearray())

    log_message(
        direction="user", chat_id=chat_id,
        text=f"[voice message, {voice.duration}s]",
        media={"type": "voice", "duration": voice.duration},
    )

    try:
        transcript = await transcribe_voice(audio_data, config.whisper_model_path)
    except TranscriptionError as e:
        await update.message.reply_text(f"Transcription failed: {e}")
        return

    if not transcript:
        await update.message.reply_text("Couldn't make out any speech in that voice message.")
        return

    # Echo so the user sees what Kai heard
    await _reply_safe(update.message, f"_Heard:_ {transcript}")

    prompt = f"[Voice message transcription]: {transcript}"
    model = claude.model

    async with get_lock(chat_id):
        _set_responding(chat_id)
        try:
            await _handle_response(update, context, chat_id, prompt, claude, model)
        finally:
            _clear_responding()


@_require_auth
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return

    chat_id = update.effective_chat.id
    prompt = update.message.text
    log_message(direction="user", chat_id=chat_id, text=prompt)
    claude = _get_claude(context)
    model = claude.model

    async with get_lock(chat_id):
        _set_responding(chat_id)
        try:
            await _handle_response(update, context, chat_id, prompt, claude, model)
        finally:
            _clear_responding()


async def _handle_response(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    prompt: str | list,
    claude: PersistentClaude,
    model: str,
) -> None:
    # Check voice mode before starting
    config: Config = context.bot_data["config"]
    voice_mode = "off"
    if config.tts_enabled:
        voice_mode = await sessions.get_setting(f"voice_mode:{chat_id}") or "off"
    voice_only = voice_mode == "only"

    # Keep activity indicator visible until the response completes
    chat_action = ChatAction.RECORD_VOICE if voice_only else ChatAction.TYPING
    typing_active = True

    async def _keep_typing():
        while typing_active:
            try:
                await context.bot.send_chat_action(chat_id=chat_id, action=chat_action)
            except Exception:
                pass
            await asyncio.sleep(4)

    typing_task = asyncio.create_task(_keep_typing())

    live_msg = None
    last_edit_time = 0.0
    last_edit_text = ""
    final_response = None

    stop_event = get_stop_event(chat_id)
    stop_event.clear()

    async for event in claude.send(prompt):
        if stop_event.is_set():
            stop_event.clear()
            if live_msg:
                await _edit_message_safe(live_msg, last_edit_text + "\n\n_(stopped)_")
            final_response = None
            break

        if event.done:
            final_response = event.response
            break

        if voice_only:
            continue

        now = time.monotonic()
        if not event.text_so_far:
            continue

        if live_msg is None:
            truncated = _truncate_for_telegram(event.text_so_far)
            live_msg = await _reply_safe(update.message, truncated)
            last_edit_time = now
            last_edit_text = event.text_so_far
        elif now - last_edit_time >= EDIT_INTERVAL and event.text_so_far != last_edit_text:
            await _edit_message_safe(live_msg, event.text_so_far)
            last_edit_time = now
            last_edit_text = event.text_so_far

    typing_active = False
    typing_task.cancel()
    try:
        await typing_task
    except asyncio.CancelledError:
        pass

    if final_response is None:
        await update.message.reply_text("Error: No response from Claude")
        return

    if not final_response.success:
        error_text = f"Error: {final_response.error}"
        if live_msg:
            await _edit_message_safe(live_msg, error_text)
        else:
            await update.message.reply_text(error_text)
        return

    if final_response.session_id:
        await sessions.save_session(chat_id, final_response.session_id, model, final_response.cost_usd)

    final_text = final_response.text
    log_message(direction="assistant", chat_id=chat_id, text=final_text)

    # Voice-only mode: synthesize and send voice, fall back to text on failure
    if voice_only and final_text:
        voice_name = await sessions.get_setting(f"voice_name:{chat_id}") or DEFAULT_VOICE
        try:
            audio = await synthesize_speech(final_text, config.piper_model_dir, voice_name)
            await context.bot.send_voice(chat_id=chat_id, voice=audio)
            return
        except TTSError as e:
            log.warning("TTS failed, falling back to text: %s", e)

    # Send text response (normal mode, or voice-only fallback)
    if live_msg:
        if len(final_text) <= 4096:
            if final_text != last_edit_text:
                await _edit_message_safe(live_msg, final_text)
        else:
            chunks = _chunk_text(final_text)
            await _edit_message_safe(live_msg, chunks[0])
            for chunk in chunks[1:]:
                await _reply_safe(update.message, chunk)
    else:
        await _send_response(update, final_text)

    # Text+voice mode: send voice note after text
    if voice_mode == "on" and final_text:
        voice_name = await sessions.get_setting(f"voice_name:{chat_id}") or DEFAULT_VOICE
        try:
            audio = await synthesize_speech(final_text, config.piper_model_dir, voice_name)
            await context.bot.send_voice(chat_id=chat_id, voice=audio)
        except TTSError as e:
            log.warning("TTS failed: %s", e)


def create_bot(config: Config) -> Application:
    app = Application.builder().token(config.telegram_bot_token).concurrent_updates(True).build()
    app.bot_data["config"] = config
    app.bot_data["claude"] = PersistentClaude(
        model=config.claude_model,
        workspace=config.claude_workspace,
        home_workspace=config.claude_workspace,
        webhook_port=config.webhook_port,
        webhook_secret=config.webhook_secret,
        max_budget_usd=config.claude_max_budget_usd,
        timeout_seconds=config.claude_timeout_seconds,
    )

    app.add_handler(CommandHandler("start", handle_start))
    app.add_handler(CommandHandler("new", handle_new))
    app.add_handler(CommandHandler("models", handle_models))
    app.add_handler(CommandHandler("model", handle_model))
    app.add_handler(CommandHandler("stats", handle_stats))
    app.add_handler(CommandHandler("help", handle_help))
    app.add_handler(CommandHandler("jobs", handle_jobs))
    app.add_handler(CommandHandler("canceljob", handle_canceljob))
    app.add_handler(CommandHandler("workspace", handle_workspace))
    app.add_handler(CommandHandler("workspaces", handle_workspaces))
    app.add_handler(CommandHandler("voice", handle_voice_command))
    app.add_handler(CommandHandler("voices", handle_voices))
    app.add_handler(CommandHandler("webhooks", handle_webhooks))
    app.add_handler(CommandHandler("stop", handle_stop))
    app.add_handler(CallbackQueryHandler(handle_model_callback, pattern=r"^model:"))
    app.add_handler(CallbackQueryHandler(handle_voice_callback, pattern=r"^voice:"))
    app.add_handler(CallbackQueryHandler(handle_workspace_callback, pattern=r"^ws:"))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.COMMAND, handle_unknown_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    return app
