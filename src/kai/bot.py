import asyncio
import base64
import functools
import json
import logging
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
from kai.chat_log import log_message
from kai.claude import PersistentClaude
from kai.config import PROJECT_ROOT, Config
from kai.locks import get_lock, get_stop_event

log = logging.getLogger(__name__)

# Minimum interval between Telegram message edits (seconds)
EDIT_INTERVAL = 2.0

# Flag file to track in-flight responses
_RESPONDING_FLAG = PROJECT_ROOT / ".responding_to"

def _line_count(path: Path) -> int:
    """Count lines in a file, returning 0 if it doesn't exist."""
    if not path.exists():
        return 0
    return len(path.read_text().splitlines())


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


async def handle_model_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    config: Config = context.bot_data["config"]
    if not _is_authorized(config, update.effective_user.id):
        await query.answer("Not authorized.")
        return

    model = query.data.removeprefix("model:")
    claude = _get_claude(context)

    name = _AVAILABLE_MODELS.get(model, model)

    if model == claude.model:
        await query.answer()
        await query.edit_message_text("No change.", reply_markup=InlineKeyboardMarkup([]))
        return

    await query.answer()
    await query.edit_message_text(
        f"Switched to {name}. Session restarted.",
        reply_markup=InlineKeyboardMarkup([]),
    )

    claude.model = model
    await claude.restart()
    await sessions.clear_session(update.effective_chat.id)


@_require_auth
async def handle_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Usage: /model <opus|sonnet|haiku>")
        return
    model = context.args[0].lower()
    if model not in _AVAILABLE_MODELS:
        await update.message.reply_text("Choose: opus, sonnet, or haiku")
        return
    claude = _get_claude(context)
    claude.model = model
    await claude.restart()
    await sessions.clear_session(update.effective_chat.id)
    await update.message.reply_text(f"Model set to {_AVAILABLE_MODELS[model]}. Session restarted.")


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


@_require_auth
async def handle_memory(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    claude = _get_claude(context)
    config: Config = context.bot_data["config"]
    home = config.claude_workspace

    # Build a list of memory file locations and their status
    lines = ["Memory files (all injected at session start):\n"]

    # 1. Claude Code auto-memory (managed by Claude Code itself)
    # Path is based on workspace path with slashes replaced by hyphens
    ws_path = str(claude.workspace)
    auto_key = ws_path.replace("/", "-")
    auto_path = Path.home() / ".claude" / "projects" / auto_key / "memory" / "MEMORY.md"
    exists = auto_path.exists()
    status = f"{_line_count(auto_path)} lines" if exists else "not created yet"
    lines.append(f"Auto-memory ({status}):\n{auto_path}")

    # 2. Home workspace memory
    home_memory = home / ".claude" / "MEMORY.md"
    exists = home_memory.exists()
    status = f"{_line_count(home_memory)} lines" if exists else "not created yet"
    lines.append(f"\nHome memory ({status}):\n{home_memory}")

    # 3. Current workspace memory (if different from home)
    if claude.workspace != home:
        ws_memory = claude.workspace / ".claude" / "MEMORY.md"
        exists = ws_memory.exists()
        status = f"{_line_count(ws_memory)} lines" if exists else "not created yet"
        lines.append(f"\nWorkspace memory ({status}):\n{ws_memory}")

    lines.append("\nAsk me about my memory in natural language to see details.")
    await update.message.reply_text("\n".join(lines))


async def _resolve_workspace_path(target: str, base: str | None) -> Path | None:
    """Resolve a workspace target to an absolute path.

    Returns None if the target is a bare name and no base is set.
    """
    if target.startswith("/") or target.startswith("~"):
        return Path(target).expanduser().resolve()
    if base:
        return (Path(base) / target).resolve()
    return None


def _short_workspace_name(path: str, base: str | None) -> str:
    """Shorten a workspace path for display."""
    if base and path.startswith(base.rstrip("/") + "/"):
        return path[len(base.rstrip("/")) + 1 :]
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
    base: str | None,
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
    base = await sessions.get_setting("workspace_base")

    if not history and current == home:
        await update.message.reply_text("No workspace history yet.\nUse /workspace new <name> to create one.")
        return

    keyboard = await _workspaces_keyboard(history, current, home, base)
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

    # Resolve target path
    if data == "home":
        path = home
        label = "Home"
    else:
        try:
            idx = int(data)
        except ValueError:
            await query.answer("Invalid selection.")
            return
        history = await sessions.get_workspace_history()
        if idx < 0 or idx >= len(history):
            await query.answer("Workspace no longer in history.")
            return
        path = Path(history[idx]["path"])
        if not path.is_dir():
            await sessions.delete_workspace_history(str(path))
            await query.answer("That workspace no longer exists.")
            history = await sessions.get_workspace_history()
            base = await sessions.get_setting("workspace_base")
            keyboard = await _workspaces_keyboard(history, str(claude.workspace), str(home), base)
            await query.edit_message_reply_markup(reply_markup=keyboard)
            return
        base = await sessions.get_setting("workspace_base")
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


@_require_auth
async def handle_workspace(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    claude = _get_claude(context)
    config: Config = context.bot_data["config"]
    home = config.claude_workspace

    # No args: show current workspace
    if not context.args:
        current = claude.workspace
        base = await sessions.get_setting("workspace_base")
        short = _short_workspace_name(str(current), base)
        if current == home:
            short = "Home"
        await update.message.reply_text(f"Workspace: {short}\n{current}")
        return

    target = " ".join(context.args)
    base = await sessions.get_setting("workspace_base")

    # "home" keyword: return to default
    if target.lower() == "home":
        await _switch_workspace(update, context, home)
        return

    # "base" keyword: show or set workspace base directory
    if target.lower().startswith("base"):
        parts = target.split(None, 1)
        if len(parts) == 1:
            if base:
                await update.message.reply_text(f"Workspace base: {base}")
            else:
                await update.message.reply_text("No workspace base set.\nUsage: /workspace base /path/to/projects")
            return
        new_base = Path(parts[1]).expanduser().resolve()
        if not new_base.is_dir():
            await update.message.reply_text(f"Not a directory:\n{new_base}")
            return
        await sessions.set_setting("workspace_base", str(new_base))
        await update.message.reply_text(f"Workspace base set to:\n{new_base}")
        return

    # "new" keyword: create a new workspace
    if target.lower().startswith("new"):
        parts = target.split(None, 1)
        if len(parts) < 2:
            await update.message.reply_text("Usage: /workspace new <name>")
            return
        name = parts[1]
        resolved = await _resolve_workspace_path(name, base)
        if resolved is None:
            await update.message.reply_text(
                "Set a workspace base first:\n/workspace base /path/to/projects\n\nOr use a full path: /workspace new /full/path"
            )
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

    # Resolve via base directory or absolute path
    resolved = await _resolve_workspace_path(target, base)
    if resolved is None:
        await update.message.reply_text(
            f"Unknown workspace: {target}\n"
            "Set a workspace base first:\n/workspace base /path/to/projects"
        )
        return
    path = resolved

    if not path.exists():
        await update.message.reply_text(f"Path does not exist:\n{path}")
        return
    if not path.is_dir():
        await update.message.reply_text(f"Not a directory:\n{path}")
        return

    await _switch_workspace(update, context, path)


@_require_auth
async def handle_webhooks(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    config: Config = context.bot_data["config"]
    running = webhook.is_running()
    status = "running" if running else "not running"
    lines = [
        f"Webhook server: {status}",
        f"Port: {config.webhook_port}",
        "",
        "Endpoints:",
        "  POST /webhook/github  (GitHub events)",
        "  POST /webhook         (generic)",
        "  POST /api/schedule    (scheduling API)",
        "  GET  /health          (health check)",
    ]
    if running:
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
        "/workspace <name> - Switch by name or path\n"
        "/workspace new <name> - Create + git init + switch\n"
        "/workspace base <path> - Set projects directory\n"
        "/workspace home - Return to default\n"
        "/workspaces - Switch workspace (inline buttons)\n"
        "/models - Choose a model\n"
        "/model <name> - Switch model directly\n"
        "/memory - Show memory file locations\n"
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
    # Keep "typing..." visible until the response completes
    typing_active = True

    async def _keep_typing():
        while typing_active:
            try:
                await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
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
    app.add_handler(CommandHandler("memory", handle_memory))
    app.add_handler(CommandHandler("workspace", handle_workspace))
    app.add_handler(CommandHandler("workspaces", handle_workspaces))
    app.add_handler(CommandHandler("webhooks", handle_webhooks))
    app.add_handler(CommandHandler("stop", handle_stop))
    app.add_handler(CallbackQueryHandler(handle_model_callback, pattern=r"^model:"))
    app.add_handler(CallbackQueryHandler(handle_workspace_callback, pattern=r"^ws:"))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.COMMAND, handle_unknown_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    return app
