"""
Telegram bot interface — command handlers, message routing, and streaming responses.

Provides functionality to:
1. Handle all Telegram slash commands (/new, /model, /workspace, /voice, etc.)
2. Process text, photo, document, and voice messages from the user
3. Stream Claude's responses in real-time with progressive message edits
4. Manage model switching, voice TTS output, and workspace navigation
5. Enforce authorization (only allowed user IDs can interact)

This module is the "presentation layer" of Kai — it receives Telegram updates,
translates them into prompts for the Claude process (claude.py), streams the
response back to the user, and handles all Telegram-specific concerns like
message length limits, Markdown fallback, inline keyboards, and typing indicators.

The response flow for a text message:
    1. User message arrives → handle_message()
    2. Message logged to JSONL history
    3. Per-chat lock acquired (prevents concurrent Claude interactions)
    4. Flag file written (for crash recovery)
    5. Prompt sent to PersistentClaude.send() → streaming begins
    6. Live message created and progressively edited (2-second intervals)
    7. Final response delivered (text, voice, or both depending on voice mode)
    8. Session saved to database (cost tracking)
    9. Flag file cleared

Handler registration order in create_bot() matters: python-telegram-bot matches
the first handler whose filters pass, so specific commands are registered before
the catch-all text message handler.
"""

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

# Minimum interval between Telegram message edits (seconds).
# Telegram rate-limits message edits; 2 seconds keeps us safely below the limit
# while still giving the user a sense of streaming output.
EDIT_INTERVAL = 2.0

# Flag file written while processing a message. If the process crashes mid-response,
# main.py detects this file at startup and notifies the user to resend.
_RESPONDING_FLAG = PROJECT_ROOT / ".responding_to"


# ── Crash recovery flag ──────────────────────────────────────────────


def _set_responding(chat_id: int) -> None:
    """Write the chat ID to the flag file, marking a response as in-flight."""
    _RESPONDING_FLAG.write_text(str(chat_id))


def _clear_responding() -> None:
    """Remove the flag file, indicating the response completed (or failed gracefully)."""
    _RESPONDING_FLAG.unlink(missing_ok=True)


# ── Update property helpers (Pyright can't narrow @property returns) ─


def _chat_id(update: Update) -> int:
    """Extract the chat ID from an update, with type narrowing for static analysis."""
    chat = update.effective_chat
    assert chat is not None
    return chat.id


def _user_id(update: Update) -> int:
    """Extract the user ID from an update, with type narrowing for static analysis."""
    user = update.effective_user
    assert user is not None
    return user.id


# ── Authorization ────────────────────────────────────────────────────


def _is_authorized(config: Config, user_id: int) -> bool:
    """Check if a Telegram user ID is in the allowed list."""
    return user_id in config.allowed_user_ids


def _require_auth(func):
    """
    Decorator that silently drops updates from unauthorized users.

    Wraps a Telegram handler function to check the sender's user ID against
    the allowed list before executing. Unauthorized messages are ignored
    without any response (to avoid revealing the bot's existence).
    """

    @functools.wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        config: Config = context.bot_data["config"]
        if not _is_authorized(config, _user_id(update)):
            return
        return await func(update, context)

    return wrapper


# ── Telegram message utilities ───────────────────────────────────────


def _truncate_for_telegram(text: str, max_len: int = 4096) -> str:
    """Truncate text to Telegram's message length limit, appending '...' if cut."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 4] + "\n..."


async def _reply_safe(msg: Message, text: str) -> Message:
    """
    Reply with Markdown formatting, falling back to plain text on parse failure.

    Telegram's Markdown parser is strict about balanced formatting characters.
    Rather than trying to escape everything, we just retry without parse_mode
    if the first attempt fails.
    """
    try:
        return await msg.reply_text(text, parse_mode=ParseMode.MARKDOWN)
    except Exception:
        return await msg.reply_text(text)


async def _edit_message_safe(msg: Message, text: str) -> None:
    """
    Edit an existing message with Markdown, falling back to plain text.

    Used during streaming to update the live response message. Silently
    ignores errors from the final fallback (e.g., message not modified,
    message deleted by user).
    """
    truncated = _truncate_for_telegram(text)
    try:
        await msg.edit_text(truncated, parse_mode=ParseMode.MARKDOWN)
    except Exception:
        try:
            await msg.edit_text(truncated)
        except Exception:
            pass


def _chunk_text(text: str, max_len: int = 4096) -> list[str]:
    """
    Split text into Telegram-safe chunks at natural break points.

    Prefers splitting at double newlines (paragraph breaks), then single
    newlines, and only falls back to hard-cutting at max_len if no break
    point is found. This keeps code blocks and paragraphs intact.

    Args:
        text: The text to split.
        max_len: Maximum length per chunk (Telegram's limit is 4096).

    Returns:
        A list of text chunks, each within max_len.
    """
    chunks = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        # Try paragraph break, then line break, then hard cut
        split_at = text.rfind("\n\n", 0, max_len)
        if split_at == -1:
            split_at = text.rfind("\n", 0, max_len)
        if split_at == -1:
            split_at = max_len
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    return chunks


async def _send_response(update: Update, text: str) -> None:
    """Send a potentially long response as multiple chunked messages."""
    assert update.message is not None
    for chunk in _chunk_text(text):
        await _reply_safe(update.message, chunk)


def _get_claude(context: ContextTypes.DEFAULT_TYPE) -> PersistentClaude:
    """Retrieve the PersistentClaude instance from bot_data."""
    return context.bot_data["claude"]


# ── Basic command handlers ───────────────────────────────────────────


@_require_auth
async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start — the initial greeting when a user first messages the bot."""
    assert update.message is not None
    await update.message.reply_text("Kai is ready. Send me a message.")


@_require_auth
async def handle_new(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle /new — kill the Claude process and start a fresh session.

    Clears the session from the database so cost tracking resets, and
    kills the subprocess so the next message launches a new one.
    """
    assert update.message is not None
    claude = _get_claude(context)
    await claude.restart()
    await sessions.clear_session(_chat_id(update))
    await update.message.reply_text("Session cleared. Starting fresh.")


# ── Model selection ──────────────────────────────────────────────────

# Available Claude models with display names (emoji prefix for visual distinction)
_AVAILABLE_MODELS = {
    "opus": "\U0001f9e0 Claude Opus 4.6",
    "sonnet": "\u26a1 Claude Sonnet 4.5",
    "haiku": "\U0001fab6 Claude Haiku 4.5",
}


def _models_keyboard(current: str) -> InlineKeyboardMarkup:
    """Build an inline keyboard with model choices, highlighting the current model."""
    buttons = []
    for key, name in _AVAILABLE_MODELS.items():
        label = f"{name} \U0001f7e2" if key == current else name
        buttons.append([InlineKeyboardButton(label, callback_data=f"model:{key}")])
    return InlineKeyboardMarkup(buttons)


@_require_auth
async def handle_models(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /models — show an inline keyboard for model selection."""
    assert update.message is not None
    claude = _get_claude(context)
    await update.message.reply_text(
        "Choose a model:",
        reply_markup=_models_keyboard(claude.model),
    )


async def _switch_model(context: ContextTypes.DEFAULT_TYPE, chat_id: int, model: str) -> None:
    """
    Switch the Claude model, restart the process, and clear the session.

    Called by both the inline keyboard callback and the /model text command.
    """
    claude = _get_claude(context)
    claude.model = model
    await claude.restart()
    await sessions.clear_session(chat_id)


async def handle_model_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle inline keyboard model selection.

    Validates authorization and the selected model, switches if different
    from current, and updates the keyboard message with confirmation text.
    """
    assert update.callback_query is not None
    query = update.callback_query
    config: Config = context.bot_data["config"]
    if not _is_authorized(config, _user_id(update)):
        await query.answer("Not authorized.")
        return

    assert query.data is not None
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
    await _switch_model(context, _chat_id(update), model)
    await query.edit_message_text(
        f"Switched to {_AVAILABLE_MODELS[model]}. Session restarted.",
        reply_markup=InlineKeyboardMarkup([]),
    )


@_require_auth
async def handle_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /model <name> — switch model directly via text command."""
    assert update.message is not None
    if not context.args:
        await update.message.reply_text("Usage: /model <opus|sonnet|haiku>")
        return
    model = context.args[0].lower()
    if model not in _AVAILABLE_MODELS:
        await update.message.reply_text("Choose: opus, sonnet, or haiku")
        return
    await _switch_model(context, _chat_id(update), model)
    await update.message.reply_text(f"Model set to {_AVAILABLE_MODELS[model]}. Session restarted.")


# ── Voice TTS ────────────────────────────────────────────────────────


def _voices_keyboard(current: str) -> InlineKeyboardMarkup:
    """Build an inline keyboard with voice choices, highlighting the current voice."""
    buttons = []
    for key, name in VOICES.items():
        label = f"{name} \U0001f7e2" if key == current else name
        buttons.append([InlineKeyboardButton(label, callback_data=f"voice:{key}")])
    return InlineKeyboardMarkup(buttons)


# Voice mode options: "off" (text only), "on" (text + voice), "only" (voice only)
_VOICE_MODES = {"off", "on", "only"}
_VOICE_MODE_LABELS = {"off": "OFF", "on": "ON (text + voice)", "only": "ONLY (voice only)"}


@_require_auth
async def handle_voice_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle /voice — toggle voice mode or set a specific voice.

    Supports multiple subcommands:
        /voice          — toggle off ↔ only
        /voice on       — enable text + voice mode
        /voice only     — enable voice-only mode (no text)
        /voice off      — disable voice
        /voice <name>   — set a specific voice (enables voice if off)
    """
    assert update.message is not None
    config: Config = context.bot_data["config"]
    if not config.tts_enabled:
        await update.message.reply_text("TTS is not enabled. Set TTS_ENABLED=true in .env")
        return

    chat_id = _chat_id(update)
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
    """Handle /voices — show an inline keyboard of available TTS voices."""
    assert update.message is not None
    config: Config = context.bot_data["config"]
    if not config.tts_enabled:
        await update.message.reply_text("TTS is not enabled. Set TTS_ENABLED=true in .env")
        return

    chat_id = _chat_id(update)
    current_voice = await sessions.get_setting(f"voice_name:{chat_id}") or DEFAULT_VOICE
    await update.message.reply_text(
        "Choose a voice:",
        reply_markup=_voices_keyboard(current_voice),
    )


async def handle_voice_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle inline keyboard voice selection.

    Sets the chosen voice in settings and auto-enables voice mode if it
    was off (defaults to "only" mode).
    """
    assert update.callback_query is not None
    query = update.callback_query
    config: Config = context.bot_data["config"]
    if not _is_authorized(config, _user_id(update)):
        await query.answer("Not authorized.")
        return

    assert query.data is not None
    voice = query.data.removeprefix("voice:")
    if voice not in VOICES:
        await query.answer("Invalid voice.")
        return

    chat_id = _chat_id(update)
    current_voice = await sessions.get_setting(f"voice_name:{chat_id}") or DEFAULT_VOICE

    if voice == current_voice:
        await query.answer()
        await query.edit_message_text("No change.", reply_markup=InlineKeyboardMarkup([]))
        return

    current_mode = await sessions.get_setting(f"voice_mode:{chat_id}") or "off"
    await sessions.set_setting(f"voice_name:{chat_id}", voice)
    # Auto-enable voice if it was off
    if current_mode == "off":
        await sessions.set_setting(f"voice_mode:{chat_id}", "only")
        current_mode = "only"
    await query.answer()
    await query.edit_message_text(
        f"Voice set to {VOICES[voice]}. Voice mode: {_VOICE_MODE_LABELS[current_mode]}",
        reply_markup=InlineKeyboardMarkup([]),
    )


# ── Info and management commands ─────────────────────────────────────


@_require_auth
async def handle_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /stats — show session info, model, cost, and process status."""
    assert update.message is not None
    claude = _get_claude(context)
    stats = await sessions.get_stats(_chat_id(update))
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
    """
    Handle /jobs — list all active scheduled jobs with their schedules.

    Formats each job with an emoji tag (bell for reminders, robot for Claude
    jobs), the job ID, name, and a human-readable schedule description.
    """
    assert update.message is not None
    jobs = await sessions.get_jobs(_chat_id(update))
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
            # Format interval in the most readable unit
            if secs >= 3600:
                detail = f"every {secs // 3600}h"
            elif secs >= 60:
                detail = f"every {secs // 60}m"
            else:
                detail = f"every {secs}s"
        elif sched == "daily":
            data = json.loads(j["schedule_data"])
            times = data.get("times", [])
            detail = f"daily at {', '.join(times)} UTC" if times else "daily"
        else:
            detail = sched
        type_tag = "\U0001f514" if j["job_type"] == "reminder" else "\U0001f916"
        lines.append(f"{type_tag} #{j['id']} {j['name']} ({detail})")
    await update.message.reply_text("Active jobs:\n" + "\n".join(lines))


@_require_auth
async def handle_canceljob(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle /canceljob <id> — permanently delete a scheduled job.

    Removes the job from both the database and APScheduler's in-memory queue.
    """
    assert update.message is not None
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
    # Remove from APScheduler's in-memory queue
    jq = context.application.job_queue
    current = jq.get_jobs_by_name(f"cron_{job_id}")
    for j in current:
        j.schedule_removal()
    await update.message.reply_text(f"Job #{job_id} cancelled.")


@_require_auth
async def handle_stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle /stop — abort the current Claude response.

    Sets the per-chat stop event (checked by the streaming loop) and kills
    the Claude process immediately. The streaming loop in _handle_response()
    sees the stop event and appends "(stopped)" to the live message.
    """
    assert update.message is not None
    chat_id = _chat_id(update)
    claude = _get_claude(context)
    stop_event = get_stop_event(chat_id)
    stop_event.set()
    claude.force_kill()
    await update.message.reply_text("Stopping...")


# ── Workspace management ─────────────────────────────────────────────


def _resolve_workspace_path(target: str, base: Path | None) -> Path | None:
    """
    Resolve a workspace name to an absolute path under the base directory.

    Only relative names are allowed (e.g., "my-project", not "/tmp/evil").
    Returns None if no base is set or if the resolved path would escape
    the base directory (path traversal prevention).

    Args:
        target: The workspace name or relative path.
        base: The WORKSPACE_BASE directory, or None if unset.

    Returns:
        The resolved absolute path, or None if invalid.
    """
    if not base:
        return None
    resolved = (base / target).resolve()
    # Prevent traversal outside the base directory
    if not str(resolved).startswith(str(base) + "/") and resolved != base:
        return None
    return resolved


def _short_workspace_name(path: str, base: Path | None) -> str:
    """
    Shorten a workspace path for display in Telegram messages and keyboards.

    If the path is under WORKSPACE_BASE, strips the base prefix to show just
    the relative name. Otherwise falls back to showing just the directory name.
    """
    base_str = str(base) if base else None
    if base_str and path.startswith(base_str.rstrip("/") + "/"):
        return path[len(base_str.rstrip("/")) + 1 :]
    return Path(path).name


async def _do_switch_workspace(context: ContextTypes.DEFAULT_TYPE, chat_id: int, path: Path) -> None:
    """
    Core workspace switch logic shared by command and callback handlers.

    Kills the Claude process (it will restart in the new directory on next
    message), clears the session, and persists the new workspace to settings.
    Switching to home deletes the setting (home is the default).
    """
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
    """
    Switch to a workspace path and send a confirmation reply.

    Wraps _do_switch_workspace with user-facing feedback including workspace
    metadata (git repo detection, CLAUDE.md presence).
    """
    assert update.message is not None
    claude = _get_claude(context)
    config: Config = context.bot_data["config"]
    home = config.claude_workspace

    if path == claude.workspace:
        await update.message.reply_text("Already in that workspace.")
        return

    await _do_switch_workspace(context, _chat_id(update), path)

    if path == home:
        await update.message.reply_text("Switched to home workspace. Session cleared.")
    else:
        # Show useful metadata about the workspace
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
    """
    Build an inline keyboard for workspace switching.

    Shows a Home button at the top, followed by recent workspace history.
    The current workspace is marked with a green dot. Each button's callback
    data is "ws:home" or "ws:<index>" where index maps to the history list.
    """
    buttons = []
    # Home button (always first)
    home_label = "\U0001f3e0 Home"
    if current_path == home_path:
        home_label += " \U0001f7e2"
    buttons.append([InlineKeyboardButton(home_label, callback_data="ws:home")])
    # History entries (skip home, already shown above)
    for i, entry in enumerate(history):
        p = entry["path"]
        if p == home_path:
            continue
        short = _short_workspace_name(p, base)
        label = short
        if p == current_path:
            label += " \U0001f7e2"
        buttons.append([InlineKeyboardButton(label, callback_data=f"ws:{i}")])
    return InlineKeyboardMarkup(buttons)


@_require_auth
async def handle_workspaces(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /workspaces — show an inline keyboard of recent workspaces."""
    assert update.message is not None
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
    """
    Handle inline keyboard workspace selection.

    Resolves the selected workspace from the callback data, validates it
    still exists, switches to it, and updates the keyboard message.
    Removes stale entries from history if the directory no longer exists.
    """
    assert update.callback_query is not None
    query = update.callback_query
    config: Config = context.bot_data["config"]
    if not _is_authorized(config, _user_id(update)):
        await query.answer("Not authorized.")
        return

    assert query.data is not None
    data = query.data.removeprefix("ws:")
    claude = _get_claude(context)
    home = config.claude_workspace
    base = config.workspace_base

    # Resolve target path from callback data
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
        # Remove stale entries where the directory no longer exists
        if not path.is_dir():
            await sessions.delete_workspace_history(str(path))
            await query.answer("That workspace no longer exists.")
            history = await sessions.get_workspace_history()
            keyboard = await _workspaces_keyboard(history, str(claude.workspace), str(home), base)
            await query.edit_message_reply_markup(reply_markup=keyboard)
            return
        label = _short_workspace_name(str(path), base)

    # Already there — dismiss the keyboard
    if path == claude.workspace:
        await query.answer()
        await query.edit_message_text("No change.", reply_markup=InlineKeyboardMarkup([]))
        return

    # Switch and confirm
    await query.answer()
    await _do_switch_workspace(context, _chat_id(update), path)
    await query.edit_message_text(
        f"Switched to {label}. Session cleared.",
        reply_markup=InlineKeyboardMarkup([]),
    )


_NO_BASE_MSG = "WORKSPACE_BASE is not set. Add it to .env and restart."


@_require_auth
async def handle_workspace(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle /workspace — show, switch, or create workspaces.

    Subcommands:
        /workspace              — show current workspace
        /workspace home         — switch to home workspace
        /workspace <name>       — switch to a workspace by name (resolved under WORKSPACE_BASE)
        /workspace new <name>   — create a new workspace with git init and switch to it

    Absolute paths and ~ expansion are rejected for security. All workspace
    names are resolved relative to WORKSPACE_BASE with traversal prevention.
    """
    assert update.message is not None
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

    # Reject absolute paths and ~ expansion for security
    if target.startswith("/") or target.startswith("~"):
        await update.message.reply_text("Absolute paths are not allowed. Use a workspace name.")
        return

    # "new" keyword: create a new workspace directory with git init
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

    # Resolve workspace name via base directory
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


# ── Server info and help ─────────────────────────────────────────────


@_require_auth
async def handle_webhooks(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /webhooks — show webhook server status and endpoint info."""
    assert update.message is not None
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
    """Handle /help — show all available commands."""
    assert update.message is not None
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
    """Handle unrecognized slash commands with a helpful redirect to /help."""
    assert update.message is not None
    await update.message.reply_text(
        f"Unknown command: {update.message.text.split()[0]}\nTry /help for available commands."
    )


# ── Media message handlers ──────────────────────────────────────────


@_require_auth
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle photo messages — download, base64-encode, and send to Claude.

    Downloads the highest-resolution version of the photo, encodes it as
    base64, and sends it to Claude as a multi-modal content block alongside
    the caption (or "What's in this image?" if no caption).
    """
    if not update.message or not update.message.photo:
        return

    chat_id = _chat_id(update)
    claude = _get_claude(context)
    model = claude.model

    # Download the largest available resolution (last in the list)
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


# File extensions treated as readable text (sent to Claude as code blocks)
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

# Image extensions that can be sent as documents (uncompressed)
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}

# Map image file extensions to MIME types for Claude's image content blocks
_IMAGE_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


@_require_auth
async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle document (file) uploads — images, text files, and code.

    Routes based on file extension:
    - Image files → base64-encoded and sent as multi-modal content
    - Text/code files → decoded as UTF-8 and sent as a code block
    - Other files → rejected with a helpful message
    """
    if not update.message or not update.message.document:
        return

    doc = update.message.document
    file_name = doc.file_name or "unknown"
    suffix = Path(file_name).suffix.lower()
    caption = update.message.caption or ""

    chat_id = _chat_id(update)
    claude = _get_claude(context)
    model = claude.model

    if suffix in _IMAGE_EXTENSIONS:
        # Handle images sent as documents (uncompressed upload)
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
        # Handle text/code files — decode and wrap in a code block
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
    """
    Handle voice messages — transcribe via whisper-cpp and send to Claude.

    Pipeline: download audio → check dependencies → transcribe → echo
    transcription to user → send to Claude as "[Voice message transcription]: ..."

    The echo step shows the user what was heard before Claude processes it,
    providing transparency and a chance to correct misheard speech.
    """
    if not update.message or not update.message.voice:
        return

    chat_id = _chat_id(update)
    claude = _get_claude(context)
    config: Config = context.bot_data["config"]

    if not config.voice_enabled:
        await update.message.reply_text("Voice messages are not enabled.")
        return

    # Check that all required external tools are available
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

    # Echo the transcription so the user sees what Kai heard
    await _reply_safe(update.message, f"_Heard:_ {transcript}")

    prompt = f"[Voice message transcription]: {transcript}"
    model = claude.model

    async with get_lock(chat_id):
        _set_responding(chat_id)
        try:
            await _handle_response(update, context, chat_id, prompt, claude, model)
        finally:
            _clear_responding()


# ── Main message handler ─────────────────────────────────────────────


@_require_auth
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle plain text messages — the primary interaction path.

    Logs the message, acquires the per-chat lock, sets the crash recovery
    flag, sends the prompt to Claude, and delegates to _handle_response()
    for streaming and delivery.
    """
    if not update.message or not update.message.text:
        return

    chat_id = _chat_id(update)
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


# ── Streaming response handler ───────────────────────────────────────


async def _handle_response(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    prompt: str | list,
    claude: PersistentClaude,
    model: str,
) -> None:
    """
    Stream Claude's response and deliver it to the user.

    This is the central response handler used by all message types (text,
    photo, document, voice). It manages the full response lifecycle:

    1. Check voice mode to determine output format
    2. Start a background typing indicator task
    3. Stream events from Claude, creating/editing a live Telegram message
    4. Handle /stop interruptions via the per-chat stop event
    5. On completion: save session, log response, deliver final text/voice
    6. Handle errors gracefully with user-visible error messages

    In voice-only mode, streaming text edits are skipped (no live message)
    and the final response is synthesized to speech via Piper TTS.

    In text+voice mode, the text response is delivered normally, then a
    voice note is sent as a follow-up.

    Args:
        update: The Telegram Update that triggered this response.
        context: Telegram callback context.
        chat_id: The Telegram chat ID.
        prompt: Text string or list of content blocks to send to Claude.
        claude: The PersistentClaude instance.
        model: Current model name (for session tracking).
    """
    assert update.message is not None
    # Check voice mode before starting
    config: Config = context.bot_data["config"]
    voice_mode = "off"
    if config.tts_enabled:
        voice_mode = await sessions.get_setting(f"voice_mode:{chat_id}") or "off"
    voice_only = voice_mode == "only"

    # Keep activity indicator visible until the response completes.
    # Telegram hides the typing indicator after ~5 seconds, so we
    # re-send it every 4 seconds in a background task.
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

    # Reset the stop event (in case /stop was sent between messages)
    stop_event = get_stop_event(chat_id)
    stop_event.clear()

    # Stream events from Claude
    async for event in claude.send(prompt):
        # Check for /stop between stream chunks
        if stop_event.is_set():
            stop_event.clear()
            if live_msg:
                await _edit_message_safe(live_msg, last_edit_text + "\n\n_(stopped)_")
            final_response = None
            break

        if event.done:
            final_response = event.response
            break

        # In voice-only mode, skip live text updates
        if voice_only:
            continue

        now = time.monotonic()
        if not event.text_so_far:
            continue

        # Create the live message on first text, then edit periodically
        if live_msg is None:
            truncated = _truncate_for_telegram(event.text_so_far)
            live_msg = await _reply_safe(update.message, truncated)
            last_edit_time = now
            last_edit_text = event.text_so_far
        elif now - last_edit_time >= EDIT_INTERVAL and event.text_so_far != last_edit_text:
            await _edit_message_safe(live_msg, event.text_so_far)
            last_edit_time = now
            last_edit_text = event.text_so_far

    # Stop the typing indicator
    typing_active = False
    typing_task.cancel()
    try:
        await typing_task
    except asyncio.CancelledError:
        pass

    # Handle error cases
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

    # Persist session info for /stats (cost accumulates across interactions)
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
        # Update the live message with the final text
        if len(final_text) <= 4096:
            if final_text != last_edit_text:
                await _edit_message_safe(live_msg, final_text)
        else:
            # Response exceeds Telegram's limit — edit first chunk, send the rest
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


# ── Application factory ─────────────────────────────────────────────


def create_bot(config: Config) -> Application:
    """
    Build and configure the Telegram Application with all handlers.

    Creates the python-telegram-bot Application, initializes the PersistentClaude
    subprocess manager, stores both in bot_data, and registers all command,
    callback, and message handlers.

    concurrent_updates=True is required so /stop can be processed while a
    message handler is blocked waiting on Claude's response.

    Handler registration order matters: specific handlers (commands, photos,
    documents, voice) are registered before the catch-all text handler.

    Args:
        config: The application Config instance.

    Returns:
        A fully configured Telegram Application ready to be started.
    """
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

    # Command handlers (alphabetical registration, but order doesn't matter for commands)
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

    # Callback query handlers for inline keyboards (pattern-matched)
    app.add_handler(CallbackQueryHandler(handle_model_callback, pattern=r"^model:"))
    app.add_handler(CallbackQueryHandler(handle_voice_callback, pattern=r"^voice:"))
    app.add_handler(CallbackQueryHandler(handle_workspace_callback, pattern=r"^ws:"))

    # Media handlers (must be before the catch-all text handler)
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))

    # Unknown command handler (catches unrecognized /commands)
    app.add_handler(MessageHandler(filters.COMMAND, handle_unknown_command))

    # Catch-all text message handler (must be last)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    return app
