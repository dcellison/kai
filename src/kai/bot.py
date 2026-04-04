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
import math
import shutil
import time
from datetime import datetime
from pathlib import Path

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Message, Update
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from kai import services, sessions, webhook
from kai.config import DATA_DIR, MAX_CONTEXT_CEILING, VALID_MODELS, Config, WorkspaceConfig
from kai.history import log_message
from kai.locks import get_lock, get_stop_event
from kai.pool import SubprocessPool
from kai.telegram_utils import chunk_text
from kai.transcribe import TranscriptionError, transcribe_voice
from kai.tts import DEFAULT_VOICE, VOICES, TTSError, synthesize_speech
from kai.workspace_utils import is_workspace_allowed

# TOTP is optional (requires pip install -e '.[totp]'). When the extra is not
# installed, is_totp_configured() returns False and the gate is fully disabled.
# All four stubs are defined in the except block so Pyright doesn't flag them
# as possibly-unbound at their call sites inside the gate.
try:
    from kai.totp import get_failure_count, get_lockout_remaining, is_totp_configured, verify_code
except ImportError:

    def is_totp_configured() -> bool:  # type: ignore[misc]
        return False

    def get_lockout_remaining() -> int:  # type: ignore[misc]
        return 0

    def verify_code(code: str, lockout_attempts: int = 3, lockout_minutes: int = 15) -> bool:  # type: ignore[misc]
        return False

    def get_failure_count() -> int:  # type: ignore[misc]
        return 0


log = logging.getLogger(__name__)

# Minimum interval between Telegram message edits (seconds).
# Telegram rate-limits message edits; 2 seconds keeps us safely below the limit
# while still giving the user a sense of streaming output.
EDIT_INTERVAL = 2.0

# Flag file written while processing a message. If the process crashes mid-response,
# main.py detects this file at startup and notifies the user to resend. Lives under
# DATA_DIR so it's writable even when source is in read-only /opt/kai/.
# Directory for per-user crash recovery flags. Each file is named by
# chat_id and exists only while that user's response is in-flight.
# Using a directory of files (not a single JSON file) avoids locking
# and allows atomic per-user create/delete.
_RESPONDING_DIR = DATA_DIR / ".responding"


# ── Crash recovery flag ──────────────────────────────────────────────


def _set_responding(chat_id: int) -> None:
    """Mark a response as in-flight for crash recovery."""
    _RESPONDING_DIR.mkdir(exist_ok=True)
    (_RESPONDING_DIR / str(chat_id)).touch()


def _clear_responding(chat_id: int) -> None:
    """Mark a response as complete for a specific user."""
    (_RESPONDING_DIR / str(chat_id)).unlink(missing_ok=True)


async def _notify_if_queued(update: Update, chat_id: int) -> bool:
    """Send a notification if the user's message will queue behind the lock.

    Called immediately before acquiring the per-chat lock. If the lock is
    already held (Kai is mid-response), sends a one-line Telegram message
    so the user knows their message was received. The notification goes
    directly to Telegram via _reply_safe - Claude never sees it. Do NOT
    add a log_message call here; the notification is purely for the user.

    Returns True if the message is queuing (lock was held), False otherwise.
    The caller uses this to decide whether to prepend a context-switch
    marker to the prompt via _prepend_queue_marker().

    There is a harmless TOCTOU gap: if the lock holder releases between
    the locked() check and the subsequent acquire, the user sees "finishing
    something up" followed by an instant response, and Claude gets a
    context-switch marker for a task that already finished. Both are
    harmless and not worth fixing.
    """
    if get_lock(chat_id).locked():
        assert update.message is not None
        await _reply_safe(
            update.message,
            "Got your message - finishing something up. /stop to interrupt.",
        )
        return True
    return False


# Prepended to prompts that waited behind the lock, so Claude focuses on the
# new message instead of continuing from the previous task's tool output.
_QUEUED_MESSAGE_MARKER = (
    "[The user sent this while you were working on something else. "
    "Their previous task is done. Focus on this new message.]\n\n"
)

# Safety-net timeout for acquiring the per-chat lock (seconds). If the
# idle timeout in claude.py doesn't fire for some reason, this prevents
# a stuck interaction from blocking all future messages indefinitely.
# Set generously: the idle timer in claude.py is the real safety net
# (fires after timeout_seconds * 5 of silence); this is a last-resort
# backstop for interactions that run legitimately long with active output.
_LOCK_ACQUIRE_TIMEOUT = 3600  # 1 hour


async def _acquire_lock_or_kill(
    chat_id: int,
    pool: "SubprocessPool",
    update: Update,
) -> asyncio.Lock | None:
    """Acquire the per-chat lock with a timeout, force-killing if stuck.

    Returns the acquired lock on success (caller must call lock.release()
    in a finally block). Returns None if the lock timed out, in which case
    the stuck Claude process was killed and the user was notified - caller
    should return without further action.

    Returns the lock object directly rather than a bool so the caller
    releases the same object that was acquired (avoids issues if get_lock
    is called again and returns a different instance).
    """
    lock = get_lock(chat_id)
    try:
        await asyncio.wait_for(lock.acquire(), timeout=_LOCK_ACQUIRE_TIMEOUT)
        return lock
    except TimeoutError:
        log.error(
            "Lock acquisition timed out for chat %d after %ds; force-killing Claude",
            chat_id,
            _LOCK_ACQUIRE_TIMEOUT,
        )
        await pool.force_kill(chat_id)
        # update.message can be None for edited messages or callback
        # queries, so guard rather than assert.
        if update.message is not None:
            await _reply_safe(
                update.message,
                "Previous task timed out and was stopped. Please send your message again.",
            )
        return None


def _prepend_queue_marker(prompt: str | list[dict[str, str]]) -> str | list[dict[str, str]]:
    """Prepend context-switch marker to a prompt that waited behind the lock.

    Handles both plain string prompts (text, document, voice) and multimodal
    content lists (photo). For lists, prepends to the first text block's text
    field and passes subsequent blocks (e.g., base64 image) through unchanged.
    """
    if isinstance(prompt, list):
        # Multimodal content (photo): prepend to the first text block
        first = prompt[0]
        return [{"type": "text", "text": _QUEUED_MESSAGE_MARKER + first["text"]}] + prompt[1:]
    return _QUEUED_MESSAGE_MARKER + prompt


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
    if the first attempt fails. Only catches BadRequest (Telegram rejecting the
    markup) - network errors and timeouts propagate normally to avoid sending
    duplicate messages.
    """
    try:
        return await msg.reply_text(text, parse_mode=ParseMode.MARKDOWN)
    except BadRequest:
        return await msg.reply_text(text)


async def _edit_message_safe(msg: Message, text: str) -> None:
    """
    Edit an existing message with Markdown, falling back to plain text.

    Used during streaming to update the live response message. On BadRequest
    (Telegram rejecting the markup), retries without parse_mode. All other
    errors are silently ignored since edits are best-effort during streaming
    (e.g., message not modified, message deleted by user, network blip).
    """
    truncated = _truncate_for_telegram(text)
    try:
        await msg.edit_text(truncated, parse_mode=ParseMode.MARKDOWN)
    except BadRequest:
        try:
            await msg.edit_text(truncated)
        except Exception:
            # Editing is best-effort during streaming; log at debug so persistent
            # issues (e.g., revoked bot token) leave a diagnostic trail
            log.debug("Failed to edit message (plain-text fallback)", exc_info=True)
    except Exception:
        log.debug("Failed to edit message", exc_info=True)


async def _send_response(update: Update, text: str) -> None:
    """Send a potentially long response as multiple chunked messages."""
    assert update.message is not None
    for chunk in chunk_text(text):
        await _reply_safe(update.message, chunk)


def _get_pool(context: ContextTypes.DEFAULT_TYPE) -> "SubprocessPool":
    """Retrieve the SubprocessPool from bot_data."""
    return context.bot_data["pool"]


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
    chat_id = _chat_id(update)
    pool = _get_pool(context)
    await pool.restart(chat_id)
    await sessions.clear_session(chat_id)
    await update.message.reply_text("Session cleared. Starting fresh.")


# ── Model selection ──────────────────────────────────────────────────

# Available Claude models with display names (emoji prefix for visual distinction)
# Keys must match VALID_MODELS in config.py (the single source of truth
# for valid model identifiers). Values are display names for Telegram.
_AVAILABLE_MODELS = {
    "opus": "\U0001f9e0 Claude Opus",
    "sonnet": "\u26a1 Claude Sonnet",
    "haiku": "\U0001fab6 Claude Haiku",
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
    pool = _get_pool(context)
    await update.message.reply_text(
        "Choose a model:",
        reply_markup=_models_keyboard(pool.get_model(_chat_id(update))),
    )


async def _switch_model(context: ContextTypes.DEFAULT_TYPE, chat_id: int, model: str) -> None:
    """
    Switch the Claude model, persist the choice, restart the process,
    and clear the session.

    Called by the inline keyboard callback, /model text command, and
    /settings model handler. The model choice is written to the DB so
    it survives restarts (behavior change from session-only).
    """
    pool = _get_pool(context)
    pool.set_model(chat_id, model)
    # Persist to settings table so the choice survives restarts
    await sessions.set_user_setting(chat_id, "model", model)
    await pool.restart(chat_id)
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

    pool = _get_pool(context)
    if model == pool.get_model(_chat_id(update)):
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


# ── Per-user settings ──────────────────────────────────────────────

# Map user-facing field names to DB storage keys. "context" is the
# user-facing name; "context_window" is the DB key. Defined once so
# the mapping isn't scattered across set/reset/display handlers.
_FIELD_ALIASES: dict[str, str] = {"context": "context_window"}


@_require_auth
async def handle_settings(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle /settings - view or modify per-user default settings.

    Dispatches to show, set, or reset based on arguments. Settings
    persist in the database and survive restarts.
    """
    assert update.message is not None
    chat_id = _chat_id(update)
    config: Config = context.bot_data["config"]

    # Parse: "/settings [field] [value...]"
    args = context.args or []
    field = args[0].lower() if args else None
    value = args[1] if len(args) > 1 else None

    # /settings - show current
    if field is None:
        await _show_settings(update, chat_id, config)
        return

    # /settings reset [field]
    if field == "reset":
        await _handle_settings_reset(update, context, chat_id, config, value)
        return

    # /settings model <name>
    if field == "model":
        if not value:
            await update.message.reply_text("Usage: /settings model <haiku|sonnet|opus>")
            return
        if value.lower() not in VALID_MODELS:
            await update.message.reply_text(f"Unknown model. Choose from: {', '.join(sorted(VALID_MODELS))}")
            return
        # Funnel through _switch_model() - same path as /model and /models
        # keyboard. _switch_model() handles DB write, instance update,
        # process restart, and session clear.
        model_key = value.lower()
        await _switch_model(context, chat_id, model_key)
        display = _AVAILABLE_MODELS.get(model_key, model_key)
        await update.message.reply_text(f"Default model set to {display}. Session restarted.")
        return

    # /settings budget <n>
    if field == "budget":
        if not value:
            await update.message.reply_text("Usage: /settings budget <amount>")
            return
        try:
            budget = float(value)
            if budget <= 0 or not math.isfinite(budget):
                raise ValueError
        except ValueError:
            await update.message.reply_text("Budget must be a positive number.")
            return
        # Enforce ceiling: users.yaml max_budget or global default.
        # 0 means "no admin ceiling" (e.g., CLAUDE_MAX_BUDGET_USD unset
        # or explicitly 0). Only enforce when a positive ceiling exists.
        user_config = config.get_user_config(chat_id)
        ceiling = (
            user_config.max_budget
            if user_config and user_config.max_budget is not None
            else config.claude_max_budget_usd
        )
        if ceiling and budget > ceiling:
            await update.message.reply_text(f"Budget cannot exceed ${ceiling:.2f} (admin limit).")
            return
        await sessions.set_user_setting(chat_id, "budget", str(budget))
        # Apply to running instance if one exists. Don't use pool.get()
        # here - it would create a new instance just to set an attribute.
        pool = _get_pool(context)
        instance = pool.get_if_exists(chat_id)
        if instance:
            instance.max_budget_usd = budget
        await update.message.reply_text(f"Default budget set to ${budget:.2f}.")
        return

    # /settings timeout <n>
    if field == "timeout":
        if not value:
            await update.message.reply_text("Usage: /settings timeout <seconds>")
            return
        try:
            timeout = int(value)
            if timeout <= 0:
                raise ValueError
        except ValueError:
            await update.message.reply_text("Timeout must be a positive integer (seconds).")
            return
        # Cap at 600s (10 minutes). The real cost guard is the budget
        # ceiling; this cap prevents a single stuck request from holding
        # the per-chat lock indefinitely.
        if timeout > 600:
            await update.message.reply_text("Timeout cannot exceed 600 seconds.")
            return
        await sessions.set_user_setting(chat_id, "timeout", str(timeout))
        # Apply to running instance if one exists (same rationale as budget)
        pool = _get_pool(context)
        instance = pool.get_if_exists(chat_id)
        if instance:
            instance.timeout_seconds = timeout
        await update.message.reply_text(f"Default timeout set to {timeout}s.")
        return

    # /settings context <n>
    if field == "context":
        if not value:
            await update.message.reply_text("Usage: /settings context <tokens>")
            return
        try:
            ctx = int(value)
            if ctx < 0:
                raise ValueError
        except ValueError:
            await update.message.reply_text("Context window must be a non-negative integer.")
            return
        if ctx != 0 and ctx < 50000:
            await update.message.reply_text("Context window must be at least 50000 tokens (or 0 for default).")
            return
        if ctx > MAX_CONTEXT_CEILING:
            await update.message.reply_text(f"Context window cannot exceed {MAX_CONTEXT_CEILING} tokens.")
            return
        await sessions.set_user_setting(chat_id, "context_window", str(ctx))
        # Context window is a CLI flag baked in at process startup
        # (passed via --settings). Must restart to take effect.
        pool = _get_pool(context)
        instance = pool.get_if_exists(chat_id)
        restarted = False
        if instance:
            instance.max_context_window = ctx
            await pool.restart(chat_id)
            await sessions.clear_session(chat_id)
            restarted = True
        label = f"{ctx:,} tokens" if ctx > 0 else "default"
        suffix = " Session restarted." if restarted else ""
        await update.message.reply_text(f"Context window set to {label}.{suffix}")
        return

    await update.message.reply_text(f"Unknown setting: {field}\nSettings: model, budget, timeout, context, reset")


async def _show_settings(update: Update, chat_id: int, config: Config) -> None:
    """Display the user's effective settings with source attribution."""
    assert update.message is not None
    db_settings = await sessions.get_user_settings(chat_id)
    user_config = config.get_user_config(chat_id)

    def _resolve(db_key: str, yaml_val: object, global_val: object, fmt: object) -> tuple[str, str]:
        """Resolve effective value and source for a setting."""
        # Wrap fmt in try/except so a corrupt DB row doesn't crash the
        # display command. Matches the defensive parsing in _restore_workspace.
        if db_key in db_settings:
            try:
                return fmt(db_settings[db_key]), "user override"  # type: ignore[operator]
            except (ValueError, TypeError):
                pass  # fall through to yaml/global
        if yaml_val is not None:
            return fmt(yaml_val), "users.yaml"  # type: ignore[operator]
        return fmt(global_val), "global default"  # type: ignore[operator]

    # Model
    yaml_model = user_config.model if user_config else None
    model, model_src = _resolve("model", yaml_model, config.claude_model, str)

    # Budget - 0 means "unlimited" (consistent with the ceiling check
    # in the budget handler, where 0 = no ceiling).
    yaml_budget = user_config.max_budget if user_config else None
    budget, budget_src = _resolve(
        "budget",
        yaml_budget,
        config.claude_max_budget_usd,
        lambda v: "unlimited" if float(v) == 0 else f"${float(v):.2f}",
    )

    # Timeout
    yaml_timeout = user_config.timeout if user_config else None
    timeout, timeout_src = _resolve("timeout", yaml_timeout, config.claude_timeout_seconds, lambda v: f"{int(v)}s")

    # Context window - handled separately from _resolve() because 0 has
    # special display semantics ("default" instead of "0 tokens") and
    # resolve_user_defaults() doesn't expose source attribution strings.
    yaml_ctx = user_config.context_window if user_config else None
    ctx_from_db = False
    try:
        if "context_window" in db_settings:
            ctx_val = int(db_settings["context_window"])
            ctx_from_db = True
        elif yaml_ctx is not None:
            ctx_val = yaml_ctx
        else:
            ctx_val = config.claude_max_context_window
    except (ValueError, TypeError):
        # Corrupt DB value - fall through to yaml/global
        ctx_val = yaml_ctx if yaml_ctx is not None else config.claude_max_context_window
    # Source attribution. When the user explicitly sets context to 0
    # (meaning "use the Claude Code default"), show "global default"
    # instead of "user override" - the intent was to revert, not override.
    # ctx_from_db is False after a corrupt DB parse, so attribution stays
    # correct on fallback (unlike checking "context_window" in db_settings).
    if ctx_from_db and ctx_val > 0:
        ctx_src = "user override"
    elif not ctx_from_db and yaml_ctx is not None:
        ctx_src = "users.yaml"
    else:
        ctx_src = "global default"
    ctx_label = f"{ctx_val:,} tokens" if ctx_val > 0 else "default"

    # Budget ceiling - show when a positive ceiling exists so the user
    # knows their limit before hitting it. Falls through to the global
    # default when no users.yaml entry exists. 0 = no limit, suppress.
    ceiling = (
        user_config.max_budget
        if user_config and user_config.max_budget is not None
        else config.claude_max_budget_usd or None  # 0 means no limit
    )
    ceiling_line = f"\n\nBudget ceiling: ${ceiling:.2f} (admin)" if ceiling else ""

    await update.message.reply_text(
        f"Your settings:\n"
        f"  Model: {model} ({model_src})\n"
        f"  Budget: {budget} ({budget_src})\n"
        f"  Timeout: {timeout} ({timeout_src})\n"
        f"  Context: {ctx_label} ({ctx_src})"
        f"{ceiling_line}"
    )


def _revert_instance_field(pool: SubprocessPool, chat_id: int, field: str, config: Config) -> None:
    """
    Write the resolved default value for a single field back onto the
    live PersistentClaude instance.

    Called before restart so that stale in-memory overrides don't
    persist after a DB entry is deleted. Resolution order mirrors
    _create_instance(): users.yaml > global config.
    """
    instance = pool.get_if_exists(chat_id)
    if not instance:
        return
    user = config.get_user_config(chat_id)
    if field == "model":
        instance.model = user.model if user and user.model else config.claude_model
    elif field == "budget":
        instance.max_budget_usd = (
            user.max_budget if user and user.max_budget is not None else config.claude_max_budget_usd
        )
    elif field == "timeout":
        instance.timeout_seconds = user.timeout if user and user.timeout is not None else config.claude_timeout_seconds
    elif field == "context_window":
        instance.max_context_window = (
            user.context_window if user and user.context_window is not None else config.claude_max_context_window
        )


async def _handle_settings_reset(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    config: Config,
    field: str | None,
) -> None:
    """
    Handle /settings reset [field].

    Always restarts the process even for non-flag settings (budget,
    timeout) where a restart isn't strictly necessary. The simplicity
    of "reset always restarts" outweighs the minor overhead of one
    extra process restart during an infrequent operation.
    """
    assert update.message is not None
    valid_fields = {"model", "budget", "timeout", "context"}

    if field:
        field = field.lower()
        if field not in valid_fields:
            await update.message.reply_text(f"Unknown field: {field}\nFields: {', '.join(sorted(valid_fields))}")
            return
        # Resolve alias (e.g., "context" -> "context_window")
        db_field = _FIELD_ALIASES.get(field, field)
        await sessions.delete_user_setting(chat_id, db_field)
        pool = _get_pool(context)
        # Write the resolved default back onto the live instance before
        # restarting. restart() preserves the Python object, so stale
        # in-memory attributes would persist without this step.
        _revert_instance_field(pool, chat_id, db_field, config)
        await pool.restart(chat_id)
        await sessions.clear_session(chat_id)
        await update.message.reply_text(f"Cleared {field} override. Using default. Session restarted.")
    else:
        await sessions.delete_all_user_settings(chat_id)
        pool = _get_pool(context)
        # Revert all four fields to their resolved defaults before
        # restarting (same rationale as single-field reset above).
        for f in ("model", "budget", "timeout", "context_window"):
            _revert_instance_field(pool, chat_id, f, config)
        await pool.restart(chat_id)
        await sessions.clear_session(chat_id)
        await update.message.reply_text("All settings cleared. Using defaults. Session restarted.")


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
            await update.message.reply_text(f"Voice mode: {_VOICE_MODE_LABELS[arg]} (voice: {VOICES[current_voice]})")
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
        await update.message.reply_text(f"Voice mode: {_VOICE_MODE_LABELS[new_mode]} (voice: {VOICES[current_voice]})")


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
    chat_id = _chat_id(update)
    pool = _get_pool(context)
    stats = await sessions.get_stats(chat_id)
    alive = pool.is_alive(chat_id)
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
    # Pass user's chat_id for ownership check - users can only cancel
    # their own jobs (prevents cross-user job manipulation).
    deleted = await sessions.delete_job(job_id, chat_id=_chat_id(update))
    if not deleted:
        await update.message.reply_text(f"Job #{job_id} not found.")
        return
    # Remove from APScheduler's in-memory queue. Daily jobs with multiple
    # times get suffixed names (cron_19_0, cron_19_1), so match both the
    # exact name and any suffixed variants — same pattern as cron.py.
    jq = context.application.job_queue
    assert jq is not None
    prefix = f"cron_{job_id}"
    current = [j for j in jq.jobs() if j.name == prefix or (j.name and j.name.startswith(f"{prefix}_"))]
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
    pool = _get_pool(context)
    stop_event = get_stop_event(chat_id)
    stop_event.set()
    await pool.force_kill(chat_id)
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
    # expanduser() handles ~ in the target path (e.g., "~/Projects/foo")
    resolved = (base / target).expanduser().resolve()
    # Resolve base too so symlinks in the base path don't bypass the check
    resolved_base = base.resolve()
    if not str(resolved).startswith(str(resolved_base) + "/") and resolved != resolved_base:
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


def _workspace_config_suffix(ws_config: WorkspaceConfig | None) -> str:
    """Build a parenthesized suffix showing workspace config details.

    Returns e.g. " (model: opus, budget: $15.00)" or "" if no config.
    """
    extras = []
    if ws_config and ws_config.model:
        extras.append(f"model: {ws_config.model}")
    if ws_config and ws_config.budget is not None:
        extras.append(f"budget: ${ws_config.budget:.2f}")
    return f" ({', '.join(extras)})" if extras else ""


async def _do_switch_workspace(context: ContextTypes.DEFAULT_TYPE, chat_id: int, path: Path) -> WorkspaceConfig | None:
    """
    Core workspace switch logic shared by command and callback handlers.

    Kills the Claude process (it will restart in the new directory on next
    message), clears the session, and persists the new workspace to settings.
    Switching to home deletes the setting (home is the default). Looks up
    per-workspace config from workspaces.yaml and passes it to Claude.

    Returns the WorkspaceConfig for the target workspace (or None) so
    callers can display config details without a redundant lookup.
    """
    pool = _get_pool(context)
    config: Config = context.bot_data["config"]
    home = config.claude_workspace

    # Layer DB overrides (from /workspace config) on top of YAML baseline.
    yaml_config = config.get_workspace_config(path)
    ws_config = await sessions.build_workspace_config(yaml_config, path, chat_id)
    await pool.change_workspace(chat_id, path, workspace_config=ws_config)
    # Per-user file confinement is handled at request time in webhook.py
    # via pool.get_workspace(chat_id), so no global update needed here.
    await sessions.clear_session(chat_id)

    if path == home:
        await sessions.delete_setting(f"workspace:{chat_id}")
    else:
        await sessions.set_setting(f"workspace:{chat_id}", str(path))
        await sessions.upsert_workspace_history(str(path), chat_id)

    return ws_config


async def _switch_workspace(update: Update, context: ContextTypes.DEFAULT_TYPE, path: Path) -> None:
    """
    Switch to a workspace path and send a confirmation reply.

    Wraps _do_switch_workspace with user-facing feedback including workspace
    metadata (git repo detection, CLAUDE.md presence).
    """
    assert update.message is not None
    pool = _get_pool(context)
    config: Config = context.bot_data["config"]
    home = config.claude_workspace

    if path == pool.get_workspace(_chat_id(update)):
        await update.message.reply_text("Already in that workspace.")
        return

    # Guard against directories deleted after startup (matches keyboard path behavior)
    if not path.is_dir():
        await update.message.reply_text("That workspace no longer exists.")
        return

    ws_config = await _do_switch_workspace(context, _chat_id(update), path)

    config_suffix = _workspace_config_suffix(ws_config)

    if path == home:
        await update.message.reply_text(f"Switched to home workspace{config_suffix}. Session cleared.")
    else:
        # Show filesystem metadata alongside config details
        notes = []
        if (path / ".git").is_dir():
            notes.append("Git repo")
        if (path / ".claude" / "CLAUDE.md").exists():
            notes.append("Has CLAUDE.md")
        note_suffix = f" ({', '.join(notes)})" if notes else ""
        await update.message.reply_text(f"Workspace: {path}{note_suffix}{config_suffix}\nSession cleared.")


def _workspaces_keyboard(
    history: list[dict],
    current_path: str,
    home_path: str,
    base: Path | None,
    allowed_workspaces: list[Path],
) -> InlineKeyboardMarkup:
    """
    Build an inline keyboard for workspace switching.

    Layout (top to bottom):
    1. Home button (always first)
    2. Allowed (pinned) workspaces from ALLOWED_WORKSPACES config, in order
    3. Recent workspace history, deduplicated against allowed workspaces and home

    The current workspace is marked with a green dot. Callback data:
    - "ws:home" for the home button
    - "ws:allowed:<index>" for pinned workspaces (index into allowed_workspaces)
    - "ws:<index>" for history entries (index into the history list)
    """
    buttons = []

    # Collect allowed paths as strings for deduplication checks below
    allowed_path_strs = {str(p) for p in allowed_workspaces}

    # Home button (always first)
    home_label = "\U0001f3e0 Home"
    if current_path == home_path:
        home_label += " \U0001f7e2"
    buttons.append([InlineKeyboardButton(home_label, callback_data="ws:home")])

    # Detect name collisions within the allowed list so labels can be disambiguated.
    # If two entries share the same directory name, show "parent/name" instead of "name".
    name_counts: dict[str, int] = {}
    for p in allowed_workspaces:
        name_counts[p.name] = name_counts.get(p.name, 0) + 1
    duplicate_names = {name for name, count in name_counts.items() if count > 1}

    # Pinned workspaces from ALLOWED_WORKSPACES (shown above history)
    for i, p in enumerate(allowed_workspaces):
        if p.name in duplicate_names:
            # Include parent directory name to make the button unambiguous
            short = f"{p.parent.name}/{p.name}"
        else:
            short = _short_workspace_name(str(p), base)
        label = short
        if str(p) == current_path:
            label += " \U0001f7e2"
        buttons.append([InlineKeyboardButton(label, callback_data=f"ws:allowed:{i}")])

    # History entries — skip home and any path already shown in the allowed section
    for i, entry in enumerate(history):
        p = entry["path"]
        if p == home_path or p in allowed_path_strs:
            continue
        short = _short_workspace_name(p, base)
        label = short
        if p == current_path:
            label += " \U0001f7e2"
        buttons.append([InlineKeyboardButton(label, callback_data=f"ws:{i}")])

    return InlineKeyboardMarkup(buttons)


# ── Workspace config (/workspace config) ───────────────────────────


async def _handle_workspace_config(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    target: str,
) -> None:
    """
    Handle /workspace config subcommands.

    Dispatches to show, set, or reset workspace configuration fields.
    All changes apply to the current workspace.
    """
    assert update.message is not None
    chat_id = _chat_id(update)
    pool = _get_pool(context)
    config: Config = context.bot_data["config"]
    workspace = pool.get_workspace(chat_id)
    workspace_str = str(workspace)

    # Parse: "config [field] [value...]"
    parts = target.split(None, 2)  # ["config"], ["config", field], or ["config", field, value]
    field = parts[1].lower() if len(parts) > 1 else None
    value = parts[2] if len(parts) > 2 else None

    # /workspace config - show current settings
    if field is None:
        await _show_workspace_config(update, workspace, config)
        return

    # /workspace config reset [field]
    if field == "reset":
        if value:
            field_to_reset = value.lower()
            await sessions.delete_workspace_config_setting(chat_id, workspace_str, field_to_reset)
            await _apply_config_change(context, chat_id, workspace, config)
            await update.message.reply_text(f"{field_to_reset} reset to default.")
        else:
            await sessions.delete_all_workspace_config(chat_id, workspace_str)
            await _apply_config_change(context, chat_id, workspace, config)
            await update.message.reply_text("All workspace config cleared. Using global defaults.")
        return

    # /workspace config model <name>
    if field == "model":
        if not value:
            await update.message.reply_text("Usage: /workspace config model <haiku|sonnet|opus>")
            return
        if value.lower() not in VALID_MODELS:
            await update.message.reply_text(f"Unknown model. Choose from: {', '.join(sorted(VALID_MODELS))}")
            return
        await sessions.set_workspace_config_setting(chat_id, workspace_str, "model", value.lower())
        await _apply_config_change(context, chat_id, workspace, config)
        await update.message.reply_text(f"Model set to {value.lower()}.")
        return

    # /workspace config budget <n>
    if field == "budget":
        if not value:
            await update.message.reply_text("Usage: /workspace config budget <amount>")
            return
        try:
            budget = float(value)
            if budget <= 0 or not math.isfinite(budget):
                raise ValueError
        except ValueError:
            await update.message.reply_text("Budget must be a positive number.")
            return
        # Enforce ceiling: per-user max_budget from users.yaml, or the
        # global claude_max_budget_usd as fallback. 0 = no ceiling.
        user_config = config.get_user_config(chat_id)
        ceiling = (
            user_config.max_budget
            if user_config and user_config.max_budget is not None
            else config.claude_max_budget_usd
        )
        if ceiling and budget > ceiling:
            await update.message.reply_text(f"Budget cannot exceed ${ceiling:.2f} (admin limit).")
            return
        await sessions.set_workspace_config_setting(chat_id, workspace_str, "budget", str(budget))
        await _apply_config_change(context, chat_id, workspace, config)
        await update.message.reply_text(f"Budget set to ${budget:.2f}.")
        return

    # /workspace config timeout <n>
    if field == "timeout":
        if not value:
            await update.message.reply_text("Usage: /workspace config timeout <seconds>")
            return
        try:
            timeout = int(value)
            if timeout <= 0:
                raise ValueError
        except ValueError:
            await update.message.reply_text("Timeout must be a positive integer (seconds).")
            return
        await sessions.set_workspace_config_setting(chat_id, workspace_str, "timeout", str(timeout))
        await _apply_config_change(context, chat_id, workspace, config)
        await update.message.reply_text(f"Timeout set to {timeout}s.")
        return

    # /workspace config env [KEY=VALUE | -KEY]
    if field == "env":
        changed = await _handle_workspace_env(update, chat_id, workspace_str, value)
        if changed:
            await _apply_config_change(context, chat_id, workspace, config)
        return

    # /workspace config prompt [text | clear]
    if field == "prompt":
        changed = await _handle_workspace_prompt(update, chat_id, workspace_str, value)
        if changed:
            await _apply_config_change(context, chat_id, workspace, config)
        return

    await update.message.reply_text(
        f"Unknown config field: {field}\nFields: model, budget, timeout, env, prompt, reset"
    )


async def _show_workspace_config(
    update: Update,
    workspace: Path,
    config: Config,
) -> None:
    """Display the effective config for the current workspace with source."""
    assert update.message is not None
    chat_id = _chat_id(update)
    yaml_config = config.get_workspace_config(workspace)
    db_settings = await sessions.get_workspace_config_settings(chat_id, str(workspace))

    lines = [f"Config for {workspace.name}:"]

    # Helper to determine source of each field
    def _source(field_name: str) -> str:
        if field_name in db_settings:
            return "user override"
        if yaml_config and getattr(yaml_config, field_name, None) is not None:
            return "workspaces.yaml"
        return "global default"

    # Model
    model = db_settings.get("model") or (yaml_config.model if yaml_config else None) or config.claude_model
    lines.append(f"  Model: {model} ({_source('model')})")

    # Budget
    try:
        budget = (
            float(db_settings["budget"])
            if "budget" in db_settings
            else yaml_config.budget
            if yaml_config and yaml_config.budget is not None
            else config.claude_max_budget_usd
        )
        lines.append(f"  Budget: ${budget:.2f} ({_source('budget')})")
    except (ValueError, TypeError):
        lines.append("  Budget: (corrupted - reset with /workspace config reset budget)")

    # Timeout
    try:
        timeout = (
            int(db_settings["timeout"])
            if "timeout" in db_settings
            else yaml_config.timeout
            if yaml_config and yaml_config.timeout is not None
            else config.claude_timeout_seconds
        )
        lines.append(f"  Timeout: {timeout}s ({_source('timeout')})")
    except (ValueError, TypeError):
        lines.append("  Timeout: (corrupted - reset with /workspace config reset timeout)")

    # Env vars (show keys only, not values - may contain secrets)
    env_keys: list[str] = []
    if yaml_config and yaml_config.env:
        env_keys.extend(yaml_config.env.keys())
    env_corrupted = False
    if "env" in db_settings:
        try:
            db_env = json.loads(db_settings["env"])
            env_keys.extend(k for k in db_env if k not in env_keys)
        except json.JSONDecodeError:
            env_corrupted = True
    if env_corrupted:
        lines.append("  Env vars: (DB override corrupted - reset to clear)")
    elif env_keys:
        lines.append(f"  Env vars: {', '.join(sorted(env_keys))}")

    # System prompt
    prompt = db_settings.get("prompt")
    if prompt:
        preview = prompt[:100] + ("..." if len(prompt) > 100 else "")
        lines.append(f"  Prompt: {preview} (user override)")
    elif yaml_config and yaml_config.system_prompt:
        preview = yaml_config.system_prompt[:100]
        if len(yaml_config.system_prompt) > 100:
            preview += "..."
        lines.append(f"  Prompt: {preview} (workspaces.yaml)")

    await update.message.reply_text("\n".join(lines))


async def _handle_workspace_env(
    update: Update,
    chat_id: int,
    workspace_str: str,
    value: str | None,
) -> bool:
    """Handle /workspace config env subcommands. Returns True if config changed."""
    assert update.message is not None

    # Load existing env vars from the database
    settings = await sessions.get_workspace_config_settings(chat_id, workspace_str)
    env: dict[str, str] = {}
    if "env" in settings:
        try:
            env = json.loads(settings["env"])
        except json.JSONDecodeError:
            # Corrupted entry; start fresh
            env = {}

    # /workspace config env - list current vars
    if not value:
        if not env:
            await update.message.reply_text("No workspace env vars set.")
        else:
            # Show keys only for security
            key_lines = [f"  {k}" for k in sorted(env.keys())]
            await update.message.reply_text("Workspace env vars:\n" + "\n".join(key_lines))
        return False

    # /workspace config env -KEY - remove a var
    if value.startswith("-"):
        key = value[1:]
        if key in env:
            del env[key]
            if env:
                await sessions.set_workspace_config_setting(chat_id, workspace_str, "env", json.dumps(env))
            else:
                await sessions.delete_workspace_config_setting(chat_id, workspace_str, "env")
            await update.message.reply_text(f"Removed {key}.")
            return True
        await update.message.reply_text(f"{key} is not set.")
        return False

    # /workspace config env KEY=VALUE - set a var
    if "=" not in value:
        await update.message.reply_text("Usage: /workspace config env KEY=VALUE")
        return False

    key, val = value.split("=", 1)
    key = key.strip()
    if not key:
        await update.message.reply_text("Key cannot be empty.")
        return False

    env[key] = val
    await sessions.set_workspace_config_setting(chat_id, workspace_str, "env", json.dumps(env))
    await update.message.reply_text(f"Set {key}.")
    return True


async def _handle_workspace_prompt(
    update: Update,
    chat_id: int,
    workspace_str: str,
    value: str | None,
) -> bool:
    """Handle /workspace config prompt subcommands. Returns True if config changed."""
    assert update.message is not None

    # Check for file attachment (caption-based prompt).
    # When a document is attached, Telegram puts the command text in
    # update.message.caption, not update.message.text. CommandHandler
    # handles caption-based dispatch, so the handler fires, but
    # context.args is populated from the caption.
    if update.message.document:
        # Reject oversized files before downloading into memory.
        # Telegram allows up to 20 MB; 100 KB is generous for a
        # system prompt that will be stored in SQLite.
        max_prompt_bytes = 100 * 1024
        file_size = update.message.document.file_size
        if file_size and file_size > max_prompt_bytes:
            await update.message.reply_text(f"File too large ({file_size // 1024}KB). Max prompt file size is 100KB.")
            return False
        file = await update.message.document.get_file()
        raw = await file.download_as_bytearray()
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            await update.message.reply_text("File must be UTF-8 text.")
            return False
        await sessions.set_workspace_config_setting(chat_id, workspace_str, "prompt", content.strip())
        await update.message.reply_text(f"Prompt set from file ({len(content)} chars).")
        return True

    # /workspace config prompt (no value) - show current
    if not value:
        settings = await sessions.get_workspace_config_settings(chat_id, workspace_str)
        prompt = settings.get("prompt")
        if prompt:
            await update.message.reply_text(f"Current prompt:\n{prompt}")
        else:
            await update.message.reply_text("No workspace prompt set.")
        return False

    # /workspace config prompt clear
    if value.strip().lower() == "clear":
        await sessions.delete_workspace_config_setting(chat_id, workspace_str, "prompt")
        await update.message.reply_text("Prompt cleared.")
        return True

    # /workspace config prompt <text>
    await sessions.set_workspace_config_setting(chat_id, workspace_str, "prompt", value.strip())
    await update.message.reply_text("Prompt set.")
    return True


async def _apply_config_change(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    workspace: Path,
    config: Config,
) -> None:
    """
    Rebuild and apply workspace config after a setting change.

    Kills the current Claude process so the next message starts fresh
    with the new config. Reuses change_workspace() since it handles
    the full reset-then-override cycle.
    """
    pool = _get_pool(context)
    yaml_config = config.get_workspace_config(workspace)
    ws_config = await sessions.build_workspace_config(yaml_config, workspace, chat_id)
    await pool.change_workspace(chat_id, workspace, workspace_config=ws_config)


@_require_auth
async def handle_workspaces(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /workspaces - show an inline keyboard of recent workspaces."""
    assert update.message is not None
    chat_id = _chat_id(update)
    config: Config = context.bot_data["config"]
    # Resolve per-user workspace access (workspace_base + allowed list)
    base, allowed = await sessions.resolve_workspace_access(chat_id, config)
    history = await sessions.get_workspace_history(chat_id)
    pool = _get_pool(context)
    current = str(pool.get_workspace(chat_id))
    home = str(config.claude_workspace)

    if not history and not allowed and current == home:
        await update.message.reply_text("No workspace history yet.\nUse /workspace new <name> to create one.")
        return

    keyboard = _workspaces_keyboard(history, current, home, base, allowed)
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
    chat_id = _chat_id(update)
    config: Config = context.bot_data["config"]
    if not _is_authorized(config, _user_id(update)):
        await query.answer("Not authorized.")
        return

    assert query.data is not None
    data = query.data.removeprefix("ws:")
    pool = _get_pool(context)
    home = config.claude_workspace

    # Resolve per-user workspace access for this user
    base, allowed = await sessions.resolve_workspace_access(chat_id, config)

    # Resolve target path from callback data
    if data == "home":
        path = home
        label = "Home"
    elif data.startswith("allowed:"):
        # Pinned workspace from the user's effective allowed list
        try:
            idx = int(data.removeprefix("allowed:"))
        except ValueError:
            await query.answer("Invalid selection.")
            await query.edit_message_text("No change.", reply_markup=InlineKeyboardMarkup([]))
            return
        if idx < 0 or idx >= len(allowed):
            await query.answer("Workspace no longer available.")
            await query.edit_message_text("No change.", reply_markup=InlineKeyboardMarkup([]))
            return
        path = allowed[idx]
        if not path.is_dir():
            await query.answer("That workspace no longer exists.")
            await query.edit_message_text("No change.", reply_markup=InlineKeyboardMarkup([]))
            return
        label = _short_workspace_name(str(path), base)
    else:
        try:
            idx = int(data)
        except ValueError:
            await query.answer("Invalid selection.")
            await query.edit_message_text("No change.", reply_markup=InlineKeyboardMarkup([]))
            return
        history = await sessions.get_workspace_history(chat_id)
        if idx < 0 or idx >= len(history):
            await query.answer("Workspace no longer in history.")
            await query.edit_message_text("No change.", reply_markup=InlineKeyboardMarkup([]))
            return
        path = Path(history[idx]["path"])
        # Reject history entries that are no longer in an allowed workspace
        # source. This handles the case where a path was removed from the
        # user's allowed list after they visited it - the history entry
        # persists but access is revoked.
        if not is_workspace_allowed(path, base, allowed):
            await sessions.delete_workspace_history(str(path), chat_id)
            await query.answer("That workspace is no longer allowed.")
            history = await sessions.get_workspace_history(chat_id)
            keyboard = _workspaces_keyboard(history, str(pool.get_workspace(chat_id)), str(home), base, allowed)
            await query.edit_message_reply_markup(reply_markup=keyboard)
            return
        # Remove stale entries where the directory no longer exists
        if not path.is_dir():
            await sessions.delete_workspace_history(str(path), chat_id)
            await query.answer("That workspace no longer exists.")
            history = await sessions.get_workspace_history(chat_id)
            keyboard = _workspaces_keyboard(history, str(pool.get_workspace(chat_id)), str(home), base, allowed)
            await query.edit_message_reply_markup(reply_markup=keyboard)
            return
        label = _short_workspace_name(str(path), base)

    # Already there — dismiss the keyboard
    if path == pool.get_workspace(chat_id):
        await query.answer()
        await query.edit_message_text("No change.", reply_markup=InlineKeyboardMarkup([]))
        return

    # Switch and confirm, showing any per-workspace config details
    await query.answer()
    ws_config = await _do_switch_workspace(context, _chat_id(update), path)
    suffix = _workspace_config_suffix(ws_config)
    await query.edit_message_text(
        f"Switched to {label}{suffix}. Session cleared.",
        reply_markup=InlineKeyboardMarkup([]),
    )


# ── Workspace allow/deny/allowed ────────────────────────────────────


async def _handle_workspace_allow(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    target: str,
) -> None:
    """Handle /workspace allow <path> - add an allowed workspace."""
    assert update.message is not None
    chat_id = _chat_id(update)
    config: Config = context.bot_data["config"]

    # Parse path from target string ("allow /path/to/dir")
    parts = target.split(None, 1)
    if len(parts) < 2:
        await update.message.reply_text("Usage: /workspace allow <path>")
        return

    raw_path = parts[1].strip()

    # Require fully absolute path. Reject ~ because expanduser() resolves
    # to the bot process's $HOME, not the requesting user's home directory.
    # In multi-user with separate os_user values, ~/projects would point to
    # the wrong location. Requiring / avoids the ambiguity entirely.
    if not raw_path.startswith("/"):
        await update.message.reply_text("Path must be absolute (start with /).")
        return

    # Resolve to canonical form
    resolved = Path(raw_path).resolve()

    # Must exist and be a directory
    if not resolved.is_dir():
        await update.message.reply_text(f"Not a directory: {resolved}")
        return

    # Check for redundancy: already under workspace_base?
    base, allowed = await sessions.resolve_workspace_access(chat_id, config)
    if base:
        resolved_base = base.resolve()
        if str(resolved).startswith(str(resolved_base) + "/") or resolved == resolved_base:
            await update.message.reply_text(
                f"Already covered by your workspace base:\n{base}\n\n"
                "Use /workspace <name> to access directories under it."
            )
            return

    # Check for duplicates (already in the effective list)
    # allowed list is pre-resolved by resolve_workspace_access()
    if resolved in allowed:
        await update.message.reply_text("Already in your allowed list.")
        return

    await sessions.add_allowed_workspace(chat_id, str(resolved))
    await update.message.reply_text(f"Added: {resolved}")


async def _handle_workspace_deny(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    target: str,
) -> None:
    """Handle /workspace deny <path> - remove an allowed workspace."""
    assert update.message is not None
    chat_id = _chat_id(update)

    parts = target.split(None, 1)
    if len(parts) < 2:
        await update.message.reply_text("Usage: /workspace deny <path>")
        return

    raw_path = parts[1].strip()

    # Same absolute-path requirement as _handle_workspace_allow:
    # relative paths resolve against cwd and will never match a stored
    # entry, producing a confusing "not in your list" response.
    if not raw_path.startswith("/"):
        await update.message.reply_text("Path must be absolute (start with /).")
        return

    resolved = Path(raw_path).resolve()

    # Check if this is a user-added path (in the database)
    removed = await sessions.remove_allowed_workspace(chat_id, str(resolved))
    if removed:
        await update.message.reply_text(f"Removed: {resolved}")
    else:
        # Check if it's a global entry (can't be removed via Telegram)
        config: Config = context.bot_data["config"]
        if resolved in [p.resolve() for p in config.allowed_workspaces]:
            await update.message.reply_text("That workspace is configured globally and cannot be removed via Telegram.")
        else:
            await update.message.reply_text("Not in your allowed workspace list.")


async def _handle_workspace_allowed(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle /workspace allowed - list all allowed workspaces."""
    assert update.message is not None
    chat_id = _chat_id(update)
    config: Config = context.bot_data["config"]
    user_config = config.get_user_config(chat_id)

    # Resolve workspace_base: users.yaml > env
    base = user_config.workspace_base if user_config and user_config.workspace_base else config.workspace_base

    # Build allowed list with source attribution. Avoids calling
    # resolve_workspace_access() + get_allowed_workspaces() which
    # would query the DB twice. Only this handler needs attribution.
    db_paths = await sessions.get_allowed_workspaces(chat_id)
    db_path_set = {p.resolve() for p in db_paths}

    # Combined list: DB first, then global, deduplicated
    seen: set[Path] = set()
    allowed: list[Path] = []
    for p in db_paths:
        resolved = p.resolve()
        if resolved not in seen:
            seen.add(resolved)
            allowed.append(resolved)
    for p in config.allowed_workspaces:
        resolved = p.resolve()
        if resolved not in seen:
            seen.add(resolved)
            allowed.append(resolved)

    lines = []
    if base:
        lines.append(f"Workspace base: {base}")
    else:
        lines.append("Workspace base: not set")

    if allowed:
        lines.append("")
        lines.append("Allowed workspaces:")
        for p in allowed:
            source = "you" if p in db_path_set else "global"
            lines.append(f"  {p} ({source})")
    elif base:
        lines.append("\nNo additional allowed paths beyond workspace base.")
    else:
        lines.append("\nNo allowed workspaces configured.")

    if not base and not allowed:
        lines.append("\nAll directories are accessible (permissive mode).")

    await update.message.reply_text("\n".join(lines))


_NO_BASE_MSG = "No workspace base configured. Set workspace_base in users.yaml or WORKSPACE_BASE in .env."


@_require_auth
async def handle_workspace(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle /workspace - show, switch, or create workspaces.

    Subcommands:
        /workspace                - show current workspace
        /workspace home           - switch to home workspace
        /workspace <name>         - switch by name (workspace_base, then allowed list)
        /workspace new <name>     - create a new workspace with git init
        /workspace allow <path>   - add an allowed workspace path
        /workspace deny <path>    - remove an allowed workspace path
        /workspace allowed        - list all allowed workspaces with sources

    Absolute paths and ~ expansion are rejected for security. Name resolution
    checks workspace_base first, then the allowed list (by directory name).
    """
    assert update.message is not None
    chat_id = _chat_id(update)
    pool = _get_pool(context)
    config: Config = context.bot_data["config"]
    home = config.claude_workspace

    # Resolve per-user workspace access (workspace_base + allowed list)
    base, allowed = await sessions.resolve_workspace_access(chat_id, config)

    # No args: show current workspace
    if not context.args:
        current = pool.get_workspace(chat_id)
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

    # Route workspace subcommands. Exact word boundary check
    # (same pattern as "config") to avoid collisions with workspace
    # names like "allowlist" or "denied-access".
    target_lower = target.lower()
    if target_lower == "allow" or target_lower.startswith("allow "):
        await _handle_workspace_allow(update, context, target)
        return
    if target_lower == "deny" or target_lower.startswith("deny "):
        await _handle_workspace_deny(update, context, target)
        return
    if target_lower == "allowed":
        await _handle_workspace_allowed(update, context)
        return

    # Reject absolute paths and ~ expansion for security
    if target.startswith("/") or target.startswith("~"):
        await update.message.reply_text("Absolute paths are not allowed. Use a workspace name.")
        return

    # "new" keyword: create a new workspace directory with git init.
    # Exact word boundary so names like "newsletter" aren't caught.
    if target_lower == "new" or target_lower.startswith("new "):
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
        rc = await proc.wait()
        if rc != 0:
            # Directory was created but git init failed (git missing,
            # permissions, etc.). Warn the user but still switch - the
            # workspace is usable without version control.
            await update.message.reply_text(
                f"Warning: git init failed (exit code {rc}). The workspace was created but has no git repo."
            )
        await _switch_workspace(update, context, resolved)
        return

    # "config" keyword: view or modify workspace settings.
    # Exact word boundary check to avoid collisions with workspace
    # names starting with "config" (e.g., "configs", "config-backup").
    if target_lower == "config" or target_lower.startswith("config "):
        await _handle_workspace_config(update, context, target)
        return

    # Try workspace_base first (base wins on name collision per spec)
    resolved: Path | None = None
    base_candidate = _resolve_workspace_path(target, base)
    if base_candidate is not None and base_candidate.is_dir():
        resolved = base_candidate

    # Fall back to allowed workspaces - match by directory name.
    # Multiple matches means the user needs to pick via /workspaces.
    if resolved is None:
        matches = [p for p in allowed if p.name == target]
        if len(matches) > 1:
            paths = "\n".join(f"  {p}" for p in matches)
            await update.message.reply_text(
                f"Multiple workspaces named '{target}':\n{paths}\nUse /workspaces to pick one."
            )
            return
        resolved = matches[0] if matches else None

    if resolved is None:
        # Give a helpful message if neither source is configured
        if not base and not allowed:
            await update.message.reply_text(_NO_BASE_MSG)
        else:
            await update.message.reply_text(f"Workspace '{target}' not found.")
        return

    await _switch_workspace(update, context, resolved)


# ── GitHub notification settings ─────────────────────────────────────


async def _show_github(update: Update, chat_id: int, config: Config) -> None:
    """Display the user's effective GitHub notification settings with source attribution."""
    assert update.message is not None
    user_config = config.get_user_config(chat_id)

    # GitHub identity (from users.yaml only, not user-settable)
    github_user = user_config.github if user_config else None

    # Resolve effective settings using the same precedence as webhook routing
    effective = await sessions.resolve_github_settings(chat_id, config)

    lines = []
    if github_user:
        lines.append(f"GitHub: {github_user}")
    else:
        lines.append("GitHub: not configured")

    # Notification destination
    notify = effective["notify_chat_id"]
    if notify and notify != chat_id:
        lines.append(f"Notifications: {notify}")
    else:
        lines.append("Notifications: this chat")

    # Feature toggles with source attribution. Read DB settings directly
    # so we can tell the user where each value comes from.
    db_settings = await sessions.get_github_db_settings(chat_id)

    def _toggle_line(
        label: str,
        db_key: str,
        yaml_val: bool | None,
        effective_val: bool,
    ) -> str:
        """Format a toggle line with its source (DB override, yaml, or global default)."""
        state = "on" if effective_val else "off"
        if db_key in db_settings:
            source = "user override"
        elif yaml_val is not None:
            source = "users.yaml"
        else:
            source = "global default"
        return f"{label}: {state} ({source})"

    yaml_pr = user_config.pr_review if user_config else None
    yaml_triage = user_config.issue_triage if user_config else None

    lines.append(
        _toggle_line(
            "PR reviews",
            "pr_review",
            yaml_pr,
            effective["pr_review"],
        )
    )
    lines.append(
        _toggle_line(
            "Issue triage",
            "issue_triage",
            yaml_triage,
            effective["issue_triage"],
        )
    )

    # Subscribed repos (from users.yaml only; self-service is #220)
    repos = effective["repos"]
    if repos:
        lines.append("")
        lines.append("Subscribed repos:")
        for repo in repos:
            lines.append(f"  {repo}")
    else:
        lines.append("\nNo repo subscriptions configured.")

    await update.message.reply_text("\n".join(lines))


async def _handle_github_notify(
    update: Update,
    chat_id: int,
    args: list[str],
) -> None:
    """Handle /github notify <chat_id|reset> - set or clear notification destination."""
    assert update.message is not None

    if not args:
        await update.message.reply_text("Usage: /github notify <chat_id> or /github notify reset")
        return

    value = args[0].lower()

    if value == "reset":
        await sessions.delete_setting(f"github_notify_chat:{chat_id}")
        await update.message.reply_text("Notification destination reset to this chat.")
        return

    # Validate chat_id is a valid integer (can be negative for groups)
    try:
        notify_id = int(value)
    except ValueError:
        await update.message.reply_text("Chat ID must be an integer.")
        return

    await sessions.set_setting(f"github_notify_chat:{chat_id}", str(notify_id))
    # Standard notifications (push, PR opened, etc.) use this immediately.
    # PR reviews and issue triage route through /api/send-message which
    # validates against allowed_user_ids (populated at startup). New group
    # chat IDs require a restart for review/triage delivery.
    await update.message.reply_text(
        f"GitHub notifications will go to chat {notify_id}.\n"
        "Note: PR reviews and issue triage will use this after restart."
    )


async def _handle_github_toggle(
    update: Update,
    chat_id: int,
    field: str,
    args: list[str],
) -> None:
    """Handle /github reviews on|off and /github triage on|off."""
    assert update.message is not None
    label = "PR reviews" if field == "pr_review" else "Issue triage"

    if not args or args[0].lower() not in ("on", "off"):
        # Usage hint uses the subcommand name, not the internal field name
        subcmd = "reviews" if field == "pr_review" else "triage"
        await update.message.reply_text(f"Usage: /github {subcmd} on|off")
        return

    value = args[0].lower() == "on"
    await sessions.set_setting(f"{field}:{chat_id}", "true" if value else "false")
    state = "enabled" if value else "disabled"
    await update.message.reply_text(f"{label} {state}.")


@_require_auth
async def handle_github(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /github - view and manage GitHub notification settings."""
    assert update.message is not None
    chat_id = _chat_id(update)
    config: Config = context.bot_data["config"]

    args = context.args or []
    subcommand = args[0].lower() if args else None

    # No subcommand: display current settings
    if subcommand is None:
        await _show_github(update, chat_id, config)
        return

    if subcommand == "notify":
        await _handle_github_notify(update, chat_id, args[1:])
        return

    if subcommand == "reviews":
        await _handle_github_toggle(update, chat_id, "pr_review", args[1:])
        return

    if subcommand == "triage":
        await _handle_github_toggle(update, chat_id, "issue_triage", args[1:])
        return

    await update.message.reply_text("Unknown subcommand. Try /github for current settings.")


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
            "  POST /api/services/*  (external service proxy)",
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
        "/workspace (or /ws) - Show current workspace\n"
        "/workspace <name> - Switch by name\n"
        "/workspace new <name> - Create + git init + switch\n"
        "/workspace home - Return to default\n"
        "/workspace allow <path> - Add an allowed workspace\n"
        "/workspace deny <path> - Remove an allowed workspace\n"
        "/workspace allowed - List your workspaces\n"
        "/workspaces - Switch workspace (inline buttons)\n"
        "/settings - Show your settings\n"
        "/settings <field> <value> - Change a setting\n"
        "/settings reset - Clear all overrides\n"
        "/github - GitHub notification settings\n"
        "/models - Choose a model\n"
        "/model <name> - Switch model (persists)\n"
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
        f"Unknown command: {(update.message.text or '').split()[0]}\nTry /help for available commands."
    )


# ── Media message handlers ──────────────────────────────────────────


def _save_upload(data: bytes, filename: str, user_id: int | None = None) -> Path:
    """
    Save file bytes to DATA_DIR/files/ with a timestamped name.

    Creates the files/ directory if it doesn't exist. Filenames are prefixed
    with a timestamp to avoid collisions and sanitized to remove slashes and
    spaces. Returns the absolute path to the saved file so Claude can
    reference it in subsequent commands.

    When user_id is provided, files are saved to a per-user subdirectory
    (DATA_DIR/files/{user_id}/) to prevent cross-user file access.
    When None, uses the shared DATA_DIR/files/ directory (backward-
    compatible for single-user deployments).

    Args:
        data: Raw file bytes to write.
        filename: Original filename from Telegram (sanitized before use).
        user_id: Optional Telegram user ID for per-user file isolation.

    Returns:
        Absolute path to the saved file.
    """
    if user_id is not None:
        files_dir = DATA_DIR / "files" / str(user_id)
    else:
        files_dir = DATA_DIR / "files"
    files_dir.mkdir(parents=True, exist_ok=True)

    # Timestamp prefix ensures unique names even if the same file is sent twice
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    # Strip directory components entirely rather than replacing slashes.
    # Path.name returns only the final component, handling "/" and "..".
    safe_name = Path(filename).name.replace(" ", "_")
    if not safe_name:
        safe_name = "unnamed_file"
    dest = files_dir / f"{ts}_{safe_name}"
    dest.write_bytes(data)
    return dest


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

    # TOTP gate: require valid session for content that invokes Claude
    if not await _check_totp(update, context):
        return

    chat_id = _chat_id(update)
    pool = _get_pool(context)
    model = pool.get_model(chat_id)

    # Download the largest available resolution (last in the list)
    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)
    data = await file.download_as_bytearray()
    raw = bytes(data)
    b64 = base64.b64encode(raw).decode()

    # Save to DATA_DIR/files/ so Claude can access the file via shell tools
    saved = _save_upload(raw, f"photo_{photo.file_unique_id}.jpg", user_id=chat_id)

    caption = update.message.caption or "What's in this image?"
    caption += f"\n[File saved to: {saved}]"
    log_message(direction="user", chat_id=chat_id, text=caption, media={"type": "photo"})
    content = [
        {"type": "text", "text": caption},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
    ]

    was_queued = await _notify_if_queued(update, chat_id)
    lock = await _acquire_lock_or_kill(chat_id, pool, update)
    if lock is None:
        return
    try:
        _set_responding(chat_id)
        try:
            await _handle_response(
                update,
                context,
                chat_id,
                _prepend_queue_marker(content) if was_queued else content,
                pool,
                model,
            )
        finally:
            _clear_responding(chat_id)
    finally:
        lock.release()


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
    Handle document (file) uploads -- images, text files, and everything else.

    All files are saved to workspace/files/ so Claude can access them via
    shell tools. Routes based on file extension for content presentation:
    - Image files -- base64-encoded and sent as multi-modal content
    - Text/code files -- decoded as UTF-8 and sent as a code block
    - Other files -- saved to disk, Claude gets the path to work with
    """
    if not update.message or not update.message.document:
        return

    # TOTP gate: require valid session for content that invokes Claude
    if not await _check_totp(update, context):
        return

    doc = update.message.document
    file_name = doc.file_name or "unknown"
    suffix = Path(file_name).suffix.lower()
    caption = update.message.caption or ""

    chat_id = _chat_id(update)
    pool = _get_pool(context)
    model = pool.get_model(chat_id)

    if suffix in _IMAGE_MEDIA_TYPES:
        # Handle images sent as documents (uncompressed upload)
        file = await context.bot.get_file(doc.file_id)
        data = await file.download_as_bytearray()
        raw = bytes(data)
        b64 = base64.b64encode(raw).decode()
        media_type = _IMAGE_MEDIA_TYPES[suffix]

        # Save to DATA_DIR/files/ so Claude can access the file via shell tools
        saved = _save_upload(raw, file_name, user_id=chat_id)
        img_caption = caption or f"What's in this image ({file_name})?"
        img_caption += f"\n[File saved to: {saved}]"

        log_message(
            direction="user",
            chat_id=chat_id,
            text=caption or file_name,
            media={"type": "document", "filename": file_name},
        )
        content = [
            {"type": "text", "text": img_caption},
            {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
        ]
    elif suffix in _TEXT_EXTENSIONS or (doc.mime_type and doc.mime_type.startswith("text/")):
        # Handle text/code files -- decode and wrap in a code block
        file = await context.bot.get_file(doc.file_id)
        data = await file.download_as_bytearray()
        raw = bytes(data)
        try:
            text_content = raw.decode("utf-8")
        except UnicodeDecodeError:
            await update.message.reply_text(f"Couldn't decode {file_name} as text.")
            return

        # Save to DATA_DIR/files/ so Claude can access the file via shell tools
        saved = _save_upload(raw, file_name, user_id=chat_id)
        header = f"File: {file_name}\n```\n{text_content}\n```\n[File saved to: {saved}]"

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
        # Any other file type -- save to disk and tell Claude the path so it
        # can work with the file via shell tools (e.g., unzip, pdftotext, etc.)
        file = await context.bot.get_file(doc.file_id)
        data = await file.download_as_bytearray()
        saved = _save_upload(bytes(data), file_name, user_id=chat_id)

        log_message(
            direction="user",
            chat_id=chat_id,
            text=caption or f"[file: {file_name}]",
            media={"type": "document", "filename": file_name},
        )
        content = (caption or f"File received: {file_name}") + f"\n[File saved to: {saved}]"

    was_queued = await _notify_if_queued(update, chat_id)
    lock = await _acquire_lock_or_kill(chat_id, pool, update)
    if lock is None:
        return
    try:
        _set_responding(chat_id)
        try:
            await _handle_response(
                update,
                context,
                chat_id,
                _prepend_queue_marker(content) if was_queued else content,
                pool,
                model,
            )
        finally:
            _clear_responding(chat_id)
    finally:
        lock.release()


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

    # TOTP gate: require valid session for content that invokes Claude
    if not await _check_totp(update, context):
        return

    chat_id = _chat_id(update)
    pool = _get_pool(context)
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
        direction="user",
        chat_id=chat_id,
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
    model = pool.get_model(chat_id)

    was_queued = await _notify_if_queued(update, chat_id)
    lock = await _acquire_lock_or_kill(chat_id, pool, update)
    if lock is None:
        return
    try:
        _set_responding(chat_id)
        try:
            await _handle_response(
                update,
                context,
                chat_id,
                _prepend_queue_marker(prompt) if was_queued else prompt,
                pool,
                model,
            )
        finally:
            _clear_responding(chat_id)
    finally:
        lock.release()


# ── Main message handler ─────────────────────────────────────────────


async def _check_totp(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """
    Check TOTP authentication if configured. Returns True if the request
    should proceed, False if a challenge was sent or access denied.

    Must be called at the top of any handler that sends user content to
    Claude. For non-text handlers (photo, document, voice), this sends
    the challenge prompt and returns False - the user must then type
    their code as a text message, which handle_message processes.

    Informational commands (/stats, /help, /jobs, etc.) do NOT need
    this gate since they don't invoke Claude with user content.
    """
    if not await asyncio.to_thread(is_totp_configured):
        return True

    assert context.user_data is not None
    assert update.effective_chat is not None
    assert update.message is not None

    totp_cfg: Config = context.bot_data["config"]
    session_min = totp_cfg.totp_session_minutes
    auth_time = context.user_data.get("totp_authenticated_at", 0)
    totp_expired = time.time() - auth_time > session_min * 60

    if not totp_expired:
        # Auth is still valid - refresh the timestamp so the session
        # timeout measures inactivity, not time since login.
        context.user_data["totp_authenticated_at"] = time.time()
        return True

    # Session expired. For non-text messages (photos, documents, voice),
    # just send the challenge prompt. The user must type their code as
    # text, which handle_message will process via the full TOTP gate.
    if not context.user_data.get("totp_pending"):
        challenge_sec = totp_cfg.totp_challenge_seconds
        context.user_data["totp_pending"] = {
            "expires_at": time.time() + challenge_sec,
        }
        await update.message.reply_text("Session expired. Enter code from authenticator.")
    return False


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

    # ── TOTP gate ────────────────────────────────────────────────────────
    # handle_message needs the full TOTP flow: check session, send
    # challenge, AND verify codes. Media handlers use _check_totp()
    # which only handles the first two steps. We keep the expiry check
    # inline here because _check_totp sets totp_pending as a side effect,
    # and we need to read the pending state before that happens.
    if await asyncio.to_thread(is_totp_configured):
        assert context.user_data is not None
        assert update.effective_chat is not None

        totp_cfg: Config = context.bot_data["config"]
        auth_time = context.user_data.get("totp_authenticated_at", 0)
        totp_expired = time.time() - auth_time > totp_cfg.totp_session_minutes * 60

        if totp_expired:
            pending = context.user_data.get("totp_pending")

            if not pending:
                # First message after expiry - send challenge via helper
                await _check_totp(update, context)
                return

            # A challenge is in flight. This text is the TOTP code.
            if time.time() > pending["expires_at"]:
                del context.user_data["totp_pending"]
                await update.message.reply_text("TOTP challenge expired. Send another message to try again.")
                return

            code = update.message.text.strip() if update.message.text else ""

            # Only treat 6-digit ASCII strings as code attempts. Other
            # messages (e.g., normal chat that arrived concurrently with the
            # challenge) are dropped with a brief reminder instead of being
            # fed to verify_code(). This prevents message deletion, spurious
            # "Invalid code" responses, and unnecessary sudo calls.
            # Note: isascii() guard is needed because isdigit() accepts
            # non-ASCII digit characters (superscripts, Arabic-Indic, etc.).
            if not (code.isascii() and code.isdigit() and len(code) == 6):
                await update.effective_chat.send_message("Authentication required. Enter your 6-digit code.")
                return

            # Delete the code message so it doesn't linger in chat
            try:
                await update.message.delete()
            except Exception:
                pass

            # Check global lockout before calling verify_code()
            lockout_remaining = await asyncio.to_thread(get_lockout_remaining)
            if lockout_remaining > 0:
                minutes = math.ceil(lockout_remaining / 60)
                await update.effective_chat.send_message(
                    f"Too many failed attempts. Locked out for {minutes} more minute{'s' if minutes != 1 else ''}."
                )
                return

            lockout_attempts = totp_cfg.totp_lockout_attempts
            lockout_minutes = totp_cfg.totp_lockout_minutes

            if await asyncio.to_thread(verify_code, code, lockout_attempts, lockout_minutes):
                del context.user_data["totp_pending"]
                context.user_data["totp_authenticated_at"] = time.time()
                await update.effective_chat.send_message("Authenticated.")
                return

            # Verification failed
            lockout_remaining = await asyncio.to_thread(get_lockout_remaining)
            if lockout_remaining > 0:
                del context.user_data["totp_pending"]
                await update.effective_chat.send_message(
                    f"Too many failed attempts. Locked out for {lockout_minutes} minutes."
                )
            else:
                remaining = lockout_attempts - await asyncio.to_thread(get_failure_count)
                await update.effective_chat.send_message(f"Invalid code. {remaining} attempt(s) remaining.")
            return

        # Auth is still valid - refresh the timestamp so the session
        # timeout measures inactivity, not time since login.
        context.user_data["totp_authenticated_at"] = time.time()
    # ── End TOTP gate ─────────────────────────────────────────────────────

    chat_id = _chat_id(update)
    prompt = update.message.text
    log_message(direction="user", chat_id=chat_id, text=prompt)
    pool = _get_pool(context)
    model = pool.get_model(chat_id)

    was_queued = await _notify_if_queued(update, chat_id)
    lock = await _acquire_lock_or_kill(chat_id, pool, update)
    if lock is None:
        return
    try:
        _set_responding(chat_id)
        try:
            await _handle_response(
                update,
                context,
                chat_id,
                _prepend_queue_marker(prompt) if was_queued else prompt,
                pool,
                model,
            )
        finally:
            _clear_responding(chat_id)
    finally:
        lock.release()


# ── Streaming response handler ───────────────────────────────────────


async def _handle_response(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    prompt: str | list,
    pool: SubprocessPool,
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

    async def _keep_typing():
        # Loop runs until the task is cancelled via typing_task.cancel().
        # No shared mutable flag needed - task cancellation is the proper
        # async mechanism and avoids fragile closure-captured booleans.
        while True:
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
    stopped_by_user = False

    try:
        # Reset the stop event (in case /stop was sent between messages)
        stop_event = get_stop_event(chat_id)
        stop_event.clear()

        # Stream events from Claude. Pass chat_id so the inner Claude
        # can include it in API calls for correct multi-user routing.
        async for event in pool.send(prompt, chat_id=chat_id):
            # Check for /stop between stream chunks
            if stop_event.is_set():
                stop_event.clear()
                stopped_by_user = True
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
    finally:
        # Always cancel the typing indicator, even if the streaming loop
        # exits with an exception. Without this, a leaked _keep_typing task
        # sends typing indicators to the chat indefinitely.
        typing_task.cancel()
        try:
            await typing_task
        except asyncio.CancelledError:
            pass

    # Handle error cases. Skip the error message if /stop was used -
    # the user already saw the "(stopped)" edit and doesn't need a false alarm.
    # Failed responses are logged to history so that after a session restart,
    # the injected history shows the message was attempted (not unanswered).
    # Without this, Claude sees an unanswered user message in history and may
    # try to address it instead of the current message.
    if final_response is None:
        if stopped_by_user:
            log_message(direction="assistant", chat_id=chat_id, text="[stopped by user]")
        else:
            log_message(direction="assistant", chat_id=chat_id, text="[no response]")
            await update.message.reply_text("Error: No response from Claude")
        return

    if not final_response.success:
        error_text = f"Error: {final_response.error}"
        log_message(direction="assistant", chat_id=chat_id, text=f"[error: {final_response.error}]")
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
            chunks = chunk_text(final_text)
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


def create_bot(config: Config, *, use_webhook: bool = True) -> Application:
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
        use_webhook: If True, suppress the default Updater (updates arrive via
            webhook POST). If False, keep the Updater for long-polling mode.

    Returns:
        A fully configured Telegram Application ready to be started.
    """
    builder = Application.builder().token(config.telegram_bot_token).concurrent_updates(True)

    # PTB's ApplicationBuilder creates an Updater by default. In webhook mode,
    # updates arrive via HTTP POST so the Updater is dead weight - suppress it.
    # In polling mode, the Updater drives the update loop and must be kept.
    if use_webhook:
        builder = builder.updater(None)

    app = builder.build()
    app.bot_data["config"] = config
    app.bot_data["pool"] = SubprocessPool(
        config=config,
        services_info=services.get_available_services(),
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
    app.add_handler(CommandHandler("settings", handle_settings))
    app.add_handler(CommandHandler("workspace", handle_workspace))
    app.add_handler(CommandHandler("ws", handle_workspace))
    app.add_handler(CommandHandler("workspaces", handle_workspaces))
    app.add_handler(CommandHandler("voice", handle_voice_command))
    app.add_handler(CommandHandler("voices", handle_voices))
    app.add_handler(CommandHandler("webhooks", handle_webhooks))
    app.add_handler(CommandHandler("github", handle_github))
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
