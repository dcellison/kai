from __future__ import annotations

import asyncio
import functools
import json
import logging
import time
from pathlib import Path

from telegram import Message, Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

import cron
import sessions
from claude import PersistentClaude
from config import Config
from locks import get_lock

log = logging.getLogger(__name__)

# Minimum interval between Telegram message edits (seconds)
EDIT_INTERVAL = 2.0

# Flag file to track in-flight responses
_RESPONDING_FLAG = Path(__file__).parent / ".responding_to"


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


@_require_auth
async def handle_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Usage: /model <opus|sonnet|haiku>")
        return
    model = context.args[0].lower()
    if model not in ("opus", "sonnet", "haiku"):
        await update.message.reply_text("Choose: opus, sonnet, or haiku")
        return
    claude = _get_claude(context)
    claude.model = model
    await claude.restart()
    await sessions.clear_session(update.effective_chat.id)
    await update.message.reply_text(f"Model set to {model}. Session restarted.")


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
async def handle_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "/new - Start a fresh session\n"
        "/model <name> - Switch model (opus, sonnet, haiku)\n"
        "/stats - Show session info and cost\n"
        "/jobs - List scheduled jobs\n"
        "/canceljob <id> - Cancel a job\n"
        "/help - This message"
    )


@_require_auth
async def handle_unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        f"Unknown command: {update.message.text.split()[0]}\n"
        "Try /help for available commands."
    )


@_require_auth
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return

    chat_id = update.effective_chat.id
    prompt = update.message.text
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
    prompt: str,
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

    async for event in claude.send(prompt):
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
        await sessions.save_session(
            chat_id, final_response.session_id, model, final_response.cost_usd
        )

    # Check for new cron job files created by Claude
    new_jobs = await cron.process_cron_files(context.application, chat_id)
    for j in new_jobs:
        log.info("Registered cron job #%d: %s", j["id"], j["name"])

    final_text = final_response.text
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
    app = Application.builder().token(config.telegram_bot_token).build()
    app.bot_data["config"] = config
    app.bot_data["claude"] = PersistentClaude(
        model=config.claude_model,
        workspace=config.claude_workspace,
        max_budget_usd=config.claude_max_budget_usd,
        timeout_seconds=config.claude_timeout_seconds,
    )

    app.add_handler(CommandHandler("start", handle_start))
    app.add_handler(CommandHandler("new", handle_new))
    app.add_handler(CommandHandler("model", handle_model))
    app.add_handler(CommandHandler("stats", handle_stats))
    app.add_handler(CommandHandler("help", handle_help))
    app.add_handler(CommandHandler("jobs", handle_jobs))
    app.add_handler(CommandHandler("canceljob", handle_canceljob))
    app.add_handler(MessageHandler(filters.COMMAND, handle_unknown_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    return app
