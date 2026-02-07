from __future__ import annotations

import asyncio
import logging
import time

from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from claude import PersistentClaude
import sessions
from config import Config

log = logging.getLogger(__name__)

# Per-chat locks to serialize messages within a conversation
_chat_locks: dict[int, asyncio.Lock] = {}

# Minimum interval between Telegram message edits (seconds)
EDIT_INTERVAL = 2.0


def _get_lock(chat_id: int) -> asyncio.Lock:
    if chat_id not in _chat_locks:
        _chat_locks[chat_id] = asyncio.Lock()
    return _chat_locks[chat_id]


def _is_authorized(config: Config, user_id: int) -> bool:
    return user_id in config.allowed_user_ids


def _truncate_for_telegram(text: str, max_len: int = 4096) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 4] + "\n..."


async def _edit_message_safe(msg, text: str) -> None:
    truncated = _truncate_for_telegram(text)
    try:
        await msg.edit_text(truncated, parse_mode=ParseMode.MARKDOWN)
    except Exception:
        try:
            await msg.edit_text(truncated)
        except Exception:
            pass


async def _send_response(update: Update, text: str) -> None:
    max_len = 4096
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

    for chunk in chunks:
        try:
            await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)
        except Exception:
            await update.message.reply_text(chunk)


def _get_claude(context: ContextTypes.DEFAULT_TYPE) -> PersistentClaude:
    return context.bot_data["claude"]


async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    config: Config = context.bot_data["config"]
    if not _is_authorized(config, update.effective_user.id):
        return
    await update.message.reply_text("Kai is ready. Send me a message.")


async def handle_new(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    config: Config = context.bot_data["config"]
    if not _is_authorized(config, update.effective_user.id):
        return
    claude = _get_claude(context)
    await claude.restart()
    await sessions.clear_session(update.effective_chat.id)
    await update.message.reply_text("Session cleared. Starting fresh.")


async def handle_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    config: Config = context.bot_data["config"]
    if not _is_authorized(config, update.effective_user.id):
        return
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


async def handle_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    config: Config = context.bot_data["config"]
    if not _is_authorized(config, update.effective_user.id):
        return
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


async def handle_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    config: Config = context.bot_data["config"]
    if not _is_authorized(config, update.effective_user.id):
        return
    await update.message.reply_text(
        "/new - Start a fresh session\n"
        "/model <name> - Switch model (opus, sonnet, haiku)\n"
        "/stats - Show session info and cost\n"
        "/help - This message"
    )


async def handle_unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    config: Config = context.bot_data["config"]
    if not _is_authorized(config, update.effective_user.id):
        return
    await update.message.reply_text(
        f"Unknown command: {update.message.text.split()[0]}\n"
        "Try /help for available commands."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    config: Config = context.bot_data["config"]
    if not _is_authorized(config, update.effective_user.id):
        return

    if not update.message or not update.message.text:
        return

    chat_id = update.effective_chat.id
    prompt = update.message.text
    claude = _get_claude(context)
    model = claude.model

    async with _get_lock(chat_id):
        # Keep "typing..." visible until the first message appears
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
                typing_active = False
                typing_task.cancel()
                truncated = _truncate_for_telegram(event.text_so_far)
                try:
                    live_msg = await update.message.reply_text(
                        truncated, parse_mode=ParseMode.MARKDOWN
                    )
                except Exception:
                    live_msg = await update.message.reply_text(truncated)
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

        final_text = final_response.text
        if live_msg:
            if len(final_text) <= 4096:
                if final_text != last_edit_text:
                    await _edit_message_safe(live_msg, final_text)
            else:
                try:
                    await live_msg.delete()
                except Exception:
                    pass
                await _send_response(update, final_text)
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
    app.add_handler(MessageHandler(filters.COMMAND, handle_unknown_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    return app
