"""
Application entry point — initializes all subsystems and runs the Telegram bot.

Provides functionality to:
1. Configure logging with appropriate levels for each subsystem
2. Load configuration and validate environment
3. Initialize the database, Telegram bot, scheduled jobs, and webhook server
4. Restore workspace from previous session
5. Notify the user if a previous response was interrupted by a crash
6. Run the event loop until shutdown (Ctrl+C or SIGTERM)
7. Clean up all resources in the correct order on exit

The startup sequence is:
    1. Load config from .env
    2. Initialize SQLite database
    3. Create the Telegram bot application
    4. Restore previous workspace (if saved in settings table)
    5. Start the Telegram updater (polling)
    6. Register slash commands in Telegram's menu
    7. Load scheduled jobs from database into APScheduler
    8. Start the webhook HTTP server
    9. Check for interrupted-response flag file
    10. Block forever on asyncio.Event().wait()

The shutdown sequence (in the finally block) reverses this order:
    webhook → updater → bot → Claude process → Telegram app → database
"""

import asyncio
import logging
from pathlib import Path

from telegram import BotCommand

from kai import cron, sessions, webhook
from kai.bot import create_bot
from kai.config import PROJECT_ROOT, load_config


def main() -> None:
    """
    Top-level entry point for the Kai bot.

    Sets up logging, loads configuration, and delegates to an async
    initialization function that manages the full application lifecycle.
    Catches KeyboardInterrupt for clean Ctrl+C shutdown and logs any
    unexpected crashes.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    # Silence noisy per-request HTTP logs and APScheduler tick logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("apscheduler.executors.default").setLevel(logging.WARNING)

    config = load_config()
    logging.info("Kai starting (model=%s, users=%s)", config.claude_model, config.allowed_user_ids)

    async def _init_and_run() -> None:
        """
        Async initialization and main event loop.

        Initializes all subsystems (database, bot, scheduler, webhooks),
        restores previous state, and blocks until shutdown. The finally
        block ensures all resources are cleaned up in reverse order.
        """
        await sessions.init_db(config.session_db_path)
        app = create_bot(config)

        # Restore workspace from previous session (persisted in settings table)
        saved_workspace = await sessions.get_setting("workspace")
        if saved_workspace:
            ws_path = Path(saved_workspace)
            base = config.workspace_base
            # Security check: reject saved workspaces outside WORKSPACE_BASE
            if base and not (
                str(ws_path.resolve()).startswith(str(base) + "/")
                or ws_path.resolve() == base
            ):
                logging.warning(
                    "Saved workspace %s is outside WORKSPACE_BASE %s, ignoring",
                    saved_workspace,
                    base,
                )
                await sessions.delete_setting("workspace")
            elif ws_path.is_dir():
                await app.bot_data["claude"].change_workspace(ws_path)
                logging.info("Restored workspace: %s", ws_path)
            else:
                logging.warning("Saved workspace no longer exists: %s", saved_workspace)
                await sessions.delete_setting("workspace")

        try:
            await app.initialize()
            await app.start()
            assert app.updater is not None
            await app.updater.start_polling()

            # Register slash command menu in Telegram's bot command list
            await app.bot.set_my_commands(
                [
                    BotCommand("models", "Choose a model"),
                    BotCommand("model", "Switch model (opus, sonnet, haiku)"),
                    BotCommand("new", "Start a fresh session"),
                    BotCommand("workspace", "Switch working directory"),
                    BotCommand("workspaces", "List recent workspaces"),
                    BotCommand("stop", "Interrupt current response"),
                    BotCommand("stats", "Show session info and cost"),
                    BotCommand("jobs", "List scheduled jobs"),
                    BotCommand("canceljob", "Cancel a scheduled job"),
                    BotCommand("voice", "Toggle voice responses / set voice"),
                    BotCommand("voices", "Choose a voice (inline buttons)"),
                    BotCommand("webhooks", "Show webhook server status"),
                    BotCommand("help", "Show available commands"),
                ]
            )

            # Reload scheduled jobs from the database into APScheduler
            await cron.init_jobs(app)

            # Start webhook and scheduling API server
            await webhook.start(app, config)

            # Check if a previous response was interrupted by a crash/restart.
            # bot.py writes this flag file when it starts processing a message
            # and deletes it when done. If it exists at startup, the process
            # crashed mid-response and the user should be notified.
            flag = PROJECT_ROOT / ".responding_to"
            try:
                chat_id = int(flag.read_text().strip())
                await app.bot.send_message(
                    chat_id, "Sorry, my previous response was interrupted. Please resend your last message."
                )
                logging.info("Notified chat %d of interrupted response", chat_id)
                flag.unlink(missing_ok=True)
            except FileNotFoundError:
                pass
            except Exception as e:
                logging.warning("Failed to send interrupted-response notice: %s", e)
                flag.unlink(missing_ok=True)

            logging.info("Kai is running. Press Ctrl+C to stop.")
            await asyncio.Event().wait()  # Block forever until shutdown signal
        finally:
            # Shutdown in reverse order of startup
            await webhook.stop()
            if app.updater is not None:
                await app.updater.stop()
            await app.stop()
            await app.bot_data["claude"].shutdown()
            await app.shutdown()
            await sessions.close_db()

    try:
        asyncio.run(_init_and_run())
    except KeyboardInterrupt:
        logging.info("Kai stopped.")
    except Exception:
        logging.exception("Kai crashed")


if __name__ == "__main__":
    main()
