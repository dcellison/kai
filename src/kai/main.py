"""
Application entry point - initializes all subsystems and runs the Telegram bot.

Provides functionality to:
1. Configure logging with daily rotation and terminal output
2. Load configuration and validate environment
3. Initialize the database, Telegram bot, scheduled jobs, and webhook server
4. Restore workspace from previous session
5. Start the Telegram transport (webhook or polling, depending on config)
6. Notify the user if a previous response was interrupted by a crash
7. Run the event loop until shutdown (Ctrl+C or SIGTERM)
8. Clean up all resources in the correct order on exit

Telegram transport mode is determined by TELEGRAM_WEBHOOK_URL in .env:
    - Set: webhook mode (Telegram POSTs updates to Kai's HTTP server)
    - Unset: polling mode (Kai pulls updates from Telegram's servers)

The startup sequence is:
    1. Load config from .env
    2. Initialize SQLite database
    3. Create the Telegram bot application (with or without Updater)
    4. Restore previous workspace (if saved in settings table)
    5. Initialize the Telegram bot and register slash commands
    6. Load scheduled jobs from database into APScheduler
    7. Start the webhook HTTP server (always runs for scheduling API, GitHub webhooks, etc.)
    8. In webhook mode: register Telegram webhook with the API
       In polling mode: start the Updater's polling loop
    9. Check for interrupted-response flag file
    10. Block forever on asyncio.Event().wait()

The shutdown sequence (in the finally block) reverses this order:
    webhook server -> polling updater (if active) -> bot -> Claude process -> Telegram app -> database
"""

import asyncio
import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from telegram import BotCommand
from telegram.error import NetworkError

from kai import cron, services, sessions, webhook
from kai.bot import _is_workspace_allowed, create_bot
from kai.config import DATA_DIR, PROJECT_ROOT, _read_protected_file, load_config


def setup_logging() -> None:
    """
    Configure root logger with file rotation and terminal output.

    Sets up two handlers on the root logger:
    - TimedRotatingFileHandler: writes to logs/kai.log, rotates at midnight,
      keeps 14 days of dated backups (kai.log.2026-02-12, etc.)
    - StreamHandler: writes to stderr for terminal visibility during `make run`
      (harmless under launchd since there's no terminal attached)

    Creates the logs/ directory if it doesn't already exist.
    """
    # Logs go under DATA_DIR so they're writable even when source is read-only
    log_dir = DATA_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")

    # Daily rotation at midnight, keep 2 weeks of history, use UTF-8 for
    # emoji and non-ASCII content in Claude responses
    file_handler = TimedRotatingFileHandler(
        filename=log_dir / "kai.log",
        when="midnight",
        backupCount=14,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    # Terminal output for interactive runs (make run, manual debugging)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)

    # Silence noisy per-request HTTP logs and APScheduler tick logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("apscheduler.executors.default").setLevel(logging.WARNING)


def main() -> None:
    """
    Top-level entry point for the Kai bot.

    Sets up logging, loads configuration, and delegates to an async
    initialization function that manages the full application lifecycle.
    Catches KeyboardInterrupt for clean Ctrl+C shutdown and logs any
    unexpected crashes.
    """
    setup_logging()

    config = load_config()
    logging.info("Kai starting (model=%s, users=%s)", config.claude_model, config.allowed_user_ids)

    # Load external service definitions. In a protected installation, services.yaml
    # lives in /etc/kai/ (root-owned). Falls back to PROJECT_ROOT for development.
    protected_yaml = _read_protected_file("/etc/kai/services.yaml")
    if protected_yaml:
        loaded = services.load_services_from_string(protected_yaml)
    else:
        loaded = services.load_services(PROJECT_ROOT / "services.yaml")
    if loaded:
        names = ", ".join(loaded.keys())
        logging.info("Loaded %d service(s): %s", len(loaded), names)

    async def _init_and_run() -> None:
        """
        Async initialization and main event loop.

        Initializes all subsystems (database, bot, scheduler, webhooks),
        restores previous state, and blocks until shutdown. The finally
        block ensures all resources are cleaned up in reverse order.
        """
        # Derive transport mode from config: webhook if URL is set, polling otherwise
        use_webhook = config.telegram_webhook_url is not None

        await sessions.init_db(config.session_db_path)
        app = create_bot(config, use_webhook=use_webhook)

        # Restore workspace from previous session (persisted in settings table)
        saved_workspace = await sessions.get_setting("workspace")
        if saved_workspace:
            ws_path = Path(saved_workspace)
            if not _is_workspace_allowed(ws_path, config):
                logging.warning(
                    "Saved workspace %s is not under WORKSPACE_BASE or ALLOWED_WORKSPACES, ignoring",
                    saved_workspace,
                )
                await sessions.delete_setting("workspace")
            elif ws_path.is_dir():
                await app.bot_data["claude"].change_workspace(ws_path)
                logging.info("Restored workspace: %s", ws_path)
            else:
                logging.warning("Saved workspace no longer exists: %s", saved_workspace)
                await sessions.delete_setting("workspace")

        try:
            # Retry initialization if the network isn't ready yet (e.g. after a
            # power outage where DNS may take a while to come back).
            for attempt in range(1, 13):
                try:
                    await app.initialize()
                    break
                except NetworkError:
                    if attempt == 12:
                        raise
                    wait = min(30, 2**attempt)
                    logging.warning(
                        "Network not ready (attempt %d/12), retrying in %ds…",
                        attempt,
                        wait,
                    )
                    await asyncio.sleep(wait)

            await app.start()

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

            # Start the HTTP server (always runs - serves scheduling API, GitHub
            # webhooks, file exchange, and health check regardless of transport mode).
            # In webhook mode, this also registers the Telegram webhook with the API.
            await webhook.start(app, config)
            # webhook.start() initializes the confinement path from config (home workspace).
            # If a non-default workspace was restored above, sync it now so send-file
            # accepts files from the restored workspace. Must come after start() because
            # start() would overwrite any earlier update_workspace() call.
            if app.bot_data["claude"].workspace != config.claude_workspace:
                webhook.update_workspace(str(app.bot_data["claude"].workspace))

            # In polling mode, start the Updater's long-polling loop. PTB's
            # start_polling() automatically calls delete_webhook() first, which
            # cleans up any stale webhook from a previous webhook-mode run.
            if not use_webhook:
                assert app.updater is not None
                await app.updater.start_polling(
                    allowed_updates=["message", "callback_query"],
                )
                logging.info("Polling started")

            # Check if a previous response was interrupted by a crash/restart.
            # bot.py writes this flag file when it starts processing a message
            # and deletes it when done. If it exists at startup, the process
            # crashed mid-response and the user should be notified.
            # Flag file is under DATA_DIR (writable) not PROJECT_ROOT (may be read-only)
            flag = DATA_DIR / ".responding_to"
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
            # Stop the polling Updater if it was running (no-op in webhook mode
            # since the Updater was suppressed at build time)
            if not use_webhook and app.updater:
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
