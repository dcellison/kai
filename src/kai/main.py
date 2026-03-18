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
from pathlib import Path

import structlog
from telegram import BotCommand
from telegram.error import NetworkError

from kai import cron, services, sessions, webhook
from kai.bot import _is_workspace_allowed, create_bot
from kai.config import DATA_DIR, PROJECT_ROOT, _read_protected_file, load_config
from kai.logging import setup_logging


def main() -> None:
    """
    Top-level entry point for the Kai bot.

    Sets up logging, loads configuration, and delegates to an async
    initialization function that manages the full application lifecycle.
    Catches KeyboardInterrupt for clean Ctrl+C shutdown and logs any
    unexpected crashes.
    """
    setup_logging(DATA_DIR / "logs")

    log = structlog.get_logger("kai.main")
    config = load_config()
    log.info("startup", model=config.claude_model, users=list(config.allowed_user_ids))

    # Load external service definitions. In a protected installation, services.yaml
    # lives in /etc/kai/ (root-owned). Falls back to PROJECT_ROOT for development.
    protected_yaml = _read_protected_file("/etc/kai/services.yaml")
    if protected_yaml:
        loaded = services.load_services_from_string(protected_yaml)
    else:
        loaded = services.load_services(PROJECT_ROOT / "services.yaml")
    if loaded:
        log.info("services.loaded", count=len(loaded), names=list(loaded.keys()))

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

        # Restore per-user workspaces from previous session
        manager = app.bot_data["claude_manager"]
        for uid in config.allowed_user_ids:
            saved_workspace = await sessions.get_setting(uid, "workspace")
            if saved_workspace:
                ws_path = Path(saved_workspace)
                if not _is_workspace_allowed(ws_path, config):
                    log.warning("workspace.disallowed", user_id=uid, workspace=saved_workspace)
                    await sessions.delete_setting(uid, "workspace")
                elif ws_path.is_dir():
                    claude = manager.get_or_create(uid)
                    ws_config = config.get_workspace_config(ws_path)
                    await claude.change_workspace(ws_path, workspace_config=ws_config)
                    webhook.update_workspace(uid, str(ws_path))
                    log.info("workspace.restored", user_id=uid, workspace=str(ws_path))
                else:
                    log.warning("workspace.missing", user_id=uid, workspace=saved_workspace)
                    await sessions.delete_setting(uid, "workspace")

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
                    log.warning("network.retry", attempt=attempt, wait_seconds=wait)
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
                    BotCommand("notifications", "Toggle notification preferences"),
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
            # If non-default workspaces were restored above, sync them now so send-file
            # accepts files from the restored workspaces. Must come after start() because
            # start() would overwrite any earlier update_workspace() call.
            for uid in config.allowed_user_ids:
                saved = await sessions.get_setting(uid, "workspace")
                if saved:
                    webhook.update_workspace(uid, saved)
                else:
                    webhook.update_workspace(uid, str(DATA_DIR / "users" / str(uid) / "home"))

            # In polling mode, start the Updater's long-polling loop. PTB's
            # start_polling() automatically calls delete_webhook() first, which
            # cleans up any stale webhook from a previous webhook-mode run.
            if not use_webhook:
                assert app.updater is not None
                await app.updater.start_polling(
                    allowed_updates=["message", "callback_query"],
                )
                log.info("polling.started")

            # Check if any previous responses were interrupted by a crash/restart.
            # bot.py writes per-user flag files when it starts processing a message
            # and deletes them when done. If they exist at startup, the process
            # crashed mid-response and the affected users should be notified.
            users_dir = DATA_DIR / "users"
            if users_dir.exists():
                for user_dir in users_dir.iterdir():
                    flag = user_dir / ".responding_to"
                    if flag.exists():
                        try:
                            chat_id = int(flag.read_text().strip())
                            await app.bot.send_message(
                                chat_id, "Sorry, my previous response was interrupted. Please resend your last message."
                            )
                            log.info("crash_recovery.notified", chat_id=chat_id)
                            flag.unlink(missing_ok=True)
                        except Exception:
                            log.exception("crash_recovery.failed")
                            flag.unlink(missing_ok=True)

            # Start periodic idle Claude instance eviction
            async def _idle_eviction_loop():
                while True:
                    await asyncio.sleep(1800)  # Every 30 minutes
                    count = await manager.evict_idle()
                    if count:
                        log.info("eviction.completed", count=count)
                    removed = await manager.cleanup_old_files()
                    if removed:
                        log.info("files.cleanup_completed", removed=removed)

            eviction_task = asyncio.create_task(_idle_eviction_loop())

            log.info("ready", mode="webhook" if use_webhook else "polling")
            await asyncio.Event().wait()  # Block forever until shutdown signal
        finally:
            # Cancel the eviction task
            try:
                eviction_task.cancel()
            except NameError:
                pass
            # Shutdown in reverse order of startup
            await webhook.stop()
            # Stop the polling Updater if it was running (no-op in webhook mode
            # since the Updater was suppressed at build time)
            if not use_webhook and app.updater:
                await app.updater.stop()
            await app.stop()
            await app.bot_data["claude_manager"].shutdown_all()
            await app.shutdown()
            await sessions.close_db()

    try:
        asyncio.run(_init_and_run())
    except KeyboardInterrupt:
        log.info("shutdown")
    except Exception:
        log.exception("crashed")


if __name__ == "__main__":
    main()
