import asyncio
import logging
from pathlib import Path

from telegram import BotCommand

from kai import cron, sessions, webhook
from kai.bot import create_bot
from kai.config import PROJECT_ROOT, load_config


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    # Silence noisy per-request and scheduler logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("apscheduler.executors.default").setLevel(logging.WARNING)

    config = load_config()
    logging.info("Kai starting (model=%s, users=%s)", config.claude_model, config.allowed_user_ids)

    async def _init_and_run() -> None:
        await sessions.init_db(config.session_db_path)
        app = create_bot(config)

        # Restore workspace from previous session
        saved_workspace = await sessions.get_setting("workspace")
        if saved_workspace:
            ws_path = Path(saved_workspace)
            if ws_path.is_dir():
                await app.bot_data["claude"].change_workspace(ws_path)
                logging.info("Restored workspace: %s", ws_path)
            else:
                logging.warning("Saved workspace no longer exists: %s", saved_workspace)
                await sessions.delete_setting("workspace")

        try:
            await app.initialize()
            await app.start()
            await app.updater.start_polling()

            # Register slash command menu in Telegram
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

            # Reload scheduled jobs from the database
            await cron.init_jobs(app)

            # Start webhook and scheduling API server
            await webhook.start(app, config)

            # Notify if previous response was interrupted by a crash/restart
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
            await asyncio.Event().wait()  # run forever
        finally:
            await webhook.stop()
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
