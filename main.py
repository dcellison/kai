import asyncio
import logging

from config import load_config
from bot import create_bot
import sessions


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    config = load_config()
    logging.info("Kai starting (model=%s, users=%s)", config.claude_model, config.allowed_user_ids)

    async def _init_and_run() -> None:
        await sessions.init_db(config.session_db_path)
        app = create_bot(config)
        try:
            await app.initialize()
            await app.start()
            await app.updater.start_polling()
            logging.info("Kai is running. Press Ctrl+C to stop.")
            await asyncio.Event().wait()  # run forever
        finally:
            await app.updater.stop()
            await app.stop()
            await app.bot_data["claude"].shutdown()
            await app.shutdown()
            await sessions.close_db()

    try:
        asyncio.run(_init_and_run())
    except KeyboardInterrupt:
        logging.info("Kai stopped.")


if __name__ == "__main__":
    main()
