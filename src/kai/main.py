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
import re
import shutil
from datetime import UTC, datetime, timedelta
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from telegram import BotCommand
from telegram.error import NetworkError

from kai import cron, services, sessions, webhook
from kai.bot import create_bot
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


def _bootstrap_memory() -> None:
    """
    Create MEMORY.md from the example template if it doesn't exist yet.

    Called once at startup. Creates the DATA_DIR/memory/ directory and
    copies MEMORY.md.example from the home workspace as a starting point.
    After this one-time setup, the inner Claude maintains the file.
    """
    memory_dir = DATA_DIR / "memory"
    memory_file = memory_dir / "MEMORY.md"
    if memory_file.exists():
        return

    memory_dir.mkdir(parents=True, exist_ok=True)
    example = PROJECT_ROOT / "home" / ".claude" / "MEMORY.md.example"
    if example.exists():
        shutil.copy2(example, memory_file)
        logging.info("Bootstrapped MEMORY.md from example template")
    else:
        # Create a minimal file so the inner Claude has something to write to
        memory_file.write_text("# Memory\n")
        logging.info("Created empty MEMORY.md (no example template found)")


# ── File cleanup ─────────────────────────────────────────────────────

# Regex to extract the upload timestamp from filenames created by
# _save_upload(): YYYYMMDD_HHMMSS_ffffff_originalname.ext
_UPLOAD_TS_RE = re.compile(r"^(\d{8}_\d{6})_")

# How often the cleanup loop runs (seconds)
_CLEANUP_INTERVAL = 86400  # 24 hours

# Initial delay before the first cleanup run (seconds)
_CLEANUP_STARTUP_DELAY = 120


def _file_age(path: Path) -> datetime | None:
    """
    Extract upload timestamp from a file's name.

    Parses the YYYYMMDD_HHMMSS prefix that _save_upload() prepends to every
    file. Returns None if the filename doesn't match the expected format
    (e.g., files placed manually or by older code paths).
    """
    match = _UPLOAD_TS_RE.match(path.name)
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S").replace(tzinfo=UTC)
        except ValueError:
            return None
    return None


async def _file_cleanup_loop(retention_days: int) -> None:
    """
    Periodically delete uploaded files older than the retention period.

    Runs once after a short startup delay, then every 24 hours. Only
    deletes files whose names contain a parseable timestamp older than
    retention_days. Files without a recognized timestamp prefix are
    left untouched.

    Args:
        retention_days: Delete files older than this many days. Must be > 0.
    """
    await asyncio.sleep(_CLEANUP_STARTUP_DELAY)

    files_dir = DATA_DIR / "files"

    while True:
        if not files_dir.is_dir():
            await asyncio.sleep(_CLEANUP_INTERVAL)
            continue

        try:
            cutoff = datetime.now(UTC) - timedelta(days=retention_days)
            deleted = 0
            errors = 0

            # Walk all files, including per-user subdirectories
            for path in files_dir.rglob("*"):
                if not path.is_file():
                    continue
                ts = _file_age(path)
                if ts is None:
                    # No recognizable timestamp - leave it alone
                    continue
                if ts < cutoff:
                    try:
                        path.unlink()
                        deleted += 1
                    except OSError:
                        errors += 1

            # Remove empty per-user directories left behind after deletion.
            # Only removes immediate subdirectories of files/ (the {chat_id}
            # dirs), not files/ itself.
            for subdir in files_dir.iterdir():
                if subdir.is_dir():
                    try:
                        subdir.rmdir()  # Only succeeds if empty
                    except OSError:
                        pass  # Not empty or permission error - skip

            if deleted or errors:
                logging.info(
                    "File cleanup: deleted %d files older than %d days%s",
                    deleted,
                    retention_days,
                    f" ({errors} errors)" if errors else "",
                )
        except Exception:
            logging.exception("File cleanup error (will retry in %ds)", _CLEANUP_INTERVAL)

        await asyncio.sleep(_CLEANUP_INTERVAL)


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
    logging.info("Kai starting (model=%s, users=%s)", config.default_model, config.allowed_user_ids)

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

        # Determine the default user (admin or first allowed user) for
        # per-user data migrations and workspace restoration. Config
        # validation ensures at least one user exists, but guard against
        # edge cases to avoid a StopIteration crash at startup.
        default_chat_id: int | None = None
        if config.user_configs:
            admins = config.get_admins()
            if admins:
                default_chat_id = admins[0].telegram_id
            else:
                default_chat_id = next(iter(config.user_configs))
        elif config.allowed_user_ids:
            default_chat_id = next(iter(config.allowed_user_ids))

        # One-time migration: rename global "workspace" setting to
        # "workspace:{chat_id}" for per-user namespacing (Phase 2).
        old_workspace = await sessions.get_setting("workspace")
        if old_workspace and default_chat_id is not None:
            await sessions.set_setting(f"workspace:{default_chat_id}", old_workspace)
            await sessions.delete_setting("workspace")
            logging.info("Migrated workspace setting to workspace:%d", default_chat_id)

        # Backfill workspace history rows from pre-Phase-2 (chat_id=0)
        if default_chat_id is not None:
            await sessions.backfill_workspace_history(default_chat_id)

        # Phase 3: per-user workspace restoration is deferred to the
        # SubprocessPool. Each user's workspace is restored lazily on
        # their first message (in pool.send()). No startup restore needed.

        # Bootstrap personal memory if it doesn't exist yet.
        # Non-fatal: a permission or disk error here should not prevent
        # the bot from starting. The memory layer is a nice-to-have.
        try:
            _bootstrap_memory()
        except OSError:
            logging.warning("Could not bootstrap MEMORY.md", exc_info=True)

        # Initialize semantic memory system (Mem0 + Qdrant).
        # Non-fatal: if Mem0 fails to initialize (missing deps, disk error,
        # model download failure), bot runs without semantic memory.
        # Import is deferred so PyTorch/sentence-transformers (~300MB RAM)
        # are not loaded when MEMORY_ENABLED=false.
        if config.memory_enabled:
            try:
                from kai.memory import init_memory
                from kai.memory import is_enabled as memory_is_enabled

                init_memory(config)
                # init_memory may silently disable memory (e.g. dimension
                # mismatch) without raising. Check actual state.
                if memory_is_enabled():
                    logging.info("Semantic memory system initialized")
                else:
                    logging.warning("Semantic memory init completed but system is disabled")
            except Exception:
                logging.warning("Could not initialize semantic memory", exc_info=True)

        # Seed Mem0 with existing MEMORY.md topic-file content on first startup
        # after memory system install. Per-user flag: each user_id gets its own
        # flag so later-added users trigger their own seed on next startup.
        # Import is_enabled fresh here rather than relying on memory_is_enabled
        # from the try block above, which is undefined if the import failed.
        try:
            from kai.memory import is_enabled as _memory_ready
            from kai.memory import seed_from_memory_md

            if config.memory_enabled and _memory_ready():
                # Collect user_ids that still need seeding. Skip any user whose
                # flag is already set (prior successful run for that user).
                user_ids_to_seed: list[str] = []
                for user_id_int in sorted(config.allowed_user_ids):
                    flag_key = f"memory_seeded:{user_id_int}"
                    if await sessions.get_setting(flag_key) is None:
                        user_ids_to_seed.append(str(user_id_int))

                if user_ids_to_seed:
                    # Run the seed in a thread executor: the seed does synchronous
                    # Qdrant I/O per entry and should not block the event loop for
                    # the minute or two the first-run migration takes. Subsequent
                    # runs are no-ops (dedup short-circuits), but we still offload
                    # to keep the startup path non-blocking.
                    loop = asyncio.get_running_loop()
                    counts = await loop.run_in_executor(
                        None,
                        lambda: seed_from_memory_md(user_ids=user_ids_to_seed),
                    )
                    # Set the per-user flag ONLY for users whose seed had no
                    # failures. Users with partial failures will retry on the
                    # next startup, and dedup will skip the already-seeded entries.
                    for user_id_str, user_counts in counts.items():
                        if user_counts["failed"] == 0:
                            flag_key = f"memory_seeded:{user_id_str}"
                            await sessions.set_setting(flag_key, "1")
                            logging.info("Memory seed flag set for user_id=%s", user_id_str)
                        else:
                            logging.warning(
                                "Memory seed for user_id=%s had %d failures; flag NOT set, will retry on next startup",
                                user_id_str,
                                user_counts["failed"],
                            )
        except Exception:
            # Migration failure is non-fatal. The bot runs without seed data;
            # Track 1 exchange ingestion continues to populate the store.
            # Also catches ImportError if mem0 is not installed.
            logging.warning("Memory seed migration failed", exc_info=True)

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
                    # Session
                    BotCommand("stop", "Interrupt current response"),
                    BotCommand("new", "Start a fresh session"),
                    # Model
                    BotCommand("models", "Choose a model"),
                    BotCommand("model", "Switch model directly"),
                    # Settings
                    BotCommand("settings", "Show or change your settings"),
                    # Workspace
                    BotCommand("workspace", "Switch working directory"),
                    BotCommand("workspaces", "List recent workspaces"),
                    # GitHub
                    BotCommand("github", "Show GitHub settings"),
                    # Voice
                    BotCommand("voice", "Toggle voice or set voice name"),
                    BotCommand("voices", "Choose a voice"),
                    # Info
                    BotCommand("stats", "Show session info and cost"),
                    BotCommand("job", "Manage scheduled jobs"),
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
            # Phase 3: per-user file confinement is handled at request
            # time via pool.get_workspace(chat_id) in webhook.py. No
            # global workspace sync needed at startup.

            # Start the subprocess pool's idle eviction task.
            app.bot_data["pool"].start()

            # Start periodic file cleanup if a retention policy is configured.
            if config.file_retention_days > 0:
                cleanup_task = asyncio.create_task(_file_cleanup_loop(config.file_retention_days))
                # Store reference to prevent GC; task self-cancels on loop shutdown
                app.bot_data["cleanup_task"] = cleanup_task

            # In polling mode, start the Updater's long-polling loop. PTB's
            # start_polling() automatically calls delete_webhook() first, which
            # cleans up any stale webhook from a previous webhook-mode run.
            if not use_webhook:
                assert app.updater is not None
                await app.updater.start_polling(
                    allowed_updates=["message", "callback_query"],
                )
                logging.info("Polling started")

            # Check for interrupted responses from a crash/restart.
            # Phase 2: check all files in the .responding directory (per-user
            # flags) instead of the old single .responding_to file.
            responding_dir = DATA_DIR / ".responding"
            try:
                flags = list(responding_dir.iterdir()) if responding_dir.is_dir() else []
            except OSError:
                flags = []
            for flag in flags:
                # Always unlink the flag first: prevents double-notify on
                # restart if send fails, and cleans up malformed files
                # (e.g., OS temp files) that would otherwise persist forever.
                flag.unlink(missing_ok=True)
                try:
                    interrupted_chat_id = int(flag.name)
                    await app.bot.send_message(
                        interrupted_chat_id,
                        "Sorry, my previous response was interrupted. Please resend your last message.",
                    )
                    logging.info("Notified chat %d of interrupted response", interrupted_chat_id)
                except Exception:
                    logging.exception("Failed to process interrupted-response flag: %s", flag.name)

            # Clean up old single-file flag if it exists (one-time migration).
            # Unlink unconditionally (same pattern as the new-style flags)
            # so malformed content doesn't persist across restarts.
            old_flag = DATA_DIR / ".responding_to"
            if old_flag.exists():
                try:
                    old_content = old_flag.read_text().strip()
                    old_flag.unlink(missing_ok=True)
                    old_chat_id = int(old_content)
                    await app.bot.send_message(
                        old_chat_id,
                        "Sorry, my previous response was interrupted. Please resend your last message.",
                    )
                    logging.info("Notified chat %d of interrupted response (old flag)", old_chat_id)
                except Exception:
                    logging.exception("Failed to process old .responding_to flag")
                    old_flag.unlink(missing_ok=True)

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
            await app.bot_data["pool"].shutdown()
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
