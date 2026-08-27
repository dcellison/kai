"""
Application entry point for the Kai core host and configured adapters.

Provides functionality to:
1. Configure logging with daily rotation and terminal output
2. Load configuration and validate environment
3. Initialize the database, transport-neutral core, adapters, jobs, and HTTP server
4. Restore workspace from previous session
5. Start the optional Telegram transport when configured
6. Notify the user if a previous response was interrupted by a crash
7. Run the event loop until shutdown (Ctrl+C or SIGTERM)
8. Clean up all resources in the correct order on exit

When Telegram is enabled, transport mode is determined by TELEGRAM_WEBHOOK_URL:
    - Set: webhook mode (Telegram POSTs updates to Kai's HTTP server)
    - Unset: polling mode (Kai pulls updates from Telegram's servers)

The startup sequence is:
    1. Load config from .env
    2. Initialize SQLite database
    3. Start the transport-neutral runtime and Workshop execution host
    4. Attach the explicit Telegram adapter when enabled
    5. Restore previous workspace (if saved in settings table)
    6. Let an enabled adapter initialize Telegram, register commands, and load jobs
    7. Start enabled adapter-owned conversation and notification delivery workers
    8. Attach the HTTP adapter for Workshop, integrations, and webhook ingress
    9. In webhook mode: register Telegram webhook with the API
       In polling mode: start the Updater's polling loop
    10. Check for interrupted-response flag file
    11. Supervise required core and delivery workers until shutdown

The shutdown sequence (in the finally block) reverses this order:
    HTTP ingress -> core-supervised adapters -> core services -> database
"""

import asyncio
import logging
import os
import re
import signal
import stat
from datetime import UTC, datetime, timedelta
from importlib import import_module
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from types import ModuleType
from typing import cast

from kai import services, sessions
from kai.application_host import KaiApplicationHost
from kai.backend_registry import load_backend_registry
from kai.config import DATA_DIR, PROJECT_ROOT, _read_protected_file, load_config
from kai.http_adapter import HttpAdapter, HttpIngressAdapter
from kai.memory_backup import run_memory_backup
from kai.workshop.bootstrap import BootstrapHuman, BootstrapNotificationChannel
from kai.workshop.delivery_policy import WorkshopDeliveryBindingPolicy
from kai.workshop.domain import RuntimeProfileId
from kai.workshop.initial_provisioning import parse_initial_provisioning
from kai.workshop.runtime_profiles import WorkshopRuntimeProfileRegistry


class _PrivateTimedRotatingFileHandler(TimedRotatingFileHandler):
    """Timed file handler whose active log is always service-private."""

    def _open(self):
        stream = super()._open()
        os.chmod(self.baseFilename, 0o600)
        return stream


def _secure_runtime_log_tree(log_dir: Path) -> None:
    """Create or repair Kai's log tree without following symlinks."""
    if log_dir.is_symlink():
        raise RuntimeError(f"Refusing symlinked log directory: {log_dir}")
    log_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not log_dir.is_dir():
        raise RuntimeError(f"Log path is not a directory: {log_dir}")
    os.chmod(log_dir, 0o700)

    for root, dirs, files in os.walk(log_dir, followlinks=False):
        root_path = Path(root)
        for name in dirs:
            child = root_path / name
            if child.is_symlink():
                raise RuntimeError(f"Refusing symlink in log directory: {child}")
            os.chmod(child, 0o700)
        for name in files:
            child = root_path / name
            child_stat = child.lstat()
            if stat.S_ISLNK(child_stat.st_mode):
                raise RuntimeError(f"Refusing symlink in log directory: {child}")
            if not stat.S_ISREG(child_stat.st_mode):
                raise RuntimeError(f"Refusing non-regular log entry: {child}")
            os.chmod(child, 0o600)


def _workshop_bootstrap_humans(
    config,
    runtime_profiles: WorkshopRuntimeProfileRegistry,
) -> tuple[BootstrapHuman, ...]:
    """Map authorized humans to their interactive direct-channel bindings."""
    return tuple(
        BootstrapHuman(
            display_name=user.name,
            role="admin" if user.role == "admin" else "member",
            transport="telegram",
            external_subject=str(user.telegram_id),
            external_channel_id=str(user.telegram_id),
            runtime_profile_id=runtime_profiles.profile_for_legacy_runtime_key(user.telegram_id).profile_id,
        )
        for user in sorted(config.user_configs.values(), key=lambda user: user.telegram_id)
        if user.runtime_profile_id is None
    )


async def _workshop_bootstrap_notification_channels(config) -> tuple[BootstrapNotificationChannel, ...]:
    """Map effective Telegram group destinations without changing live routing."""
    members_by_group: dict[int, set[str]] = {}
    for user in sorted(config.user_configs.values(), key=lambda item: item.telegram_id):
        effective = await sessions.resolve_github_settings(user.telegram_id, config)
        destination = effective["notify_chat_id"]
        if destination >= 0:
            continue
        members_by_group.setdefault(destination, set()).add(str(user.telegram_id))
    return tuple(
        BootstrapNotificationChannel(
            transport="telegram",
            external_channel_id=str(destination),
            member_external_subjects=tuple(sorted(members)),
        )
        for destination, members in sorted(members_by_group.items())
    )


def _workshop_registered_backend_ids(config) -> frozenset[str]:
    """Resolve the protected registry, with configured dev backends as fallback."""
    registry = load_backend_registry()
    if registry:
        return frozenset(registry)
    configured = {config.default_backend}
    configured.update(user.backend for user in config.user_configs.values() if user.backend)
    return frozenset(configured)


def _load_telegram_adapter_module() -> ModuleType:
    """Load the optional Telegram adapter with an operator-facing failure."""
    try:
        return import_module("kai.telegram_adapter")
    except ModuleNotFoundError as exc:
        if exc.name == "telegram" or (exc.name is not None and exc.name.startswith("telegram.")):
            raise SystemExit(
                "The Telegram adapter is enabled, but its optional dependency is not installed. "
                "Install Kai with the 'telegram' extra or disable Telegram in KAI_ENABLED_ADAPTERS."
            ) from None
        raise


def _delivery_policy(config, telegram_adapter_module: ModuleType | None) -> WorkshopDeliveryBindingPolicy:
    """Compose adapter declarations outside the transport-neutral host."""
    capabilities = (
        (telegram_adapter_module.TELEGRAM_DELIVERY_CAPABILITIES,) if telegram_adapter_module is not None else ()
    )
    return WorkshopDeliveryBindingPolicy(
        frozenset(item.transport for item in capabilities),
        capabilities,
    )


def setup_logging() -> None:
    """
    Configure root logger with file rotation and terminal output.

    Sets up two handlers on the root logger:
    - TimedRotatingFileHandler: writes to logs/kai.log, rotates at midnight,
      keeps 14 days of dated backups (kai.log.2026-02-12, etc.)
    - StreamHandler: writes to stderr for terminal visibility during `make run`
      (harmless under launchd since there's no terminal attached)

    Creates or repairs the service-private logs/ tree before opening a file.
    """
    # Logs go under DATA_DIR so they're writable even when source is read-only
    log_dir = DATA_DIR / "logs"
    _secure_runtime_log_tree(log_dir)

    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")

    # Daily rotation at midnight, keep 2 weeks of history, use UTF-8 for
    # emoji and non-ASCII content in Claude responses
    file_handler = _PrivateTimedRotatingFileHandler(
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

    # Silence noisy per-request HTTP logs and routine scheduler execution logs.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("apscheduler.executors.default").setLevel(logging.WARNING)


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

            # Walk all files, including canonical principal and legacy
            # configured-user subdirectories.
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


# ── Memory backup ────────────────────────────────────────────────────

# How often the backup loop wakes up (seconds). The freshness floor in
# run_memory_backup (MIN_SNAPSHOT_AGE) is what actually spaces the
# snapshots; the wake interval just has to be at most nightly.
_MEMORY_BACKUP_INTERVAL = 86400  # 24 hours

# Initial delay before the first backup attempt (seconds). Long enough
# to stay clear of startup work, short enough that a restarted service
# still snapshots promptly when the last snapshot is stale.
_MEMORY_BACKUP_STARTUP_DELAY = 300


# Budget for finishing in-flight memory work at shutdown. The installed
# launchd wrapper and ExitTimeOut provide a larger bounded generation grace;
# this inner budget remains deliberately short so memory cannot consume it all.
_MEMORY_DRAIN_TIMEOUT_S = 10.0
_LAUNCHER_PARENT_POLL_SECONDS = 0.5


def _configured_launcher_pid() -> int | None:
    """Return the root-owned launcher's generation PID when configured."""
    raw = os.environ.get("KAI_SERVICE_GENERATION", "").strip()
    if not raw.isdigit():
        return None
    pid = int(raw)
    return pid if pid > 1 else None


async def _watch_launcher_parent(parent_pid: int, stop_requested: asyncio.Event) -> None:
    """Request graceful shutdown if launchd kills only the wrapper shell."""
    while not stop_requested.is_set():
        if os.getppid() != parent_pid:
            logging.warning(
                "Kai launcher generation %d disappeared; beginning graceful shutdown",
                parent_pid,
            )
            stop_requested.set()
            return
        try:
            await asyncio.wait_for(
                stop_requested.wait(),
                timeout=_LAUNCHER_PARENT_POLL_SECONDS,
            )
        except TimeoutError:
            continue


async def _drain_pending_memory_work() -> None:
    """
    Await in-flight extraction and episode tasks before the loop dies.

    Both stages of memory ingestion are fire-and-forget tasks
    (`_pending_memory_tasks` holds stage-1 extraction wrappers,
    `_pending_episode_tasks` holds stage-2 episode generation).
    Without this drain, closing the event loop cancels them silently
    and every restart discards whatever the extractor was in the
    middle of learning from the last exchange.

    Runs after `core_host.stop()`: the adapters are down, so no new
    ingestion can start while the drain waits; the loop below only
    has to chase tasks that already existed plus their direct
    descendants. The loop shape (re-collect, wait, repeat) exists
    because completing a stage-1 task can SPAWN a stage-2 task; a
    single `asyncio.wait` over the initial snapshot would drain
    stage 1 and then close the loop on the freshly spawned stage 2.

    On deadline, survivors are cancelled and one WARNING states what
    was dropped; losing a turn's facts on a slow provider call is
    acceptable, losing them silently is the audit finding.
    """
    from kai import memory_extraction
    from kai.conversation_compatibility import _pending_memory_tasks

    loop = asyncio.get_running_loop()
    deadline = loop.time() + _MEMORY_DRAIN_TIMEOUT_S
    while True:
        pending = _pending_memory_tasks | memory_extraction._pending_episode_tasks
        if not pending:
            return
        remaining = deadline - loop.time()
        if remaining <= 0:
            break
        logging.info("Draining %d in-flight memory task(s) before shutdown", len(pending))
        _, not_done = await asyncio.wait(pending, timeout=remaining)
        if not_done:
            break
    leftover = _pending_memory_tasks | memory_extraction._pending_episode_tasks
    for task in leftover:
        task.cancel()
    await asyncio.gather(*leftover, return_exceptions=True)
    logging.warning(
        "Shutdown drain exceeded %.0fs; cancelled %d in-flight memory task(s); "
        "facts or episodes from those turns are lost",
        _MEMORY_DRAIN_TIMEOUT_S,
        len(leftover),
    )


def _close_semantic_memory() -> None:
    """Release Mem0/Qdrant resources and always clear canonical authority."""
    from kai.memory import close_memory, configure_memory_authority

    try:
        close_memory()
    except Exception:
        logging.exception("Semantic memory stopped with an error")
    finally:
        configure_memory_authority(None)


async def _memory_backup_loop() -> None:
    """
    Nightly snapshot of the semantic memory corpus.

    Runs once after a short startup delay, then every 24 hours,
    delegating the actual snapshot, freshness skip, and retention to
    kai.memory_backup.run_memory_backup in a worker thread (the copy is
    blocking I/O). Failures are logged loudly and the loop keeps
    running: one bad night must not end backups.
    """
    await asyncio.sleep(_MEMORY_BACKUP_STARTUP_DELAY)

    while True:
        memory_dir = DATA_DIR / "memory"
        if memory_dir.is_dir():
            try:
                snapshot = await asyncio.to_thread(
                    run_memory_backup,
                    memory_dir,
                    DATA_DIR / "backups" / "memory",
                    datetime.now(UTC),
                )
                if snapshot is not None:
                    logging.info("Memory backup: snapshot written to %s", snapshot)
            except Exception:
                logging.exception("Memory backup FAILED - the memory corpus has no fresh snapshot")
        await asyncio.sleep(_MEMORY_BACKUP_INTERVAL)


def main() -> None:
    """
    Top-level entry point for the Kai bot.

    Sets up logging, then delegates the entire startup and run
    lifecycle to `_start` under a single SystemExit choke point: the
    fail-closed startup gates (config validation, binary resolution,
    users.yaml validation) raise SystemExit with an actionable
    message, but that text only reaches stderr, which launchd
    redirects to a separate file. Without the CRITICAL re-log, the
    main log ends mid-startup with no explanation and a gate exit
    reads as a hang to anyone tailing it. The re-raise keeps the
    exit code and stderr behavior unchanged; clean exits (None or
    integer codes) stay silent.
    """
    setup_logging()

    try:
        _start()
    except SystemExit as e:
        if isinstance(e.code, str) and e.code.strip():
            logging.critical("Startup failed: %s", e.code)
        raise


def _start() -> None:
    """
    Load configuration and run the application lifecycle.

    Loads configuration, then delegates to an async initialization
    function that manages the full application lifecycle. Catches
    KeyboardInterrupt for clean Ctrl+C shutdown and converts unexpected
    crashes into a non-zero process exit after logging them. SystemExit
    propagates to `main`'s choke point, which mirrors gate messages
    into the main log.
    """
    config = load_config()
    logging.info("Kai starting (model=%s, users=%s)", config.default_model, config.allowed_user_ids)
    telegram_adapter_module = _load_telegram_adapter_module() if config.telegram_enabled else None

    # Load external service definitions. In a protected installation, services.yaml
    # lives in /etc/kai/ (root-owned). Falls back to PROJECT_ROOT for development.
    protected_yaml = _read_protected_file("/etc/kai/services.yaml")
    if protected_yaml is not None:
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
        # Derive Telegram transport only when that optional adapter is enabled.
        use_webhook = config.telegram_enabled and config.telegram_webhook_url is not None

        await sessions.init_db(config.session_db_path)

        # Seed the non-authoritative Workshop identity/channel projection.
        # Notification groups are outbound-only channels: they share existing
        # principals but never create an inbound identity or alter live GitHub
        # routing through this migration.
        runtime_profiles = WorkshopRuntimeProfileRegistry.load(config)
        initial_plan = parse_initial_provisioning(config.initial_workshop_provisioning)
        workshop_bootstrap = await sessions.bootstrap_workshop_foundation(
            (_workshop_bootstrap_humans(config, runtime_profiles) if config.telegram_enabled else ()),
            workshop_id=initial_plan.workshop_id if initial_plan is not None else None,
        )
        if initial_plan is not None:
            await sessions.reconcile_initial_workshop_human(
                initial_plan,
                runtime_profiles,
            )
        if config.telegram_enabled:
            for user in sorted(
                config.user_configs.values(),
                key=lambda item: item.telegram_id,
            ):
                if user.runtime_profile_id is None:
                    continue
                await sessions.link_workshop_transport_profile(
                    runtime_profiles,
                    runtime_profile_id=RuntimeProfileId(user.runtime_profile_id),
                    transport="telegram",
                    external_subject=str(user.telegram_id),
                    external_channel_id=str(user.telegram_id),
                )
            notification_channels = await _workshop_bootstrap_notification_channels(config)
            if notification_channels:
                await sessions.bootstrap_workshop_foundation(
                    (),
                    notification_channels=notification_channels,
                    workshop_id=initial_plan.workshop_id if initial_plan is not None else None,
                )
        logging.info(
            "Workshop bootstrap ready (humans=%d, channels=%d, new_events=%d, existing_events=%d)",
            workshop_bootstrap.human_count,
            workshop_bootstrap.channel_count,
            workshop_bootstrap.created_events,
            workshop_bootstrap.existing_events,
        )
        principal_storage = await sessions.load_workshop_principal_storage_registry(runtime_profiles)
        from kai.backend import configure_principal_storage_namespaces

        configure_principal_storage_namespaces(principal_storage)
        logging.info(
            "Workshop principal storage ready (namespaces=%d)",
            len(principal_storage.namespaces),
        )
        internal_api_contexts = await sessions.load_workshop_internal_api_context_registry(runtime_profiles)
        logging.info(
            "Workshop internal API authority ready (contexts=%d)",
            len(internal_api_contexts.contexts),
        )
        channel_history = await sessions.load_workshop_channel_history_registry(runtime_profiles)
        from kai.history import configure_channel_history_namespaces

        configure_channel_history_namespaces(channel_history)
        logging.info(
            "Workshop channel history ready (namespaces=%d)",
            len(channel_history.namespaces),
        )

        # Load user-registered memory projects into the detection
        # cache. Must follow init_db (the rows live in the session
        # DB) and precede message handling, because both detection
        # call sites read the merged registry on the very first
        # turn. After startup the /project handlers keep the cache
        # in lockstep with the DB; this is the only bulk load.
        from kai.memory_projects import load_db_registry

        load_db_registry(await sessions.get_memory_project_rows())

        core_host: KaiApplicationHost | None = None
        telegram_adapter: HttpIngressAdapter | None = None
        cleanup_task: asyncio.Task[None] | None = None
        memory_backup_task: asyncio.Task[None] | None = None
        launcher_watch_task: asyncio.Task[None] | None = None

        # Determine the default compatibility user (admin or first user) for
        # legacy per-user migrations. Workshop-only installations may have no
        # users.yaml entries, in which case these migrations are unnecessary.
        admins = config.get_admins()
        if admins:
            default_chat_id: int | None = admins[0].telegram_id
        elif config.user_configs:
            default_chat_id = next(iter(config.user_configs))
        else:
            default_chat_id = None

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

        execution_state, execution_migration = await sessions.initialize_workshop_execution_state(runtime_profiles)
        logging.info(
            "Workshop canonical execution state ready "
            "(profiles=%d, newly_migrated=%d, settings=%d, workspace_settings=%d, "
            "history=%d, grants=%d)",
            len(execution_state.namespaces),
            execution_migration.newly_migrated,
            execution_migration.settings,
            execution_migration.workspace_settings,
            execution_migration.history,
            execution_migration.grants,
        )
        operational_migration = await sessions.initialize_workshop_operational_state(
            execution_state,
            config,
            runtime_profiles,
        )
        logging.info(
            "Workshop canonical operational state ready "
            "(profiles=%d, newly_migrated=%d, jobs=%d, github_subscriptions=%d)",
            operational_migration.profiles,
            operational_migration.newly_migrated,
            operational_migration.jobs,
            operational_migration.github_subscriptions,
        )
        # Phase 3: per-user workspace restoration is deferred to the
        # SubprocessPool. Each user's workspace is restored lazily on
        # their first message (in pool.send()). No startup restore needed.

        # Personal MEMORY.md is now per-human under memory/<principal_id>/
        # and is bootstrapped lazily by the backend on first session
        # (backend.ensure_user_memory). No startup bootstrap is needed;
        # that call is intentionally gone. See issue #347.

        # Semantic memory (Mem0 + Qdrant). init_memory() is the single
        # entry point; it no-ops when MEMORY_ENABLED is False, so this
        # call is safe to run unconditionally. Structural safeguards
        # (user-only Track 1 embedding, source-weighted retrieval,
        # scoped delete primitive) shipped alongside this re-enable in
        # spec §320 / epic #306; Haiku extraction (Phase 2) is gated
        # separately via MEMORY_EXTRACTION_ENABLED.
        memory_authority_enabled = False
        try:
            from kai.memory import configure_memory_authority, init_memory, is_enabled

            configure_memory_authority(execution_state)
            init_memory(config)
        except Exception:
            logging.warning("Could not initialize semantic memory", exc_info=True)
        else:
            memory_authority_enabled = is_enabled()
            if memory_authority_enabled:
                memory_migration = await sessions.initialize_workshop_memory_authority(execution_state)
                logging.info(
                    "Workshop canonical memory authority ready "
                    "(profiles=%d, newly_migrated=%d, moved=%d, stamped=%d, total=%d)",
                    memory_migration.profiles,
                    memory_migration.newly_migrated,
                    memory_migration.moved,
                    memory_migration.stamped,
                    memory_migration.total,
                )
                try:
                    from kai.memory_scope_review import refresh_scope_review_status

                    scope_status = await asyncio.to_thread(refresh_scope_review_status)
                    observed = scope_status.get("observed", {})
                    logging.info(
                        "Semantic memory scope census ready "
                        "(legacy_default=%d, reviewed_quarantine=%d, unreviewed=%d, invalid=%d)",
                        int(observed.get("raw_legacy_default", 0)),
                        int(observed.get("quarantined", 0)),
                        int(observed.get("unreviewed", 0)),
                        int(observed.get("invalid_quarantined", 0)),
                    )
                except Exception:
                    logging.warning("Could not refresh semantic memory scope census", exc_info=True)

        runtime_key_cutover = await sessions.initialize_workshop_runtime_key_cutover(
            execution_state,
            memory_enabled=memory_authority_enabled,
        )
        logging.info(
            "Workshop canonical runtime-key cutover ready "
            "(profiles=%d, newly_recorded=%d, archived_keys=%d, legacy_reads=disabled)",
            runtime_key_cutover.profiles,
            runtime_key_cutover.newly_recorded,
            runtime_key_cutover.archived_keys,
        )

        try:
            core_host = KaiApplicationHost(
                config=config,
                runtime_profiles=runtime_profiles,
                execution_state=execution_state,
                principal_storage=principal_storage,
                internal_api_contexts=internal_api_contexts,
                services_info=services.get_available_services(),
                registered_backend_ids=_workshop_registered_backend_ids(config),
                delivery_policy=_delivery_policy(config, telegram_adapter_module),
                client_voice_capabilities=(
                    (telegram_adapter_module.voice_capability(config),) if telegram_adapter_module is not None else ()
                ),
            )
            core_services = await core_host.start()
            logging.info("Kai core application host is ready")

            if config.telegram_enabled:
                assert telegram_adapter_module is not None
                telegram_adapter = cast(
                    HttpIngressAdapter,
                    telegram_adapter_module.TelegramAdapter(
                        config,
                        core_services,
                        use_webhook=use_webhook,
                    ),
                )
                await core_host.attach_adapter("telegram", telegram_adapter)

            http_adapter = HttpAdapter(
                config,
                core_host,
                core_services,
                telegram_adapter,
            )
            await core_host.attach_adapter("http", http_adapter)
            # Phase 3: per-user file confinement is handled at request
            # time via pool.get_effective_workspace(chat_id) in
            # webhook.py. No global workspace sync needed at startup.

            # Start periodic file cleanup if a retention policy is configured.
            if config.file_retention_days > 0:
                cleanup_task = asyncio.create_task(_file_cleanup_loop(config.file_retention_days))

            # Start nightly memory backups. Unconditional: the loop
            # no-ops when DATA_DIR/memory does not exist, and gating on
            # memory_enabled would leave the MEMORY.md fallback store
            # (used when memory is disabled) without backups.
            memory_backup_task = asyncio.create_task(_memory_backup_loop())

            logging.info("Kai is running. Press Ctrl+C to stop.")
            # launchd stops the service with SIGTERM (the launcher
            # script forwards it to this process group). Python's
            # default SIGTERM action terminates the interpreter
            # WITHOUT running finally blocks, so before this handler
            # existed every service stop skipped the shutdown path
            # below entirely; the graceful path only ever ran on
            # Ctrl+C. The handler turns SIGTERM into an event so the
            # supervision wait returns and the finally block (host
            # stop, memory drain, db close) actually executes.
            stop_requested = asyncio.Event()
            asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, stop_requested.set)
            if launcher_pid := _configured_launcher_pid():
                launcher_watch_task = asyncio.create_task(
                    _watch_launcher_parent(launcher_pid, stop_requested),
                    name="kai-launcher-parent-watch",
                )
            host_supervision = asyncio.create_task(core_host.wait())
            sigterm_wait = asyncio.create_task(stop_requested.wait())
            done, pending = await asyncio.wait(
                {host_supervision, sigterm_wait},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            if sigterm_wait in done:
                logging.info("SIGTERM received; shutting down")
            # Preserve the pre-existing failure semantics: a
            # supervised worker's crash must still propagate out of
            # _init_and_run as an exception, not be flattened into a
            # clean-looking exit by the wait restructure.
            if host_supervision in done:
                supervision_error = host_supervision.exception()
                if supervision_error is not None:
                    raise supervision_error
        finally:
            # Shutdown in reverse order of startup
            if launcher_watch_task is not None:
                launcher_watch_task.cancel()
                await asyncio.gather(launcher_watch_task, return_exceptions=True)
            if core_host is not None:
                try:
                    await core_host.stop()
                except Exception:
                    logging.exception("Kai core application host stopped with an error")
            if cleanup_task is not None:
                cleanup_task.cancel()
                await asyncio.gather(cleanup_task, return_exceptions=True)
            if memory_backup_task is not None:
                memory_backup_task.cancel()
                await asyncio.gather(memory_backup_task, return_exceptions=True)
            # After host stop (no new ingestion), before the db and
            # memory authority go away (a draining extraction still
            # stores through them). Guarded like core_host.stop()
            # above: an exception here must not skip close_db.
            try:
                await _drain_pending_memory_work()
            except Exception:
                logging.exception("Shutdown memory drain failed")
            _close_semantic_memory()
            await sessions.close_db()

    try:
        asyncio.run(_init_and_run())
    except KeyboardInterrupt:
        logging.info("Kai stopped.")
    except Exception as exc:
        logging.exception("Kai crashed")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
