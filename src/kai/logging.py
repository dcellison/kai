"""
Structured logging and audit trail for multi-user Kai.

Provides:
1. structlog-based processor pipeline with contextvars for automatic context injection
2. Dual log files: kai.log (operational, JSON, 14d) and audit.log (user actions, JSON, 30d)
3. Context binding via bind_user_context() — sets user_id, session_id, correlation_id
4. Decorators @log_handler / @log_route for automatic entry/exit logging
5. Typed audit functions (audit_user_message, audit_assistant_response, etc.)
6. Session tracking (get_session_id / reset_session_id) — moved here from bot.py

Architecture:
    Entry points (bot.py, webhook.py, cron.py) call bind_user_context() or use decorators.
    structlog's merge_contextvars processor auto-injects user_id etc. into every log line.
    Two file handlers (kai.log + audit.log) and one console handler are configured on
    the stdlib root logger. structlog wraps stdlib so existing getLogger() calls work.

    kai.log:   all events (JSON), 14-day rotation
    audit.log: only kai.audit.* events (JSON), 30-day rotation, filtered via _AuditFilter
    console:   human-readable colored output for interactive runs
"""

import functools
import logging
import time
import uuid
from contextvars import ContextVar
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

import structlog

# ── Context variables ─────────────────────────────────────────────────
# Each asyncio Task inherits a copy of the parent's context, so concurrent
# users get isolated values automatically.

ctx_user_id: ContextVar[int | None] = ContextVar("ctx_user_id", default=None)
ctx_session_id: ContextVar[str | None] = ContextVar("ctx_session_id", default=None)
ctx_correlation_id: ContextVar[str | None] = ContextVar("ctx_correlation_id", default=None)
ctx_operation: ContextVar[str | None] = ContextVar("ctx_operation", default=None)

# ── Per-user session tracking ─────────────────────────────────────────
# Moved here from bot.py to avoid circular imports (decorators need session_id).

# Thread-safety: This dict is only read/written from the main asyncio event
# loop thread (get_session_id/reset_session_id are called from async handlers,
# never from asyncio.to_thread). CPython's GIL protects dict operations, but
# the single-thread guarantee is the actual safety contract.
_user_sessions: dict[int, str] = {}


def get_session_id(user_id: int) -> str:
    """Get or create a session ID for this user."""
    if user_id not in _user_sessions:
        _user_sessions[user_id] = uuid.uuid4().hex[:12]
    return _user_sessions[user_id]


def reset_session_id(user_id: int) -> None:
    """Reset session ID (called on /new, workspace switch)."""
    _user_sessions.pop(user_id, None)


# ── Context binding ───────────────────────────────────────────────────


def bind_user_context(
    user_id: int | None = None,
    session_id: str | None = None,
    operation: str | None = None,
) -> None:
    """Bind user context for the current async task.

    Sets ContextVars and calls structlog.contextvars.bind_contextvars()
    so all subsequent log calls in this task include user_id, session_id, etc.
    Clears any previously bound contextvars first to prevent stale keys
    leaking across nested or retried calls within the same asyncio Task.
    """
    structlog.contextvars.clear_contextvars()
    correlation_id = uuid.uuid4().hex[:10]

    if user_id is not None:
        ctx_user_id.set(user_id)
        if session_id is None:
            session_id = get_session_id(user_id)
    if session_id is not None:
        ctx_session_id.set(session_id)
    ctx_correlation_id.set(correlation_id)
    if operation is not None:
        ctx_operation.set(operation)

    # Bind to structlog so processors inject these automatically
    bound: dict = {}
    if user_id is not None:
        bound["user_id"] = user_id
    if session_id is not None:
        bound["session_id"] = session_id
    bound["correlation_id"] = correlation_id
    if operation is not None:
        bound["operation"] = operation
    structlog.contextvars.bind_contextvars(**bound)


# ── Audit filter ──────────────────────────────────────────────────────


class _AuditFilter(logging.Filter):
    """Only pass log records from kai.audit and its children."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.name == "kai.audit" or record.name.startswith("kai.audit.")


# ── Logging setup ─────────────────────────────────────────────────────


def setup_logging(log_dir: Path, *, debug: bool = False) -> None:
    """Configure structlog + stdlib logging with dual file output.

    Creates three handlers on the root logger:
    - kai.log (JSON, TimedRotatingFileHandler, 14 days)
    - audit.log (JSON, TimedRotatingFileHandler, 30 days, kai.audit.* only)
    - stderr (ConsoleRenderer, colored, human-readable)

    Args:
        log_dir: Directory for log files (created if needed).
        debug: If True, set root logger to DEBUG level.
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    # ── structlog configuration ──────────────────────────────────────
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # ── Formatters ───────────────────────────────────────────────────
    json_formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
        foreign_pre_chain=shared_processors,
    )

    console_formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        foreign_pre_chain=shared_processors,
    )

    # ── Handlers ─────────────────────────────────────────────────────

    # kai.log — operational log (all events), JSON, 14-day rotation
    kai_handler = TimedRotatingFileHandler(
        filename=log_dir / "kai.log",
        when="midnight",
        backupCount=14,
        encoding="utf-8",
    )
    kai_handler.setFormatter(json_formatter)

    # audit.log — audit trail (kai.audit.* only), JSON, 30-day rotation
    audit_handler = TimedRotatingFileHandler(
        filename=log_dir / "audit.log",
        when="midnight",
        backupCount=30,
        encoding="utf-8",
    )
    audit_handler.setFormatter(json_formatter)
    audit_handler.addFilter(_AuditFilter())

    # Console — human-readable colored output
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)

    # ── Root logger ──────────────────────────────────────────────────
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if debug else logging.INFO)
    root.addHandler(kai_handler)
    root.addHandler(audit_handler)
    root.addHandler(console_handler)

    # Silence noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("apscheduler.executors.default").setLevel(logging.WARNING)
    logging.getLogger("aiohttp.access").setLevel(logging.WARNING)


# ── Decorators ────────────────────────────────────────────────────────


def log_handler(name: str):
    """Decorator for Telegram bot handlers.

    Automatically extracts user_id from update.effective_user, binds context,
    and logs handler.completed / handler.failed with duration_ms.
    """

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(update, context, *args, **kwargs):
            user_id = None
            if update.effective_user:
                user_id = update.effective_user.id
            bind_user_context(user_id=user_id, operation=name)
            log = structlog.get_logger("kai.bot")
            t0 = time.monotonic()
            try:
                result = await func(update, context, *args, **kwargs)
                elapsed = int((time.monotonic() - t0) * 1000)
                log.info("handler.completed", handler=name, duration_ms=elapsed)
                return result
            except Exception:
                elapsed = int((time.monotonic() - t0) * 1000)
                log.error("handler.failed", handler=name, duration_ms=elapsed, exc_info=True)
                raise

        return wrapper

    return decorator


def log_route(name: str):
    """Decorator for aiohttp route handlers.

    Binds context (user_id from X-User-Id header if present),
    logs route.completed / route.failed with status/duration_ms.
    """

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(request, *args, **kwargs):
            user_id = None
            raw = request.headers.get("X-User-Id")
            if raw:
                try:
                    user_id = int(raw)
                except ValueError:
                    pass
            bind_user_context(user_id=user_id, operation=name)
            log = structlog.get_logger("kai.webhook")
            t0 = time.monotonic()
            try:
                response = await func(request, *args, **kwargs)
                elapsed = int((time.monotonic() - t0) * 1000)
                log.info(
                    "route.completed",
                    route=name,
                    status=response.status,
                    duration_ms=elapsed,
                    remote=request.remote,
                )
                return response
            except Exception:
                elapsed = int((time.monotonic() - t0) * 1000)
                log.error(
                    "route.failed",
                    route=name,
                    duration_ms=elapsed,
                    remote=request.remote,
                    exc_info=True,
                )
                raise

        return wrapper

    return decorator


# ── Audit functions ───────────────────────────────────────────────────
# All write to kai.audit logger — automatically captured by both kai.log
# and audit.log (via _AuditFilter).

# NOTE: Logger created at import time. structlog handles this via lazy
# configuration, but audit functions must not be called before setup_logging().
_audit = structlog.get_logger("kai.audit")


def audit_user_message(
    user_id: int,
    chat_id: int,
    text: str,
    media: dict | None = None,
) -> None:
    """Log a user message to the audit trail."""
    _audit.info(
        "user.message",
        user_id=user_id,
        chat_id=chat_id,
        text=text[:2000],
        has_media=media is not None,
        media_type=media.get("type") if media else None,
    )


def audit_assistant_response(
    user_id: int,
    chat_id: int,
    text: str,
    *,
    cost_usd: float = 0.0,
    duration_ms: int = 0,
    model: str = "",
    session_id: str | None = None,
) -> None:
    """Log a Claude response to the audit trail."""
    _audit.info(
        "assistant.response",
        user_id=user_id,
        chat_id=chat_id,
        text_length=len(text),
        text_preview=text[:500],
        cost_usd=cost_usd,
        duration_ms=duration_ms,
        model=model,
        claude_session_id=session_id,
    )


def audit_auth_event(user_id: int, event: str, **kwargs) -> None:
    """Log an authentication event (TOTP challenge/success/fail/lockout)."""
    _audit.info(f"auth.{event}", user_id=user_id, **kwargs)


def audit_webhook_event(event_type: str, source: str | None = None, **kwargs) -> None:
    """Log a webhook event (GitHub, generic)."""
    _audit.info(f"webhook.{event_type}", source=source, **kwargs)


def audit_job_event(job_id: int, user_id: int, event: str, **kwargs) -> None:
    """Log a scheduled job event (fired/completed/failed)."""
    _audit.info(f"job.{event}", job_id=job_id, user_id=user_id, **kwargs)


def audit_service_call(
    service_name: str,
    user_id: int | None = None,
    status: int = 0,
    duration_ms: int = 0,
    **kwargs,
) -> None:
    """Log an external service call."""
    _audit.info(
        "service.call",
        service=service_name,
        user_id=user_id,
        status=status,
        duration_ms=duration_ms,
        **kwargs,
    )
