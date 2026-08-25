"""Telegram webhook ingress hosted by Kai's transport-neutral HTTP listener."""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import re
import time

from aiohttp import web
from telegram import Update
from telegram.ext import Application

from kai import sessions
from kai.config import Config

log = logging.getLogger(__name__)

_BACKGROUND_TASK_DRAIN_TIMEOUT = 30.0
_ERROR_RECENCY_THRESHOLD = 600
_HEALTH_CHECK_INTERVAL = 300
_TELEGRAM_UPDATE_MAX_ATTEMPTS = 5


class TelegramWebhookIngress:
    """Own Telegram webhook routes, durable update dispatch, and API registration."""

    def __init__(self, application: Application, config: Config) -> None:
        webhook_url = config.telegram_webhook_url
        webhook_secret = config.telegram_webhook_secret
        if not webhook_url or not webhook_secret:
            raise RuntimeError("Telegram webhook mode requires a URL and non-empty secret")
        admins = config.get_admins()
        if admins:
            notification_chat_id = admins[0].telegram_id
        elif config.user_configs:
            fallback = next(iter(config.user_configs.values()))
            notification_chat_id = fallback.telegram_id
            log.warning(
                "No admin users defined in users.yaml; using %s (telegram_id: %d) "
                "for Telegram webhook-health notifications.",
                fallback.name,
                fallback.telegram_id,
            )
        else:
            raise RuntimeError("Telegram webhook mode requires at least one configured Telegram user")

        self._application = application
        self._bot = application.bot
        self._webhook_url = webhook_url
        self._webhook_secret = webhook_secret
        self._notification_chat_id = notification_chat_id
        self._registered = False
        self._background_tasks: set[asyncio.Task[None]] = set()
        self._queue_worker_task: asyncio.Task[None] | None = None
        self._queue_worker_active_row_id: int | None = None
        self._health_monitor_task: asyncio.Task[None] | None = None

    def register_routes(self, app: web.Application) -> None:
        """Publish only the Telegram-owned ingress route on a shared app."""
        app.router.add_post("/webhook/telegram", self.handle_update)

    async def start(self) -> None:
        """Recover queued work, register the remote webhook, and monitor health."""
        requeued = await sessions.requeue_processing_telegram_updates()
        if requeued:
            log.info("Requeued %d unfinished Telegram update(s) from previous run", requeued)
        self._ensure_update_queue_worker()

        max_attempts = 5
        for attempt in range(1, max_attempts + 1):
            try:
                await self._bot.set_webhook(
                    url=self._webhook_url,
                    secret_token=self._webhook_secret,
                    allowed_updates=["message", "callback_query"],
                )
                self._registered = True
                log.info("Registered Telegram webhook: %s", self._webhook_url)
                break
            except Exception:
                if attempt == max_attempts:
                    log.exception("Failed to register webhook after %d attempts", max_attempts)
                    raise
                wait = 2**attempt
                log.warning(
                    "Webhook registration attempt %d/%d failed, retrying in %ds",
                    attempt,
                    max_attempts,
                    wait,
                )
                await asyncio.sleep(wait)

        self._health_monitor_task = asyncio.create_task(self._webhook_health_loop())

    async def stop(self) -> None:
        """Stop monitoring, deregister Telegram, and drain accepted updates."""
        if self._health_monitor_task is not None:
            self._health_monitor_task.cancel()
            try:
                await self._health_monitor_task
            except asyncio.CancelledError:
                pass
            self._health_monitor_task = None

        if self._registered:
            try:
                await self._bot.delete_webhook()
                log.info("Deregistered Telegram webhook")
            except Exception:
                log.warning("Failed to deregister Telegram webhook (will re-register on next start)")
            self._registered = False
        await self._drain_background_tasks()

    async def handle_update(self, request: web.Request) -> web.Response:
        """Authenticate and durably enqueue one Telegram webhook update."""
        provided = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
        if not hmac.compare_digest(provided, self._webhook_secret):
            log.warning("Telegram update: invalid secret")
            return web.Response(status=401, text="Invalid secret")

        try:
            data = await request.json()
        except json.JSONDecodeError:
            log.warning("Telegram update: malformed JSON")
            return web.Response(status=200)

        update_id = data.get("update_id") if isinstance(data, dict) else None
        if isinstance(update_id, bool) or not isinstance(update_id, int):
            log.warning("Telegram update: missing or invalid update_id")
            return web.Response(status=200)

        try:
            row_id, _inserted = await sessions.enqueue_telegram_update(update_id, json.dumps(data))
        except Exception:
            log.exception("Failed to persist Telegram update %s", update_id)
            return web.Response(status=500, text="Failed to enqueue update")

        try:
            self._ensure_update_queue_worker()
        except Exception:
            log.exception("Failed to start Telegram update queue worker")
        else:
            try:
                self._dispatch_priority_stop(row_id, data)
            except Exception:
                log.exception("Failed to dispatch priority Telegram /stop row %s", row_id)
        return web.Response(status=200)

    async def _process_queued_update(self, row: sessions.TelegramUpdateQueueRow) -> None:
        row_id = row["id"]
        try:
            data = json.loads(row["payload"])
            update = Update.de_json(data, self._bot)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            log.exception("Discarding malformed queued Telegram update %s", row_id)
            await sessions.discard_telegram_update(row_id, error)
            return

        if update is None:
            await sessions.complete_telegram_update(row_id)
            return

        try:
            await self._application.process_update(update)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if row["attempt_count"] >= _TELEGRAM_UPDATE_MAX_ATTEMPTS:
                log.exception(
                    "Discarding Telegram update queue row %s after %d failed attempt(s)",
                    row_id,
                    row["attempt_count"],
                )
                await sessions.discard_telegram_update(row_id, error)
                return
            log.exception("Telegram update queue row %s failed; retrying later", row_id)
            await sessions.retry_telegram_update(row_id, error)
            return
        await sessions.complete_telegram_update(row_id)

    async def _update_queue_worker(self) -> None:
        try:
            while True:
                row = await sessions.claim_next_telegram_update()
                if row is None:
                    return
                self._queue_worker_active_row_id = row["id"]
                try:
                    await self._process_queued_update(row)
                finally:
                    self._queue_worker_active_row_id = None
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("Telegram update queue worker crashed")

    def _queue_worker_done(self, task: asyncio.Task[None]) -> None:
        if self._queue_worker_task is task:
            self._queue_worker_task = None
        self._background_tasks.discard(task)

    def _ensure_update_queue_worker(self) -> None:
        if self._queue_worker_task is not None and not self._queue_worker_task.done():
            return
        task = asyncio.create_task(self._update_queue_worker())
        self._queue_worker_task = task
        self._background_tasks.add(task)
        task.add_done_callback(self._queue_worker_done)

    @staticmethod
    def is_stop_update(data: object) -> bool:
        """Return whether a raw Telegram update contains a /stop command."""
        if not isinstance(data, dict):
            return False
        message = data.get("message")
        if not isinstance(message, dict):
            return False
        text = message.get("text")
        return isinstance(text, str) and re.match(r"^/stop(?:@[A-Za-z0-9_]+)?(?:\s|$)", text) is not None

    async def _process_priority_update(self, row_id: int) -> None:
        try:
            row = await sessions.claim_telegram_update(row_id)
            if row is not None:
                await self._process_queued_update(row)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("Priority Telegram update queue task crashed for row %s", row_id)

    def _dispatch_priority_stop(self, row_id: int, data: object) -> None:
        if self._queue_worker_active_row_id is None or not self.is_stop_update(data):
            return
        task = asyncio.create_task(self._process_priority_update(row_id))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _drain_background_tasks(self) -> None:
        pending = {task for task in self._background_tasks if not task.done()}
        if not pending:
            return
        log.info("Waiting for %d Telegram webhook task(s) to finish", len(pending))
        done, still_pending = await asyncio.wait(pending, timeout=_BACKGROUND_TASK_DRAIN_TIMEOUT)
        for task in done:
            try:
                task.result()
            except asyncio.CancelledError:
                pass
            except Exception:
                log.exception("Telegram webhook task failed during shutdown")
            finally:
                self._background_tasks.discard(task)
        if still_pending:
            log.warning(
                "Cancelling %d Telegram webhook task(s) after %.1fs shutdown timeout",
                len(still_pending),
                _BACKGROUND_TASK_DRAIN_TIMEOUT,
            )
            for task in still_pending:
                task.cancel()
            await asyncio.gather(*still_pending, return_exceptions=True)
            self._background_tasks.difference_update(still_pending)

    async def _webhook_health_loop(self) -> None:
        await asyncio.sleep(_HEALTH_CHECK_INTERVAL)
        previous_pending = 0
        consecutive_failures = 0
        failure_notified = False
        while True:
            try:
                info = await self._bot.get_webhook_info()
                needs_reregister = False
                reason = ""
                if not info.url:
                    needs_reregister = True
                    reason = "webhook URL is empty"
                elif info.last_error_date:
                    error_age = time.time() - info.last_error_date.timestamp()
                    if error_age < _ERROR_RECENCY_THRESHOLD:
                        needs_reregister = True
                        reason = f"recent error ({int(error_age)}s ago): {info.last_error_message or 'unknown'}"
                current_pending = info.pending_update_count or 0
                if not needs_reregister and current_pending > 0 and previous_pending > 0:
                    needs_reregister = True
                    reason = (
                        f"pending_update_count stuck at {current_pending} (was {previous_pending} on previous check)"
                    )
                previous_pending = current_pending
                if needs_reregister:
                    log.warning("Webhook health: %s - re-registering", reason)
                    await self._bot.delete_webhook()
                    await self._bot.set_webhook(
                        url=self._webhook_url,
                        secret_token=self._webhook_secret,
                        allowed_updates=["message", "callback_query"],
                    )
                    log.info("Webhook re-registered (self-healing)")
                    previous_pending = 0
                consecutive_failures = 0
                failure_notified = False
            except Exception:
                log.exception("Webhook health check failed")
                consecutive_failures += 1
                if consecutive_failures >= 3 and not failure_notified:
                    try:
                        await self._bot.send_message(
                            self._notification_chat_id,
                            "Webhook health monitor has failed 3 consecutive checks. Self-healing may be degraded.",
                        )
                    except Exception:
                        log.warning("Could not send health monitor failure notification")
                    failure_notified = True
            await asyncio.sleep(_HEALTH_CHECK_INTERVAL)


__all__ = ["TelegramWebhookIngress"]
