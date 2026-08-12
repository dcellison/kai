"""Operator-invoked qualification commands for Kai Workshop."""

from __future__ import annotations

import argparse
import asyncio
import os
import stat
from datetime import timedelta
from pathlib import Path

from telegram import Bot
from telegram.error import TelegramError

from kai.config import DATA_DIR, load_config
from kai.workshop.delivery_authority import (
    DeliveryAuthorityError,
    WorkshopConversationDeliveryAuthority,
)
from kai.workshop.delivery_fragments import WorkshopDeliveryFragments
from kai.workshop.delivery_outbox import (
    QUALIFICATION_PURPOSE,
    DeliveryState,
    DeliveryTargetNotFoundError,
    WorkshopDeliveryOutbox,
)
from kai.workshop.delivery_qualification import DeliveryQualificationError, WorkshopDeliveryQualification
from kai.workshop.domain import DeliveryId
from kai.workshop.store import WorkshopEventStore
from kai.workshop.telegram_delivery import (
    TelegramWorkOutcome,
    WorkshopTelegramDeliveryAdapter,
    WorkshopTelegramDeliveryWorker,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m kai workshop")
    commands = parser.add_subparsers(dest="command", required=True)
    qualification = commands.add_parser(
        "delivery-qualification",
        help="deliberately exercise one durable Telegram delivery",
    )
    actions = qualification.add_subparsers(dest="action", required=True)

    prepare = actions.add_parser("prepare", help="queue the latest canonical Kai reply without sending it")
    prepare.add_argument("--telegram-user-id", required=True, type=int)

    prepare_group = actions.add_parser(
        "prepare-notification-group",
        help="atomically create and queue one outbound-only notification-group qualification",
    )
    prepare_group.add_argument("--telegram-chat-id", required=True, type=int)

    for action in ("status", "run"):
        action_parser = actions.add_parser(action)
        action_parser.add_argument("--delivery-id", required=True)

    interrupted = actions.add_parser(
        "simulate-interruption",
        help="claim without sending so expiry/restart recovery can be qualified",
    )
    interrupted.add_argument("--delivery-id", required=True)
    interrupted.add_argument("--lease-seconds", type=int, default=5, choices=range(1, 301), metavar="1..300")

    authority = commands.add_parser(
        "delivery-authority",
        help="inspect or deactivate the live conversation-delivery authority",
    )
    authority_actions = authority.add_subparsers(dest="action", required=True)
    authority_actions.add_parser("status")
    deactivate = authority_actions.add_parser(
        "deactivate",
        help="deactivate only after all non-terminal work is reconciled",
    )
    deactivate.add_argument(
        "--acknowledge-terminal-failures",
        action="store_true",
        help="explicitly acknowledge retained terminal failure evidence",
    )
    return parser


def _delivery_id(value: str) -> DeliveryId:
    try:
        return DeliveryId(value)
    except (TypeError, ValueError) as exc:
        raise DeliveryQualificationError("Invalid delivery ID") from exc


def _print_state(state: DeliveryState) -> None:
    print(f"Delivery: {state.delivery_id}")
    print(f"Status: {state.status}")
    print(f"Attempts: {state.attempt_count}/{state.max_attempts}")
    if state.last_error_code is not None:
        print(f"Last outcome: {state.last_error_code}")


def _qualification_database(data_dir: Path) -> Path:
    database = data_dir / "kai.db"
    try:
        metadata = database.lstat()
    except FileNotFoundError as exc:
        raise DeliveryQualificationError(
            "The deployed Kai database was not found; set KAI_DATA_DIR to the deployed data directory"
        ) from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise DeliveryQualificationError("The deployed Kai database is not a regular file")
    if metadata.st_uid != os.geteuid():
        raise DeliveryQualificationError("Run this command as the OS account that owns the deployed Kai database")
    return database


async def _run(args: argparse.Namespace) -> int:
    store = await WorkshopEventStore.open(_qualification_database(DATA_DIR))
    try:
        if args.command == "delivery-authority":
            authority = WorkshopConversationDeliveryAuthority(store)
            if args.action == "status":
                active = await authority.active_epoch_in_transaction(required=False)
                print(f"Conversation delivery authority: {'active' if active is not None else 'inactive'}")
                return 0
            await authority.deactivate(
                acknowledge_terminal_failures=args.acknowledge_terminal_failures,
            )
            print("Conversation delivery authority: deactivated")
            print("No delivery work was deleted or reassigned.")
            return 0

        qualification = WorkshopDeliveryQualification(store)
        if args.action == "prepare":
            result = await qualification.prepare(args.telegram_user_id)
            _print_state(result.delivery)
            if result.inserted:
                print("Prepared only; no Telegram message was sent.")
            else:
                print("The latest canonical reply was already prepared; no Telegram message was sent.")
            return 0
        if args.action == "prepare-notification-group":
            result = await qualification.prepare_notification_group(args.telegram_chat_id)
            _print_state(result.delivery)
            if result.inserted:
                print("Prepared notification-group qualification only; no Telegram message was sent.")
            else:
                print("The notification-group qualification was already prepared; no Telegram message was sent.")
            return 0

        delivery_id = _delivery_id(args.delivery_id)
        if args.action == "status":
            _print_state(await qualification.status(delivery_id))
            return 0

        worker_id = f"qualification:{os.getpid()}"
        if args.action == "simulate-interruption":
            claim = await qualification.simulate_interruption(
                delivery_id,
                worker_id=worker_id,
                lease_duration=timedelta(seconds=args.lease_seconds),
            )
            print(f"Delivery: {claim.delivery_id}")
            print("Status: leased (simulated interruption before send)")
            print(f"Attempt: {claim.attempt_number}")
            print(f"Lease expires: {claim.lease_expires_at.isoformat()}")
            print("No Telegram message was sent.")
            return 0

        config = load_config()
        try:
            async with Bot(config.telegram_bot_token) as bot:
                worker = WorkshopTelegramDeliveryWorker(
                    WorkshopDeliveryOutbox(store),
                    WorkshopDeliveryFragments(store),
                    WorkshopTelegramDeliveryAdapter(bot),
                    worker_id=worker_id,
                    purpose=QUALIFICATION_PURPOSE,
                )
                result = await worker.run_delivery(delivery_id)
        except TelegramError as exc:
            raise DeliveryQualificationError("Telegram client initialization failed before delivery") from exc
        state = await qualification.status(delivery_id)
        _print_state(state)
        if result.outcome == TelegramWorkOutcome.SUCCEEDED:
            print("Qualification message delivered through the Workshop outbox.")
            return 0
        if result.outcome == TelegramWorkOutcome.RETRY_SCHEDULED:
            print("Delivery retry scheduled; run this command again after the due time.")
            return 2
        if result.outcome == TelegramWorkOutcome.FAILED:
            print("Delivery failed terminally; automatic resend is disabled.")
            return 1
        print("The selected delivery is not currently claimable; inspect its status.")
        return 2
    finally:
        await store.close()


def cli(args: list[str]) -> None:
    parsed = _parser().parse_args(args)
    try:
        code = asyncio.run(_run(parsed))
    except DeliveryAuthorityError as exc:
        raise SystemExit(f"Workshop delivery authority failed: {exc}") from exc
    except (DeliveryQualificationError, DeliveryTargetNotFoundError) as exc:
        raise SystemExit(f"Workshop delivery qualification failed: {exc}") from exc
    raise SystemExit(code)
