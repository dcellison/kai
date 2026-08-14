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
from kai.workshop.client_access import WorkshopClientAccess, WorkshopClientAccessError
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
from kai.workshop.domain import (
    ChannelId,
    DeliveryId,
    DeviceId,
    EnrollmentGrantId,
    PrincipalId,
    WorkshopId,
)
from kai.workshop.human_provisioning import (
    WorkshopHumanProvisioner,
    WorkshopHumanProvisioningError,
)
from kai.workshop.runtime_assignments import (
    WorkshopRuntimeAssignmentError,
    WorkshopRuntimeAssignmentService,
)
from kai.workshop.runtime_profiles import WorkshopRuntimeProfileError, WorkshopRuntimeProfileRegistry
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

    client_access = commands.add_parser(
        "client-access",
        help="issue or revoke human Workshop client credentials",
    )
    client_actions = client_access.add_subparsers(dest="action", required=True)
    client_actions.add_parser(
        "list-humans",
        help="list canonical humans and their direct channels",
    )
    provision = client_actions.add_parser(
        "provision-human",
        help="create a canonical human and direct channel without transport or runtime access",
    )
    provision.add_argument("--provisioning-key", required=True)
    provision.add_argument("--display-name", required=True)
    provision.add_argument("--role", required=True, choices=("admin", "member"))
    provision.add_argument("--workshop-id")
    client_actions.add_parser(
        "list-runtime-profiles",
        help="list protected transport-neutral runtime profiles",
    )
    assign_runtime = client_actions.add_parser(
        "assign-runtime",
        help="assign one protected runtime profile to a human's direct-channel agent",
    )
    assign_runtime.add_argument("--principal-id", required=True)
    assign_runtime.add_argument("--channel-id", required=True)
    assign_runtime.add_argument("--runtime-profile-id", required=True)
    issue = client_actions.add_parser(
        "issue-enrollment",
        help="issue one short-lived enrollment token for a canonical human",
    )
    issue_identity = issue.add_mutually_exclusive_group(required=True)
    issue_identity.add_argument("--principal-id")
    issue_identity.add_argument("--telegram-user-id", type=int)
    issue.add_argument("--channel-id")
    revoke_device = client_actions.add_parser(
        "revoke-device",
        help="revoke one client device and all of its sessions",
    )
    revoke_device_identity = revoke_device.add_mutually_exclusive_group(required=True)
    revoke_device_identity.add_argument("--principal-id")
    revoke_device_identity.add_argument("--telegram-user-id", type=int)
    revoke_device.add_argument("--device-id", required=True)
    revoke_enrollment = client_actions.add_parser(
        "revoke-enrollment",
        help="revoke one unredeemed enrollment grant",
    )
    revoke_enrollment_identity = revoke_enrollment.add_mutually_exclusive_group(required=True)
    revoke_enrollment_identity.add_argument("--principal-id")
    revoke_enrollment_identity.add_argument("--telegram-user-id", type=int)
    revoke_enrollment.add_argument("--grant-id", required=True)
    return parser


def _delivery_id(value: str) -> DeliveryId:
    try:
        return DeliveryId(value)
    except (TypeError, ValueError) as exc:
        raise DeliveryQualificationError("Invalid delivery ID") from exc


def _device_id(value: str) -> DeviceId:
    try:
        return DeviceId(value)
    except (TypeError, ValueError) as exc:
        raise WorkshopClientAccessError("Invalid device ID") from exc


def _enrollment_grant_id(value: str) -> EnrollmentGrantId:
    try:
        return EnrollmentGrantId(value)
    except (TypeError, ValueError) as exc:
        raise WorkshopClientAccessError("Invalid enrollment grant ID") from exc


def _principal_id(value: str) -> PrincipalId:
    try:
        return PrincipalId(value)
    except (TypeError, ValueError) as exc:
        raise WorkshopClientAccessError("Invalid principal ID") from exc


def _channel_id(value: str) -> ChannelId:
    try:
        return ChannelId(value)
    except (TypeError, ValueError) as exc:
        raise WorkshopClientAccessError("Invalid channel ID") from exc


def _workshop_id(value: str) -> WorkshopId:
    try:
        return WorkshopId(value)
    except (TypeError, ValueError) as exc:
        raise WorkshopHumanProvisioningError("Invalid Workshop ID") from exc


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
        if args.command == "client-access":
            access = WorkshopClientAccess(store)
            if args.action == "list-humans":
                humans = await access.list_humans()
                if not humans:
                    print("No canonical Workshop humans are available.")
                    return 0
                for human in humans:
                    print(f"Human: {human.display_name}")
                    print(f"Principal: {human.principal_id}")
                    if human.direct_channels:
                        for channel_id in human.direct_channels:
                            print(f"Direct channel: {channel_id}")
                    else:
                        print("Direct channel: unavailable")
                return 0
            if args.action == "provision-human":
                human = await WorkshopHumanProvisioner(store).provision(
                    args.provisioning_key,
                    args.display_name,
                    args.role,
                    workshop_id=(_workshop_id(args.workshop_id) if args.workshop_id is not None else None),
                )
                print(f"Human: {human.display_name}")
                print(f"Principal: {human.principal_id}")
                print(f"Workshop: {human.workshop_id}")
                print(f"Direct channel: {human.channel_id}")
                print(f"Role: {human.role}")
                print(f"Provisioning key: {human.provisioning_key}")
                print(f"Status: {'created' if human.created else 'already provisioned'}")
                print("Transport access: not assigned")
                print("Runtime access: not assigned")
                return 0
            if args.action == "list-runtime-profiles":
                profiles = WorkshopRuntimeProfileRegistry.load(load_config()).profiles
                for profile in profiles:
                    print(f"Runtime profile: {profile.profile_id}")
                    print(f"Runtime name: {profile.display_name}")
                    print(f"OS user: {profile.os_user or 'Kai service account'}")
                    print(f"Backend: {profile.backend}")
                    print(f"Provider: {profile.provider}")
                return 0
            if args.action == "assign-runtime":
                profiles = WorkshopRuntimeProfileRegistry.load(load_config())
                assignment = await WorkshopRuntimeAssignmentService(store, profiles).assign(
                    _principal_id(args.principal_id),
                    _channel_id(args.channel_id),
                    args.runtime_profile_id,
                )
                print(f"Principal: {assignment.principal_id}")
                print(f"Direct channel: {assignment.channel_id}")
                print(f"Agent: {assignment.agent_id}")
                print(f"Runtime profile: {assignment.runtime_profile_id}")
                print(f"Status: {'assigned' if assignment.created else 'already assigned'}")
                print("Runtime authority is explicit channel-agent policy, not human or transport identity.")
                return 0
            if args.action == "issue-enrollment":
                if args.principal_id is not None:
                    if args.channel_id is None:
                        raise WorkshopClientAccessError("--channel-id is required with --principal-id")
                    issued = await access.issue_enrollment(
                        _principal_id(args.principal_id),
                        _channel_id(args.channel_id),
                    )
                else:
                    if args.channel_id is not None:
                        raise WorkshopClientAccessError("--channel-id cannot be used with --telegram-user-id")
                    issued = await access.issue_enrollment_for_telegram(args.telegram_user_id)
                print(f"Enrollment: {issued.grant.grant_id}")
                print(f"Channel: {issued.channel_id}")
                print(f"Expires: {issued.grant.expires_at.isoformat()}")
                print(f"Token: {issued.grant.token}")
                print("The token is shown once; Kai stores only its hash.")
                return 0
            if args.action == "revoke-device":
                device_id = _device_id(args.device_id)
                if args.principal_id is not None:
                    await access.revoke_device(_principal_id(args.principal_id), device_id)
                else:
                    await access.revoke_device_for_telegram(args.telegram_user_id, device_id)
                print(f"Device: {device_id}")
                print("Status: revoked (all device sessions revoked)")
                return 0
            grant_id = _enrollment_grant_id(args.grant_id)
            if args.principal_id is not None:
                await access.revoke_enrollment(_principal_id(args.principal_id), grant_id)
            else:
                await access.revoke_enrollment_for_telegram(args.telegram_user_id, grant_id)
            print(f"Enrollment: {grant_id}")
            print("Status: revoked")
            return 0

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
    except WorkshopClientAccessError as exc:
        raise SystemExit(f"Workshop client access failed: {exc}") from exc
    except WorkshopHumanProvisioningError as exc:
        raise SystemExit(f"Workshop human provisioning failed: {exc}") from exc
    except WorkshopRuntimeAssignmentError as exc:
        raise SystemExit(f"Workshop runtime assignment failed: {exc}") from exc
    except WorkshopRuntimeProfileError as exc:
        raise SystemExit(f"Workshop runtime profile failed: {exc}") from exc
    except (DeliveryQualificationError, DeliveryTargetNotFoundError) as exc:
        if parsed.command == "client-access":
            raise SystemExit(f"Workshop client access failed: {exc}") from exc
        raise SystemExit(f"Workshop delivery qualification failed: {exc}") from exc
    raise SystemExit(code)
