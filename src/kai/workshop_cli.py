"""Operator commands for Kai Workshop administration and exports."""

from __future__ import annotations

import argparse
import asyncio
import os
import stat
from pathlib import Path

from kai.config import DATA_DIR, load_config
from kai.workshop.client_access import WorkshopClientAccess, WorkshopClientAccessError
from kai.workshop.domain import (
    ChannelId,
    DeviceId,
    EnrollmentGrantId,
    PrincipalId,
    WorkshopId,
)
from kai.workshop.human_provisioning import (
    WorkshopHumanProvisioner,
    WorkshopHumanProvisioningError,
)
from kai.workshop.integration_notifications import (
    DEFAULT_INTEGRATION_ROUTE,
    GENERIC_INTEGRATION_SOURCE,
    IntegrationNotificationError,
    WorkshopIntegrationNotificationService,
)
from kai.workshop.runtime_assignments import (
    WorkshopRuntimeAssignmentError,
    WorkshopRuntimeAssignmentService,
)
from kai.workshop.runtime_profiles import WorkshopRuntimeProfileError, WorkshopRuntimeProfileRegistry
from kai.workshop.store import WorkshopEventStore
from kai.workshop.transcript_export import (
    CanonicalTranscriptExportError,
    build_canonical_transcript_export,
)


class WorkshopOperatorCommandError(RuntimeError):
    """A Workshop operator command cannot safely access deployed state."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m kai workshop")
    commands = parser.add_subparsers(dest="command", required=True)
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
    transcript = commands.add_parser(
        "transcript",
        help="export canonical Workshop conversation history",
    )
    transcript_actions = transcript.add_subparsers(dest="action", required=True)
    export = transcript_actions.add_parser(
        "export",
        help="write one canonical channel transcript as NDJSON to standard output",
    )
    export.add_argument("--channel-id", required=True)
    integration_route = commands.add_parser(
        "integration-route",
        help="inspect or assign the canonical generic-webhook destination",
    )
    integration_route_actions = integration_route.add_subparsers(dest="action", required=True)
    integration_route_actions.add_parser("status")
    integration_route_actions.add_parser(
        "reconcile",
        help="seed the route from canonical admin policy when unambiguous",
    )
    set_route = integration_route_actions.add_parser(
        "set",
        help="assign the generic/default route to a canonical channel",
    )
    set_route.add_argument("--channel-id", required=True)
    return parser


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


def _transcript_channel_id(value: str) -> ChannelId:
    try:
        return ChannelId(value)
    except (TypeError, ValueError) as exc:
        raise CanonicalTranscriptExportError("Invalid channel ID") from exc


def _workshop_id(value: str) -> WorkshopId:
    try:
        return WorkshopId(value)
    except (TypeError, ValueError) as exc:
        raise WorkshopHumanProvisioningError("Invalid Workshop ID") from exc


def _deployed_database(data_dir: Path) -> Path:
    database = data_dir / "kai.db"
    try:
        metadata = database.lstat()
    except FileNotFoundError as exc:
        raise WorkshopOperatorCommandError(
            "The deployed Kai database was not found; set KAI_DATA_DIR to the deployed data directory"
        ) from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise WorkshopOperatorCommandError("The deployed Kai database is not a regular file")
    if metadata.st_uid != os.geteuid():
        raise WorkshopOperatorCommandError("Run this command as the OS account that owns the deployed Kai database")
    return database


async def _run(args: argparse.Namespace) -> int:
    store = await WorkshopEventStore.open(_deployed_database(DATA_DIR))
    try:
        if args.command == "integration-route":
            service = WorkshopIntegrationNotificationService(store)
            if args.action == "set":
                try:
                    channel_id = ChannelId(args.channel_id)
                except (TypeError, ValueError) as exc:
                    raise IntegrationNotificationError("Invalid canonical channel ID") from exc
                status = await service.set_route(
                    source=GENERIC_INTEGRATION_SOURCE,
                    route_name=DEFAULT_INTEGRATION_ROUTE,
                    channel_id=channel_id,
                )
            elif args.action == "reconcile":
                status = await service.reconcile_default_generic_route()
            else:
                status = await service.route_status(
                    source=GENERIC_INTEGRATION_SOURCE,
                    route_name=DEFAULT_INTEGRATION_ROUTE,
                )
            print(f"Integration route: {status.source}/{status.route_name}")
            print(f"Status: {status.state}")
            print(f"Channel: {status.channel_id or 'unavailable'}")
            print(f"Detail: {status.detail}")
            return 0 if status.state == "active" else 2

        if args.command == "transcript":
            export = await build_canonical_transcript_export(
                store,
                _transcript_channel_id(args.channel_id),
            )
            print(export.ndjson(), end="")
            return 0

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

        raise WorkshopOperatorCommandError(f"Unsupported Workshop command: {args.command}")
    finally:
        await store.close()


def cli(args: list[str]) -> None:
    parsed = _parser().parse_args(args)
    try:
        code = asyncio.run(_run(parsed))
    except WorkshopClientAccessError as exc:
        raise SystemExit(f"Workshop client access failed: {exc}") from exc
    except WorkshopHumanProvisioningError as exc:
        raise SystemExit(f"Workshop human provisioning failed: {exc}") from exc
    except WorkshopRuntimeAssignmentError as exc:
        raise SystemExit(f"Workshop runtime assignment failed: {exc}") from exc
    except WorkshopRuntimeProfileError as exc:
        raise SystemExit(f"Workshop runtime profile failed: {exc}") from exc
    except IntegrationNotificationError as exc:
        raise SystemExit(f"Workshop integration route failed: {exc}") from exc
    except CanonicalTranscriptExportError as exc:
        raise SystemExit(f"Workshop transcript export failed: {exc}") from exc
    except WorkshopOperatorCommandError as exc:
        raise SystemExit(f"Workshop operator command failed: {exc}") from exc
    raise SystemExit(code)
