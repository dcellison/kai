"""Operator commands for Kai Workshop administration and exports."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pwd
import stat
from pathlib import Path

from kai import sessions
from kai.backend_registry import load_backend_registry
from kai.config import DATA_DIR, load_config, models_for_backend_policy
from kai.workshop.claude_model_discovery import ClaudeModelDiscoveryAdapter
from kai.workshop.client_access import WorkshopClientAccess, WorkshopClientAccessError
from kai.workshop.codex_model_discovery import CodexModelDiscoveryAdapter
from kai.workshop.delivery_policy import WorkshopDeliveryBindingPolicy
from kai.workshop.domain import (
    ChannelId,
    DeviceId,
    EnrollmentGrantId,
    PrincipalId,
    WorkshopId,
)
from kai.workshop.execution_state import WorkshopExecutionStateRegistry
from kai.workshop.goose_model_discovery import GooseModelDiscoveryAdapter
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
from kai.workshop.model_catalogue import (
    ModelCatalogueError,
    ModelCatalogueRefreshResult,
    WorkshopModelCatalogueService,
)
from kai.workshop.model_discovery_inventory import (
    ModelDiscoveryBackendInventory,
    WorkshopModelDiscoveryInventoryService,
)
from kai.workshop.opencode_model_discovery import OpenCodeModelDiscoveryAdapter
from kai.workshop.pi_model_discovery import PiModelDiscoveryAdapter
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
    catalogue = commands.add_parser(
        "model-catalogue",
        help="inspect, refresh, or administer canonical model catalogues",
    )
    catalogue_actions = catalogue.add_subparsers(dest="action", required=True)
    catalogue_actions.add_parser("status", help="show aggregate catalogue state without discovery")
    catalogue_list = catalogue_actions.add_parser("list", help="list one protected catalogue lane")
    _add_catalogue_lane_arguments(catalogue_list)
    catalogue_refresh = catalogue_actions.add_parser("refresh", help="refresh one protected catalogue lane")
    _add_catalogue_lane_arguments(catalogue_refresh)
    catalogue_actions.add_parser("refresh-all", help="intentionally refresh every protected lane")
    catalogue_upsert = catalogue_actions.add_parser(
        "upsert",
        help="add or update one policy-bounded operator model entry",
    )
    _add_catalogue_lane_arguments(catalogue_upsert)
    catalogue_upsert.add_argument("--model-id", required=True)
    catalogue_upsert.add_argument("--display-label", required=True)
    catalogue_upsert.add_argument("--capabilities-json", default="{}")
    catalogue_deactivate = catalogue_actions.add_parser(
        "deactivate",
        help="deactivate one operator-managed model entry",
    )
    _add_catalogue_lane_arguments(catalogue_deactivate)
    catalogue_deactivate.add_argument("--model-id", required=True)
    return parser


def _add_catalogue_lane_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--runtime-profile-id", required=True)
    parser.add_argument("--option-id", required=True)


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


def _telegram_subject(value: int) -> str:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 2**63 - 1:
        raise WorkshopClientAccessError("Telegram user ID must be a positive signed 64-bit integer")
    return str(value)


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


async def _open_operator_model_catalogue(
    database: Path,
    store: WorkshopEventStore,
) -> tuple[WorkshopModelCatalogueService, WorkshopModelDiscoveryInventoryService]:
    """Open deployed catalogue authority without starting a runtime or model."""
    config = load_config()
    runtime_profiles = WorkshopRuntimeProfileRegistry.load(config)
    execution_state = await WorkshopExecutionStateRegistry.from_store(store, runtime_profiles)
    selected_backends: dict[str, tuple[str, str]] = {}
    selected_models: dict[str, str] = {}
    for namespace in execution_state.namespaces:
        profile = runtime_profiles.resolve(namespace.runtime_profile_id)
        settings = await sessions.read_canonical_execution_settings(database, namespace)
        option_id = settings.get("backend", f"{profile.backend}:{profile.provider}")
        try:
            option = profile.backend_option(option_id)
        except WorkshopRuntimeProfileError as exc:
            raise WorkshopOperatorCommandError(
                f"Runtime profile {profile.profile_id} has an invalid canonical backend selection"
            ) from exc
        selected_backends[str(profile.profile_id)] = (option.backend, option.provider)
        selected_models[str(profile.profile_id)] = settings.get("model", option.model)

    backend_registry = load_backend_registry()
    if not backend_registry:
        raise WorkshopOperatorCommandError("The deployed backend registry is unavailable")
    service_os_user = pwd.getpwuid(os.geteuid()).pw_name
    inventory = WorkshopModelDiscoveryInventoryService(
        config=config,
        runtime_profiles=runtime_profiles,
        execution_state=execution_state,
        backend_registry=backend_registry,
        selected_backend=lambda profile_id: selected_backends[str(profile_id)],
        service_os_user=service_os_user,
    )

    async def selected_model(profile_id: object) -> str:
        return selected_models[str(profile_id)]

    catalogue_store = await WorkshopEventStore.open(database)
    catalogue = WorkshopModelCatalogueService(
        catalogue_store,
        inventory,
        selected_model=selected_model,
        curated_models=lambda lane: models_for_backend_policy(
            lane.backend,
            lane.provider,
            allowed_models=lane.allowed_models,
        ),
        adapters={
            "claude": ClaudeModelDiscoveryAdapter(),
            "codex": CodexModelDiscoveryAdapter(service_os_user=service_os_user),
            "goose": GooseModelDiscoveryAdapter(service_os_user=service_os_user),
            "opencode": OpenCodeModelDiscoveryAdapter(service_os_user=service_os_user),
            "pi": PiModelDiscoveryAdapter(service_os_user=service_os_user),
        },
    )
    return catalogue, inventory


def _catalogue_lane(
    inventory: WorkshopModelDiscoveryInventoryService,
    runtime_profile_id: str,
    option_id: str,
) -> ModelDiscoveryBackendInventory:
    for profile in inventory.inventories:
        if str(profile.runtime_profile_id) != runtime_profile_id:
            continue
        for lane in profile.backends:
            if lane.option_id == option_id.strip().lower():
                return lane
    raise WorkshopOperatorCommandError("The requested model catalogue lane does not exist")


async def _require_matching_catalogue_environment(
    store: WorkshopEventStore,
    lane: ModelDiscoveryBackendInventory,
) -> None:
    """Refuse refresh when the CLI lacks the deployed auth environment."""
    async with store.connection.execute(
        "SELECT cache_key FROM workshop_model_catalogue_refreshes "
        "WHERE runtime_profile_id = ? AND backend = ? AND provider = ? AND active = 1",
        (lane.cache_inputs.runtime_profile_id, lane.backend, lane.provider),
    ) as cursor:
        row = await cursor.fetchone()
    if row is not None and str(row[0]) != lane.cache_key:
        raise WorkshopOperatorCommandError(
            "The CLI environment does not match the running service's non-secret catalogue context; "
            "run the command with the deployed Kai environment instead of invalidating its cache"
        )


def _print_catalogue_refresh(result: ModelCatalogueRefreshResult) -> None:
    print(f"Runtime profile: {result.runtime_profile_id}")
    print(f"Backend option: {result.option_id}")
    print(f"Status: {result.status.value}")
    print(f"Generation: {result.generation}")
    print(f"Discovered models: {result.discovered_models}")
    print(f"Last-known-good preserved: {'yes' if result.preserved_last_known_good else 'no'}")


async def _run_model_catalogue(
    args: argparse.Namespace,
    database: Path,
    store: WorkshopEventStore,
) -> int:
    catalogue, inventory = await _open_operator_model_catalogue(database, store)
    authority = catalogue.operator_authority()
    try:
        if args.action == "status":
            snapshots = await catalogue.inspect_all_as_operator(authority)
            diagnostics = inventory.operator_diagnostics
            refreshed = sum(snapshot.refresh is not None for snapshot in snapshots)
            stale = sum(snapshot.stale for snapshot in snapshots)
            failed = sum(
                snapshot.refresh is not None and snapshot.refresh.status.value not in {"succeeded", "refreshing"}
                for snapshot in snapshots
            )
            print(
                "Model catalogue: "
                f"profiles={diagnostics.profiles}, contexts={diagnostics.options}, "
                f"refreshed={refreshed}, stale={stale}, failed={failed}"
            )
            print(
                "Discovery readiness: "
                f"selected={diagnostics.selected}, selectable={diagnostics.selectable}, "
                f"unavailable={diagnostics.unavailable}, misconfigured={diagnostics.misconfigured}"
            )
            print("Discovery invoked: no")
            return 0

        lane = None
        if args.action != "refresh-all":
            lane = _catalogue_lane(inventory, args.runtime_profile_id, args.option_id)
        if args.action == "list":
            assert lane is not None
            snapshot = await catalogue.inspect_as_operator(
                authority,
                lane.cache_inputs.runtime_profile_id,
                lane.option_id,
            )
            print(f"Runtime profile: {snapshot.runtime_profile_id}")
            print(f"Backend option: {snapshot.option_id}")
            print(f"Refresh status: {snapshot.refresh.status.value if snapshot.refresh else 'not refreshed'}")
            print(f"Stale: {'yes' if snapshot.stale else 'no'}")
            for entry in snapshot.entries:
                sources = ",".join(item.source for item in entry.provenances)
                print(
                    f"{entry.model_id}\t{entry.status.value}\t"
                    f"{'selectable' if entry.selectable else 'retained-only'}\t{sources}"
                )
            return 0
        if args.action == "refresh":
            assert lane is not None
            await _require_matching_catalogue_environment(store, lane)
            _print_catalogue_refresh(
                await catalogue.refresh_as_operator(
                    authority,
                    lane.cache_inputs.runtime_profile_id,
                    lane.option_id,
                )
            )
            return 0
        if args.action == "refresh-all":
            for profile in inventory.inventories:
                for candidate in profile.backends:
                    await _require_matching_catalogue_environment(store, candidate)
            results = await catalogue.refresh_all(authority)
            for index, result in enumerate(results):
                if index:
                    print()
                _print_catalogue_refresh(result)
            return 0 if all(result.status.value == "succeeded" for result in results) else 2
        assert lane is not None
        if args.action == "upsert":
            try:
                capabilities = json.loads(args.capabilities_json)
            except json.JSONDecodeError as exc:
                raise WorkshopOperatorCommandError("Capabilities must be valid JSON") from exc
            if not isinstance(capabilities, dict):
                raise WorkshopOperatorCommandError("Capabilities must be a JSON object")
            await catalogue.upsert_operator_entry(
                authority,
                lane.cache_inputs.runtime_profile_id,
                lane.option_id,
                model_id=args.model_id,
                display_label=args.display_label,
                capabilities=capabilities,
            )
            print(f"Runtime profile: {lane.cache_inputs.runtime_profile_id}")
            print(f"Backend option: {lane.option_id}")
            print(f"Model: {args.model_id}")
            print("Operator entry: active (created or updated)")
            return 0
        deactivated = await catalogue.deactivate_operator_entry(
            authority,
            lane.cache_inputs.runtime_profile_id,
            lane.option_id,
            model_id=args.model_id,
        )
        print(f"Runtime profile: {lane.cache_inputs.runtime_profile_id}")
        print(f"Backend option: {lane.option_id}")
        print(f"Model: {args.model_id}")
        print(f"Operator entry: {'deactivated' if deactivated else 'already inactive or absent'}")
        return 0
    finally:
        await catalogue.close()


async def _run(args: argparse.Namespace) -> int:
    database = _deployed_database(DATA_DIR)
    store = await WorkshopEventStore.open(database)
    try:
        if args.command == "model-catalogue":
            return await _run_model_catalogue(args, database, store)
        if args.command == "integration-route":
            service = WorkshopIntegrationNotificationService(
                store,
                WorkshopDeliveryBindingPolicy.disabled(),
            )
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
                    subject = _telegram_subject(args.telegram_user_id)
                    principal_id, channel_id = await access.resolve_external_direct_human(
                        provider="telegram",
                        external_subject=subject,
                        transport="telegram",
                        external_channel_id=subject,
                    )
                    issued = await access.issue_enrollment(principal_id, channel_id)
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
                    subject = _telegram_subject(args.telegram_user_id)
                    principal_id, _ = await access.resolve_external_direct_human(
                        provider="telegram",
                        external_subject=subject,
                        transport="telegram",
                        external_channel_id=subject,
                    )
                    await access.revoke_device(principal_id, device_id)
                print(f"Device: {device_id}")
                print("Status: revoked (all device sessions revoked)")
                return 0
            grant_id = _enrollment_grant_id(args.grant_id)
            if args.principal_id is not None:
                await access.revoke_enrollment(_principal_id(args.principal_id), grant_id)
            else:
                subject = _telegram_subject(args.telegram_user_id)
                principal_id, _ = await access.resolve_external_direct_human(
                    provider="telegram",
                    external_subject=subject,
                    transport="telegram",
                    external_channel_id=subject,
                )
                await access.revoke_enrollment(principal_id, grant_id)
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
    except ModelCatalogueError as exc:
        raise SystemExit(f"Workshop model catalogue failed: {exc}") from exc
    except IntegrationNotificationError as exc:
        raise SystemExit(f"Workshop integration route failed: {exc}") from exc
    except CanonicalTranscriptExportError as exc:
        raise SystemExit(f"Workshop transcript export failed: {exc}") from exc
    except WorkshopOperatorCommandError as exc:
        raise SystemExit(f"Workshop operator command failed: {exc}") from exc
    raise SystemExit(code)
