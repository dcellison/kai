"""Principal-scoped opt-in task routing and immutable run decisions."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from kai.workshop.agent_definitions import load_agent_definition_revision
from kai.workshop.domain import ChannelId, RunId, RuntimeProfileId
from kai.workshop.routing_eligibility import (
    RoutingEligibilityAuthority,
    RoutingTaskClass,
    RuntimeCapability,
    RuntimeEligibilityCandidate,
    RuntimeEligibilityReport,
    WorkshopRoutingEligibilityService,
)
from kai.workshop.run_execution_authority import RunExecutionSelection
from kai.workshop.run_lifecycle import DurableRun
from kai.workshop.store import WorkshopEventStore


class RoutingPolicyError(RuntimeError):
    """A routing policy or durable decision is invalid."""


class RoutingPolicyConflictError(RoutingPolicyError):
    """A policy revision changed before a requested mutation."""


class RoutingFallback(StrEnum):
    SELECTED = "selected"
    FAIL_CLOSED = "fail_closed"


class RoutingDecisionDisposition(StrEnum):
    SELECTED_DEFAULT = "selected_default"
    ROUTED = "routed"
    FALLBACK_SELECTED = "fallback_selected"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class RoutingPolicyEntry:
    task_class: RoutingTaskClass
    backend_option_id: str | None
    fallback: RoutingFallback
    revision: int

    @property
    def enabled(self) -> bool:
        return self.backend_option_id is not None


@dataclass(frozen=True, slots=True)
class RoutingPolicySnapshot:
    version: int
    authority: RoutingEligibilityAuthority
    entries: tuple[RoutingPolicyEntry, ...]
    authorized_options: dict[RoutingTaskClass, tuple[str, ...]]
    eligible_options: dict[RoutingTaskClass, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class RunRoutingDecision:
    run_id: RunId
    runtime_profile_id: RuntimeProfileId
    requested_task_class: RoutingTaskClass | None
    requested_backend_option_id: str | None
    selected_backend_option_id: str | None
    disposition: RoutingDecisionDisposition
    reason_code: str
    policy_revision: int | None
    selection: RunExecutionSelection
    evidence_version: int
    decided_at: datetime

    @property
    def rejected(self) -> bool:
        return self.disposition == RoutingDecisionDisposition.REJECTED


class WorkshopRoutingPolicyService:
    """Own opt-in policy and one immutable routing decision per run."""

    def __init__(
        self,
        store: WorkshopEventStore,
        eligibility: WorkshopRoutingEligibilityService,
        database_lock: asyncio.Lock,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._store = store
        self._eligibility = eligibility
        self._database_lock = database_lock
        self._clock = clock or (lambda: datetime.now(UTC))

    def authority_for_principal_channel(self, principal_id, channel_id) -> RoutingEligibilityAuthority:
        return self._eligibility.authority_for_principal_channel(principal_id, channel_id)

    async def inspect(self, authority: RoutingEligibilityAuthority) -> RoutingPolicySnapshot:
        reports = {
            task_class: await self._eligibility.inspect(authority, task_class) for task_class in RoutingTaskClass
        }
        async with self._database_lock:
            stored = await self._load_policy_entries(authority.runtime_profile_id)
        entries = tuple(
            stored.get(task_class) or RoutingPolicyEntry(task_class, None, RoutingFallback.SELECTED, 0)
            for task_class in RoutingTaskClass
        )
        return RoutingPolicySnapshot(
            version=1,
            authority=authority,
            entries=entries,
            authorized_options={
                task_class: tuple(candidate.option_id for candidate in report.candidates)
                for task_class, report in reports.items()
            },
            eligible_options={
                task_class: tuple(candidate.option_id for candidate in report.candidates if candidate.eligible)
                for task_class, report in reports.items()
            },
        )

    async def update(
        self,
        authority: RoutingEligibilityAuthority,
        *,
        task_class: RoutingTaskClass,
        backend_option_id: str | None,
        fallback: RoutingFallback,
        expected_revision: int,
    ) -> RoutingPolicySnapshot:
        if isinstance(expected_revision, bool) or expected_revision < 0:
            raise RoutingPolicyError("expected_revision must be a non-negative integer")
        option = backend_option_id.strip().lower() if backend_option_id is not None else None
        if option == "":
            option = None
        report = await self._eligibility.inspect(authority, task_class)
        authorized = {candidate.option_id for candidate in report.candidates}
        if option is not None and option not in authorized:
            raise RoutingPolicyError("Routing target is not authorized for this principal")
        when = self._timestamp(self._clock())
        async with self._database_lock:
            current = (await self._load_policy_entries(authority.runtime_profile_id)).get(task_class)
            current_revision = current.revision if current is not None else 0
            if current_revision != expected_revision:
                raise RoutingPolicyConflictError("Routing policy changed; reload before saving")
            revision = current_revision + 1
            await self._store.connection.execute(
                "INSERT INTO workshop_routing_policies ("
                "runtime_profile_id, task_class, backend_option_id, fallback, revision, updated_at"
                ") VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(runtime_profile_id, task_class) DO UPDATE SET "
                "backend_option_id=excluded.backend_option_id, fallback=excluded.fallback, "
                "revision=excluded.revision, updated_at=excluded.updated_at",
                (
                    authority.runtime_profile_id,
                    task_class.value,
                    option,
                    fallback.value,
                    revision,
                    when,
                ),
            )
            await self._store.connection.commit()
        return await self.inspect(authority)

    async def decide_for_run(
        self,
        run: DurableRun,
        runtime_profile_id: RuntimeProfileId,
    ) -> RunRoutingDecision:
        """Load or create a decision while the execution owner holds its DB lock."""
        existing = await self.load_decision(run.run_id)
        if existing is not None:
            if existing.runtime_profile_id != runtime_profile_id:
                raise RoutingPolicyError("Stored routing decision has a conflicting runtime profile")
            return existing

        requested_task = await self._requested_task_class(run)
        async with self._store.connection.execute(
            "SELECT kind FROM channels WHERE id = ? AND workshop_id = ?",
            (run.channel_id, run.workshop_id),
        ) as cursor:
            channel_row = await cursor.fetchone()
        if channel_row is None:
            raise RoutingPolicyError("Run channel is unavailable")
        if run.sponsor_principal_id is None:
            raise RoutingPolicyError("Agent run has no owner runtime sponsor")
        async with self._store.connection.execute(
            "SELECT owner_direct_channel_id FROM agent_definitions "
            "WHERE agent_id = ? AND owner_principal_id = ? "
            "AND owner_runtime_profile_id = ? AND lifecycle_state = 'active'",
            (run.agent_id, run.sponsor_principal_id, runtime_profile_id),
        ) as cursor:
            owner_row = await cursor.fetchone()
        owner_channel_id = (
            ChannelId(str(owner_row[0])) if owner_row is not None and owner_row[0] is not None else run.channel_id
        )
        authority = self._eligibility.authority_for_sponsored_channel(
            run.sponsor_principal_id,
            owner_channel_id,
            run.agent_id,
            runtime_profile_id,
        )
        shared_channel = True
        if authority.runtime_profile_id != runtime_profile_id:
            raise RoutingPolicyError("Run does not match canonical routing profile")
        if authority.agent_id != run.agent_id:
            raise RoutingPolicyError("Run does not match canonical routing authority")
        agent_requirements: tuple[RuntimeCapability, ...] = ()
        if run.agent_definition_revision_id is not None:
            revision = await load_agent_definition_revision(self._store, run.agent_definition_revision_id)
            if revision is None or revision.agent_id != run.agent_id:
                raise RoutingPolicyError("Run agent definition revision is unavailable")
            agent_requirements = tuple(
                RuntimeCapability(value)
                for value in revision.capabilities
                if value in {capability.value for capability in RuntimeCapability}
            )

        if requested_task is None:
            report = await self._inspect_run_authority(
                authority,
                RoutingTaskClass.CONVERSATION,
                additional_required=agent_requirements,
                shared_channel=shared_channel,
            )
            selected = self._selected_candidate(report)
            decision = self._decision(
                run,
                runtime_profile_id,
                requested_task=None,
                requested_option=None,
                selected_option=selected.option_id,
                disposition=RoutingDecisionDisposition.SELECTED_DEFAULT,
                reason_code="task_class_not_requested",
                policy_revision=None,
                candidate=selected,
                evidence_version=report.version,
            )
            return await self._store_decision(decision)

        report = await self._inspect_run_authority(
            authority,
            requested_task,
            additional_required=agent_requirements,
            shared_channel=shared_channel,
        )
        policy = (await self._load_policy_entries(runtime_profile_id)).get(requested_task)
        if policy is None or policy.backend_option_id is None:
            selected = self._selected_candidate(report)
            decision = self._decision(
                run,
                runtime_profile_id,
                requested_task=requested_task,
                requested_option=None,
                selected_option=None,
                disposition=RoutingDecisionDisposition.REJECTED,
                reason_code="routing_not_enabled",
                policy_revision=policy.revision if policy is not None else None,
                candidate=selected,
                evidence_version=report.version,
            )
            return await self._store_decision(decision)

        target = next(
            (candidate for candidate in report.candidates if candidate.option_id == policy.backend_option_id),
            None,
        )
        if target is None:
            raise RoutingPolicyError("Stored routing policy references an unauthorized backend option")
        if target.eligible:
            decision = self._decision(
                run,
                runtime_profile_id,
                requested_task=requested_task,
                requested_option=target.option_id,
                selected_option=target.option_id,
                disposition=RoutingDecisionDisposition.ROUTED,
                reason_code="configured_route_eligible",
                policy_revision=policy.revision,
                candidate=target,
                evidence_version=report.version,
            )
            return await self._store_decision(decision)

        if policy.fallback == RoutingFallback.SELECTED:
            selected = self._selected_candidate(report)
            if selected.eligible:
                decision = self._decision(
                    run,
                    runtime_profile_id,
                    requested_task=requested_task,
                    requested_option=target.option_id,
                    selected_option=selected.option_id,
                    disposition=RoutingDecisionDisposition.FALLBACK_SELECTED,
                    reason_code="configured_route_ineligible",
                    policy_revision=policy.revision,
                    candidate=selected,
                    evidence_version=report.version,
                )
                return await self._store_decision(decision)
            rejection_reason = "selected_fallback_ineligible"
        else:
            rejection_reason = self._rejection_reason(target)
        decision = self._decision(
            run,
            runtime_profile_id,
            requested_task=requested_task,
            requested_option=target.option_id,
            selected_option=None,
            disposition=RoutingDecisionDisposition.REJECTED,
            reason_code=rejection_reason,
            policy_revision=policy.revision,
            candidate=target,
            evidence_version=report.version,
        )
        return await self._store_decision(decision)

    async def _inspect_run_authority(
        self,
        authority: RoutingEligibilityAuthority,
        task_class: RoutingTaskClass,
        *,
        additional_required: tuple[RuntimeCapability, ...],
        shared_channel: bool,
    ) -> RuntimeEligibilityReport:
        if shared_channel:
            return await self._eligibility.inspect_sponsored_channel(
                authority,
                task_class,
                additional_required=additional_required,
            )
        return await self._eligibility.inspect(
            authority,
            task_class,
            additional_required=additional_required,
        )

    async def load_decision(self, run_id: RunId) -> RunRoutingDecision | None:
        async with self._store.connection.execute(
            "SELECT run_id, runtime_profile_id, requested_task_class, "
            "requested_backend_option_id, selected_backend_option_id, disposition, reason_code, "
            "policy_revision, backend, provider, model, evidence_version, decided_at "
            "FROM workshop_run_routing_decisions WHERE run_id = ?",
            (run_id,),
        ) as cursor:
            row = await cursor.fetchone()
        return None if row is None else self._decision_from_row(row)

    async def _load_policy_entries(
        self,
        runtime_profile_id: RuntimeProfileId,
    ) -> dict[RoutingTaskClass, RoutingPolicyEntry]:
        async with self._store.connection.execute(
            "SELECT task_class, backend_option_id, fallback, revision "
            "FROM workshop_routing_policies WHERE runtime_profile_id = ? ORDER BY task_class",
            (runtime_profile_id,),
        ) as cursor:
            rows = list(await cursor.fetchall())
        return {
            RoutingTaskClass(str(row[0])): RoutingPolicyEntry(
                RoutingTaskClass(str(row[0])),
                str(row[1]) if row[1] is not None else None,
                RoutingFallback(str(row[2])),
                int(row[3]),
            )
            for row in rows
        }

    async def _requested_task_class(self, run: DurableRun) -> RoutingTaskClass | None:
        async with self._store.connection.execute(
            "SELECT e.metadata_json FROM messages m "
            "JOIN event_log e ON e.position = m.created_event_position WHERE m.id = ?",
            (run.inbound_message_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            raise RoutingPolicyError("Run inbound message is unavailable")
        metadata = json.loads(str(row[0]))
        value = metadata.get("routing_task_class")
        return None if value is None else RoutingTaskClass(str(value))

    async def _store_decision(self, decision: RunRoutingDecision) -> RunRoutingDecision:
        await self._store.connection.execute(
            "INSERT OR IGNORE INTO workshop_run_routing_decisions ("
            "run_id, runtime_profile_id, requested_task_class, requested_backend_option_id, "
            "selected_backend_option_id, disposition, reason_code, policy_revision, backend, "
            "provider, model, evidence_version, decided_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                decision.run_id,
                decision.runtime_profile_id,
                decision.requested_task_class.value if decision.requested_task_class else None,
                decision.requested_backend_option_id,
                decision.selected_backend_option_id,
                decision.disposition.value,
                decision.reason_code,
                decision.policy_revision,
                decision.selection.backend,
                decision.selection.provider,
                decision.selection.model,
                decision.evidence_version,
                self._timestamp(decision.decided_at),
            ),
        )
        await self._store.connection.commit()
        stored = await self.load_decision(decision.run_id)
        if stored is None:
            raise RoutingPolicyError("Routing decision was not stored")
        if stored != decision:
            raise RoutingPolicyError("Stored routing decision conflicts with the requested decision")
        return stored

    def _decision(
        self,
        run: DurableRun,
        runtime_profile_id: RuntimeProfileId,
        *,
        requested_task: RoutingTaskClass | None,
        requested_option: str | None,
        selected_option: str | None,
        disposition: RoutingDecisionDisposition,
        reason_code: str,
        policy_revision: int | None,
        candidate: RuntimeEligibilityCandidate,
        evidence_version: int,
    ) -> RunRoutingDecision:
        return RunRoutingDecision(
            run_id=run.run_id,
            runtime_profile_id=runtime_profile_id,
            requested_task_class=requested_task,
            requested_backend_option_id=requested_option,
            selected_backend_option_id=selected_option,
            disposition=disposition,
            reason_code=reason_code,
            policy_revision=policy_revision,
            selection=RunExecutionSelection(
                backend=candidate.backend,
                provider=candidate.provider or None,
                model=candidate.model_id,
            ),
            evidence_version=evidence_version,
            decided_at=self._now(),
        )

    @staticmethod
    def _selected_candidate(report: RuntimeEligibilityReport) -> RuntimeEligibilityCandidate:
        selected = tuple(candidate for candidate in report.candidates if candidate.selected)
        if len(selected) != 1:
            raise RoutingPolicyError("Routing inventory does not identify exactly one selected backend")
        return selected[0]

    @staticmethod
    def _rejection_reason(candidate: RuntimeEligibilityCandidate) -> str:
        for reason in candidate.reasons:
            if reason.code != "eligible" and reason.code != "runtime_unverified":
                return reason.code
        return "configured_route_ineligible"

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise RoutingPolicyError("Routing clock must be timezone-aware")
        return value.astimezone(UTC)

    @staticmethod
    def _timestamp(value: datetime) -> str:
        if value.tzinfo is None or value.utcoffset() is None:
            raise RoutingPolicyError("Routing timestamp must be timezone-aware")
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _decision_from_row(row) -> RunRoutingDecision:
        return RunRoutingDecision(
            run_id=RunId(str(row[0])),
            runtime_profile_id=RuntimeProfileId(str(row[1])),
            requested_task_class=RoutingTaskClass(str(row[2])) if row[2] is not None else None,
            requested_backend_option_id=str(row[3]) if row[3] is not None else None,
            selected_backend_option_id=str(row[4]) if row[4] is not None else None,
            disposition=RoutingDecisionDisposition(str(row[5])),
            reason_code=str(row[6]),
            policy_revision=int(row[7]) if row[7] is not None else None,
            selection=RunExecutionSelection(
                backend=str(row[8]),
                provider=str(row[9]) if row[9] is not None else None,
                model=str(row[10]),
            ),
            evidence_version=int(row[11]),
            decided_at=datetime.fromisoformat(str(row[12]).replace("Z", "+00:00")).astimezone(UTC),
        )
