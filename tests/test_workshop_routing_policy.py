from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from kai.workshop.bootstrap import (
    BootstrapHuman,
    bootstrap_default_workshop,
    bootstrap_human_principal_id,
)
from kai.workshop.conversation_commands import WorkshopConversationCommandService
from kai.workshop.domain import ChannelId
from kai.workshop.inbound import ClientInboundMessage
from kai.workshop.routing_eligibility import (
    CapabilityAssessment,
    CapabilitySupport,
    EligibilityReason,
    RoutingEligibilityAuthority,
    RoutingTaskClass,
    RuntimeCapability,
    RuntimeEligibilityCandidate,
    RuntimeEligibilityReport,
)
from kai.workshop.routing_policy import (
    RoutingDecisionDisposition,
    RoutingFallback,
    RoutingPolicyConflictError,
    WorkshopRoutingPolicyService,
)
from kai.workshop.store import WorkshopEventStore
from tests.workshop_profiles import profile_id

_NOW = datetime(2026, 8, 29, 15, 0, tzinfo=UTC)


def _candidate(option_id: str, *, selected: bool, eligible: bool) -> RuntimeEligibilityCandidate:
    backend, provider = option_id.split(":", 1)
    return RuntimeEligibilityCandidate(
        option_id=option_id,
        backend=backend,
        provider=provider,
        allowed_services=(),
        model_id=f"{backend}-model",
        model_source="current_selection" if selected else "protected_default",
        selected=selected,
        eligible=eligible,
        capabilities=(
            CapabilityAssessment(
                RuntimeCapability.TEXT_GENERATION,
                CapabilitySupport.SUPPORTED if eligible else CapabilitySupport.UNKNOWN,
                "test evidence",
            ),
        ),
        reasons=(EligibilityReason("eligible" if eligible else "capability_unknown", "test"),),
    )


class _Eligibility:
    def __init__(self, authority: RoutingEligibilityAuthority) -> None:
        self.authority = authority
        self.candidates = (
            _candidate("claude:anthropic", selected=True, eligible=True),
            _candidate("codex:openai", selected=False, eligible=True),
        )

    def authority_for_principal_channel(self, principal_id, channel_id):
        if principal_id != self.authority.principal_id or channel_id != self.authority.channel_id:
            raise RuntimeError("access denied")
        return self.authority

    def authority_for_principal_runtime(self, principal_id, runtime_profile_id):
        raise AssertionError("Run routing must resolve the exact channel lane")

    async def inspect(self, authority, task_class, *, additional_required=()):
        assert not additional_required or additional_required[0].value == "text_generation"
        assert authority == self.authority
        canonical_task = RoutingTaskClass(task_class)
        return RuntimeEligibilityReport(
            version=1,
            task_class=canonical_task,
            required_capabilities=(RuntimeCapability.TEXT_GENERATION,),
            principal_id=authority.principal_id,
            channel_id=authority.channel_id,
            agent_id=authority.agent_id,
            runtime_profile_id=authority.runtime_profile_id,
            workspace="/workspace",
            candidates=self.candidates,
        )


async def _fixture(path: Path, *, task_class: str | None = "coding"):
    store = await WorkshopEventStore.open(path)
    result = await bootstrap_default_workshop(
        store,
        (
            BootstrapHuman(
                display_name="Daniel",
                role="admin",
                transport="telegram",
                external_subject="101",
                external_channel_id="101",
                runtime_profile_id=profile_id(101),
            ),
        ),
    )
    principal_id = bootstrap_human_principal_id(result.workshop_id, "telegram", "101")
    channel_token = hashlib.sha256(b"telegram\x00101").hexdigest()
    channel_id = ChannelId.derived(result.workshop_id, f"direct-channel:{channel_token}")
    accepted = await WorkshopConversationCommandService(store).accept_client(
        ClientInboundMessage(
            principal_id=principal_id,
            channel_id=channel_id,
            client_message_id="routing-policy-test",
            body="Run the explicit task",
            occurred_at=_NOW,
            routing_task_class=task_class,
        )
    )
    authority = RoutingEligibilityAuthority(
        principal_id,
        channel_id,
        result.agent_id,
        profile_id(101),
    )
    eligibility = _Eligibility(authority)
    service = WorkshopRoutingPolicyService(store, eligibility, asyncio.Lock())  # type: ignore[arg-type]
    return store, accepted.run, authority, eligibility, service


@pytest.mark.asyncio
async def test_explicit_route_is_opt_in_durable_and_reused_on_recovery(tmp_path: Path) -> None:
    store, run, authority, eligibility, service = await _fixture(tmp_path / "kai.db")
    try:
        rejected = await service.decide_for_run(run, profile_id(101))
        assert rejected.disposition == RoutingDecisionDisposition.REJECTED
        assert rejected.reason_code == "routing_not_enabled"

        # A decision is immutable even if policy changes after acceptance.
        snapshot = await service.update(
            authority,
            task_class=RoutingTaskClass.CODING,
            backend_option_id="codex:openai",
            fallback=RoutingFallback.FAIL_CLOSED,
            expected_revision=0,
        )
        assert snapshot.entries[1].revision == 1
        assert await service.decide_for_run(run, profile_id(101)) == rejected

        eligibility.candidates = (
            eligibility.candidates[0],
            _candidate("codex:openai", selected=False, eligible=False),
        )
        assert await service.decide_for_run(run, profile_id(101)) == rejected
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_policy_routes_or_uses_explicit_selected_fallback(tmp_path: Path) -> None:
    store, run, authority, _, service = await _fixture(tmp_path / "kai.db")
    try:
        await service.update(
            authority,
            task_class=RoutingTaskClass.CODING,
            backend_option_id="codex:openai",
            fallback=RoutingFallback.SELECTED,
            expected_revision=0,
        )
        routed = await service.decide_for_run(run, profile_id(101))
        assert routed.disposition == RoutingDecisionDisposition.ROUTED
        assert routed.selected_backend_option_id == "codex:openai"

        fallback_store, fallback_run, fallback_authority, restarted_eligibility, restarted = await _fixture(
            tmp_path / "fallback.db",
        )
        try:
            await restarted.update(
                fallback_authority,
                task_class=RoutingTaskClass.CODING,
                backend_option_id="codex:openai",
                fallback=RoutingFallback.SELECTED,
                expected_revision=0,
            )
            restarted_eligibility.candidates = (
                restarted_eligibility.candidates[0],
                _candidate("codex:openai", selected=False, eligible=False),
            )
            fallback = await restarted.decide_for_run(fallback_run, profile_id(101))
            assert fallback.disposition == RoutingDecisionDisposition.FALLBACK_SELECTED
            assert fallback.selected_backend_option_id == "claude:anthropic"
        finally:
            await fallback_store.close()
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_policy_revision_conflicts_and_default_messages_preserve_selection(tmp_path: Path) -> None:
    store, run, authority, _, service = await _fixture(
        tmp_path / "kai.db",
        task_class=None,
    )
    try:
        await service.update(
            authority,
            task_class=RoutingTaskClass.VISION,
            backend_option_id="codex:openai",
            fallback=RoutingFallback.FAIL_CLOSED,
            expected_revision=0,
        )
        with pytest.raises(RoutingPolicyConflictError):
            await service.update(
                authority,
                task_class=RoutingTaskClass.VISION,
                backend_option_id=None,
                fallback=RoutingFallback.SELECTED,
                expected_revision=0,
            )
        disabled = await service.update(
            authority,
            task_class=RoutingTaskClass.VISION,
            backend_option_id=None,
            fallback=RoutingFallback.SELECTED,
            expected_revision=1,
        )
        assert disabled.entries[2].backend_option_id is None
        assert disabled.entries[2].revision == 2
        with pytest.raises(RoutingPolicyConflictError):
            await service.update(
                authority,
                task_class=RoutingTaskClass.VISION,
                backend_option_id="codex:openai",
                fallback=RoutingFallback.SELECTED,
                expected_revision=0,
            )
        decision = await service.decide_for_run(run, profile_id(101))
        assert decision.disposition == RoutingDecisionDisposition.SELECTED_DEFAULT
        assert decision.selected_backend_option_id == "claude:anthropic"
    finally:
        await store.close()
