from pathlib import Path

import pytest

from kai.backend import USER_MESSAGE_MARKER, assemble_turn_context, build_session_context
from kai.config import WorkspaceConfig
from kai.context_authority import (
    BACKEND_CONTEXT_CONTRACTS,
    CONTEXT_AUTHORITY_CONTRACT,
    ContextProtocolRole,
    NativeInstructionPolicy,
    NativeInstructionSource,
)
from kai.workshop.context_manifests import (
    ContextAuthorityClass,
    ContextDeliveryRole,
    ContextOwnerKind,
    ContextRefreshClass,
    ContextSourceDescriptor,
    ContextSourceKind,
    ContextSourceState,
    ContextTrustClass,
)


def test_every_conversational_backend_has_one_common_protocol_contract():
    assert set(BACKEND_CONTEXT_CONTRACTS) == {"claude", "codex", "goose", "opencode", "pi"}
    assert {
        (contract.kai_controlled_role, contract.current_input_role, contract.fallback)
        for contract in BACKEND_CONTEXT_CONTRACTS.values()
    } == {
        (
            ContextProtocolRole.KAI_CONTEXT,
            ContextProtocolRole.USER_INPUT,
            "labelled_native_user_context",
        )
    }
    assert BACKEND_CONTEXT_CONTRACTS["codex"].native_instruction_policy is (
        NativeInstructionPolicy.PROVIDER_GLOBAL_ONLY
    )
    assert all(
        contract.native_instruction_policy is NativeInstructionPolicy.DISABLED
        for backend, contract in BACKEND_CONTEXT_CONTRACTS.items()
        if backend != "codex"
    )


@pytest.mark.asyncio
async def test_equivalent_canonical_fixture_has_identical_shape_on_all_backends():
    session = "\n\n".join(
        (
            CONTEXT_AUTHORITY_CONTRACT,
            "[Your principal policy and instructions:]\nPRINCIPAL",
            "## Workspace Instructions\n\nWORKSPACE",
            "[Conversation context:]\nHISTORY",
        )
    )
    outputs = []
    for backend, contract in BACKEND_CONTEXT_CONTRACTS.items():
        outputs.append(
            await assemble_turn_context(
                "CURRENT",
                chat_id=None,
                session_context=session,
                agent_definition_context="[Agent definition:]\nAGENT",
                collaboration_context="[Attempt authority:]\nATTEMPT",
                backend_name=backend,
                ambient_context_discovery_enabled=(
                    contract.native_instruction_policy is not NativeInstructionPolicy.DISABLED
                ),
                native_instruction_policy=contract.native_instruction_policy,
            )
        )
    assert len(set(outputs)) == 1
    rendered = outputs[0]
    assert isinstance(rendered, str)
    assert rendered.startswith(CONTEXT_AUTHORITY_CONTRACT)
    assert rendered.endswith(f"{USER_MESSAGE_MARKER}\n\nCURRENT")


def test_explicit_workspace_policy_is_observed_and_principal_precedence_is_declared(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    revisions: list[str | None] = []
    rendered = build_session_context(
        workspace=workspace,
        home_workspace=workspace,
        api=type("Api", (), {"webhook_secret": "", "webhook_port": 8080, "services_info": []})(),
        workspace_config=WorkspaceConfig(path=workspace, system_prompt="WORKSPACE POLICY"),
        chat_id=None,
        data_dir=tmp_path / "data",
        workspace_policy_observer=revisions.append,
    )
    assert rendered.startswith(CONTEXT_AUTHORITY_CONTRACT)
    assert "## Workspace Instructions\n\nWORKSPACE POLICY" in rendered
    assert len(revisions) == 1 and revisions[0] is not None


def test_provider_native_manifest_round_trip_contains_only_redacted_path():
    path = Path("/Users/daniel/.codex/AGENTS.md")
    native = NativeInstructionSource.redacted(
        path,
        scope="provider_global",
    )
    descriptor = ContextSourceDescriptor(
        kind=ContextSourceKind.PROVIDER_NATIVE,
        owner_kind=ContextOwnerKind.PROVIDER,
        owner_id="openai",
        scope="provider",
        trust_class=ContextTrustClass.PROVIDER_CONTROLLED,
        authority_class=ContextAuthorityClass.PROVIDER,
        refresh_class=ContextRefreshClass.PROVIDER_CONTROLLED,
        delivery_role=ContextDeliveryRole.PROVIDER_NATIVE,
        state=ContextSourceState.PROVIDER_CONTROLLED,
        reason="provider_global_sources_admitted",
        delivery_shape="allowlisted_provider_global",
        native_instruction_sources=(native,),
    )
    payload = descriptor.payload()
    assert str(path) not in repr(payload)
    assert ContextSourceDescriptor.from_payload(payload) == descriptor
