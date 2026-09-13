"""Shared internal-API contract registry tests."""

from kai import webhook
from kai.backend import ApiContext, render_attempt_capability_guidance, render_persistent_capability_guidance
from kai.internal_api_contracts import INTERNAL_API_CONTRACTS, InternalAPIOperation, contract
from kai.internal_api_scopes import InternalAPIScope


def test_every_internal_handler_declares_the_registry_operation() -> None:
    handlers = {
        InternalAPIOperation.JOB_CREATE: webhook._handle_schedule,
        InternalAPIOperation.JOB_LIST: webhook._handle_get_jobs,
        InternalAPIOperation.JOB_GET: webhook._handle_get_job,
        InternalAPIOperation.JOB_UPDATE: webhook._handle_update_job,
        InternalAPIOperation.JOB_DELETE: webhook._handle_delete_job,
        InternalAPIOperation.SERVICE_CALL: webhook._handle_service_call,
        InternalAPIOperation.MESSAGE_SEND: webhook._handle_send_message,
        InternalAPIOperation.FILE_SEND: webhook._handle_send_file,
        InternalAPIOperation.MEMORY_ADD: webhook._handle_memory_add,
        InternalAPIOperation.MEMORY_SEARCH: webhook._handle_memory_search,
        InternalAPIOperation.MEMORY_STATS: webhook._handle_memory_stats,
        InternalAPIOperation.MEMORY_DELETE_ALL: webhook._handle_memory_delete_all,
        InternalAPIOperation.AGENT_DELEGATE: webhook._handle_agent_delegation,
        InternalAPIOperation.CONTEXT_READ: webhook._handle_collaboration_context,
        InternalAPIOperation.REACTION_SET: webhook._handle_collaboration_reaction,
        InternalAPIOperation.COLLABORATION_MESSAGE: webhook._handle_collaboration_message,
        InternalAPIOperation.COLLABORATION_ARTIFACT: webhook._handle_collaboration_artifact,
    }

    assert set(handlers) == set(INTERNAL_API_CONTRACTS)
    assert all(handler.__kai_internal_api_operation__ is operation for operation, handler in handlers.items())


def test_rendered_contract_contains_the_server_method_path_and_fields() -> None:
    for item in INTERNAL_API_CONTRACTS.values():
        rendered = item.render(8123)
        assert f"{item.method} http://localhost:8123{item.path}" in rendered
        assert all(field in rendered for field in item.required_fields)
        assert all(field in rendered for field in item.optional_fields)


def test_shared_lane_gets_no_persistent_private_api_guidance(tmp_path) -> None:
    api = ApiContext(
        webhook_port=8080,
        webhook_secret="secret",
        scopes=frozenset({InternalAPIScope.COLLABORATION_INVOKE}),
    )

    assert (
        render_persistent_capability_guidance(
            api,
            memory_enabled=True,
            outbox_path=tmp_path / "outbox",
            services_info=[],
        )
        == ""
    )


def test_attempt_guidance_contains_only_effective_collaboration_operations() -> None:
    api = ApiContext(
        webhook_port=8080,
        webhook_secret="secret",
        scopes=frozenset({InternalAPIScope.COLLABORATION_INVOKE}),
    )

    rendered = render_attempt_capability_guidance(api, ("reaction",))

    assert contract(InternalAPIOperation.REACTION_SET).path in rendered
    assert contract(InternalAPIOperation.AGENT_DELEGATE).path not in rendered
    assert contract(InternalAPIOperation.COLLABORATION_MESSAGE).path not in rendered
    assert "X-Kai-Collaboration-Proof" not in rendered


def test_shared_publication_endpoint_describes_only_the_granted_message_kind() -> None:
    api = ApiContext(
        webhook_port=8080,
        webhook_secret="secret",
        scopes=frozenset({InternalAPIScope.COLLABORATION_INVOKE}),
    )

    progress = render_attempt_capability_guidance(api, ("progress_publish",))
    reply = render_attempt_capability_guidance(api, ("thread_reply",))

    assert "kind to 'progress'" in progress
    assert "thread_reply" not in progress
    assert "kind to 'thread_reply'" in reply
    assert "progress" not in reply
