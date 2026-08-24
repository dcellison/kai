"""Tests for principal-bound internal API credentials."""

import pytest

from kai.internal_api_auth import InternalAPIAuth, InternalAPIScope
from kai.workshop.domain import AgentId, ChannelId, PrincipalId
from kai.workshop.internal_api_contexts import WorkshopInternalAPIExecutionContext
from tests.workshop_profiles import profile_id


def _context(runtime_config_id: int) -> WorkshopInternalAPIExecutionContext:
    return WorkshopInternalAPIExecutionContext(
        principal_id=PrincipalId(f"prn_{runtime_config_id:032x}"),
        channel_id=ChannelId(f"chn_{runtime_config_id:032x}"),
        agent_id=AgentId(f"agt_{runtime_config_id:032x}"),
        runtime_profile_id=profile_id(runtime_config_id),
        _runtime_config_id=runtime_config_id,
    )


def test_agent_credentials_are_unique_and_resolve_server_side() -> None:
    """Each user receives a distinct token that resolves to only that user."""
    context_123 = _context(123)
    context_456 = _context(456)
    auth = InternalAPIAuth.for_execution_contexts({context_123, context_456})

    token_123 = auth.agent_credential_for(context_123)
    token_456 = auth.agent_credential_for(context_456)
    principal_123 = auth.authenticate(token_123)
    principal_456 = auth.authenticate(token_456)

    assert token_123 != token_456
    assert principal_123 is not None
    assert principal_456 is not None
    assert principal_123.principal_id == context_123.principal_id
    assert principal_123.channel_id == context_123.channel_id
    assert principal_123.agent_id == context_123.agent_id
    assert principal_123.runtime_profile_id == context_123.runtime_profile_id
    assert principal_456.principal_id == context_456.principal_id
    assert auth.authenticate("not-a-token") is None


def test_persistent_agent_profile_is_explicit_and_excludes_delete_all() -> None:
    """Persistent agents retain normal APIs but cannot wipe all memory."""
    context = _context(123)
    auth = InternalAPIAuth.for_execution_contexts({context})
    principal = auth.authenticate(auth.agent_credential_for(context))

    assert principal is not None
    assert principal.scopes == frozenset(
        {
            InternalAPIScope.JOBS_READ,
            InternalAPIScope.JOBS_WRITE,
            InternalAPIScope.MESSAGES_SEND,
            InternalAPIScope.FILES_SEND,
            InternalAPIScope.MEMORY_READ,
            InternalAPIScope.MEMORY_ADD,
        }
    )
    assert principal.allowed_services == frozenset()
    assert not principal.allows(InternalAPIScope.SERVICES_CALL)
    assert not principal.allows(InternalAPIScope.MEMORY_DELETE_ALL)


def test_persistent_agent_service_scope_requires_explicit_names() -> None:
    """Service scope and resources are issued only from the per-user allowlist."""
    context_123 = _context(123)
    context_456 = _context(456)
    auth = InternalAPIAuth.for_execution_contexts(
        {context_123, context_456},
        allowed_services_by_profile={context_123.runtime_profile_id: {"perplexity", "weather"}},
    )

    principal_123 = auth.authenticate(auth.agent_credential_for(context_123))
    principal_456 = auth.authenticate(auth.agent_credential_for(context_456))

    assert principal_123 is not None
    assert principal_123.allows(InternalAPIScope.SERVICES_CALL)
    assert principal_123.allows_service("perplexity")
    assert principal_123.allows_service("weather")
    assert not principal_123.allows_service("billing")
    assert principal_456 is not None
    assert not principal_456.allows(InternalAPIScope.SERVICES_CALL)
    assert not principal_456.allows_service("perplexity")


def test_notification_credential_has_only_message_scope() -> None:
    """One-shot notification agents cannot access jobs, files, services, or memory."""
    auth = InternalAPIAuth()
    context = _context(123)
    principal = auth.authenticate(auth.notification_credential_for(context))

    assert principal is not None
    assert principal.principal_id == context.principal_id
    assert principal.allowed_services == frozenset()
    assert principal.allows(InternalAPIScope.MESSAGES_SEND)
    assert not principal.allows(InternalAPIScope.JOBS_READ)
    assert not principal.allows(InternalAPIScope.JOBS_WRITE)
    assert not principal.allows(InternalAPIScope.SERVICES_CALL)
    assert not principal.allows(InternalAPIScope.FILES_SEND)
    assert not principal.allows(InternalAPIScope.MEMORY_READ)
    assert not principal.allows(InternalAPIScope.MEMORY_ADD)
    assert not principal.allows(InternalAPIScope.MEMORY_DELETE_ALL)


def test_fixed_test_credentials_must_be_unique() -> None:
    """Ambiguous credential-to-principal mappings are rejected."""
    with pytest.raises(ValueError, match="unique"):
        InternalAPIAuth({_context(123): "same", _context(456): "same"})
