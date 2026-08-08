"""Tests for principal-bound internal API credentials."""

import pytest

from kai.internal_api_auth import InternalAPIAuth, InternalAPIScope


def test_agent_credentials_are_unique_and_resolve_server_side() -> None:
    """Each user receives a distinct token that resolves to only that user."""
    auth = InternalAPIAuth.for_users({123, 456})

    token_123 = auth.agent_credential_for(123)
    token_456 = auth.agent_credential_for(456)
    principal_123 = auth.authenticate(token_123)
    principal_456 = auth.authenticate(token_456)

    assert token_123 != token_456
    assert principal_123 is not None
    assert principal_456 is not None
    assert principal_123.chat_id == 123
    assert principal_456.chat_id == 456
    assert auth.authenticate("not-a-token") is None


def test_persistent_agent_profile_is_explicit_and_excludes_delete_all() -> None:
    """Persistent agents retain normal APIs but cannot wipe all memory."""
    auth = InternalAPIAuth.for_users({123})
    principal = auth.authenticate(auth.agent_credential_for(123))

    assert principal is not None
    assert principal.scopes == frozenset(
        {
            InternalAPIScope.JOBS_READ,
            InternalAPIScope.JOBS_WRITE,
            InternalAPIScope.SERVICES_CALL,
            InternalAPIScope.MESSAGES_SEND,
            InternalAPIScope.FILES_SEND,
            InternalAPIScope.MEMORY_READ,
            InternalAPIScope.MEMORY_ADD,
        }
    )
    assert not principal.allows(InternalAPIScope.MEMORY_DELETE_ALL)


def test_notification_credential_has_only_message_scope() -> None:
    """One-shot notification agents cannot access jobs, files, services, or memory."""
    auth = InternalAPIAuth()
    principal = auth.authenticate(auth.notification_credential_for(-100123))

    assert principal is not None
    assert principal.chat_id == -100123
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
        InternalAPIAuth({123: "same", 456: "same"})
