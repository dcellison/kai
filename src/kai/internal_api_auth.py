"""Process-local authentication and authorization for Kai's internal HTTP API.

Persistent and one-shot agents need a narrow way to call the loopback API, but
must never receive an external webhook signing secret or choose another user's
identity. This module issues random, process-lifetime credentials that resolve
to a server-owned principal and an explicit set of API scopes.
"""

from __future__ import annotations

import hmac
import secrets
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum

from kai.workshop.domain import AgentId, ChannelId, PrincipalId, RuntimeProfileId
from kai.workshop.internal_api_contexts import WorkshopInternalAPIExecutionContext


class InternalAPIScope(StrEnum):
    """Operations an internal API credential may perform."""

    JOBS_READ = "jobs:read"
    JOBS_WRITE = "jobs:write"
    SERVICES_CALL = "services:call"
    MESSAGES_SEND = "messages:send"
    FILES_SEND = "files:send"
    MEMORY_READ = "memory:read"
    MEMORY_ADD = "memory:add"
    MEMORY_DELETE_ALL = "memory:delete-all"


# Keep the persistent-agent base profile explicit. Constructing it from the
# enum would silently grant every future capability to every long-lived agent.
# SERVICES_CALL is added only when that principal has at least one explicitly
# allowed service name.
_PERSISTENT_AGENT_BASE_SCOPES = frozenset(
    {
        InternalAPIScope.JOBS_READ,
        InternalAPIScope.JOBS_WRITE,
        InternalAPIScope.MESSAGES_SEND,
        InternalAPIScope.FILES_SEND,
        InternalAPIScope.MEMORY_READ,
        InternalAPIScope.MEMORY_ADD,
    }
)
_NOTIFICATION_SCOPES = frozenset({InternalAPIScope.MESSAGES_SEND})


@dataclass(frozen=True)
class InternalAPIPrincipal:
    """Identity and authority resolved from an internal API credential."""

    principal_id: PrincipalId
    channel_id: ChannelId
    agent_id: AgentId
    runtime_profile_id: RuntimeProfileId
    scopes: frozenset[InternalAPIScope]
    allowed_services: frozenset[str] = frozenset()

    def allows(self, scope: InternalAPIScope) -> bool:
        """Return whether this principal has the requested API scope."""
        return scope in self.scopes

    def allows_service(self, service_name: str) -> bool:
        """Return whether this principal may call one named service."""
        return self.allows(InternalAPIScope.SERVICES_CALL) and service_name in self.allowed_services


class InternalAPIAuth:
    """Issue and resolve random credentials for internal API principals.

    Credentials live only for the lifetime of the outer Kai process. Persistent
    backends are lazy children of that same process, so they receive the current
    credential whenever they start and do not require an at-rest token store.

    Args:
        agent_credentials: Optional fixed per-context credentials. Production code
            leaves this unset and receives cryptographically random tokens; the
            explicit form keeps direct handler tests deterministic.
        allowed_services_by_profile: Explicit service names each protected
            runtime may call. Omitted profiles receive no service-call scope.
    """

    def __init__(
        self,
        agent_credentials: Mapping[WorkshopInternalAPIExecutionContext, str] | None = None,
        *,
        allowed_services_by_profile: Mapping[RuntimeProfileId, Iterable[str]] | None = None,
    ) -> None:
        self._principals_by_credential: dict[str, InternalAPIPrincipal] = {}
        self._credentials_by_principal: dict[InternalAPIPrincipal, str] = {}
        self._allowed_services_by_profile = {
            profile_id: frozenset(name for name in names if name)
            for profile_id, names in (allowed_services_by_profile or {}).items()
        }

        for context, credential in (agent_credentials or {}).items():
            self._register(
                credential,
                self._agent_principal(context),
            )

    @classmethod
    def for_execution_contexts(
        cls,
        contexts: Iterable[WorkshopInternalAPIExecutionContext],
        *,
        allowed_services_by_profile: Mapping[RuntimeProfileId, Iterable[str]] | None = None,
    ) -> InternalAPIAuth:
        """Create one credential for every canonical execution context."""
        auth = cls(allowed_services_by_profile=allowed_services_by_profile)
        for context in sorted(set(contexts), key=lambda item: item.runtime_profile_id):
            auth.agent_credential_for(context)
        return auth

    def agent_credential_for(self, context: WorkshopInternalAPIExecutionContext) -> str:
        """Return the scoped credential for one canonical agent context."""
        return self._credential_for(self._agent_principal(context))

    def notification_credential_for(self, context: WorkshopInternalAPIExecutionContext) -> str:
        """Return a send-message-only credential for a notification agent."""
        return self._credential_for(self._principal(context, _NOTIFICATION_SCOPES))

    def revoke_agent_context(self, context: WorkshopInternalAPIExecutionContext) -> None:
        """Revoke the process-local credential for one retired execution lane."""
        principal = self._agent_principal(context)
        credential = self._credentials_by_principal.pop(principal, None)
        if credential is not None:
            self._principals_by_credential.pop(credential, None)

    def authenticate(self, credential: str) -> InternalAPIPrincipal | None:
        """Resolve a credential to its server-owned principal, if valid.

        Every candidate is compared with ``hmac.compare_digest``. The store is
        intentionally small (normally one or two credentials per configured
        user), so avoiding a normal dictionary lookup keeps the comparison
        behavior independent of secret prefix or string hash behavior.
        """
        if not credential:
            return None

        matched: InternalAPIPrincipal | None = None
        for expected, principal in self._principals_by_credential.items():
            if hmac.compare_digest(credential, expected):
                matched = principal
        return matched

    def _agent_principal(
        self,
        context: WorkshopInternalAPIExecutionContext,
    ) -> InternalAPIPrincipal:
        """Build the persistent-agent principal for one canonical context."""
        allowed_services = self._allowed_services_by_profile.get(context.runtime_profile_id, frozenset())
        scopes = set(_PERSISTENT_AGENT_BASE_SCOPES)
        if allowed_services:
            scopes.add(InternalAPIScope.SERVICES_CALL)
        return self._principal(
            context,
            frozenset(scopes),
            allowed_services,
        )

    @staticmethod
    def _principal(
        context: WorkshopInternalAPIExecutionContext,
        scopes: frozenset[InternalAPIScope],
        allowed_services: frozenset[str] = frozenset(),
    ) -> InternalAPIPrincipal:
        return InternalAPIPrincipal(
            principal_id=context.principal_id,
            channel_id=context.channel_id,
            agent_id=context.agent_id,
            runtime_profile_id=context.runtime_profile_id,
            scopes=scopes,
            allowed_services=allowed_services,
        )

    def _credential_for(self, principal: InternalAPIPrincipal) -> str:
        """Return an existing credential for a principal or issue a new one."""
        existing = self._credentials_by_principal.get(principal)
        if existing is not None:
            return existing

        while True:
            credential = secrets.token_urlsafe(32)
            if credential not in self._principals_by_credential:
                break
        self._register(credential, principal)
        return credential

    def _register(self, credential: str, principal: InternalAPIPrincipal) -> None:
        """Register one non-empty, unique credential/principal pair."""
        if not credential:
            raise ValueError("Internal API credentials must not be empty")
        if credential in self._principals_by_credential:
            raise ValueError("Internal API credentials must be unique")
        if principal in self._credentials_by_principal:
            raise ValueError("Each internal API principal must have one credential")

        self._principals_by_credential[credential] = principal
        self._credentials_by_principal[principal] = credential
